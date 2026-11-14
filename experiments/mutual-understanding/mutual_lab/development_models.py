"""Isolated development encoders; existing research profiles/loaders stay unchanged.

The bidirectional encoder and its width-matched unidirectional control retain
the same text embedding, producer, visual projection and output dimension.
They must be saved in development-only checkpoints by the experiment runner.
Importing parameter descriptions does not require the optional tensor packages.
"""

from dataclasses import asdict

from .profiles import ModelProfile

ENCODERS = ("last", "bidirectional", "wide_last")


def _gru_parameters(inputs: int, hidden: int, layers: int, directions: int = 1) -> int:
    return sum(directions * (3 * hidden * (inputs if layer == 0 else directions * hidden)
                             + 3 * hidden * hidden + 6 * hidden)
               for layer in range(layers))


def _understanding_parameters(profile, vocab_size, hidden, directions):
    return (vocab_size * profile.embedding_dim
            + _gru_parameters(profile.embedding_dim, hidden, profile.layers, directions)
            + profile.hidden_dim * (hidden * directions + 1))


def development_model_spec(profile: ModelProfile, encoder="last", vocab_size=None) -> dict:
    """Describe actual parameter counts and the capacity-control matching rule."""
    profile.validate()
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown development understanding encoder: {encoder}")
    expected_vocabulary = 259 if profile.tokenizer == "utf8" else 65539
    if vocab_size is None:
        vocab_size = expected_vocabulary
    if type(vocab_size) is not int or vocab_size != expected_vocabulary:
        raise ValueError("Vocabulary must match the profile's fixed numeric tokenizer")
    dimension = profile.hidden_dim
    target = _understanding_parameters(profile, vocab_size, dimension, 2)
    hidden = dimension
    directions = 2 if encoder == "bidirectional" else 1
    if encoder == "wide_last":
        # Minimize the exact count difference; ties choose the smaller width.
        hidden = min(range(dimension, 3 * dimension + 1), key=lambda width:
                     (abs(_understanding_parameters(profile, vocab_size, width, 1) - target), width))
    understanding = _understanding_parameters(profile, vocab_size, hidden, directions)
    producer = (vocab_size * profile.embedding_dim
                + _gru_parameters(profile.embedding_dim + dimension, dimension, profile.layers)
                + vocab_size * (dimension + 1))
    embedding = vocab_size * profile.embedding_dim
    return {"schema_version": 1, "kind": "development_encoder_comparison",
            "excluded_from_research": True, "encoder_variant": encoder,
            "base_model_profile": asdict(profile), "vocab_size": vocab_size,
            "embedding_dim": profile.embedding_dim, "output_dim": dimension,
            "recurrent_hidden_per_direction": hidden, "directions": directions,
            "layers": profile.layers,
            "readout": "final_forward_and_backward" if directions == 2 else "final_forward",
            "understanding_parameters": understanding, "producer_parameters": producer,
            "parameter_count": understanding + producer,
            "bidirectional_understanding_target": target,
            "understanding_count_relative_difference_from_bidirectional": abs(understanding - target) / target,
            "nonembedding_count_relative_difference_from_bidirectional":
                abs(understanding - target) / (target - embedding),
            "capacity_matching_rule": "closest exact understanding parameter count; smaller recurrent width breaks ties",
            "shared_initialization": ["understanding.embedding", "producer", "visual_projection"],
            "initialization_source": "random initialization only; no pretrained language or vision weights"}


def build_development_model(profile: ModelProfile, seed: int, encoder="last"):
    """Build a NativeModel-compatible development model with isolated encoder changes.

    ``last`` returns the original implementation without changing any tensors.
    The other variants replace only the recurrent understanding encoder/head.
    Every factory invocation restores the caller's global Torch RNG state.
    """
    specification = development_model_spec(profile, encoder)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Development initialization seed must be an integer in [0, 2**32-1]")
    from .neural import NativeModel, torch, nn, F
    model = NativeModel(profile, seed)
    if encoder != "last":
        class DevelopmentUnderstanding(nn.Module):
            def __init__(self, embedding):
                super().__init__()
                self.embedding = embedding
                self.gru = nn.GRU(profile.embedding_dim,
                                 specification["recurrent_hidden_per_direction"], profile.layers,
                                 batch_first=True, bidirectional=encoder == "bidirectional")
                incoming = specification["recurrent_hidden_per_direction"] * specification["directions"]
                self.head = nn.Linear(incoming, profile.hidden_dim)

            def forward(self, ids, lengths):
                packed = nn.utils.rnn.pack_padded_sequence(self.embedding(ids), lengths,
                                                          batch_first=True, enforce_sorted=False)
                _, hidden = self.gru(packed)
                state = torch.cat((hidden[-2], hidden[-1]), dim=-1) if encoder == "bidirectional" else hidden[-1]
                return F.normalize(self.head(state), dim=-1)

        # The original initialization happens first, preserving exactly the
        # producer, pixels and embedding across the three encoder conditions.
        with torch.random.fork_rng():
            torch.manual_seed((seed + 17011) % (2**32))
            model.understanding = DevelopmentUnderstanding(model.understanding.embedding)
    model.development_spec = specification
    model.development_only = True
    if sum(parameter.numel() for parameter in model.parameters()) != specification["parameter_count"]:
        raise RuntimeError("Development model parameter count differs from its recorded specification")
    return model
