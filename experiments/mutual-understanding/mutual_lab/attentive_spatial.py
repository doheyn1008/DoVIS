"""Conditional development decoder: consult the frozen image grid at each token.

This is an unactivated candidate, not a qualified native or acquisition model.
The base language/visual tensors are retained. Additional decoder parameters
and a changed visual readout are explicit architecture confounds, not a matched
capacity result. Importing the factory does not require tensor dependencies.
"""

from __future__ import annotations

import hashlib
import math


def build_attentive_spatial_model(profile, initialization_seed, bundle, condition="B1"):
    """Build a development-only SpatialNativeModel with post-GRU cross-attention.

    Teacher forcing uses one parallel GRU call. Inference preserves the same
    greedy token loop and trace contract as NativeModel. The zero-initialized
    residual vocabulary projection makes initial logits exactly match the
    original global-vector producer; attention becomes trainable as that
    projection starts learning. No Scene labels enter either path.
    """
    from .neural import torch, nn, F
    from .spatial_prior import build_spatial_model, png_tensor

    model = build_spatial_model(profile, initialization_seed, bundle, condition)
    original = model.producer
    dimension, vocabulary = profile.hidden_dim, model.tokenizer.vocab_size

    class GridProducer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding, self.gru, self.output = original.embedding, original.gru, original.output
            self.layers = original.layers
            self.query = nn.Linear(dimension, dimension, bias=False)
            self.key = nn.Linear(34, dimension, bias=False)
            self.value = nn.Linear(34, dimension, bias=False)
            self.attention_output = nn.Linear(dimension, vocabulary, bias=False)
            nn.init.zeros_(self.attention_output.weight)

        def forward(self, ids, visual, hidden=None, spatial=None):
            if spatial is None:
                raise ValueError("The attention decoder requires its image grid; use teacher_forced_logits")
            if spatial.ndim != 3 or spatial.shape[0] != ids.shape[0] or spatial.shape[1:] != (24, 34):
                raise ValueError("Expected one 24-cell spatial grid per token sequence")
            if hidden is None:
                hidden = visual.unsqueeze(0).repeat(self.layers, 1, 1).contiguous()
            inputs = torch.cat((self.embedding(ids), visual[:, None, :].expand(-1, ids.shape[1], -1)), -1)
            states, hidden = self.gru(inputs, hidden)
            weights = torch.softmax(self.query(states) @ self.key(spatial).transpose(1, 2)
                                    / math.sqrt(dimension), dim=-1)
            context = weights @ self.value(spatial)
            return self.output(states) + self.attention_output(context), hidden

    # Initialize only the new readout in an isolated RNG scope. The inherited
    # embedding, GRU, vocabulary head, understanding path and prior stay exact.
    with torch.random.fork_rng():
        torch.manual_seed((initialization_seed + 90011) % 2**32)
        model.producer = GridProducer()
    coordinates = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 6),
                                             indexing="ij"), dim=-1).reshape(24, 2)
    model.register_buffer("spatial_coordinates", coordinates)
    model.spatial_grid_cache = {}

    class AttentiveSpatialModel(type(model)):
        @torch.no_grad()
        def visual_grid(self, pngs):
            if any(parameter.requires_grad for parameter in self.visual_encoder.parameters()):
                raise RuntimeError("The spatial encoder must remain frozen")
            self.visual_encoder.eval()
            missing = {hashlib.sha256(png).digest(): png for png in pngs
                       if hashlib.sha256(png).digest() not in self.spatial_grid_cache}
            if missing:
                keys = list(missing)
                features = self.visual_encoder.features(png_tensor([missing[key] for key in keys]))
                cells = features.flatten(2).transpose(1, 2)
                grid = torch.cat((cells, self.spatial_coordinates[None].expand(len(keys), -1, -1)), -1)
                self.spatial_grid_cache.update((key, value.detach().clone()) for key, value in zip(keys, grid))
            return torch.stack([self.spatial_grid_cache[hashlib.sha256(png).digest()] for png in pngs])

        def teacher_forced_logits(self, ids, pngs):
            if ids.ndim != 2 or len(pngs) != ids.shape[0]:
                raise ValueError("Token and PNG batches must have the same size")
            logits, _ = self.producer(ids, self.visual(pngs), spatial=self.visual_grid(pngs))
            return logits

        def production_loss(self, examples):
            ids, _ = self.tokenizer.batch([example["text"] for example in examples])
            logits = self.teacher_forced_logits(ids[:, :-1], [example["image"] for example in examples])
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=0)

        @torch.no_grad()
        def generate_with_trace(self, png):
            visual, spatial = self.visual([png]), self.visual_grid([png])
            current, hidden, output = torch.tensor([[1]]), None, []
            terminated = False
            for _ in range(self.tokenizer.maximum - 2):
                logits, hidden = self.producer(current, visual, hidden, spatial=spatial)
                token = int(logits[0, -1].argmax())
                if token == 2:
                    terminated = True
                    break
                output.append(token)
                current = torch.tensor([[token]])
            invalid_encoding = False
            if self.tokenizer.kind == "utf8":
                try:
                    bytes(token - 3 for token in output if token >= 3).decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    invalid_encoding = True
            else:
                invalid_encoding = any(0xD800 <= token - 3 <= 0xDFFF for token in output)
            return {"text": self.tokenizer.decode(output), "token_ids": output, "terminated": terminated,
                    "invalid_encoding": invalid_encoding, "special_tokens_in_body": sum(token < 3 for token in output),
                    "output_units": len(output), "unit_limit": self.tokenizer.maximum - 2}

    # Preserve the constructed model's modules and buffers rather than making
    # another random initialization. Only state_dict tensors are checkpointed.
    model.__class__ = AttentiveSpatialModel
    extra = sum(parameter.numel() for name, parameter in model.producer.named_parameters()
                if name.startswith(("query.", "key.", "value.", "attention_output.")))
    model.development_spec = {**model.development_spec, "kind": "development_spatial_decoder_attention",
        "decoder_variant": "post_gru_grid_cross_attention", "decoder_extra_parameters": extra,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_language_parameters": sum(parameter.numel() for parameter in model.parameters()
                                              if parameter.requires_grad),
        "capacity_matched": False, "initial_residual_vocabulary_projection": "zeros",
        "spatial_keys_values": "24 frozen CNN cells with fixed row/column coordinates; no labels",
        "interpretation": "Decoder visual-readout and parameter-count change; not isolated attention-only capacity evidence"}
    return model
