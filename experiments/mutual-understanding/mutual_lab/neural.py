"""Optional CPU GRU learner: L1-trained understanding is the L2 learning path.

No downloaded weights/tokenizer. Model-facing arguments contain pixels and text only.
The autoregressive L1 producer is a separate network, frozen after native training.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

try:
    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError("Neural runs require: python -m pip install -e '.[neural]'") from exc

from .agents import SharedImage
from .config import AgentConfig, ExperimentConfig
from .images import scene_png
from .profiles import ModelProfile
from .storage import revision, write_json
from .world import make_research_dataset, native_captions, variants


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def digest(value) -> str:
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(str((str(item.dtype), tuple(item.shape))).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, bytes):
            h.update(len(item).to_bytes(8, "big")); h.update(item)
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(str(len(item)).encode())
            for part in item:
                visit(part)
        else:
            h.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\x00")

    visit(value)
    return h.hexdigest()


class Tokenizer:
    """Fixed numeric encoding; never fits on either language's vocabulary."""
    def __init__(self, profile: ModelProfile):
        self.kind = profile.tokenizer
        self.maximum = profile.max_sequence_length
        self.vocab_size = 259 if self.kind == "utf8" else 65539

    def encode(self, text: str) -> list[int]:
        units = list(text.encode("utf-8")) if self.kind == "utf8" else [ord(c) for c in text]
        if self.kind == "unicode-bmp" and any(c > 65535 for c in units):
            raise ValueError("unicode-bmp profile accepts BMP code points only")
        if len(units) + 2 > self.maximum:
            raise ValueError("Sentence exceeds model maximum; truncation is not permitted")
        return [1, *(unit + 3 for unit in units), 2]

    def decode(self, ids: list[int]) -> str:
        units = [i - 3 for i in ids if i >= 3]
        if self.kind == "utf8":
            return bytes(units).decode("utf-8", errors="ignore")
        return "".join(chr(i) for i in units)

    def batch(self, texts: list[str]) -> tuple[torch.Tensor, list[int]]:
        encoded = [self.encode(t) for t in texts]
        result = torch.zeros(len(texts), max(map(len, encoded)), dtype=torch.long)
        for i, ids in enumerate(encoded):
            result[i, :len(ids)] = torch.tensor(ids)
        return result, list(map(len, encoded))


class Understanding(nn.Module):
    def __init__(self, p: ModelProfile, vocab: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, p.embedding_dim, padding_idx=0)
        self.gru = nn.GRU(p.embedding_dim, p.hidden_dim, p.layers, batch_first=True)
        self.head = nn.Linear(p.hidden_dim, p.hidden_dim)

    def forward(self, ids, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(ids), lengths,
                                                  batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return F.normalize(self.head(hidden[-1]), dim=-1)


class Producer(nn.Module):
    def __init__(self, p: ModelProfile, vocab: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, p.embedding_dim, padding_idx=0)
        self.gru = nn.GRU(p.embedding_dim + p.hidden_dim, p.hidden_dim, p.layers, batch_first=True)
        self.output = nn.Linear(p.hidden_dim, vocab)
        self.layers = p.layers

    def forward(self, ids, visual, hidden=None):
        if hidden is None:
            hidden = visual.unsqueeze(0).repeat(self.layers, 1, 1).contiguous()
        inputs = torch.cat((self.embedding(ids), visual[:, None, :].expand(-1, ids.shape[1], -1)), -1)
        states, hidden = self.gru(inputs, hidden)
        return self.output(states), hidden


class NativeModel(nn.Module):
    def __init__(self, profile: ModelProfile, seed: int):
        super().__init__()
        self.tokenizer = Tokenizer(profile)
        # fork_rng restores the caller's global RNG; paired language models start identically.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.understanding = Understanding(profile, self.tokenizer.vocab_size)
            self.producer = Producer(profile, self.tokenizer.vocab_size)
            projection = torch.randn(16 * 12 * 3, profile.hidden_dim) / math.sqrt(16 * 12 * 3)
        self.register_buffer("visual_projection", projection)
        self.visual_cache = {}

    def visual(self, pngs: list[bytes]) -> torch.Tensor:
        vectors = []
        for png in pngs:
            key = hashlib.sha256(png).digest()
            if key not in self.visual_cache:
                with Image.open(io.BytesIO(png)) as image:
                    pixels = np.asarray(image.convert("RGB").resize((16, 12), Image.Resampling.BILINEAR),
                                        dtype=np.float32).copy()
                pixels = (pixels - np.array([247, 249, 252], dtype=np.float32)) / 255
                self.visual_cache[key] = F.normalize(torch.from_numpy(pixels.reshape(-1)) @ self.visual_projection, dim=0)
            vectors.append(self.visual_cache[key])
        return torch.stack(vectors)

    def grounding_loss(self, examples: list[dict]):
        ids, lengths = self.tokenizer.batch([e["text"] for e in examples])
        text = self.understanding(ids, lengths)
        images = self.visual([png for e in examples for png in e["choices"]]).reshape(len(examples), 4, -1)
        logits = torch.einsum("bh,bkh->bk", text, images) / .1
        return F.cross_entropy(logits, torch.tensor([e["answer"] for e in examples]))

    def production_loss(self, examples: list[dict]):
        ids, _ = self.tokenizer.batch([e["text"] for e in examples])
        logits, _ = self.producer(ids[:, :-1], self.visual([e["image"] for e in examples]))
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=0)

    @torch.no_grad()
    def generate(self, png: bytes) -> str:
        visual = self.visual([png])
        current, hidden, output = torch.tensor([[1]]), None, []
        for _ in range(self.tokenizer.maximum - 2):
            logits, hidden = self.producer(current, visual, hidden)
            token = int(logits[0, -1].argmax())
            if token == 2:
                break
            output.append(token)
            current = torch.tensor([[token]])
        return self.tokenizer.decode(output)

    @torch.no_grad()
    def predict(self, text: str, pngs: list[bytes]) -> int:
        ids, lengths = self.tokenizer.batch([text])
        scores = self.understanding(ids, lengths) @ self.visual(pngs).T
        return int(scores[0].argmax())


def native_split_digest(dataset) -> str:
    return digest({"train": sorted(s.key for s in dataset.train),
                   "qualification": sorted(p.scene.key for p in dataset.native_test)})


def checkpoint_provenance(payload: dict) -> dict:
    """Separate the original L1 audit from this checkpoint's actual acquisition state.

    Derived from existing payload fields, so older checkpoints and state digests
    remain compatible. A zero-exposure acquisition snapshot is still resume-only.
    """
    counts = payload.get("counters", {})
    exposures = counts.get("exposures", 0)
    updates = counts.get("updates", 0)
    replay_entries = len(payload.get("foreign_buffer", []))
    return {"native_metadata_scope": "original L1 training only",
            "origin_native_checkpoint_id": payload["metadata"]["native_checkpoint_id"],
            "current_state_kind": payload["kind"], "partner_exposures": exposures,
            "acquisition_optimizer_updates": updates, "foreign_replay_entries": replay_entries,
            "eligible_for_fresh_experiment": payload["kind"] == "native" and
                exposures == 0 and updates == 0 and replay_entries == 0 and
                payload["metadata"]["foreign_training_languages"] == []}


def save_checkpoint(path: str | Path, payload: dict) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), {"format": "mutual-gru-v1", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                         "kind": payload["kind"], "metadata": payload["metadata"],
                                         "provenance": checkpoint_provenance(payload)})
    return path


def load_checkpoint(path: str | Path, *, expected_kind: str | None = None) -> dict:
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["format"] != "mutual-gru-v1" or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint format or SHA256 mismatch")
    if expected_kind not in {None, "native", "acquisition"}:
        raise ValueError("Unknown expected checkpoint kind")
    if expected_kind is not None and manifest["kind"] != expected_kind:
        raise ValueError(f"Expected {expected_kind} checkpoint, got {manifest['kind']}. "
                         "New experiments require native_checkpoint; learned states require restore()/resume.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["metadata"] != manifest["metadata"] or state["kind"] != manifest["kind"]:
        raise ValueError("Checkpoint manifest does not match saved state")
    provenance = checkpoint_provenance(state)
    if "provenance" in manifest and manifest["provenance"] != provenance:
        raise ValueError("Checkpoint provenance does not match actual acquisition state")
    if expected_kind == "native" and not provenance["eligible_for_fresh_experiment"]:
        raise ValueError("Native checkpoint contains foreign acquisition history; use restore()/resume")
    return state


def train_native(config: ExperimentConfig, language: str, path: str | Path, *, initialization_seed: int,
                 native_training_seed: int, model_replica_id: str = "r0", progress=None) -> Path:
    config.validate()
    from .languages import LANGUAGES
    if language not in LANGUAGES or LANGUAGES[language]["corpus"] != config.native_training_profile.corpus_version:
        raise ValueError("Native corpus is not ready for this language/profile")
    configure_cpu()
    data = make_research_dataset(config)
    model = NativeModel(config.model_profile, initialization_seed)
    initial = copy.deepcopy(model.understanding.state_dict())
    rng = random.Random(native_training_seed)
    examples = []
    for scene in data.train:
        choices = [scene_png(s) for s in variants(scene.group)]
        for caption in native_captions(scene, language):
            model.tokenizer.encode(caption)  # Validate the whole corpus before training.
            examples.append({"text": caption, "image": scene_png(scene), "choices": choices,
                             "answer": variants(scene.group).index(scene)})
    profile = config.native_training_profile
    optimizer = torch.optim.Adam(model.parameters(), lr=profile.learning_rate)
    loss_value = None
    for step in range(profile.steps):
        batch = rng.choices(examples, k=profile.batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = model.grounding_loss(batch) + model.production_loss(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss_value = float(loss.detach())
        if progress and (step + 1) % 100 == 0:
            progress({"native_step": step + 1, "loss": loss_value, "language": language})
    comprehension, production, qualification_rows = 0, 0, []
    for probe in data.native_test:
        captions = native_captions(probe.scene, language)
        for caption in captions:
            comprehension += model.predict(caption, [scene_png(s) for s in probe.choices]) == probe.correct_choice - 1
        generated = model.generate(scene_png(probe.scene))
        valid = generated in captions
        production += valid
        qualification_rows.append({"scene_key": probe.scene.key, "generated": generated, "valid_native_caption": valid})
    comprehension /= len(data.native_test) * 4
    production /= len(data.native_test)
    metadata = {"language": language, "model_profile": asdict(config.model_profile),
                "native_training_profile": asdict(profile), "initialization_seed": initialization_seed,
                "native_training_seed": native_training_seed, "model_replica_id": model_replica_id,
                "initial_understanding_sha256": digest(initial), "native_data_sha256": digest(examples),
                "native_split_sha256": native_split_digest(data),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "understanding_parameter_count": sum(p.numel() for p in model.understanding.parameters()),
                "training_revision": revision(),
                "native_examples_seen": profile.steps * profile.batch_size,
                "native_comprehension": comprehension, "native_production": production,
                "native_eligible": comprehension >= profile.comprehension_min and production >= profile.production_min,
                "qualification": qualification_rows, "last_training_loss": loss_value,
                "foreign_training_languages": [], "training_languages": [language],
                "tokenizer_source": "fixed numeric UTF-8/BMP encoding; no fitted vocabulary or pretrained weights",
                "vision_source": "fixed pixel pooling and initial random projection; no text/vision pretraining",
                "corpus_status": "authored controlled natural-language pilot; not open-domain proficiency",
                "generation_policy": "greedy, max_sequence_length-2 units; incomplete/invalid UTF-8 bytes dropped",
                "torch_version": str(torch.__version__)}
    metadata["native_checkpoint_id"] = digest({"weights": model.state_dict(), "metadata": metadata})
    return save_checkpoint(path, {"kind": "native", "metadata": metadata, "weights": model.state_dict(),
                                 "initial_understanding": initial, "native_buffer": examples,
                                 "native_optimizer": optimizer.state_dict(), "native_rng": rng.getstate()})


class NeuralAgent:
    """L1 understanding weights continue learning; only pixels + partner text cross the boundary."""
    persistent = True

    def __init__(self, config: AgentConfig, experiment: ExperimentConfig):
        state = load_checkpoint(config.native_checkpoint, expected_kind="native")
        self._initialize(config, experiment, state)

    def _initialize(self, config: AgentConfig, experiment: ExperimentConfig, state: dict):
        configure_cpu()
        metadata = state["metadata"]
        if metadata["language"] != config.language or metadata["model_profile"] != asdict(experiment.model_profile):
            raise ValueError("Checkpoint language/architecture/tokenizer is incompatible")
        if metadata["native_training_profile"] != asdict(experiment.native_training_profile):
            raise ValueError("Checkpoint native training profile is incompatible")
        if experiment.model_replica_id != "unassigned" and metadata["model_replica_id"] != experiment.model_replica_id:
            raise ValueError("Checkpoint independent model replica does not match experiment")
        if metadata["native_split_sha256"] != native_split_digest(make_research_dataset(experiment)):
            raise ValueError("Checkpoint native split differs; prepare a new L1 model for this split")
        if metadata["training_languages"] != [config.language] or metadata["foreign_training_languages"]:
            raise ValueError("Checkpoint does not have monolingual training provenance")
        self.config, self.experiment = config, experiment
        self.metadata = copy.deepcopy(metadata)
        self.initial_checkpoint_provenance = checkpoint_provenance(state)
        self.model = NativeModel(experiment.model_profile, metadata["initialization_seed"])
        self.model.load_state_dict(state["weights"])
        if state["kind"] == "native" and experiment.acquisition_profile.reset_l1_understanding:
            self.model.understanding.load_state_dict(state["initial_understanding"])
        self.model.producer.requires_grad_(False)
        self.native_buffer = copy.deepcopy(state["native_buffer"])
        self.foreign_buffer = []
        self.rng = random.Random(experiment.effective_seed("dialogue"))
        self.optimizer = torch.optim.Adam(self.model.understanding.parameters(),
                                          lr=experiment.acquisition_profile.learning_rate)
        self.exposures = self.updates = self.native_replay_exposures = self.foreign_replay_exposures = 0
        self.characters = self.bytes_received = self.input_units = self.training_sequence_units = 0
        if state["kind"] == "acquisition":
            if state["acquisition_profile"] != asdict(experiment.acquisition_profile):
                raise ValueError("Acquisition checkpoint has a different learning rule")
            self.optimizer.load_state_dict(state["optimizer"])
            self.foreign_buffer = copy.deepcopy(state["foreign_buffer"])
            self.rng.setstate(state["rng"])
            for key, value in state["counters"].items():
                setattr(self, key, value)
        elif state["kind"] != "native":
            raise ValueError("Unknown checkpoint kind")

    def speak(self, image: SharedImage, partner_utterance=None, *, frozen=False):
        return self.model.generate(image.png)

    def observe(self, image: SharedImage, utterance: str, choices: tuple[SharedImage, ...]):
        pngs = [p.png for p in choices]
        if len(pngs) != 4 or pngs.count(image.png) != 1:
            raise ValueError("Grounding requires the actual shared image and three different visual alternatives")
        item = {"text": utterance, "image": image.png, "choices": pngs, "answer": pngs.index(image.png)}
        units = len(self.model.tokenizer.encode(utterance)) - 2
        self.exposures += 1
        self.characters += len(utterance)
        self.bytes_received += len(utterance.encode("utf-8"))
        self.input_units += units
        p = self.experiment.acquisition_profile
        for _ in range(p.updates_per_exposure):
            replay = self.rng.choices(self.foreign_buffer, k=p.foreign_replay_per_update) if self.foreign_buffer else []
            native = self.rng.choices(self.native_buffer, k=p.native_replay_per_update)
            batch = [item, *replay, *native]
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model.grounding_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.understanding.parameters(), 1)
            self.optimizer.step()
            self.updates += 1
            self.native_replay_exposures += len(native)
            self.foreign_replay_exposures += len(replay)
            self.training_sequence_units += sum(len(self.model.tokenizer.encode(e["text"])) for e in batch)
        self.foreign_buffer.append(item)
        self.foreign_buffer = self.foreign_buffer[-p.replay_capacity:]

    def choose(self, utterance: str, choices: tuple[SharedImage, ...], rng):
        answer = self.model.predict(utterance, [image.png for image in choices]) + 1
        return answer, str(answer)

    def counters(self):
        return {key: getattr(self, key) for key in ("exposures", "updates", "native_replay_exposures",
                "foreign_replay_exposures", "characters", "bytes_received", "input_units", "training_sequence_units")}

    def learning_state(self):
        return {"kind": "acquisition", "metadata": self.metadata, "weights": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "rng": self.rng.getstate(),
                "native_buffer": self.native_buffer, "foreign_buffer": self.foreign_buffer,
                "counters": self.counters(), "acquisition_profile": asdict(self.experiment.acquisition_profile)}

    def state_digest(self):
        return digest(self.learning_state())

    def save(self, path):
        return save_checkpoint(path, self.learning_state())

    @classmethod
    def restore(cls, path, config, experiment):
        state = load_checkpoint(path, expected_kind="acquisition")
        agent = cls.__new__(cls)
        agent._initialize(config, experiment, state)
        return agent
