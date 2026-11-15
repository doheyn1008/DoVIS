"""Explicit development-only online understanding updates for frozen H1 learners.

This module deliberately lives outside the frozen mutual_lab package. It does
not produce a native/acquisition checkpoint accepted by the research runner.
Only PNG bytes and an observed utterance enter an update; no Scene is accepted.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import random


FORMAT = "h1-development-acquisition-v1"
FACTORY = "mutual_lab.producer_context_study.RecordingFactory.H1"
SCOPE = {"kind": "development_only", "excluded_from_research": True,
         "eligible_for_l2": False, "eligible_for_research": False,
         "eligible_for_fresh_experiment": False}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, name + " must be an integer")


def _png(value):
    _require(type(value) is bytes and value.startswith(b"\x89PNG\r\n\x1a\n"), "Supply PNG bytes only")
    return value


def _caches(model):
    return {name: value for name, value in vars(model).items()
            if name.endswith("_cache") and isinstance(value, dict)}


class H1DevelopmentLearner:
    """Continue only H1's understanding weights in an excluded pilot.

    A fresh L2 Adam is intentional. The L1 optimizer is neither restored nor
    converted. Full understanding reset uses the exact H1 constructor and same
    image-only prior/initialization, rather than the legacy NativeModel class.
    """

    @classmethod
    def from_checkpoint(cls, checkpoint, prior, *, allow_development_training=False,
                        language=None, learning_rate=.001, seed=89007,
                        reset_understanding=False):
        _require(allow_development_training is True, "Explicit development-training opt-in is required")
        _require(type(reset_understanding) is bool, "reset_understanding must be boolean")
        _integer(seed, "seed")
        _require(seed <= 2**32 - 1, "seed is outside the supported range")
        _require(type(learning_rate) in (int, float) and math.isfinite(learning_rate)
                 and learning_rate > 0, "learning_rate must be positive and finite")
        from mutual_lab.mastery_infer import load_development_model
        from mutual_lab.neural import torch, digest
        model = load_development_model(checkpoint, prior)
        metadata = model.mastery_inference_metadata
        _require(metadata["model_factory"] == FACTORY and metadata["phase"] == "C"
                 and metadata["condition"] == "C1", "Only explicit H1/C1 development checkpoints are supported")
        _require(language is None or language == metadata["language"], "Checkpoint language differs")
        checkpoint, prior = Path(checkpoint), Path(prior)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if reset_understanding:
            from mutual_lab.producer_context import build_producer_context_model
            from mutual_lab.spatial_prior import load_prior_bundle
            from mutual_lab.profiles import ModelProfile
            with torch.random.fork_rng():
                fresh = build_producer_context_model(ModelProfile(**payload["plan"]["model_profile"]),
                    payload["seed_pair"][0], load_prior_bundle(prior),
                    condition=payload["prior_metadata"]["condition"], context="object_moments")
                model.understanding.load_state_dict(fresh.understanding.state_dict(), strict=True)
            del fresh
        model.requires_grad_(False)
        model.understanding.requires_grad_(True)
        model.eval()
        self = cls.__new__(cls)
        self.model = model
        self.language = metadata["language"]
        self.options = {"learning_rate": float(learning_rate), "seed": seed,
                        "reset_understanding": reset_understanding}
        self.bindings = {"checkpoint_sha256": _sha(checkpoint),
            "checkpoint_manifest_sha256": _sha(checkpoint.with_suffix(".json")),
            "prior_sha256": _sha(prior), "prior_manifest_sha256": _sha(prior.with_suffix(".json")),
            "adapter_source_sha256": _sha(__file__), "factory": FACTORY,
            "language": self.language, "l1_step": metadata["step"],
            "verified_model_sources": copy.deepcopy(metadata["verified_source_sha256"]),
            "initial_model_sha256": digest(model.state_dict())}
        self.optimizer = torch.optim.Adam(model.understanding.parameters(), lr=float(learning_rate))
        self._optimizer_groups = copy.deepcopy(self.optimizer.state_dict()["param_groups"])
        self.rng = random.Random(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.torch_rng = torch.random.get_rng_state().clone()
        self._counters = {name: 0 for name in ("exposures", "updates", "characters", "bytes_received",
            "input_units", "training_sequence_units", "native_replay_exposures")}
        self._frozen = self._frozen_digest()
        return self

    def _frozen_digest(self):
        from mutual_lab.neural import digest
        return digest({name: value for name, value in self.model.state_dict().items()
                       if not name.startswith("understanding.")})

    def _verify_frozen(self):
        _require(self._frozen_digest() == self._frozen, "Frozen producer/visual state changed")
        _require(all(p.requires_grad == name.startswith("understanding.")
                     for name, p in self.model.named_parameters()), "Unexpected trainable parameter flags")

    def counters(self):
        return copy.deepcopy(self._counters)

    def branch_digests(self):
        from mutual_lab.neural import digest
        return {"understanding": digest(self.model.understanding.state_dict()),
                "producer": digest(self.model.producer.state_dict()),
                "visual": digest(self.model.visual_encoder.state_dict()), "frozen": self._frozen_digest()}

    def cache_digest(self):
        from mutual_lab.neural import digest
        return digest(_caches(self.model))

    def warm(self, pngs):
        """Explicitly materialize a caller-validated development image pool."""
        from mutual_lab.neural import torch
        _require(isinstance(pngs, (list, tuple)) and pngs, "Supply a nonempty PNG pool")
        pngs = [_png(png) for png in pngs]
        before = self.state_digest()
        with torch.no_grad(), torch.random.fork_rng():
            for png in pngs:
                self.model.visual([png]); self.model.visual_grid([png])
                self.model.understanding_objects([png]); self.model.producer_visual([png])
        self._verify_frozen()
        _require(self.state_digest() == before, "Feature warming changed learning state")

    def _read_only(self, operation):
        from mutual_lab.neural import torch
        self._verify_frozen()
        before = self.state_digest()
        cache = copy.deepcopy(_caches(self.model))
        try:
            with torch.no_grad(), torch.random.fork_rng():
                result = operation()
        finally:
            for name, values in cache.items():
                getattr(self.model, name).clear(); getattr(self.model, name).update(values)
        self._verify_frozen()
        _require(self.state_digest() == before, "Read-only inference changed learning state")
        return result

    def emit(self, png):
        return self._read_only(lambda: copy.deepcopy(self.model.generate_with_trace(_png(png))))

    def predict(self, text, choices_png):
        return self.predict_many([text], [choices_png])[0]

    def predict_many(self, texts, choices_pngs):
        _require(isinstance(texts, (list, tuple)) and texts and isinstance(choices_pngs, (list, tuple))
                 and len(texts) == len(choices_pngs), "Supply one choice sequence per utterance")
        choices = []
        for text, row in zip(texts, choices_pngs):
            self._text(text)
            _require(isinstance(row, (list, tuple)) and len(row) in (2, 4), "Use two or four PNG choices")
            row = [_png(png) for png in row]
            _require(len(set(row)) == len(row), "Comprehension choices must be distinct")
            choices.append(row)
        _require(len({len(row) for row in choices}) == 1, "A comprehension batch needs a consistent choice count")
        return self._read_only(lambda: self.model.comprehension_logits(list(texts), choices).argmax(dim=-1).tolist())

    def _text(self, text):
        _require(type(text) is str and bool(text.strip()), "Supply a nonempty observed utterance")
        text.encode("utf-8", errors="strict")
        self.model.tokenizer.encode(text)

    def _example(self, value):
        _require(isinstance(value, dict) and set(value) == {"text", "choices", "answer"},
                 "Replay must contain only text, choices and answer")
        self._text(value["text"])
        choices = value["choices"]
        _require(isinstance(choices, (list, tuple)) and len(choices) == 4, "Replay requires four PNG choices")
        choices = [_png(png) for png in choices]
        _require(len(set(choices)) == 4 and type(value["answer"]) is int and 0 <= value["answer"] < 4,
                 "Replay choices/answer are invalid")
        return {"text": value["text"], "choices": choices, "answer": value["answer"]}

    def update(self, peer_text, target_png, negative_pngs, *, updates=1, native_replay=()):
        from mutual_lab.neural import torch, nn
        self._verify_frozen(); self._text(peer_text); _integer(updates, "updates")
        _require(isinstance(negative_pngs, (list, tuple)) and len(negative_pngs) == 3,
                 "Supply exactly three negative PNGs")
        choices = [_png(target_png), *[_png(png) for png in negative_pngs]]
        _require(len(set(choices)) == 4, "Target and negative PNGs must be distinct")
        _require(isinstance(native_replay, (list, tuple)), "Replay must be an explicit sequence")
        native = [self._example(value) for value in native_replay]
        snapshot = self.state_dict()
        losses = []
        try:
            self._counters["exposures"] += 1
            self._counters["characters"] += len(peer_text)
            self._counters["bytes_received"] += len(peer_text.encode("utf-8"))
            self._counters["input_units"] += len(self.model.tokenizer.encode(peer_text)) - 2
            for _ in range(updates):
                shuffled = list(choices); self.rng.shuffle(shuffled)
                batch = [{"text": peer_text, "choices": shuffled, "answer": shuffled.index(target_png)}, *native]
                with torch.random.fork_rng():
                    torch.random.set_rng_state(self.torch_rng)
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = self.model.grounding_loss(batch)
                    _require(bool(torch.isfinite(loss).item()), "Nonfinite acquisition loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.understanding.parameters(), 1., error_if_nonfinite=True)
                    self.optimizer.step()
                    self.torch_rng = torch.random.get_rng_state().clone()
                self._counters["updates"] += 1
                self._counters["native_replay_exposures"] += len(native)
                self._counters["training_sequence_units"] += sum(len(self.model.tokenizer.encode(x["text"])) for x in batch)
                losses.append(float(loss.detach()))
            self._verify_frozen()
            _require(all(bool(torch.isfinite(v).all()) for v in self.model.understanding.state_dict().values()),
                     "Nonfinite understanding tensor")
        except BaseException:
            self.load_state_dict(snapshot)
            raise
        return {**SCOPE, "losses": losses, "counters": self.counters()}

    def state_dict(self):
        return copy.deepcopy({"format": FORMAT, **SCOPE, "bindings": self.bindings,
            "options": self.options, "language": self.language, "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "rng": self.rng.getstate(),
            "torch_rng": self.torch_rng, "counters": self._counters,
            "feature_caches": _caches(self.model), "frozen_sha256": self._frozen})

    def state_digest(self):
        from mutual_lab.neural import digest
        state = self.state_dict(); del state["feature_caches"]
        return digest(state)

    def load_state_dict(self, state):
        """Validate a new-format pilot snapshot before mutating this learner."""
        from mutual_lab.neural import torch, digest
        _require(isinstance(state, dict) and state.get("format") == FORMAT, "Not an H1 development acquisition snapshot")
        _require(all(state.get(k) == v and type(state.get(k)) is type(v) for k, v in SCOPE.items()),
                 "Acquisition snapshot cannot become research/native eligible")
        _require(digest(state.get("bindings")) == digest(self.bindings) and digest(state.get("options")) == digest(self.options)
                 and state.get("language") == self.language, "Acquisition source/checkpoint/prior/options binding differs")
        weights = state.get("weights", {})
        reference = self.model.state_dict()
        _require(set(weights) == set(reference), "Acquisition tensor names differ")
        for name, value in weights.items():
            _require(isinstance(value, torch.Tensor) and value.shape == reference[name].shape
                     and value.dtype == reference[name].dtype and bool(torch.isfinite(value).all()),
                     "Invalid acquisition tensor: " + name)
        _require(state.get("frozen_sha256") == self._frozen == digest({n: v for n, v in weights.items()
                 if not n.startswith("understanding.")}), "Snapshot changed frozen producer/visual state")
        counts = state.get("counters", {})
        _require(set(counts) == set(self._counters) and all(type(v) is int and v >= 0 for v in counts.values()),
                 "Invalid acquisition counters")
        _require(counts["updates"] > 0 or not state.get("optimizer", {}).get("state"), "Untrained L2 Adam is not empty")
        optimizer = state.get("optimizer", {})
        _require(digest(optimizer.get("param_groups")) == digest(self._optimizer_groups), "Acquisition Adam options differ")
        parameters = list(self.model.understanding.parameters())
        ids = [i for group in self._optimizer_groups for i in group["params"]]
        states = optimizer.get("state", {})
        expected = set(ids) if counts["updates"] else set()
        _require(isinstance(states, dict) and all(type(i) is int for i in states) and set(states) == expected,
                 "Acquisition Adam state coverage differs")
        for index, parameter in zip(ids, parameters):
            if index not in states:
                continue
            entry = states[index]
            _require(isinstance(entry, dict) and set(entry) == {"step", "exp_avg", "exp_avg_sq"}, "Adam moment fields differ")
            _require(isinstance(entry["step"], torch.Tensor) and entry["step"].ndim == 0
                     and entry["step"].dtype != torch.bool and not entry["step"].is_complex()
                     and bool(torch.isfinite(entry["step"]).item())
                     and float(entry["step"]) == counts["updates"], "Acquisition Adam step differs")
            for key in ("exp_avg", "exp_avg_sq"):
                value = entry[key]
                _require(isinstance(value, torch.Tensor) and value.is_floating_point()
                         and value.shape == parameter.shape and value.dtype == parameter.dtype
                         and bool(torch.isfinite(value).all()), "Invalid acquisition Adam moment")
        check_rng = random.Random()
        try:
            check_rng.setstate(state["rng"])
            generator = torch.Generator(); generator.set_state(state["torch_rng"])
        except (TypeError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            raise ValueError("Invalid acquisition RNG state") from exc
        cache = state.get("feature_caches", {})
        _require(isinstance(cache, dict) and set(cache) == set(_caches(self.model)), "Feature cache names differ")
        current_caches = _caches(self.model)
        for name, values in cache.items():
            _require(isinstance(values, dict) and all(isinstance(k, bytes) and len(k) == 32
                     and isinstance(v, torch.Tensor) and not v.requires_grad and bool(torch.isfinite(v).all())
                     for k, v in values.items()), "Invalid frozen feature cache")
            _require(all(key in current_caches[name] and value.shape == current_caches[name][key].shape
                         and value.dtype == current_caches[name][key].dtype
                         and torch.equal(value, current_caches[name][key]) for key, value in values.items()),
                     "Snapshot feature cache differs; prewarm the same canonical PNG pool before restore")
        self.model.load_state_dict(weights, strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(optimizer))
        self.rng.setstate(state["rng"]); self.torch_rng = state["torch_rng"].clone()
        self._counters = copy.deepcopy(counts)
        for name, values in cache.items():
            getattr(self.model, name).clear(); getattr(self.model, name).update(copy.deepcopy(values))
        self._verify_frozen()
