"""Versioned, independent controls for architecture, L1 training and acquisition."""

from dataclasses import dataclass
import math

RESEARCH_EVALUATION_MAX_ITEMS = 96


def bounded(value, name, low, high, *, integer=False):
    if (type(value) not in ({int} if integer else {int, float})
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be in [{low}, {high}]")


@dataclass(frozen=True)
class ModelProfile:
    id: str = "tiny-gru-v1"
    architecture: str = "gru"
    embedding_dim: int = 32
    hidden_dim: int = 64
    layers: int = 1
    tokenizer: str = "utf8"
    max_sequence_length: int = 256

    def validate(self):
        if self.architecture != "gru" or self.tokenizer not in {"utf8", "unicode-bmp"}:
            raise ValueError("Supported model: GRU with utf8 or fixed unicode-bmp input")
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("model_profile.id is required")
        for key, low, high in (("embedding_dim", 8, 512), ("hidden_dim", 16, 2048),
                               ("layers", 1, 4), ("max_sequence_length", 128, 1024)):
            bounded(getattr(self, key), key, low, high, integer=True)


MODEL_PROFILES = {
    "tiny-gru-v1": ModelProfile(),
    "gru-12m-v1": ModelProfile("gru-12m-v1", embedding_dim=128, hidden_dim=768, layers=2),
}


@dataclass(frozen=True)
class NativeTrainingProfile:
    id: str = "equal-budget-v1"
    steps: int = 600
    batch_size: int = 16
    learning_rate: float = .003
    comprehension_min: float = .8
    production_min: float = .8
    budget_rule: str = "equal_steps"
    corpus_version: str = "shapes-natural-v1"

    def validate(self):
        if self.budget_rule != "equal_steps" or self.corpus_version != "shapes-natural-v1":
            raise ValueError("Unsupported native training/corpus profile")
        bounded(self.steps, "native steps", 0, 1_000_000, integer=True)
        bounded(self.batch_size, "native batch size", 1, 256, integer=True)
        bounded(self.learning_rate, "native learning rate", 1e-6, .1)
        bounded(self.comprehension_min, "native comprehension minimum", .25, 1)
        bounded(self.production_min, "native production minimum", 0, 1)


@dataclass(frozen=True)
class AcquisitionProfile:
    id: str = "grounded-adam-v1"
    optimizer: str = "adam"
    learning_rate: float = .001
    updates_per_exposure: int = 1
    native_replay_per_update: int = 1
    foreign_replay_per_update: int = 1
    replay_capacity: int = 128
    association: str = "aligned"
    reset_l1_understanding: bool = False

    def validate(self):
        if self.optimizer != "adam" or self.association not in {"aligned", "shuffled"}:
            raise ValueError("Unsupported optimizer or association control")
        bounded(self.learning_rate, "acquisition learning rate", 1e-6, .1)
        for key in ("updates_per_exposure", "native_replay_per_update", "foreign_replay_per_update"):
            bounded(getattr(self, key), key, 0, 100, integer=True)
        bounded(self.replay_capacity, "replay capacity", 1, 10000, integer=True)
        if type(self.reset_l1_understanding) is not bool:
            raise ValueError("reset_l1_understanding must be boolean")


@dataclass(frozen=True)
class EvaluationProfile:
    id: str = "four-choice-v2"
    threshold_scale: str = "raw"
    pretest_items: int = 24
    final_items: int = 24
    native_items: int = 16
    chance_margin: float = .15
    common_horizon: int | None = None

    def validate(self):
        if self.threshold_scale not in {"raw", "corrected"} or self.id != "four-choice-v2":
            raise ValueError("Unsupported evaluation profile or threshold scale")
        for key in ("pretest_items", "final_items", "native_items"):
            bounded(getattr(self, key), key, 4, RESEARCH_EVALUATION_MAX_ITEMS, integer=True)
            if getattr(self, key) % 4:
                raise ValueError(f"{key} must be divisible by four")
        bounded(self.chance_margin, "chance equivalence margin (raw accuracy)", .001, .24)
        if self.common_horizon is not None:
            bounded(self.common_horizon, "common horizon", 1, 100000, integer=True)
