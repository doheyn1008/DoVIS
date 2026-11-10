"""A small compositional visual world, with an unordered-object-pair holdout."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass

from .config import ExperimentConfig

COLORS = ("red", "blue", "green")
SHAPES = ("circle", "square", "triangle")
RELATIONS = ("left_of", "above")


@dataclass(frozen=True, order=True)
class Object:
    color: str
    shape: str


@dataclass(frozen=True)
class Scene:
    subject: Object
    relation: str
    object: Object

    @property
    def group(self) -> tuple[Object, Object]:
        return tuple(sorted((self.subject, self.object)))

    @property
    def key(self) -> str:
        return ":".join(self.features)

    @property
    def features(self) -> tuple[str, ...]:
        return (self.subject.color, self.subject.shape, self.relation,
                self.object.color, self.object.shape)

    def to_dict(self) -> dict:
        return asdict(self)


def variants(pair: tuple[Object, Object]) -> list[Scene]:
    a, b = pair
    return [Scene(x, relation, y) for x, y in ((a, b), (b, a)) for relation in RELATIONS]


@dataclass(frozen=True)
class Probe:
    scene: Scene
    choices: tuple[Scene, ...]
    correct_choice: int  # Researcher-only; never passed to endpoint agents.


@dataclass(frozen=True)
class Dataset:
    train: tuple[Scene, ...]
    probes: tuple[Probe, ...]
    holdout: tuple[Scene, ...]

    def to_dict(self) -> dict:
        return {
            "split_unit": "unordered pair of colored shapes",
            "train": [s.to_dict() for s in self.train],
            "holdout": [s.to_dict() for s in self.holdout],
            "probes": [{"scene": p.scene.to_dict(), "choices": [s.to_dict() for s in p.choices],
                        "correct_choice": p.correct_choice} for p in self.probes],
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def make_dataset(config: ExperimentConfig) -> Dataset:
    rng = random.Random(config.seed)
    objects = [Object(c, s) for c in COLORS for s in SHAPES]
    pairs = list(itertools.combinations(objects, 2))
    rng.shuffle(pairs)
    heldout_count = max(1, round(len(pairs) * config.holdout_fraction))
    heldout = [scene for pair in pairs[:heldout_count] for scene in variants(pair)]
    train = [scene for pair in pairs[heldout_count:] for scene in variants(pair)]
    if config.eval_items > len(heldout):
        raise ValueError(f"eval_items={config.eval_items} exceeds holdout size {len(heldout)}; "
                         "increase holdout_fraction or decrease eval_items")
    rng.shuffle(train)
    rng.shuffle(heldout)
    answer_positions = list(range(4)) * (config.eval_items // 4)
    rng.shuffle(answer_positions)
    probes = []
    for scene, position in zip(heldout, answer_positions):
        others = [s for s in variants(scene.group) if s != scene]
        rng.shuffle(others)
        others.insert(position, scene)
        probes.append(Probe(scene, tuple(others), position + 1))
    return Dataset(tuple(train), tuple(probes), tuple(heldout))


_WORDS = {
    "en": {"red": "red", "blue": "blue", "green": "green", "circle": "circle",
           "square": "square", "triangle": "triangle", "left_of": "to the left of", "above": "above"},
    "ko": {"red": "빨간", "blue": "파란", "green": "초록", "circle": "원",
           "square": "사각형", "triangle": "삼각형", "left_of": "왼쪽에", "above": "위쪽에"},
    "ja": {"red": "赤い", "blue": "青い", "green": "緑の", "circle": "円",
           "square": "四角", "triangle": "三角", "left_of": "左に", "above": "上に"},
}


def controlled_caption(scene: Scene, language: str) -> str:
    """DEMO speaker only: deliberately restricted templates, not a natural-language model."""
    words = _WORDS[language]
    a = f"{words[scene.subject.color]} {words[scene.subject.shape]}"
    b = f"{words[scene.object.color]} {words[scene.object.shape]}"
    rel = words[scene.relation]
    if language == "en":
        return f"The {a} is {rel} the {b}."
    if language == "ko":
        return f"{a}은 {b}의 {rel} 있다."
    # Spaces make the demo's simple tokenizer explicit; this is controlled Japanese.
    return f"{a} は {b} の {rel} ある。"
