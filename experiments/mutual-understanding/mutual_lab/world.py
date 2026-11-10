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
    rng = random.Random(config.effective_seed("scene_split"))
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
    "es": {"red": "rojo", "blue": "azul", "green": "verde", "circle": "círculo",
           "square": "cuadrado", "triangle": "triángulo", "left_of": "a la izquierda de", "above": "encima de"},
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
    if language == "es":
        return f"El {words[scene.subject.shape]} {words[scene.subject.color]} está {rel}l " \
               f"{words[scene.object.shape]} {words[scene.object.color]}."
    # Spaces make the demo's simple tokenizer explicit; this is controlled Japanese.
    return f"{a} は {b} の {rel} ある。"


def native_captions(scene: Scene, language: str) -> tuple[str, ...]:
    """Small auditable L1 corpus, four surface forms per meaning; no translated input.

    Each training invocation receives ONLY the selected language's returned strings.
    These authored sentences are a bounded natural-language task, not open-domain data.
    """
    w = _WORDS[language]
    a = f"{w[scene.subject.color]} {w[scene.subject.shape]}"
    b = f"{w[scene.object.color]} {w[scene.object.shape]}"
    if language == "en":
        rel = w[scene.relation]
        inverse = "to the right of" if scene.relation == "left_of" else "below"
        return (f"The {a} is {rel} the {b}.", f"A {a} is {rel} a {b}.",
                f"There is a {a} {rel} the {b}.", f"The {b} is {inverse} the {a}.")
    if language == "ko":
        particle = "은"  # all three nouns end in a consonant
        subject = "이"  # 원, 사각형, 삼각형
        inverse = "오른쪽에" if scene.relation == "left_of" else "아래쪽에"
        return (f"{a}{particle} {b}의 {w[scene.relation]} 있다.",
                f"{a}{subject} {b}의 {w[scene.relation]} 놓여 있다.",
                f"{b}의 {w[scene.relation]} {a}{subject} 있다.",
                f"{b}은 {a}의 {inverse} 있다.")
    if language == "ja":
        a, b = a.replace(" ", ""), b.replace(" ", "")
        inverse = "右に" if scene.relation == "left_of" else "下に"
        return (f"{a}は{b}の{w[scene.relation]}ある。", f"{b}の{w[scene.relation]}{a}がある。",
                f"{a}が{b}の{w[scene.relation]}置かれている。", f"{b}は{a}の{inverse}ある。")
    if language == "es":
        a = f"{w[scene.subject.shape]} {w[scene.subject.color]}"
        b = f"{w[scene.object.shape]} {w[scene.object.color]}"
        rel = "a la izquierda del" if scene.relation == "left_of" else "encima del"
        inverse = "a la derecha del" if scene.relation == "left_of" else "debajo del"
        return (f"El {a} está {rel} {b}.", f"Hay un {a} {rel} {b}.",
                f"Se ve un {a} {rel} {b}.", f"El {b} está {inverse} {a}.")
    raise ValueError(f"No native corpus for {language}")


@dataclass(frozen=True)
class ResearchDataset:
    train: tuple[Scene, ...]
    probes: tuple[Probe, ...]
    pretest: tuple[Probe, ...]
    final: tuple[Probe, ...]
    native_test: tuple[Probe, ...]

    def view(self, split: str) -> Dataset:
        probes = {"curve": self.probes, "pretest": self.pretest,
                  "final": self.final, "native": self.native_test}[split]
        return Dataset(self.train, probes, tuple(p.scene for p in probes))

    def to_dict(self):
        return {"split_version": "pair-disjoint-five-way-v1", "split_unit": "unordered object pair",
                "train": [s.to_dict() for s in self.train],
                **{name: self.view(name).to_dict()["probes"] for name in ("curve", "pretest", "final", "native")}}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def make_research_dataset(config: ExperimentConfig) -> ResearchDataset:
    """Reserve native qualification, first-contact screening, curve and final pairs up front."""
    split_rng = random.Random(config.effective_seed("scene_split"))
    choice_rng = random.Random(config.effective_seed("evaluation"))
    pairs = list(itertools.combinations([Object(c, s) for c in COLORS for s in SHAPES], 2))
    split_rng.shuffle(pairs)
    e = config.evaluation_profile
    sizes = (config.eval_items, e.pretest_items, e.final_items, e.native_items)
    if sum(sizes) // 4 >= len(pairs):
        raise ValueError("Research split leaves no training pairs; reduce the four evaluation sizes")
    splits = []
    offset = 0
    for size in sizes:
        scenes = [s for pair in pairs[offset:offset + size // 4] for s in variants(pair)]
        offset += size // 4
        positions = list(range(4)) * (size // 4)
        choice_rng.shuffle(positions)
        probes = []
        for scene, position in zip(scenes, positions):
            choices = [s for s in variants(scene.group) if s != scene]
            choice_rng.shuffle(choices)
            choices.insert(position, scene)
            probes.append(Probe(scene, tuple(choices), position + 1))
        splits.append(tuple(probes))
    train = tuple(s for pair in pairs[offset:] for s in variants(pair))
    return ResearchDataset(train, *splits)
