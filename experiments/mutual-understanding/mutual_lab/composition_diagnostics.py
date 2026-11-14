"""Development-only composition splits and binary, attribute-specific probes.

Every anchor and alternative belongs to the research TRAIN pool. These probes
are separate from the original four-choice score and never qualify a learner.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass

from .config import integer
from .images import scene_png
from .languages import LANGUAGES
from .native_preparation import PreparationPlan
from .world import COLORS, SHAPES, Object, Scene, make_research_dataset, native_captions, variants

SPLIT_VERSION = "seeded-first-coverage-partners-at-least-two-v1"
CONTRAST_VERSION = "composition-binary-v1"
CATEGORIES = ("role_reversal", "color_only", "shape_only", "binding_swap")


def partner_counts(scenes):
    pairs = {scene.group for scene in scenes}
    counts = Counter(obj for pair in pairs for obj in pair)
    return [{"object": asdict(obj), "distinct_partners": counts[obj]}
            for obj in sorted({Object(c, s) for c in COLORS for s in SHAPES})]


def split_manifest(splits):
    pairs = {name: [[asdict(obj) for obj in pair]
                    for pair in sorted({scene.group for scene in scenes})]
             for name, scenes in splits.items()}
    encoded = json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()
    # A manifest describes its input; it must not label an original split as
    # selected by the balanced policy. The caller records the selection policy.
    return {"split_version": "composition-split-manifest-v1", "pair_keys": pairs,
            "pair_sha256": hashlib.sha256(encoded).hexdigest(),
            "minimum_train_distinct_partners": min(
                (row["distinct_partners"] for row in partner_counts(splits["train"])), default=0),
            "partner_counts": {name: partner_counts(scenes) for name, scenes in splits.items()}}


def balanced_splits(config, plan=PreparationPlan()):
    """Use the same seeded pair order as the original policy, with extra coverage.

    The first eligible subset is selected by semantics only. No model score or
    language is consulted. A bounded search rejects impossible custom plans.
    """
    plan.validate(config)
    required = {Object(color, shape) for color in COLORS for shape in SHAPES}
    if plan.train_pairs < len(required):
        raise ValueError("Two distinct partners per object require at least nine training pairs")
    pairs = sorted({scene.group for scene in make_research_dataset(config).train})
    if plan.train_pairs + plan.dev_pairs > len(pairs):
        raise ValueError("Balanced subsets exceed the research TRAIN pool")
    degree = Counter(obj for pair in pairs for obj in pair)
    if any(degree[obj] < 2 for obj in required):
        raise ValueError("The research TRAIN pool cannot provide two partners for every object")
    random.Random(plan.data_seed).shuffle(pairs)
    for index, dev in enumerate(itertools.combinations(pairs, plan.dev_pairs)):
        if index >= 100000:
            raise ValueError("Coverage search exceeded the bounded development search limit")
        train = tuple(pair for pair in pairs if pair not in dev)[:plan.train_pairs]
        counts = Counter(obj for pair in train for obj in pair)
        dev_objects = set(itertools.chain.from_iterable(dev))
        if (all(counts[obj] >= 2 for obj in required)
                and {obj.color for obj in dev_objects} == set(COLORS)
                and {obj.shape for obj in dev_objects} == set(SHAPES)):
            return {name: tuple(scene for pair in pool for scene in variants(pair))
                    for name, pool in (("train", train), ("dev", dev))}
    raise ValueError("No coverage-valid, two-partner split exists")


@dataclass(frozen=True)
class ContrastProbe:
    category: str
    scene: Scene
    choices: tuple[Scene, Scene]
    correct_choice: int  # Zero-based; researcher-side only.

    def to_dict(self):
        return {"category": self.category, "scene": self.scene.to_dict(),
                "choices": [scene.to_dict() for scene in self.choices],
                "correct_choice": self.correct_choice}


def is_contrast(anchor, alternative, category):
    """Definitions preserve the spatial axis; role reversal flips its direction."""
    if anchor == alternative or anchor.relation != alternative.relation:
        return False
    a, b, x, y = anchor.subject, anchor.object, alternative.subject, alternative.object
    if category == "role_reversal":
        return (x, y) == (b, a)
    if category == "color_only":
        return (a.shape, b.shape) == (x.shape, y.shape) and ((a.color != x.color) + (b.color != y.color)) == 1
    if category == "shape_only":
        return (a.color, b.color) == (x.color, y.color) and ((a.shape != x.shape) + (b.shape != y.shape)) == 1
    if category == "binding_swap":
        return (a.color != b.color and a.shape != b.shape
                and (x.shape, y.shape) == (a.shape, b.shape)
                and (x.color, y.color) == (b.color, a.color))
    raise ValueError("Unknown composition contrast category")


def build_contrast_probes(config, scenes, seed=39001):
    """Return binary probes with balanced correct positions per category.

    Callers supply their held-out development scenes. This function validates
    research-TRAIN membership; the caller owns the learner's fit/dev boundary.
    Counterfactuals may use any research TRAIN scene. Missing
    eligible contrasts are reported, rather than filled from protected pools.
    Image and semantic duplicates are forbidden within each choice set; reusing
    an anchor across diagnostic categories is deliberate.
    """
    config.validate()
    integer(seed, "contrast seed", 0, 2**32 - 1)
    anchors = tuple(sorted(scenes, key=lambda scene: scene.key))
    if len(set(anchors)) != len(anchors):
        raise ValueError("Duplicate contrast anchor scenes")
    allowed = tuple(sorted(make_research_dataset(config).train, key=lambda scene: scene.key))
    if not set(anchors) <= set(allowed):
        raise ValueError("Contrast anchors must belong to the research TRAIN pool")
    image_hashes = {scene: hashlib.sha256(scene_png(scene)).hexdigest() for scene in allowed}
    probes, availability = [], {}
    for offset, category in enumerate(CATEGORIES):
        rng = random.Random(seed + offset)
        candidates, missing, duplicate_images = [], [], 0
        for scene in anchors:
            alternatives = [other for other in allowed if is_contrast(scene, other, category)]
            unique = [other for other in alternatives if image_hashes[other] != image_hashes[scene]]
            duplicate_images += len(alternatives) - len(unique)
            if not unique:
                missing.append(scene.key)
                continue
            candidates.append((scene, unique[rng.randrange(len(unique))]))
        positions = ([0, 1] * ((len(candidates) + 1) // 2))[:len(candidates)]
        rng.shuffle(positions)
        for (scene, alternative), position in zip(candidates, positions):
            choices = (scene, alternative) if position == 0 else (alternative, scene)
            probes.append(ContrastProbe(category, scene, choices, position))
        availability[category] = {"anchors": len(anchors), "available": len(candidates),
            "unavailable": len(missing), "unavailable_scene_keys": missing,
            "duplicate_images_rejected": duplicate_images,
            "correct_position_counts": [positions.count(0), positions.count(1)]}
    return {"version": CONTRAST_VERSION, "excluded_from_research": True,
            "choice_count": 2, "chance_accuracy": .5, "seed": seed,
            "anchor_scene_keys": [scene.key for scene in anchors],
            "availability": availability, "probes": tuple(probes),
            "scope": "Supplied development anchors; all alternatives from research TRAIN only"}


def contrast_manifest(probe_set):
    return {**probe_set, "probes": [probe.to_dict() for probe in probe_set["probes"]]}


def score_contrasts(model, probe_set, language):
    """Read-only inference; the caller can additionally hash its learning state.

    The learner receives only one native caption and two PNGs, with no scene
    labels, diagnostic category or correct answer. No optimizer or RNG is used.
    """
    if language not in LANGUAGES:
        raise ValueError("Unknown contrast language")
    rows = []
    counts = {category: {"hits": 0, "items": 0} for category in CATEGORIES}
    for probe in probe_set["probes"]:
        images = [scene_png(scene) for scene in probe.choices]
        for template_index, text in enumerate(native_captions(probe.scene, language)):
            prediction = model.predict(text, images)
            if type(prediction) is not int or prediction not in (0, 1):
                raise ValueError("A binary contrast prediction must be zero or one")
            correct = prediction == probe.correct_choice
            counts[probe.category]["items"] += 1
            counts[probe.category]["hits"] += int(correct)
            rows.append({"category": probe.category, "scene": probe.scene.to_dict(),
                         "text": text, "template_index": template_index,
                         "choices": [scene.key for scene in probe.choices],
                         "answer": probe.correct_choice, "prediction": prediction,
                         "correct": correct, "predicted_scene": probe.choices[prediction].to_dict()})
    for values in counts.values():
        accuracy = values["hits"] / values["items"] if values["items"] else None
        values.update(accuracy=accuracy, corrected_accuracy=2 * accuracy - 1 if accuracy is not None else None)
    return {"version": CONTRAST_VERSION, "excluded_from_research": True,
            "language": language, "choice_count": 2, "chance_accuracy": .5,
            "availability": probe_set["availability"], "categories": counts, "predictions": rows,
            "interpretation": "Additional development diagnostics; not comparable to the original four-choice raw score"}
