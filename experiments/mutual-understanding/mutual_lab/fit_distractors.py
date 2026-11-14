"""Four-choice L1 training negatives from an explicit, fit-only image pool.

Evaluation contrast builders are deliberately not reused: they may see all
research-TRAIN pairs, including the held-out development combinations.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from types import MappingProxyType

from .composition_diagnostics import CATEGORIES, is_contrast
from .config import integer
from .images import scene_png
from .world import Scene

CATALOG_VERSION = "explicit-fit-negatives-v1"
SAMPLING_RULE = "omit-category-per-four-exposures-rotate-position-per-exposure-v1"


def _sha(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class FitCatalog:
    scenes: tuple[Scene, ...]
    pngs: tuple[bytes, ...]
    png_sha256: tuple[str, ...]
    candidates: tuple[tuple[tuple[int, ...], ...], ...]
    by_key: object
    by_png: object

    def manifest(self):
        availability = {}
        for category_index, category in enumerate(CATEGORIES):
            counts = [len(per_anchor[category_index]) for per_anchor in self.candidates]
            availability[category] = {
                "anchors": len(self.scenes), "available": sum(count > 0 for count in counts),
                "unavailable": sum(count == 0 for count in counts),
                "candidate_count_by_anchor": dict(zip((s.key for s in self.scenes), counts)),
            }
        semantic_pool = [scene.to_dict() for scene in self.scenes]
        pool = [{"scene": scene.to_dict(), "scene_key": scene.key, "png_sha256": checksum}
                for scene, checksum in zip(self.scenes, self.png_sha256)]
        encoded = json.dumps(pool, sort_keys=True, separators=(",", ":")).encode()
        return {"version": CATALOG_VERSION, "sampling_rule": SAMPLING_RULE,
                "scope": "Only the explicitly supplied fit scenes; no global or evaluation pool lookup",
                "choice_count": 4, "chance_accuracy": .25,
                "scene_count": len(self.scenes), "pair_count": len({s.group for s in self.scenes}),
                "semantic_pool": semantic_pool, "pool": pool,
                "pool_sha256": _sha(encoded), "availability": availability,
                "color_and_shape_available": sum(bool(row[1]) and bool(row[2]) for row in self.candidates),
                "duplicate_semantics": 0, "duplicate_pngs": 0,
                "fallback": "Sample remaining unique fit PNGs without replacement; never drop an anchor",
                "position_rule": "global exposure modulo4; omitted category is floor(exposure/4) modulo4"}


def build_fit_catalog(fit_scenes, render_fn=None):
    """Build without consulting a configuration, language, dev or protected pool.

    The caller owns the fit/dev split. Supplying this pool explicitly makes the
    complete allowed image set available for boundary checks and provenance.
    """
    scenes = tuple(fit_scenes)
    if any(not isinstance(scene, Scene) for scene in scenes):
        raise ValueError("Fit catalog requires Scene values")
    if len(scenes) != len(set(scenes)):
        raise ValueError("Duplicate semantic scenes in fit catalog")
    scenes = tuple(sorted(scenes, key=lambda scene: scene.key))
    renderer = scene_png if render_fn is None else render_fn
    if not callable(renderer):
        raise ValueError("Fit renderer must be callable")
    pngs = tuple(renderer(scene) for scene in scenes)
    if any(not isinstance(png, bytes) for png in pngs):
        raise ValueError("Fit renderer must return PNG bytes")
    checksums = tuple(_sha(png) for png in pngs)
    if len(set(checksums)) != len(checksums):
        raise ValueError("Duplicate PNGs in fit catalog")
    if len(scenes) < 4:
        raise ValueError("Four unique fit scenes are required for four-choice training")
    candidates = tuple(tuple(tuple(j for j, alternative in enumerate(scenes)
                                   if is_contrast(anchor, alternative, category))
                             for category in CATEGORIES) for anchor in scenes)
    return FitCatalog(scenes, pngs, checksums, candidates,
                      MappingProxyType({scene.key: i for i, scene in enumerate(scenes)}),
                      MappingProxyType({checksum: i for i, checksum in enumerate(checksums)}))


def _empty_audit():
    result = {key: 0 for key in ("examples", "distractor_slots", "fallback_slots", "fallback_examples",
                                 "semantic_duplicate_choices", "png_duplicate_choices", "selected_actual_other")}
    for prefix in ("requested", "unavailable", "selected_by_policy", "selected_actual", "omitted"):
        result.update({f"{prefix}_{category}": 0 for category in CATEGORIES})
    result.update({f"answer_position_{position}": 0 for position in range(4)})
    return result


def make_training_examples(batch, catalog, negative_rng, exposure_offset=0, policy="attributes"):
    """Return a batch with the same positives/texts and exactly four fit PNGs.

    ``same_pair`` copies the original example dictionaries and leaves choices,
    labels and the negative RNG untouched. ``attributes`` attempts three of the
    four contrast categories, rotating the omitted category once per four
    exposures. Positive position rotates every exposure, crossing all omission
    types and positions over a 16-exposure cycle. Unavailable categories use
    unique fit-only fallback images; no meaning or caption is resampled.

    The flat numeric audit can be summed over batches without special handling.
    Category-attempt sources and actual contrast membership are separate.
    """
    if not isinstance(catalog, FitCatalog):
        raise ValueError("Use an explicit FitCatalog")
    if policy not in {"same_pair", "attributes"}:
        raise ValueError("Unknown fit-negative policy")
    integer(exposure_offset, "negative exposure offset", 0, 2**63 - 1)
    if policy == "attributes" and not isinstance(negative_rng, random.Random):
        raise ValueError("Attribute negatives need their own random.Random instance")
    audit, output = _empty_audit(), []
    for row_index, example in enumerate(batch):
        key = example.get("_scene_key")
        if key not in catalog.by_key:
            raise ValueError("Training anchor is outside the explicit fit catalog")
        positive = catalog.by_key[key]
        if example.get("image") != catalog.pngs[positive]:
            raise ValueError("Training positive image does not match its fit scene key")
        prepared = dict(example)
        if policy == "same_pair":
            choices, answer = example.get("choices", ()), example.get("answer")
            if len(choices) != 4 or type(answer) is not int or not 0 <= answer < 4:
                raise ValueError("Baseline training requires four choices and a zero-based answer")
            if choices[answer] != catalog.pngs[positive]:
                raise ValueError("Baseline answer does not identify the positive image")
            try:
                indices = [catalog.by_png[_sha(png)] for png in choices]
            except KeyError as exc:
                raise ValueError("A baseline training choice is outside the fit catalog") from exc
            if len(set(indices)) != 4:
                raise ValueError("Duplicate baseline training choices")
        else:
            exposure = exposure_offset + row_index
            omitted = (exposure // 4) % len(CATEGORIES)
            audit[f"omitted_{CATEGORIES[omitted]}"] += 1
            selected = []
            for offset in (1, 2, 3):
                category_index = (omitted + offset) % len(CATEGORIES)
                category = CATEGORIES[category_index]
                audit[f"requested_{category}"] += 1
                eligible = [i for i in catalog.candidates[positive][category_index] if i not in selected]
                if eligible:
                    selected.append(eligible[negative_rng.randrange(len(eligible))])
                    audit[f"selected_by_policy_{category}"] += 1
                else:
                    audit[f"unavailable_{category}"] += 1
            fallback = 3 - len(selected)
            if fallback:
                remaining = [i for i in range(len(catalog.scenes)) if i != positive and i not in selected]
                selected.extend(negative_rng.sample(remaining, fallback))
                audit["fallback_slots"] += fallback
                audit["fallback_examples"] += 1
            answer = exposure % 4
            indices = selected[:]
            indices.insert(answer, positive)
            prepared["choices"] = [catalog.pngs[i] for i in indices]
            prepared["answer"] = answer
        # These assertions cover both policies before the training code sees a batch.
        if len(indices) != 4 or len(set(indices)) != 4:
            raise ValueError("Training choices must be four distinct fit scenes/PNGs")
        audit["examples"] += 1
        audit["distractor_slots"] += 3
        audit[f"answer_position_{answer}"] += 1
        audit["anchor_exposures:" + key] = audit.get("anchor_exposures:" + key, 0) + 1
        anchor = catalog.scenes[positive]
        for index in indices:
            if index == positive:
                continue
            candidate = catalog.scenes[index]
            matching = [category for category in CATEGORIES if is_contrast(anchor, candidate, category)]
            if matching:
                for category in matching:
                    audit[f"selected_actual_{category}"] += 1
            else:
                audit["selected_actual_other"] += 1
            field = "candidate_exposures:" + candidate.key
            audit[field] = audit.get(field, 0) + 1
        output.append(prepared)
    return output, audit
