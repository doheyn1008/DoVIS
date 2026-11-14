"""A separately versioned development world; original protected pairs stay out.

Five colors and the same three shapes add 69 pairs without consuming the old
22 protected pairs. Existing fit/dev pairs keep their original membership.
This module does not train models or change the original world/renderer.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from dataclasses import asdict

from .composition_diagnostics import CATEGORIES, ContrastProbe, balanced_splits, is_contrast
from .config import integer
from .diagnostics import normalized_surface
from .images import BACKGROUND, HEIGHT, RGB as BASE_RGB, WIDTH, encode_png
from .native_preparation import PreparationPlan
from .world import COLORS as BASE_COLORS, RELATIONS, SHAPES, Object, Scene, _WORDS, make_research_dataset, variants

VERSION = "five-colors-three-shapes-v1"
COLORS = (*BASE_COLORS, "yellow", "purple")
RGB = {**BASE_RGB, "yellow": (226, 180, 42), "purple": (139, 86, 195)}
EXTRA_WORDS = {"ko": {"yellow": "노란", "purple": "보라색"},
               "ja": {"yellow": "黄色い", "purple": "紫の"},
               "en": {"yellow": "yellow", "purple": "purple"},
               "es": {"yellow": "amarillo", "purple": "morado"}}
WORDS = {language: {**words, **EXTRA_WORDS[language]} for language, words in _WORDS.items()}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _protected_pairs(config):
    research = make_research_dataset(config)
    return {probe.scene.group for name in ("native", "pretest", "curve", "final")
            for probe in research.view(name).probes}


def _coverage(scenes, allowed):
    return {category: {scene for scene in scenes
                       if any(is_contrast(scene, other, category) for other in allowed)}
            for category in CATEGORIES}


def _select(config, data_seed):
    integer(data_seed, "expanded development data seed", 0, 2**32 - 1)
    base = balanced_splits(config, PreparationPlan())
    old_fit, old_dev = ({s.group for s in base[name]} for name in ("train", "dev"))
    protected = _protected_pairs(config)
    if len(old_fit) != 10 or len(old_dev) != 4 or len(protected) != 22:
        raise ValueError("This world version requires the frozen 10/4/22 base partition")
    objects = tuple(sorted(Object(color, shape) for color in COLORS for shape in SHAPES))
    new_pairs = sorted(pair for pair in itertools.combinations(objects, 2)
                       if any(obj.color not in BASE_COLORS for obj in pair))
    if len(new_pairs) != 69:
        raise ValueError("Expanded world definition no longer has 69 new pairs")
    allowed_pairs = set(new_pairs) | old_fit | old_dev
    if allowed_pairs & protected:
        raise ValueError("Protected pairs entered the expanded development pool")
    allowed = tuple(scene for pair in sorted(allowed_pairs) for scene in variants(pair))
    rng = random.Random(data_seed)
    for attempt in range(1, 10001):
        order = new_pairs.copy()
        rng.shuffle(order)
        fit, dev = old_fit | set(order[24:]), old_dev | set(order[:24])
        fit_counts, dev_counts = (Counter(obj for pair in pool for obj in pair) for pool in (fit, dev))
        if min(fit_counts[obj] for obj in objects) < 4 or min(dev_counts[obj] for obj in objects) < 2:
            continue
        scenes = tuple(scene for pair in sorted(dev) for scene in variants(pair))
        coverage = _coverage(scenes, allowed)
        if any(len(anchors) < 8 or len({scene.group for scene in anchors}) < 2
               for anchors in coverage.values()):
            continue
        return {"train": tuple(scene for pair in sorted(fit) for scene in variants(pair)),
                "dev": scenes}, {"selection_attempt": attempt,
                    "selection_rule": "first-valid-seeded-new-pair-permutation-v1",
                    "data_seed": data_seed, "old_fit_pairs_preserved": len(old_fit),
                    "old_dev_pairs_preserved": len(old_dev), "protected_pairs_excluded": len(protected)}
    raise ValueError("No coverage-valid expanded development split within the fixed search bound")


def development_splits(config, data_seed=59003):
    """Return train/dev scenes chosen by semantics only, never by model scores."""
    return _select(config, data_seed)[0]


def development_manifest(config, data_seed=59003):
    splits, selection = _select(config, data_seed)
    pairs = {name: [[asdict(obj) for obj in pair]
                    for pair in sorted({scene.group for scene in scenes})]
             for name, scenes in splits.items()}
    counts = {name: Counter(obj for pair in {s.group for s in scenes} for obj in pair)
              for name, scenes in splits.items()}
    coverage = _coverage(splits["dev"], (*splits["train"], *splits["dev"]))
    return {"version": VERSION, "excluded_from_research": True, **selection,
            "colors": list(COLORS), "shapes": list(SHAPES), "pair_keys": pairs,
            "pair_sha256": _hash(pairs), "scene_counts": {k: len(v) for k, v in splits.items()},
            "rgb_palette": {color: list(rgb) for color, rgb in RGB.items()},
            "png_pool_sha256": {name: _hash([
                {"scene_key": scene.key, "sha256": hashlib.sha256(render_scene(scene)).hexdigest()}
                for scene in scenes]) for name, scenes in splits.items()},
            "partner_counts": {name: [{"object": asdict(obj), "distinct_partners": value}
                                      for obj, value in sorted(counter.items())]
                               for name, counter in counts.items()},
            "contrast_availability": {category: {"anchors": len(anchors),
                "object_pairs": len({scene.group for scene in anchors})}
                for category, anchors in coverage.items()},
            "interpretation": "New development population; not an expansion of the protected research evaluation"}


def _validate_scene(scene):
    if (scene.relation not in RELATIONS or scene.subject == scene.object
            or any(obj.color not in RGB or obj.shape not in SHAPES for obj in (scene.subject, scene.object))):
        raise ValueError("Scene is outside the versioned five-color development world")


@functools.lru_cache(maxsize=1024)
def render_scene(scene):
    _validate_scene(scene)
    pixels = bytearray(BACKGROUND * (WIDTH * HEIGHT))
    centers = ((65, 88), (191, 88)) if scene.relation == "left_of" else ((128, 43), (128, 133))
    for obj, (cx, cy) in zip((scene.subject, scene.object), centers):
        radius = 28
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                dx, dy = x - cx, y - cy
                inside = (obj.shape == "square" or
                          (obj.shape == "circle" and dx * dx + dy * dy <= radius * radius) or
                          (obj.shape == "triangle" and abs(dx) <= (dy + radius) / 2))
                if inside:
                    offset = (y * WIDTH + x) * 3
                    pixels[offset:offset + 3] = bytes(RGB[obj.color])
    return encode_png(pixels, WIDTH, HEIGHT)


def captions(scene, language):
    _validate_scene(scene)
    if language not in WORDS:
        raise ValueError("Unknown development language")
    words = WORDS[language]
    a = f"{words[scene.subject.color]} {words[scene.subject.shape]}"
    b = f"{words[scene.object.color]} {words[scene.object.shape]}"
    if language == "en":
        rel = words[scene.relation]
        inverse = "to the right of" if scene.relation == "left_of" else "below"
        return (f"The {a} is {rel} the {b}.", f"A {a} is {rel} a {b}.",
                f"There is a {a} {rel} the {b}.", f"The {b} is {inverse} the {a}.")
    if language == "ko":
        inverse = "오른쪽에" if scene.relation == "left_of" else "아래쪽에"
        return (f"{a}은 {b}의 {words[scene.relation]} 있다.",
                f"{a}이 {b}의 {words[scene.relation]} 놓여 있다.",
                f"{b}의 {words[scene.relation]} {a}이 있다.", f"{b}은 {a}의 {inverse} 있다.")
    if language == "ja":
        a, b = a.replace(" ", ""), b.replace(" ", "")
        inverse = "右に" if scene.relation == "left_of" else "下に"
        return (f"{a}は{b}の{words[scene.relation]}ある。", f"{b}の{words[scene.relation]}{a}がある。",
                f"{a}が{b}の{words[scene.relation]}置かれている。", f"{b}は{a}の{inverse}ある。")
    a = f"{words[scene.subject.shape]} {words[scene.subject.color]}"
    b = f"{words[scene.object.shape]} {words[scene.object.color]}"
    rel = "a la izquierda del" if scene.relation == "left_of" else "encima del"
    inverse = "a la derecha del" if scene.relation == "left_of" else "debajo del"
    return (f"El {a} está {rel} {b}.", f"Hay un {a} {rel} {b}.",
            f"Se ve un {a} {rel} {b}.", f"El {b} está {inverse} {a}.")


def make_examples(model, scenes, language):
    rows = []
    for scene in scenes:
        choices = variants(scene.group)
        for text in captions(scene, language):
            model.tokenizer.encode(text)
            rows.append({"text": text, "image": render_scene(scene),
                         "choices": [render_scene(other) for other in choices],
                         "answer": choices.index(scene), "_scene_key": scene.key,
                         "_first_caption": captions(scene, language)[0]})
    return rows


def build_development_contrasts(config, splits, seed=59011):
    integer(seed, "expanded contrast seed", 0, 2**32 - 1)
    allowed = tuple(sorted((*splits["train"], *splits["dev"]), key=lambda scene: scene.key))
    if len(set(allowed)) != len(allowed) or {s.group for s in allowed} & _protected_pairs(config):
        raise ValueError("Expanded contrast pool repeats scenes or includes protected pairs")
    for scene in allowed:
        _validate_scene(scene)
    anchors = tuple(sorted(splits["dev"], key=lambda scene: scene.key))
    hashes = {scene: hashlib.sha256(render_scene(scene)).hexdigest() for scene in allowed}
    availability, probes = {}, []
    for offset, category in enumerate(CATEGORIES):
        rng = random.Random(seed + offset)
        candidates, missing = [], []
        for scene in anchors:
            alternatives = [other for other in allowed if is_contrast(scene, other, category)
                            and hashes[other] != hashes[scene]]
            if not alternatives:
                missing.append(scene.key)
            else:
                candidates.append((scene, alternatives[rng.randrange(len(alternatives))]))
        positions = ([0, 1] * ((len(candidates) + 1) // 2))[:len(candidates)]
        rng.shuffle(positions)
        for (scene, alternative), position in zip(candidates, positions):
            choices = (scene, alternative) if position == 0 else (alternative, scene)
            probes.append(ContrastProbe(category, scene, choices, position))
        availability[category] = {"anchors": len(anchors), "available": len(candidates),
            "available_object_pairs": len({scene.group for scene, _ in candidates}),
            "unavailable": len(missing), "unavailable_scene_keys": missing,
            "correct_position_counts": [positions.count(0), positions.count(1)]}
    return {"version": VERSION + "-binary", "excluded_from_research": True,
            "choice_count": 2, "chance_accuracy": .5, "seed": seed,
            "availability": availability, "probes": tuple(probes),
            "scope": "Expanded dev anchors; alternatives only from the new fit/dev development pool"}


def score_development_contrasts(model, probes, language):
    counts = {category: {"hits": 0, "items": 0} for category in CATEGORIES}
    rows = []
    for probe in probes["probes"]:
        for index, text in enumerate(captions(probe.scene, language)):
            predicted = model.predict(text, [render_scene(scene) for scene in probe.choices])
            if type(predicted) is not int or predicted not in (0, 1):
                raise ValueError("Binary predictions must be zero or one")
            correct = predicted == probe.correct_choice
            counts[probe.category]["hits"] += int(correct)
            counts[probe.category]["items"] += 1
            rows.append({"category": probe.category, "scene": probe.scene.to_dict(), "text": text,
                "template_index": index, "answer": probe.correct_choice, "prediction": predicted,
                "correct": correct, "predicted_scene": probe.choices[predicted].to_dict()})
    for values in counts.values():
        accuracy = values["hits"] / values["items"] if values["items"] else None
        values.update(accuracy=accuracy, corrected_accuracy=2 * accuracy - 1 if accuracy is not None else None)
    return {"version": probes["version"], "excluded_from_research": True, "language": language,
            "choice_count": 2, "chance_accuracy": .5, "availability": probes["availability"],
            "categories": counts, "predictions": rows}


@functools.lru_cache(maxsize=4)
def semantic_index(language):
    # Researcher-side interpretation only; no protected images are scored or shown.
    index = {}
    objects = [Object(color, shape) for color in COLORS for shape in SHAPES]
    for pair in itertools.combinations(objects, 2):
        for scene in variants(pair):
            for text in captions(scene, language):
                index.setdefault(normalized_surface(text), set()).add(scene)
    return index


def grade_generation(trace, expected, language):
    exact = trace["text"] in captions(expected, language)
    meanings = semantic_index(language).get(normalized_surface(trace["text"]), set())
    parsed = next(iter(meanings)) if len(meanings) == 1 else None
    errors = []
    if parsed is None:
        errors.append("ambiguous_surface" if meanings else "unrecognized_surface")
    elif parsed != expected:
        if parsed.subject == expected.object and parsed.object == expected.subject:
            errors.append("subject_object_order")
        if parsed.subject.color != expected.subject.color or parsed.object.color != expected.object.color:
            errors.append("color_error")
        if parsed.subject.shape != expected.subject.shape or parsed.object.shape != expected.object.shape:
            errors.append("shape_error")
        if parsed.group != expected.group:
            errors.append("object_identity_error")
        if parsed.relation != expected.relation:
            errors.append("relation_error")
    elif not exact:
        errors.append("surface_only")
    if not trace["terminated"]:
        errors.append("no_eos")
    if trace["invalid_encoding"]:
        errors.append("invalid_encoding")
    if trace["special_tokens_in_body"]:
        errors.append("special_tokens_in_body")
    return {"scene": expected.to_dict(), **trace, "exact_match": exact,
            "recognized_semantic_correct": parsed == expected if parsed is not None else None,
            "recognized_meaning": parsed.to_dict() if parsed else None, "error_flags": errors}


def evaluate_development(model, examples, scenes, language, batch_size):
    """Evaluate this version's own captions/PNGs without touching learning state."""
    import torch
    import torch.nn.functional as F
    comprehension_loss, production_loss, hits, token_hits, token_count = 0.0, 0.0, 0, 0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            comprehension_loss += float(model.grounding_loss(batch)) * len(batch)
            ids, _ = model.tokenizer.batch([row["text"] for row in batch])
            targets = ids[:, 1:]
            pngs = [row["image"] for row in batch]
            teacher_forced_logits = getattr(model, "teacher_forced_logits", None)
            if callable(teacher_forced_logits):
                logits = teacher_forced_logits(ids[:, :-1], pngs)
            else:
                logits, _ = model.producer(ids[:, :-1], model.visual(pngs))
            production_loss += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                                    ignore_index=0, reduction="sum"))
            mask = targets != 0
            token_hits += int(((logits.argmax(-1) == targets) & mask).sum())
            token_count += int(mask.sum())
            hits += sum(model.predict(row["text"], row["choices"]) == row["answer"] for row in batch)
        generated = [grade_generation(model.generate_with_trace(render_scene(scene)), scene, language) for scene in scenes]
    count = len(generated)
    valid = sum(row["exact_match"] and row["terminated"] and not row["invalid_encoding"]
                and not row["special_tokens_in_body"] for row in generated)
    metrics = {"comprehension_loss": comprehension_loss / len(examples),
               "production_loss": production_loss / token_count,
               "teacher_forced_token_accuracy": token_hits / token_count,
               "comprehension_hits": hits, "comprehension_items": len(examples),
               "comprehension_accuracy": hits / len(examples), "generation_scenes": count,
               "generation_exact_hits": sum(row["exact_match"] for row in generated),
               "generation_exact_accuracy": sum(row["exact_match"] for row in generated) / count,
               "valid_generation_hits": valid, "valid_generation_accuracy": valid / count,
               "recognized_semantic_correct": sum(row["recognized_semantic_correct"] is True for row in generated),
               "semantic_success_lower_bound": sum(row["recognized_semantic_correct"] is True for row in generated) / count,
               "semantic_unknown_count": sum(row["recognized_semantic_correct"] is None for row in generated),
               "generation_error_counts": dict(Counter(flag for row in generated for flag in row["error_flags"]))}
    if not all(math.isfinite(metrics[key]) for key in
               ("comprehension_loss", "production_loss", "teacher_forced_token_accuracy")):
        raise ValueError("Expanded development evaluation produced non-finite metrics")
    return metrics, generated
