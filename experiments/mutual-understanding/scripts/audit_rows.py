"""Independent, tensor-free raw row audit for the fixed H confirmation world.

No model is fitted or scored here. All saved observations are retained. Surface
and UTF-8 flags are regraded from saved token IDs; observed EOS is bound to the
saved known-generator trace (and can separately be checked by model restore).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

CATEGORIES = ("role_reversal", "color_only", "shape_only", "binding_swap")
COUNTS = {"role_reversal": (448, 112, 28), "color_only": (448, 112, 28),
          "shape_only": (432, 108, 27), "binding_swap": (240, 60, 15)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def scene_id(scene):
    return canonical(scene.to_dict() if hasattr(scene, "to_dict") else scene)


def close(actual, expected, message):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), message)


def typed_equal(actual, expected):
    """JSON booleans and integer counts must not impersonate one another."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(typed_equal(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(typed_equal(a, b) for a, b in zip(actual, expected)))
    if type(expected) is float:
        return (type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12))
    return type(actual) is type(expected) and actual == expected


def exact_counter(rows, expected, key, label):
    actual = Counter(key(row) for row in rows)
    require(actual == Counter(expected), label + " exact row identity multiset differs")


def check_metrics(reported, expected, label):
    require(isinstance(reported, dict), label + " metrics absent")
    for key, value in expected.items():
        actual = reported.get(key)
        if type(value) is float:
            close(actual, value, label + ": " + key)
        elif type(value) is int:
            require(type(actual) is int and actual == value, label + ": " + key)
        else:
            require(typed_equal(actual, value), label + ": " + key)


def regrade_trace(row, scene, language, unit_limit=254):
    from mutual_lab.development_world import grade_generation
    ids = row.get("token_ids")
    require(isinstance(ids, list) and all(type(t) is int and 0 <= t <= 258 for t in ids),
            "Generation token IDs must be fixed UTF-8 vocabulary integers")
    require(2 not in ids, "EOS must not be recorded inside the greedy body")
    require(type(row.get("terminated")) is bool, "Observed EOS flag must be boolean")
    require(type(row.get("unit_limit")) is int and row["unit_limit"] == unit_limit,
            "Generation limit changed")
    require(len(ids) < unit_limit if row["terminated"] else len(ids) == unit_limit,
            "Trace length contradicts observed EOS/maximum")
    body = bytes(t - 3 for t in ids if t >= 3)
    try:
        body.decode("utf-8", errors="strict")
        invalid = False
    except UnicodeDecodeError:
        invalid = True
    trace = {"text": body.decode("utf-8", errors="ignore"), "token_ids": ids,
             "terminated": row["terminated"], "invalid_encoding": invalid,
             "special_tokens_in_body": sum(t < 3 for t in ids),
             "output_units": len(ids), "unit_limit": unit_limit}
    for name, value in trace.items():
        require(type(row.get(name)) is type(value) and row[name] == value,
                "Saved generation trace differs from tokens: " + name)
    graded = grade_generation(trace, scene, language)
    for name in ("scene", "exact_match", "recognized_semantic_correct", "recognized_meaning", "error_flags"):
        actual, expected = row.get(name), graded[name]
        require(type(actual) is type(expected) and actual == expected,
                "Generation grading mismatch: " + name)
    return graded


def audit_rows(directory, raw, splits, language, steps=(0, 2400, 4800, 9600), *, config=None):
    """Validate exact saved identities and independently recompute all six scores.

    ``points`` contains the independently reconstructed metric subset and full
    candidate flags, keyed by string step. Losses are checked for finiteness,
    not recomputed without a full model replay. Raises ValueError on any defect.
    """
    import mutual_lab
    from mutual_lab.config import ExperimentConfig
    from mutual_lab.development_world import build_development_contrasts, captions
    from mutual_lab.world import variants

    directory = Path(directory)
    require(language in ("ko", "ja", "en", "es") and raw.get("language") == language, "Language differs")
    require(tuple(steps) == (0, 2400, 4800, 9600), "Fixed saved budgets changed")
    require(set(splits) == {"train", "dev"} and {k: len(v) for k, v in splits.items()} == {"train": 220, "dev": 112},
            "H world requires 220 fit and 112 development scenes")
    plan = raw.get("plan", {})
    require(plan.get("comprehension_min") == .9 and plan.get("production_min") == .9
            and plan.get("contrast_min") == .9 and plan.get("min_contrast_anchors") == 8
            and plan.get("min_contrast_pairs") == 2, "Six-score or coverage criteria changed")
    profile = plan.get("model_profile", {})
    require(profile.get("tokenizer") == "utf8" and profile.get("max_sequence_length") == 256,
            "Expected fixed UTF-8 tokenizer and generation bound")
    if config is None:
        config = ExperimentConfig.load(Path(mutual_lab.__file__).resolve().parents[1] / "configs/neural.example.json")
    probes = build_development_contrasts(config, splits)
    require(set(probes["availability"]) == set(CATEGORIES), "Unexpected binary diagnostic categories")
    for name, (items, anchors, pairs) in COUNTS.items():
        a = probes["availability"][name]
        require(a["available"] == anchors and a["available_object_pairs"] == pairs and items == anchors * 4,
                "Fixed diagnostic coverage differs: " + name)

    gpath, upath = directory / "generations.json", directory / "comprehension_predictions.json"
    generations, native = json.loads(gpath.read_text()), json.loads(upath.read_text())
    require(isinstance(generations, list) and isinstance(native, list), "Saved raw rows must be arrays")
    def gid(row):
        require(type(row.get("step")) is int, "Generation step must be integer")
        return row["step"], row.get("split"), scene_id(row.get("scene"))
    def uid(row):
        require(type(row.get("step")) is int and type(row.get("template_index")) is int,
                "Native step/template must be integer")
        return row["step"], row.get("split"), scene_id(row.get("scene")), row["template_index"], row.get("text")
    expected_g = [(step, split, scene_id(scene)) for step in steps for split, scenes in splits.items() for scene in scenes]
    expected_u = [(step, split, scene_id(scene), form, text) for step in steps
                  for split, scenes in splits.items() for scene in scenes
                  for form, text in enumerate(captions(scene, language))]
    exact_counter(generations, expected_g, gid, "Generation")
    exact_counter(native, expected_u, uid, "Native comprehension")
    curves = raw.get("curves")
    require(isinstance(curves, list) and [p.get("step") for p in curves] == list(steps)
            and all(type(p.get("step")) is int for p in curves), "Missing, duplicate or unexpected saved curve budgets")
    require(len({scene_id(s) for ss in splits.values() for s in ss}) == 332, "Repeated split scene")
    scenes_by_id = {scene_id(s): s for ss in splits.values() for s in ss}
    gen_by = {}; native_by = {}
    for row in generations: gen_by.setdefault((row["step"], row["split"]), []).append(row)
    for row in native: native_by.setdefault((row["step"], row["split"]), []).append(row)
    points = {}; total_binary = 0
    for point in curves:
        step = point["step"]; result = {"step": step}
        for split, scenes in splits.items():
            gs = [regrade_trace(row, scenes_by_id[scene_id(row["scene"])], language)
                  for row in gen_by[(step, split)]]
            us = native_by[(step, split)]; hits = 0
            for row in us:
                scene = scenes_by_id[scene_id(row["scene"])]; alternatives = variants(scene.group)
                answer, predicted = alternatives.index(scene), row.get("prediction")
                require(type(row.get("answer")) is int and row["answer"] == answer, "Native answer differs")
                require(type(predicted) is int and 0 <= predicted < 4, "Invalid four-choice prediction")
                correct = predicted == answer
                require(type(row.get("correct")) is bool and row["correct"] is correct, "Native correctness flag differs")
                require(row.get("predicted_scene") == alternatives[predicted].to_dict(), "Native predicted scene differs")
                hits += correct
            valid = sum(g["exact_match"] and g["terminated"] and not g["invalid_encoding"]
                        and not g["special_tokens_in_body"] for g in gs)
            exact = sum(g["exact_match"] for g in gs); semantic = sum(g["recognized_semantic_correct"] is True for g in gs)
            metrics = {"comprehension_hits": hits, "comprehension_items": len(us), "comprehension_accuracy": hits / len(us),
                "generation_scenes": len(gs), "generation_exact_hits": exact, "generation_exact_accuracy": exact / len(gs),
                "valid_generation_hits": valid, "valid_generation_accuracy": valid / len(gs),
                "recognized_semantic_correct": semantic, "semantic_success_lower_bound": semantic / len(gs),
                "semantic_unknown_count": sum(g["recognized_semantic_correct"] is None for g in gs),
                "generation_error_counts": dict(Counter(e for g in gs for e in g["error_flags"]))}
            check_metrics(point.get(split), metrics, f"{step}/{split}")
            for field in ("comprehension_loss", "production_loss", "teacher_forced_token_accuracy"):
                value = point[split].get(field)
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0
                        and (field != "teacher_forced_token_accuracy" or value <= 1), "Nonfinite/invalid auxiliary metric")
            result[split] = metrics

        contrast = point.get("contrasts", {}); br = contrast.get("predictions")
        require(contrast.get("language") == language and contrast.get("choice_count") == 2
                and contrast.get("chance_accuracy") == .5 and contrast.get("excluded_from_research") is True
                and contrast.get("version") == probes["version"] and typed_equal(contrast.get("availability"), probes["availability"]),
                "Binary diagnostic header/availability differs")
        require(isinstance(br, list), "Binary predictions missing")
        lookup = {}; expected_b = []
        for probe in probes["probes"]:
            for form, text in enumerate(captions(probe.scene, language)):
                key = (probe.category, scene_id(probe.scene), form, text)
                expected_b.append(key); lookup[key] = probe
        def bid(row):
            require(type(row.get("template_index")) is int, "Binary template must be integer")
            return row.get("category"), scene_id(row.get("scene")), row["template_index"], row.get("text")
        exact_counter(br, expected_b, bid, f"Binary {step}")
        counts = {k: {"hits": 0, "items": 0} for k in CATEGORIES}
        for row in br:
            key = bid(row); probe = lookup[key]; predicted = row.get("prediction")
            require(type(row.get("answer")) is int and row["answer"] == probe.correct_choice, "Binary answer differs")
            require(type(predicted) is int and predicted in (0, 1), "Invalid binary prediction")
            correct = predicted == probe.correct_choice
            require(type(row.get("correct")) is bool and row["correct"] is correct, "Binary correct flag differs")
            require(row.get("predicted_scene") == probe.choices[predicted].to_dict(), "Binary predicted scene differs")
            counts[probe.category]["items"] += 1; counts[probe.category]["hits"] += correct
        require(set(contrast.get("categories", {})) == set(CATEGORIES), "Binary category set differs")
        details = {}
        for category, metrics in counts.items():
            require(metrics["items"] == COUNTS[category][0], "Binary denominator differs")
            accuracy = metrics["hits"] / metrics["items"]
            metrics.update(accuracy=accuracy, corrected_accuracy=2 * accuracy - 1)
            check_metrics(contrast["categories"][category], metrics, f"{step}/{category}")
            _, anchors, pairs = COUNTS[category]
            details[category] = {"hits": metrics["hits"], "items": metrics["items"], "accuracy": accuracy,
                                 "available_anchors": anchors, "available_pairs": pairs,
                                 "coverage_sufficient": anchors >= 8 and pairs >= 2, "score_target_met": accuracy >= .9}
        score = (result["dev"]["comprehension_accuracy"] >= .9 and result["dev"]["valid_generation_accuracy"] >= .9
                 and all(d["score_target_met"] for d in details.values()))
        candidate = {"status": "development_candidate" if score else "targets_not_met", "score_targets_met": score,
                     "coverage_sufficient": True, "categories": details,
                     "eligible_for_fresh_experiment": False, "excluded_from_research": True}
        require(typed_equal(point.get("candidate"), candidate), "Candidate flags/thresholds differ from independently recomputed six scores")
        result.update(contrasts=counts, candidate=candidate); points[str(step)] = result; total_binary += len(br)
    return {"status": "pass", "points": points,
            "generation_rows": len(generations), "native_rows": len(native), "binary_rows": total_binary,
            "generation_file_sha256": hashlib.sha256(gpath.read_bytes()).hexdigest(),
            "native_file_sha256": hashlib.sha256(upath.read_bytes()).hexdigest(),
            "fixed_final_qualified": points[str(steps[-1])]["candidate"]["status"] == "development_candidate",
            "generation_scope": "Token/encoding/surface/semantic regrade of saved traces; EOS observation separately bound by trusted generator and bounded restore"}
