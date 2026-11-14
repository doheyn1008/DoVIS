"""Conditional fresh-L1-seed validation of one already fixed development model.

Importing this module performs no training or activation. Every explicit job
starts one fresh attentive/B1/mixed_blocks language model; it never resumes the
discovery checkpoint, retries a seed, or qualifies a research learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

from .composition_diagnostics import CATEGORIES
from .diagnostics import safe_json
from .fit_distractors import negative_policy_manifest
from .mastery_lab import MasteryPlan, run_trial
from .native_preparation import LANGUAGES_ORDER
from .negative_policy_study import FixedLanguageFactory, _mixed_branch_counts
from .storage import now

VERSION = "candidate-validation-v1"
SEED_PAIRS = ((69007, 69009), (69017, 69019), (69027, 69029))
FACTORY = "mutual_lab.negative_policy_study.FixedLanguageFactory.attentive"
DATA_VERSION = "five-colors-three-shapes-v1"
PAIR_SHA256 = "1c3c190107ce23ce8245fb17c402c17d21f836ed4e96b5b13172b082a6ed26f6"
PNG_SHA256 = {"train": "c98f8b6cd69301ab53508e31386e037b7131176f5d802286895acc02db358502",
              "dev": "df2535b4a95f2a26c7b3c743a0da42693904d30d60bbbb07a609dd587a18ab11"}
AVAILABILITY = {"role_reversal": (112, 28), "color_only": (112, 28),
                "shape_only": (108, 27), "binding_swap": (60, 15)}
SCOPE = ("Development-only fresh L1 initialization/training seed repeats; one shared frozen visual prior. "
         "Not independent visual replications, open-domain L1 fluency, or language-distance evidence")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _reference_plan():
    return MasteryPlan.load(Path(__file__).resolve().parents[1] / "configs/mastery-C-plan.json")


def _validate_plan(plan):
    if plan.seed_pairs != SEED_PAIRS or plan.b_seed_pair != SEED_PAIRS[0]:
        raise ValueError("Validation requires the three predeclared fresh L1 seed pairs")
    current, reference = asdict(plan), asdict(_reference_plan())
    for key in ("seed_pairs", "b_seed_pair", "metadata"):
        current.pop(key); reference.pop(key)
    if _canonical(current) != _canonical(reference):
        raise ValueError("Validation must retain the C plan's data, model, budgets, thresholds and optimization controls")


def _validate_request(record, plan):
    if record is None:
        return
    if not isinstance(record, dict) or set(record) != {"sha256", "content"}:
        raise ValueError("Invalid pinned candidate-request record")
    content = record["content"]
    if (not isinstance(content, dict) or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64):
        raise ValueError("Invalid pinned candidate-request provenance")
    expected = {"schema_version": 1, "model": "attentive", "fixed_prior_condition": "B1",
                "policy": "mixed_blocks", "fixed_final_step": plan.budgets[-1], "data_version": DATA_VERSION}
    if any(content.get(key) != value for key, value in expected.items()):
        raise ValueError("Candidate request differs from the fixed validation model")


def _load_request(path, plan, *, plan_path=None, prior_path=None):
    if path is None:
        return None
    content_bytes = Path(path).read_bytes()
    record = {"sha256": hashlib.sha256(content_bytes).hexdigest(), "content": json.loads(content_bytes)}
    _validate_request(record, plan)
    for key, source in (("validation_plan_sha256", plan_path),
                        ("prior_manifest_sha256", Path(prior_path).with_suffix(".json") if prior_path else None)):
        expected = record["content"].get(key)
        if source is not None and expected != hashlib.sha256(Path(source).read_bytes()).hexdigest():
            raise ValueError(f"Pinned candidate request has a different {key}")
    return record


def run_candidate_validation(config, language, replica, plan, output="runs/candidate-validation", progress=print,
                             *, prior_bundle=None, candidate_request=None):
    """Execute exactly one fixed fresh-seed trial, preserving every outcome."""
    plan.validate(config); _validate_plan(plan); _validate_request(candidate_request, plan)
    if language not in LANGUAGES_ORDER or type(replica) is not int or replica not in range(len(SEED_PAIRS)):
        raise ValueError("Supply one known language and replica 0, 1 or 2")
    if prior_bundle is None:
        raise ValueError("Validation requires the same explicit frozen visual prior bundle")
    pair = SEED_PAIRS[replica]
    # The raw checkpoint records this per-trial B/C seed pair, allowing the
    # inference loader to verify seed provenance without accepting a resume.
    trial_plan = replace(plan, b_seed_pair=pair)
    factory = FixedLanguageFactory("attentive", "B1")
    directory = Path(output).resolve() / f"candidate-validation-{language}-r{replica}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    suite = {"schema_version": 1, "kind": VERSION, "created_at": now(), "status": "running",
        "language": language, "replica": replica, "seed_pair": list(pair), "plan": asdict(plan),
        "model": "attentive", "fixed_prior_condition": "B1", "policy": "mixed_blocks",
        "negative_policy_specification": negative_policy_manifest("mixed_blocks"),
        "source_candidate": "E1 at the predeclared final budget", "candidate_request": candidate_request,
        "engine_adapter": {"phase": "C", "condition": "C1"}, "data_version": DATA_VERSION,
        "fresh_initialization": True, "resumed": False, "scratch_language_training": True,
        "shared_visual_prior": True, "independent_visual_replications": False,
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "interpretation": SCOPE, "initial_state": {}}
    begun = time.monotonic()

    def snapshot():
        suite.update(initial_state=factory.initial_state or {}, elapsed_seconds=time.monotonic() - begun)
        safe_json(directory / "candidate-validation.json", suite)

    snapshot()
    try:
        path = run_trial(config, language, "C", "C1", replica, pair, trial_plan, directory / "raw-engine",
            progress, policy="mixed_blocks", prior_bundle=prior_bundle,
            remaining_seconds=plan.max_trial_seconds, model_factory=factory)
        raw = json.loads((path / "report.json").read_text())
        suite.update(status=raw["status"], engine_report=raw,
                     raw_report_path=str((path / "report.json").relative_to(directory)))
    except KeyboardInterrupt as exc:
        from .composition_suite import ArmCancelled
        suite["status"] = "cancelled"
        try:
            if isinstance(exc, ArmCancelled):
                raw = getattr(exc, "report", None)
                if not isinstance(raw, Mapping):
                    try:
                        raw = json.loads(exc.report_path.read_text())
                    except Exception:
                        raw = None
                if isinstance(raw, Mapping):
                    suite["engine_report"] = dict(raw)
                try:
                    suite["raw_report_path"] = str(exc.report_path.relative_to(directory))
                except Exception:
                    pass
        finally:
            try:
                snapshot()
            finally:
                raise exc
    except Exception as exc:
        suite.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    snapshot()
    return directory


def _rate(metrics, hits_key, items_key, accuracy_key, expected_items):
    hits, items, accuracy = (metrics.get(key) for key in (hits_key, items_key, accuracy_key))
    return (type(hits) is int and type(items) is int and items == expected_items and 0 <= hits <= items
            and type(accuracy) in (int, float) and math.isfinite(accuracy)
            and math.isclose(accuracy, hits / items, abs_tol=1e-12))


def _meets(value, threshold):
    return type(value) in (int, float) and math.isfinite(value) and value >= threshold


def _point_checks(point, plan):
    dev, candidate = point.get("dev", {}), point.get("candidate", {})
    categories = point.get("contrasts", {}).get("categories", {})
    declared = candidate.get("categories", {})
    checks = {"dev_count_consistency": _rate(dev, "comprehension_hits", "comprehension_items", "comprehension_accuracy", 448)
              and _rate(dev, "valid_generation_hits", "generation_scenes", "valid_generation_accuracy", 112),
        "all_categories": set(categories) == set(CATEGORIES) == set(declared),
        "candidate_development_only": candidate.get("excluded_from_research") is True
            and candidate.get("eligible_for_fresh_experiment") is False}
    for category in CATEGORIES:
        anchors, pairs = AVAILABILITY[category]
        details, metrics = declared.get(category, {}), categories.get(category, {})
        checks[category + "_coverage"] = (details.get("available_anchors") == anchors
            and details.get("available_pairs") == pairs and anchors >= plan.min_contrast_anchors
            and pairs >= plan.min_contrast_pairs and details.get("coverage_sufficient") is True)
        checks[category + "_counts"] = (_rate(metrics, "hits", "items", "accuracy", anchors * 4)
            and all(details.get(key) == metrics.get(key) for key in ("hits", "items", "accuracy")))
        checks[category + "_score_flag"] = (details.get("score_target_met") is
            _meets(metrics.get("accuracy"), plan.contrast_min))
    scores = (checks["dev_count_consistency"] and dev["comprehension_accuracy"] >= plan.comprehension_min
        and dev["valid_generation_accuracy"] >= plan.production_min
        and all(checks[category + "_counts"] and categories[category]["accuracy"] >= plan.contrast_min
                for category in CATEGORIES))
    covered = all(checks[category + "_coverage"] for category in CATEGORIES)
    checks["candidate_flags_consistent"] = (candidate.get("score_targets_met") is bool(scores)
        and candidate.get("coverage_sufficient") is bool(covered)
        and candidate.get("status") == ("indeterminate_coverage" if not covered else
                                       "development_candidate" if scores else "targets_not_met"))
    return checks, bool(scores and covered and all(checks.values()))


def aggregate_candidate_validations(suites, plan=None, *, candidate_request=None):
    """Require all twelve fixed final checkpoints; retain failures and absences."""
    if plan is None:
        plan = MasteryPlan.from_dict(suites[0]["plan"]) if suites else replace(
            _reference_plan(), seed_pairs=SEED_PAIRS, b_seed_pair=SEED_PAIRS[0])
    _validate_plan(plan); _validate_request(candidate_request, plan)
    indexed, rows, failures, trial_checks = {}, [], [], []
    for suite in suites:
        if suite.get("kind") != VERSION:
            raise ValueError("Expected candidate-validation.json, not discovery or raw engine reports")
        language, replica = suite.get("language"), suite.get("replica")
        if language not in LANGUAGES_ORDER or type(replica) is not int or replica not in range(3):
            raise ValueError("Unknown validation language or replica")
        key = (replica, language)
        if key in indexed:
            raise ValueError("Duplicate validation attempt; retry/winner selection is not allowed")
        indexed[key] = suite
        if _canonical(suite.get("plan")) != _canonical(asdict(plan)):
            raise ValueError("Validation plans differ")
        request = suite.get("candidate_request")
        _validate_request(request, plan)
        if candidate_request is None and len(indexed) == 1:
            candidate_request = request
        if request != candidate_request:
            raise ValueError("Pinned discovery candidate provenance differs across validation jobs")
        raw = suite.get("engine_report", {})
        pair, initial = SEED_PAIRS[replica], suite.get("initial_state", {})
        expected_trial_plan = asdict(replace(plan, b_seed_pair=pair))
        expected_steps = [0, *plan.budgets]
        points = raw.get("curves", [])
        benchmark = raw.get("benchmark")
        checks = {"fixed_model": suite.get("model") == "attentive" and suite.get("fixed_prior_condition") == "B1"
            and suite.get("policy") == "mixed_blocks" and suite.get("negative_policy_specification") == negative_policy_manifest("mixed_blocks")
            and raw.get("model_factory") == FACTORY and raw.get("policy") == "mixed_blocks",
            "fresh_language_training": suite.get("fresh_initialization") is True and suite.get("resumed") is False
                and suite.get("scratch_language_training") is True and suite.get("shared_visual_prior") is True
                and suite.get("independent_visual_replications") is False,
            "declared_seed": suite.get("seed_pair") == list(pair) == raw.get("seed_pair"),
            "engine_adapter": raw.get("phase") == "C" and raw.get("condition") == "C1"
                and raw.get("language") == language and raw.get("replica") == replica,
            "trial_plan": _canonical(raw.get("plan")) == _canonical(expected_trial_plan),
            "completed": suite.get("status") == "completed" and raw.get("status") == "completed",
            "all_budgets": [point.get("step") for point in points] == expected_steps
                and raw.get("training_steps_completed") == plan.budgets[-1],
            "all_checkpoint_manifests": [item.get("step") for item in raw.get("checkpoints", [])] == expected_steps
                and all(item.get("format") == "mutual-mastery-development-v1" and item.get("kind") == "development_only"
                    and item.get("excluded_from_research") is True and item.get("eligible_for_fresh_experiment") is False
                    and type(item.get("step")) is int
                    and item.get("path") == f"step-{item.get('step'):06d}.pt"
                    and isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64
                    for item in raw.get("checkpoints", [])),
            "same_development_population": raw.get("data_version") == DATA_VERSION
                and raw.get("split", {}).get("version") == DATA_VERSION
                and raw.get("split", {}).get("pair_sha256") == PAIR_SHA256
                and raw.get("split", {}).get("scene_counts") == {"train": 220, "dev": 112}
                and raw.get("split", {}).get("png_pool_sha256") == PNG_SHA256,
            "initialization_captured": bool(initial) and initial.get("model") == "attentive"
                and initial.get("understanding_sha256") == raw.get("initial_understanding_sha256")
                and initial.get("producer_sha256") == raw.get("initial_producer_sha256")
                and initial.get("prior_sha256") == raw.get("initial_prior_sha256")
                and initial.get("parameter_count") == raw.get("parameter_count")
                and initial.get("trainable_parameter_count") == raw.get("trainable_parameter_count"),
            "frozen_prior": bool(raw.get("initial_prior_sha256"))
                and raw.get("initial_prior_sha256") == raw.get("final_prior_sha256")
                and raw.get("prior_metadata", {}).get("condition") == "B1"
                and raw.get("prior_metadata", {}).get("prior_sha256") == raw.get("initial_prior_sha256"),
            "benchmark_reset": plan.benchmark_steps == 0 or (isinstance(benchmark, Mapping)
                and benchmark.get("steps") == plan.benchmark_steps
                and benchmark.get("initial_weights_restored") is True),
            "native_only_input": raw.get("training_languages") == [language]
                and raw.get("foreign_training_languages") == [] and raw.get("protected_splits_used") == [],
            "development_only": suite.get("excluded_from_research") is True and suite.get("eligible_for_fresh_experiment") is False
                and raw.get("excluded_from_research") is True and raw.get("eligible_for_fresh_experiment") is False,
            "fixed_training_controls": raw.get("optimization") == "independent-branch-clipping-adam-v1"
                and raw.get("clip_norm_per_branch") == 1 and raw.get("choice_count") == 4
                and raw.get("training_chance_accuracy") == .25 and raw.get("comprehension_forms") == 4
                and raw.get("production_forms") == 1 and raw.get("teacher_forced_evaluation_forms") == 4
                and raw.get("negative_sampling_seed") == pair[1] + plan.negative_seed_offset
                and raw.get("model_profile") == asdict(plan.model_profile)
                and type(raw.get("parameter_count")) is int and raw["parameter_count"] > 0
                and type(raw.get("trainable_parameter_count")) is int
                and 0 < raw["trainable_parameter_count"] < raw["parameter_count"]}
        exposure = raw.get("negative_exposures", {})
        total = plan.budgets[-1] * raw.get("batch_size", 0) if type(raw.get("batch_size")) is int else 0
        checks["negative_exposure_counts"] = (total > 0 and exposure.get("examples") == total
            and exposure.get("distractor_slots") == total * 3 and exposure.get("semantic_duplicate_choices") == 0
            and exposure.get("png_duplicate_choices") == 0 and all(exposure.get(f"branch_{branch}_examples") == count
                for branch, count in _mixed_branch_counts(total).items()))
        final_pass = False
        for point in points:
            point_checks, passes = _point_checks(point, plan)
            if not all(point_checks.values()):
                checks["point_" + str(point.get("step"))] = False
            if point.get("step") == plan.budgets[-1]:
                final_pass = passes
            rows.append({"language": language, "replica": replica, "seed_pair": list(pair),
                "step": point.get("step"), "status": suite.get("status"), "dev": point.get("dev", {}),
                "contrasts": point.get("contrasts", {}).get("categories", {}),
                "candidate": point.get("candidate", {}), "point_checks": point_checks})
        trial_checks.append({"language": language, "replica": replica, "checks": checks,
                             "final_candidate": final_pass and all(checks.values())})
        if not all(checks.values()):
            failures.append({"language": language, "replica": replica, "status": suite.get("status"),
                             "failed_checks": [key for key, passed in checks.items() if not passed],
                             "error": suite.get("error", raw.get("error"))})
    missing = [{"replica": replica, "language": language} for replica in range(3)
               for language in LANGUAGES_ORDER if (replica, language) not in indexed]
    blocks, initial_u, initial_p, priors, signatures = [], [], [], [], []
    for replica, pair in enumerate(SEED_PAIRS):
        block_suites = [indexed[(replica, language)] for language in LANGUAGES_ORDER if (replica, language) in indexed]
        raws = [suite.get("engine_report", {}) for suite in block_suites]
        fields = ("initial_understanding_sha256", "initial_producer_sha256", "initial_prior_sha256", "meaning_stream_sha256")
        block_checks = {key: len(raws) == 4 and all(bool(raw.get(key)) for raw in raws)
            and len({_canonical(raw.get(key)) for raw in raws}) == 1 for key in fields}
        for raw in raws:
            priors.append(raw.get("initial_prior_sha256"))
            signatures.append({key: raw.get(key) for key in ("config_sha256", "source_sha256", "model_profile",
                "model_factory", "parameter_count", "trainable_parameter_count", "batch_size", "learning_rate", "split", "prior_metadata")})
        if raws:
            initial_u.append(raws[0].get("initial_understanding_sha256"))
            initial_p.append(raws[0].get("initial_producer_sha256"))
        block_trials = [item for item in trial_checks if item["replica"] == replica]
        blocks.append({"replica": replica, "seed_pair": list(pair), "reported_languages": [s["language"] for s in block_suites],
            "checks": block_checks, "final_candidate_count": sum(item["final_candidate"] for item in block_trials),
            "all_four_final_candidates": len(block_trials) == 4 and all(item["final_candidate"] for item in block_trials)})
    global_checks = {
        "all_twelve_jobs": len(indexed) == 12 and not missing,
        "all_seed_blocks_matched": all(all(block["checks"].values()) for block in blocks),
        "distinct_language_initializations_across_seed_blocks": len(initial_u) == 3 and all(initial_u)
            and len(set(initial_u)) == 3 and len(initial_p) == 3 and all(initial_p) and len(set(initial_p)) == 3,
        "one_shared_frozen_visual_prior": len(priors) == 12 and all(priors) and len(set(priors)) == 1,
        "same_protocol_provenance": len(signatures) == 12 and bool(signatures[0].get("source_sha256"))
            and bool(signatures[0].get("config_sha256")) and len({_canonical(s) for s in signatures}) == 1,
        "all_trials_valid": len(trial_checks) == 12 and all(all(item["checks"].values()) for item in trial_checks),
        "all_twelve_fixed_final_candidates": len(trial_checks) == 12 and all(item["final_candidate"] for item in trial_checks)}
    completed = not missing and not failures and all(value for key, value in global_checks.items()
        if key != "all_twelve_fixed_final_candidates")
    return {"schema_version": 1, "kind": VERSION + "-comparison", "status": "completed" if completed else "incomplete",
        "robust_development_candidate": all(global_checks.values()), "fixed_final_step": plan.budgets[-1],
        "model": "attentive", "fixed_prior_condition": "B1", "policy": "mixed_blocks", "plan": asdict(plan),
        "candidate_request": candidate_request, "expected_jobs": 12, "reported_jobs": len(indexed),
        "missing_trials": missing, "failed_trials": failures, "checks": global_checks,
        "seed_blocks": blocks, "trial_checks": trial_checks, "rows": rows,
        "final_candidate_count": sum(item["final_candidate"] for item in trial_checks),
        "noncandidate_final_trials": [{"language": item["language"], "replica": item["replica"]}
            for item in trial_checks if not item["final_candidate"]],
        "selection": "All predeclared final checkpoints required; no winner, retry, best-step selection or automatic research promotion",
        "shared_visual_prior": True, "independent_visual_replications": False,
        "excluded_from_research": True, "eligible_for_fresh_experiment": False, "interpretation": SCOPE}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/neural.example.json")
    parser.add_argument("--plan", type=Path, default=root / "configs/mastery-validation-plan.json")
    parser.add_argument("--language", choices=LANGUAGES_ORDER)
    parser.add_argument("--replica", type=int, choices=range(3))
    parser.add_argument("--prior-bundle", type=Path)
    parser.add_argument("--candidate-request", type=Path)
    parser.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = MasteryPlan.load(args.plan)
    request = _load_request(args.candidate_request, plan, plan_path=args.plan,
                            prior_path=args.prior_bundle if args.aggregate_dir is None else None)
    if args.aggregate_dir is not None:
        paths = sorted(args.aggregate_dir.rglob("candidate-validation.json"))
        result = aggregate_candidate_validations([json.loads(path.read_text()) for path in paths], plan,
                                                 candidate_request=request)
        destination = args.output or Path("runs/candidate-validation-comparison.json")
        destination.parent.mkdir(parents=True, exist_ok=True); safe_json(destination, result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result["status"] == "completed" else 1
    if args.language is None or args.replica is None or args.prior_bundle is None:
        parser.error("--language, --replica and --prior-bundle are required for explicit validation execution")
    from .config import ExperimentConfig
    from .spatial_prior import load_prior_bundle
    path = run_candidate_validation(ExperimentConfig.load(args.config), args.language, args.replica, plan,
        args.output or "runs/candidate-validation", prior_bundle=load_prior_bundle(args.prior_bundle),
        candidate_request=request)
    print(path, flush=True)
    return 0 if json.loads((path / "candidate-validation.json").read_text())["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
