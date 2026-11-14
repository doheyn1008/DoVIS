"""Unactivated E comparison: original negatives versus alternating fit negatives.

Only the comprehension distractor policy differs. The C development world,
frozen prior, language architecture, initial weights, native input streams,
optimization and budgets are fixed. E0 uses same_pair; E1 uses mixed_blocks.
Raw C0/C1 labels select a fixed B0/B1 prior and are adapter metadata only.
Importing this module does not train, select a model, or schedule any work.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from .composition_diagnostics import CATEGORIES
from .diagnostics import safe_json
from .fit_distractors import MIXED_BLOCK_EXPOSURES, negative_policy_manifest
from .mastery_lab import MasteryPlan, run_trial
from .native_preparation import LANGUAGES_ORDER
from .storage import now

VERSION = "negative-policy-study-v1"
POLICIES = {"E0": "same_pair", "E1": "mixed_blocks"}
ADAPTER = {"engine_phase": "C", "mapping": {"B0": "C0", "B1": "C1"},
    "meaning": "Both E arms use the SAME raw C condition for the fixed prior; C0/C1 are not negative-policy conditions",
    "raw_artifacts": "Raw C reports/checkpoints remain unchanged; negative-policy-study.json declares the E axis"}


class FixedLanguageFactory:
    """Construct one trial and immediately capture its complete initialization."""

    def __init__(self, model, prior_condition):
        self.__module__ = 'mutual_lab.negative_policy_study'
        if model not in ("standard", "attentive") or prior_condition not in ("B0", "B1"):
            raise ValueError("Unknown fixed model or prior")
        self.model, self.prior_condition, self.initial_state = model, prior_condition, None
        self.__qualname__ = "FixedLanguageFactory." + model

    def __call__(self, profile, initialization_seed, bundle, condition="B1"):
        if condition != self.prior_condition or self.initial_state is not None:
            raise ValueError("Factory must construct exactly one trial with the declared prior")
        from .attentive_spatial import build_attentive_spatial_model
        from .neural import digest
        from .spatial_prior import build_spatial_model, spatial_prior_hash
        builder = build_spatial_model if self.model == "standard" else build_attentive_spatial_model
        model = builder(profile, initialization_seed, bundle, condition=condition)
        self.initial_state = {"model": self.model,
            "understanding_sha256": digest(model.understanding.state_dict()),
            "producer_sha256": digest(model.producer.state_dict()), "prior_sha256": spatial_prior_hash(model),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "capacity_matched": True, "input_scope": "PNG pixels and native text; no Scene metadata or oracle labels"}
        return model


def _mixed_branch_counts(exposures):
    cycles, tail = divmod(exposures, 2 * MIXED_BLOCK_EXPOSURES)
    same = cycles * MIXED_BLOCK_EXPOSURES + min(tail, MIXED_BLOCK_EXPOSURES)
    return {"same_pair": same, "attributes": exposures - same}


def _candidate_coverage(report):
    return [{"step": point["step"], "categories": {category:
        {key: value[key] for key in ("available_anchors", "available_pairs", "coverage_sufficient")}
        for category, value in point["candidate"]["categories"].items()}}
        for point in report.get("curves", [])]


def negative_policy_pair_audit(reports, prior_condition, model):
    """Require decoder invariance while allowing understanding to change."""
    if prior_condition not in ("B0", "B1") or model not in ("standard", "attentive"):
        raise ValueError("Unknown fixed model or prior")
    names = [r["negative_policy_condition"] for r in reports]
    if len(names) != len(set(names)) or set(names) - POLICIES.keys():
        raise ValueError("Duplicate or unknown negative-policy arm")
    pair = {r["negative_policy_condition"]: r for r in reports}
    if any(name not in pair or pair[name]["status"] != "completed" for name in POLICIES):
        return {"status": "incomplete", "checks": {}}
    left, right = (pair[name] for name in POLICIES)
    a, b = left["engine_report"], right["engine_report"]
    init_a, init_b = left["initial_state"], right["initial_state"]
    keys = ("language", "replica", "seed_pair", "plan_sha256", "split", "data_sha256", "config_sha256",
        "source_sha256", "model_profile", "model_factory", "parameter_count", "trainable_parameter_count",
        "batch_size", "learning_rate", "negative_sampling_seed", "optimization", "clip_norm_per_branch",
        "choice_count", "training_chance_accuracy", "comprehension_forms", "production_forms", "teacher_forced_evaluation_forms",
        "initial_understanding_sha256", "initial_producer_sha256", "initial_prior_sha256",
        "final_producer_sha256", "final_prior_sha256", "meaning_stream_sha256", "comprehension_caption_stream_sha256",
        "environment_sha256", "frozen_feature_policy", "frozen_feature_tables")
    checks = {key: key in a and key in b and a[key] == b[key] for key in keys}
    checks.update({
        "complete_initial_state": init_a == init_b and bool(init_a),
        "captured_initialization": all(init.get("understanding_sha256") == raw.get("initial_understanding_sha256")
            and init.get("producer_sha256") == raw.get("initial_producer_sha256")
            and init.get("parameter_count") == raw.get("parameter_count")
            and init.get("trainable_parameter_count") == raw.get("trainable_parameter_count")
            for init, raw in ((init_a, a), (init_b, b))),
        "fixed_model_and_prior": all(record.get("model") == model and record.get("fixed_prior_condition") == prior_condition
            and record["initial_state"].get("model") == model for record in (left, right)),
        "frozen_common_prior": all(init.get("prior_sha256") == raw.get("initial_prior_sha256") == raw.get("final_prior_sha256")
            for init, raw in ((init_a, a), (init_b, b))),
        "negative_policy_axis": all(record.get("negative_policy") == POLICIES[name]
            and record["engine_report"].get("policy") == POLICIES[name]
            and record.get("negative_policy_specification") == negative_policy_manifest(POLICIES[name])
            for name, record in pair.items()),
        "engine_adapter": all(raw.get("phase") == "C" and raw.get("condition") == ADAPTER["mapping"][prior_condition]
            and raw.get("status") == "completed" for raw in (a, b)),
        "independent_branch_clipping": all(raw.get("optimization") == "independent-branch-clipping-adam-v1"
            and raw.get("clip_norm_per_branch") == 1 for raw in (a, b)),
        "input_boundary": all(raw.get("training_languages") == [raw.get("language")]
            and raw.get("foreign_training_languages") == [] and raw.get("protected_splits_used") == []
            and raw.get("excluded_from_research") is True and raw.get("eligible_for_fresh_experiment") is False
            for raw in (a, b)),
        "all_budgets": all(len(raw.get("curves", [])) == len(raw["plan"]["budgets"]) + 1
            and {p["step"] for p in raw["curves"]} == {0, *raw["plan"]["budgets"]}
            and raw.get("training_steps_completed") == raw["plan"]["budgets"][-1] for raw in (a, b)),
        "candidate_coverage": _candidate_coverage(a) == _candidate_coverage(b),
        "candidate_development_only": all(p["candidate"].get("eligible_for_fresh_experiment") is False
            and p["candidate"].get("excluded_from_research") is True
            and set(p["candidate"]["categories"]) == set(CATEGORIES) for raw in (a, b) for p in raw.get("curves", [])),
        "four_choices": all(raw.get("choice_count") == 4 and raw.get("training_chance_accuracy") == .25 for raw in (a, b)),
        "deterministic_frozen_features": all(raw.get("frozen_feature_policy") == "per_png_singleton_v1"
            and isinstance(raw.get("frozen_feature_tables", {}).get("step_0"), dict)
            and raw["frozen_feature_tables"]["step_0"].get("policy") == "per_png_singleton_v1"
            and raw["frozen_feature_tables"].get("final") == raw["frozen_feature_tables"]["step_0"]
            for raw in (a, b)),
    })
    expected = a["training_steps_completed"] * a["batch_size"]
    checks["negative_exposure_totals"] = all(raw.get("negative_exposures", {}).get("examples") == expected
        and raw["negative_exposures"].get("distractor_slots") == expected * 3
        and raw["negative_exposures"].get("semantic_duplicate_choices") == 0
        and raw["negative_exposures"].get("png_duplicate_choices") == 0 for raw in (a, b))
    checks["positive_exposure_counts"] = ({key: value for key, value in a["negative_exposures"].items()
        if key.startswith("anchor_exposures:")} == {key: value for key, value in b["negative_exposures"].items()
        if key.startswith("anchor_exposures:")})
    checks["mixed_branch_schedule"] = all(b["negative_exposures"].get(f"branch_{branch}_examples") == count
        for branch, count in _mixed_branch_counts(expected).items())
    return {"status": "matched" if all(checks.values()) else "mismatch", "checks": checks}


def run_negative_policy_study(config, language, plan=MasteryPlan(), output="runs/negative-policy-study", progress=print,
                              *, prior_bundle=None, prior_condition="B1", model="standard"):
    """Explicit future E execution; fixes the architecture before either arm."""
    plan.validate(config)
    if language not in LANGUAGES_ORDER or prior_bundle is None:
        raise ValueError("Supply a known language and one common frozen prior bundle")
    FixedLanguageFactory(model, prior_condition)  # Validate before output creation.
    directory = Path(output).resolve() / f"negative-policy-{language}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    suite = {"schema_version": 1, "kind": VERSION, "phase": "E", "study_phase": "E", "axis": "negative_policy",
        "language": language, "model": model, "fixed_prior_condition": prior_condition, "plan": asdict(plan),
        "created_at": now(), "status": "running", "engine_adapter": ADAPTER, "capacity_matched": True,
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "interpretation": "Development-only negative-policy comparison; fixed model/prior, one b_seed_pair, no automatic winner or independent replication claim",
        "reports": [], "pair_audit": {"status": "incomplete", "checks": {}}}
    begun = time.monotonic()

    def snapshot():
        suite["pair_audit"] = negative_policy_pair_audit(suite["reports"], prior_condition, model)
        suite["elapsed_seconds"] = time.monotonic() - begun
        safe_json(directory / "negative-policy-study.json", suite)

    snapshot()
    for condition, policy in POLICIES.items():
        factory = FixedLanguageFactory(model, prior_condition)
        record = {"negative_policy_condition": condition, "negative_policy": policy,
            "negative_policy_specification": negative_policy_manifest(policy), "fixed_prior_condition": prior_condition,
            "model": model, "replica": 0, "seed_pair": list(plan.b_seed_pair), "status": "running", "initial_state": {}}
        remaining = plan.max_suite_seconds - (time.monotonic() - begun)
        try:
            if remaining <= 0:
                record["status"] = "suite_time_limit"
            else:
                path = run_trial(config, language, "C", ADAPTER["mapping"][prior_condition], 0, plan.b_seed_pair, plan,
                    directory / "raw-engine", progress, policy=policy, prior_bundle=prior_bundle,
                    remaining_seconds=remaining, model_factory=factory)
                raw = json.loads((path / "report.json").read_text())
                record.update(status=raw["status"], engine_report=raw,
                    raw_report_path=str((path / "report.json").relative_to(directory)))
        except KeyboardInterrupt as exc:
            from .composition_suite import ArmCancelled
            record.update(status="cancelled", initial_state=factory.initial_state or {})
            suite["reports"].append(record); suite["status"] = "cancelled"
            try:
                if isinstance(exc, ArmCancelled):
                    raw = getattr(exc, "report", None)
                    if not isinstance(raw, Mapping):
                        try:
                            raw = json.loads(exc.report_path.read_text())
                        except Exception:
                            raw = None
                    if isinstance(raw, Mapping):
                        record["engine_report"] = dict(raw)
                    try:
                        record["raw_report_path"] = str(exc.report_path.relative_to(directory))
                    except Exception:
                        pass
            finally:
                try:
                    snapshot()
                finally:
                    # Preserve the original interrupt even when persistence fails.
                    raise exc
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record["initial_state"] = factory.initial_state or {}
        suite["reports"].append(record); snapshot()
    suite["status"] = "completed" if suite["pair_audit"]["status"] == "matched" else "incomplete"
    snapshot()
    return directory


def aggregate_negative_policy_studies(suites):
    """Read only E manifests; keep per-budget counts and all incomplete arms."""
    seen, protocols, rows, failed, deltas, signatures = set(), set(), [], [], [], {}
    shared_prior = None
    for suite in suites:
        if suite.get("kind") != VERSION or suite.get("study_phase") != "E" or suite.get("axis") != "negative_policy":
            raise ValueError("Expected negative-policy-study.json; raw C or decoder reports are not E studies")
        language = suite["language"]
        if language not in LANGUAGES_ORDER or language in seen:
            raise ValueError("Unknown or duplicate language")
        seen.add(language)
        protocols.add(json.dumps({key: suite[key] for key in ("plan", "model", "fixed_prior_condition")}, sort_keys=True))
        plan, reports = MasteryPlan.from_dict(suite["plan"]), suite["reports"]
        audit = negative_policy_pair_audit(reports, suite["fixed_prior_condition"], suite["model"])
        if audit["status"] != "matched":
            failed.append({"language": language, "status": "pair_" + audit["status"], "checks": audit["checks"]})
        for condition in sorted(POLICIES.keys() - {r["negative_policy_condition"] for r in reports}):
            failed.append({"language": language, "condition": condition, "status": "missing"})
        paired = {}
        for record in reports:
            condition, status, raw = record["negative_policy_condition"], record["status"], record.get("engine_report", {})
            if raw and (raw.get("language") != language or raw.get("replica") != 0
                or raw.get("seed_pair") != list(plan.b_seed_pair) or raw.get("plan") != suite["plan"]):
                raise ValueError("Raw engine metadata differs from its E manifest")
            if status != "completed":
                failed.append({"language": language, "condition": condition, "status": status})
            else:
                signature = {key: raw.get(key) for key in ("split", "config_sha256", "source_sha256", "model_profile",
                    "model_factory", "seed_pair", "policy", "optimization", "clip_norm_per_branch")}
                signature["initial_state"] = record["initial_state"]
                if signatures.setdefault(condition, signature) != signature:
                    raise ValueError("E provenance differs across languages")
                if shared_prior is None:
                    shared_prior = raw.get("initial_prior_sha256")
                if raw.get("initial_prior_sha256") != shared_prior:
                    raise ValueError("All E languages/arms must share one frozen visual prior")
                if len(raw.get("curves", [])) != len(plan.budgets) + 1 or {p["step"] for p in raw["curves"]} != {0, *plan.budgets}:
                    failed.append({"language": language, "condition": condition, "status": "missing_budget"})
            for point in raw.get("curves", []):
                row = {"language": language, "condition": condition, "negative_policy": POLICIES[condition],
                    "step": point["step"], "status": status, "dev": point["dev"],
                    "contrasts": point["contrasts"]["categories"], "candidate": point["candidate"]}
                rows.append(row); paired[(condition, point["step"])] = row
        if audit["status"] == "matched":
            for step in (0, *plan.budgets):
                a, b = (paired[(condition, step)] for condition in POLICIES)
                deltas.append({"language": language, "step": step,
                    "comprehension_delta": b["dev"]["comprehension_accuracy"] - a["dev"]["comprehension_accuracy"],
                    "generation_delta": b["dev"]["valid_generation_accuracy"] - a["dev"]["valid_generation_accuracy"],
                    "contrast_deltas": {category: b["contrasts"][category]["accuracy"] - a["contrasts"][category]["accuracy"]
                        if a["contrasts"][category]["accuracy"] is not None and b["contrasts"][category]["accuracy"] is not None else None
                        for category in a["contrasts"]}})
    if len(protocols) > 1:
        raise ValueError("E plans, fixed models or selected priors differ")
    missing, counts = [language for language in LANGUAGES_ORDER if language not in seen], []
    budgets = (0, *MasteryPlan.from_dict(suites[0]["plan"]).budgets) if suites else ()
    for condition in POLICIES:
        for step in budgets:
            available = [row for row in rows if row["condition"] == condition and row["step"] == step]
            completed = [row for row in available if row["status"] == "completed"]
            counts.append({"condition": condition, "step": step, "expected_languages": len(LANGUAGES_ORDER),
                "reported_rows": len(available), "completed_rows": len(completed),
                **{key: sum(row["dev"][key] for row in completed) for key in
                   ("comprehension_hits", "comprehension_items", "valid_generation_hits", "generation_scenes")}})
    return {"schema_version": 1, "kind": VERSION + "-comparison", "study_phase": "E", "axis": "negative_policy",
        "status": "incomplete" if missing or failed else "completed", "missing_languages": missing,
        "failed_trials": failed, "rows": rows, "budget_counts": counts, "paired_deltas": deltas,
        "model": suites[0]["model"] if suites else None,
        "fixed_prior_condition": suites[0]["fixed_prior_condition"] if suites else None,
        "negative_policies": {condition: negative_policy_manifest(policy) for condition, policy in POLICIES.items()},
        "capacity_matched": True, "selection": "undetermined; this comparison has no automatic policy selection rule",
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "interpretation": "Development-only; same fixed architecture and shared prior are not independent replications"}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/neural.example.json")
    parser.add_argument("--plan", type=Path, default=root / "configs/mastery-C-plan.json")
    parser.add_argument("--language", choices=LANGUAGES_ORDER)
    parser.add_argument("--prior-bundle", type=Path)
    parser.add_argument("--prior-condition", choices=("B0", "B1"), default="B1")
    parser.add_argument("--model", choices=("standard", "attentive"), default="standard")
    parser.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.aggregate_dir is not None:
        paths = sorted(args.aggregate_dir.rglob("negative-policy-study.json"))
        if not paths:
            raise ValueError("No negative-policy-study.json manifests found")
        result = aggregate_negative_policy_studies([json.loads(path.read_text()) for path in paths])
        destination = args.output or Path("runs/negative-policy-comparison.json")
        destination.parent.mkdir(parents=True, exist_ok=True); safe_json(destination, result)
        print(json.dumps(result), flush=True); return 0
    if args.language is None or args.prior_bundle is None:
        parser.error("--language and --prior-bundle are required for explicit E execution")
    from .config import ExperimentConfig
    from .spatial_prior import load_prior_bundle
    path = run_negative_policy_study(ExperimentConfig.load(args.config), args.language, MasteryPlan.load(args.plan),
        args.output or "runs/negative-policy-study", prior_bundle=load_prior_bundle(args.prior_bundle),
        prior_condition=args.prior_condition, model=args.model)
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
