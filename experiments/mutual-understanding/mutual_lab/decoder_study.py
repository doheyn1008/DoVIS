"""Conditional decoder-axis development comparison, never activated on import.

D0 retains global-vector conditioning. D1 adds post-GRU grid attention and
parameters. Both use one selected, identical B0 or B1 frozen visual prior.
The unchanged C engine supplies the five-color data/training protocol: its
C0/C1 labels are adapter labels only, not this study's comparison conditions.
Raw C reports and checkpoints are preserved and remain development-only.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from .diagnostics import safe_json
from .mastery_lab import MasteryPlan, run_trial
from .native_preparation import LANGUAGES_ORDER
from .storage import now

VERSION = "decoder-study-v1"
CONDITIONS = ("D0", "D1")
ADAPTER = {
    "engine_phase": "C",
    "mapping": {"B0": "C0", "B1": "C1"},
    "meaning": "Both decoder arms use the SAME raw C condition selected by the fixed visual prior; C labels are engine adapter metadata, not decoder arms",
    "raw_artifacts": "Raw C reports/checkpoints are unchanged; decoder-study.json is the study-axis manifest",
}


class RecordingFactory:
    """Capture the actual trial initialization, without constructing audit models."""

    def __init__(self, decoder_condition, prior_condition):
        self.__module__ = 'mutual_lab.decoder_study'
        if decoder_condition not in CONDITIONS or prior_condition not in ("B0", "B1"):
            raise ValueError("Unknown decoder or fixed-prior condition")
        self.decoder_condition, self.prior_condition = decoder_condition, prior_condition
        self.__qualname__ = "RecordingFactory." + decoder_condition
        self.initial_state = None

    def __call__(self, profile, initialization_seed, bundle, condition="B1"):
        if condition != self.prior_condition or self.initial_state is not None:
            raise ValueError("Factory must construct exactly one trial with its declared common prior")
        from .attentive_spatial import build_attentive_spatial_model
        from .neural import digest
        from .spatial_prior import build_spatial_model, spatial_prior_hash
        builder = build_spatial_model if self.decoder_condition == "D0" else build_attentive_spatial_model
        model = builder(profile, initialization_seed, bundle, condition=condition)
        base = {name: value for name, value in model.producer.state_dict().items()
                if name.startswith(("embedding.", "gru.", "output."))}
        extra = sum(parameter.numel() for name, parameter in model.producer.named_parameters()
                    if not name.startswith(("embedding.", "gru.", "output.")))
        self.initial_state = {
            "understanding_sha256": digest(model.understanding.state_dict()),
            "base_producer_sha256": digest(base), "prior_sha256": spatial_prior_hash(model),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "decoder_extra_parameter_count": extra,
            "residual_vocabulary_projection_zero": (self.decoder_condition == "D1" and
                bool((model.producer.attention_output.weight == 0).all())),
            "capacity_matched": False,
            "input_scope": "PNG pixels and native-language text only; no Scene metadata or oracle labels",
        }
        return model


def decoder_pair_audit(reports, prior_condition):
    """Check decoder-only effects without pretending full producer hashes match."""
    if prior_condition not in ("B0", "B1"):
        raise ValueError("Unknown fixed prior")
    if len({r["decoder_condition"] for r in reports}) != len(reports) or any(
            r["decoder_condition"] not in CONDITIONS for r in reports):
        raise ValueError("Duplicate or unknown decoder arm")
    pair = {report["decoder_condition"]: report for report in reports}
    if any(name not in pair or pair[name]["status"] != "completed" for name in CONDITIONS):
        return {"status": "incomplete", "checks": {}}
    left, right = (pair[name] for name in CONDITIONS)
    a, b = left["engine_report"], right["engine_report"]
    init_a, init_b = left["initial_state"], right["initial_state"]
    keys = ("language", "replica", "seed_pair", "plan_sha256", "split", "data_sha256",
            "config_sha256", "source_sha256", "model_profile", "batch_size", "learning_rate",
            "policy", "negative_sampling_seed", "negative_exposures", "optimization", "clip_norm_per_branch",
            "initial_understanding_sha256", "meaning_stream_sha256", "comprehension_caption_stream_sha256",
            "environment_sha256", "final_understanding_sha256", "initial_prior_sha256", "final_prior_sha256")
    checks = {key: key in a and key in b and a[key] == b[key] for key in keys}
    checks.update({
        "base_producer_initialization": init_a.get("base_producer_sha256") == init_b.get("base_producer_sha256")
            and init_a.get("base_producer_sha256") is not None,
        "captured_understanding": all(init.get("understanding_sha256") == report.get("initial_understanding_sha256")
            for init, report in ((init_a, a), (init_b, b))),
        "frozen_common_prior": all(report.get("initial_prior_sha256") == report.get("final_prior_sha256")
            == init.get("prior_sha256") for init, report in ((init_a, a), (init_b, b))),
        "engine_adapter": all(report.get("phase") == "C" and report.get("condition") == ADAPTER["mapping"][prior_condition]
            for report in (a, b)),
        "engine_completion": all(report.get("status") == "completed" for report in (a, b)),
        "decoder_axis": all(report.get("fixed_prior_condition") == prior_condition for report in (left, right)),
        "input_boundary": all(report.get("training_languages") == [report.get("language")] and
            report.get("foreign_training_languages") == [] and report.get("protected_splits_used") == [] and
            report.get("excluded_from_research") is True and report.get("eligible_for_fresh_experiment") is False
            for report in (a, b)),
        "independent_branch_clipping": all(report.get("optimization") == "independent-branch-clipping-adam-v1"
            and report.get("clip_norm_per_branch") == 1 for report in (a, b)),
        "zero_residual_initialization": init_b.get("residual_vocabulary_projection_zero") is True,
        "parameter_accounting": (init_a.get("decoder_extra_parameter_count") == 0 and
            init_b.get("decoder_extra_parameter_count", 0) > 0 and
            init_b.get("parameter_count", 0) - init_a.get("parameter_count", 0) == init_b.get("decoder_extra_parameter_count") and
            init_b.get("trainable_parameter_count", 0) - init_a.get("trainable_parameter_count", 0) == init_b.get("decoder_extra_parameter_count") and
            all(init.get("parameter_count") == report.get("parameter_count") and
                init.get("trainable_parameter_count") == report.get("trainable_parameter_count")
                for init, report in ((init_a, a), (init_b, b)))),
        "all_budgets": all(len(report.get("curves", [])) == len(report["plan"]["budgets"]) + 1 and
            {point["step"] for point in report["curves"]} == {0, *report["plan"]["budgets"]} and
            report.get("training_steps_completed") == report["plan"]["budgets"][-1] for report in (a, b)),
    })
    return {"status": "matched" if all(checks.values()) else "mismatch", "checks": checks}


def run_decoder_study(config, language, plan=MasteryPlan(), output="runs/decoder-study", progress=print,
                      *, policy="same_pair", prior_bundle=None, prior_condition="B1"):
    """Explicitly run a future D pair; importing/preparing this module runs nothing."""
    plan.validate(config)
    if language not in LANGUAGES_ORDER or policy not in ("same_pair", "attributes"):
        raise ValueError("Unknown language or distractor policy")
    if prior_condition not in ("B0", "B1") or prior_bundle is None:
        raise ValueError("Supply one explicit B0/B1 common frozen prior bundle")
    directory = Path(output).resolve() / f"decoder-{language}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    suite = {"schema_version": 1, "kind": VERSION, "phase": "D", "axis": "decoder",
        "language": language, "fixed_prior_condition": prior_condition, "policy": policy,
        "plan": asdict(plan), "status": "running", "created_at": now(),
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "engine_adapter": ADAPTER, "capacity_matched": False,
        "interpretation": "Architecture plus additional decoder parameters; shared visual prior and development selection are not independent replication or L1 research qualification",
        "seed_replication": "One declared b_seed_pair per language; shared frozen visual prior across arms/languages",
        "reports": [], "pair_audit": {"status": "incomplete", "checks": {}}}
    begun = time.monotonic()

    def snapshot():
        suite["pair_audit"] = decoder_pair_audit(suite["reports"], prior_condition)
        suite["elapsed_seconds"] = time.monotonic() - begun
        safe_json(directory / "decoder-study.json", suite)

    snapshot()
    for condition in CONDITIONS:
        factory = RecordingFactory(condition, prior_condition)
        record = {"decoder_condition": condition, "fixed_prior_condition": prior_condition,
                  "replica": 0, "seed_pair": list(plan.b_seed_pair), "initial_state": {}, "status": "running"}
        remaining = plan.max_suite_seconds - (time.monotonic() - begun)
        try:
            if remaining <= 0:
                record["status"] = "suite_time_limit"
            else:
                path = run_trial(config, language, "C", ADAPTER["mapping"][prior_condition], 0,
                    plan.b_seed_pair, plan, directory / "raw-engine", progress, policy=policy,
                    prior_bundle=prior_bundle, remaining_seconds=remaining, model_factory=factory)
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
                    # Persistence must never replace the interrupt or start D1.
                    raise exc
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record["initial_state"] = factory.initial_state or {}
        suite["reports"].append(record); snapshot()
    suite["status"] = "completed" if suite["pair_audit"]["status"] == "matched" else "incomplete"
    snapshot()
    return directory


def aggregate_decoder_studies(suites):
    """Aggregate only decoder manifests, preserving failed and missing coverage."""
    seen, plans, fixed_priors, rows, failed, deltas, signatures = set(), set(), set(), [], [], [], {}
    shared_prior = None
    for suite in suites:
        if suite.get("kind") != VERSION or suite.get("phase") != "D" or suite.get("axis") != "decoder":
            raise ValueError("Expected decoder-study.json; raw C engine reports are not decoder studies")
        language = suite["language"]
        if language not in LANGUAGES_ORDER or language in seen:
            raise ValueError("Unknown or duplicate language")
        seen.add(language); plans.add(json.dumps(suite["plan"], sort_keys=True))
        fixed_priors.add(suite["fixed_prior_condition"])
        plan = MasteryPlan.from_dict(suite["plan"])
        reports = suite["reports"]
        audit = decoder_pair_audit(reports, suite["fixed_prior_condition"])
        if audit["status"] != "matched":
            failed.append({"language": language, "status": "pair_" + audit["status"], "checks": audit["checks"]})
        present = {report["decoder_condition"] for report in reports}
        for condition in sorted(set(CONDITIONS) - present):
            failed.append({"language": language, "condition": condition, "status": "missing"})
        paired = {}
        for record in reports:
            condition, status = record["decoder_condition"], record["status"]
            raw = record.get("engine_report", {})
            if raw and (raw.get("language") != language or raw.get("replica") != 0 or
                        raw.get("seed_pair") != list(plan.b_seed_pair) or raw.get("plan") != suite["plan"]):
                raise ValueError("Raw engine metadata differs from its decoder-study manifest")
            if status != "completed":
                failed.append({"language": language, "condition": condition, "status": status})
            else:
                initial = record["initial_state"]
                signature = {key: raw.get(key) for key in ("split", "config_sha256", "source_sha256", "model_profile",
                    "seed_pair", "policy", "optimization", "clip_norm_per_branch")}
                signature["initial_state"] = initial
                previous = signatures.setdefault(condition, signature)
                if previous != signature:
                    raise ValueError("Decoder study provenance differs across languages")
                if shared_prior is None:
                    shared_prior = raw.get("initial_prior_sha256")
                if raw.get("initial_prior_sha256") != shared_prior:
                    raise ValueError("Languages/arms must share the same frozen visual prior")
                if len(raw.get("curves", [])) != len(plan.budgets) + 1 or {p["step"] for p in raw["curves"]} != {0, *plan.budgets}:
                    failed.append({"language": language, "condition": condition, "status": "missing_budget"})
            for point in raw.get("curves", []):
                row = {"language": language, "condition": condition, "step": point["step"], "status": status,
                       "dev": point["dev"], "contrasts": point["contrasts"]["categories"],
                       "candidate": point["candidate"], "parameter_counts": record["initial_state"]}
                rows.append(row); paired[(condition, point["step"])] = row
        if audit["status"] == "matched":
            for step in (0, *plan.budgets):
                a, b = (paired[(condition, step)] for condition in CONDITIONS)
                deltas.append({"language": language, "step": step,
                    "comprehension_delta": b["dev"]["comprehension_accuracy"] - a["dev"]["comprehension_accuracy"],
                    "generation_delta": b["dev"]["valid_generation_accuracy"] - a["dev"]["valid_generation_accuracy"]})
    if len(plans) > 1 or len(fixed_priors) > 1:
        raise ValueError("Decoder plans or selected visual priors differ")
    missing = [language for language in LANGUAGES_ORDER if language not in seen]
    counts = []
    budgets = (0, *MasteryPlan.from_dict(suites[0]["plan"]).budgets) if suites else ()
    for condition in CONDITIONS:
        for step in budgets:
            available = [row for row in rows if row["condition"] == condition and row["step"] == step]
            completed = [row for row in available if row["status"] == "completed"]
            counts.append({"condition": condition, "step": step, "expected_languages": len(LANGUAGES_ORDER),
                "reported_rows": len(available), "completed_rows": len(completed),
                **{name: sum(row["dev"][name] for row in completed) for name in
                   ("comprehension_hits", "comprehension_items", "valid_generation_hits", "generation_scenes")}})
    return {"schema_version": 1, "kind": VERSION + "-comparison", "phase": "D", "axis": "decoder",
        "status": "incomplete" if missing or failed else "completed", "missing_languages": missing,
        "failed_trials": failed, "rows": rows, "budget_counts": counts, "paired_deltas": deltas,
        "capacity_matched": False, "fixed_prior_condition": next(iter(fixed_priors), None),
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "interpretation": "Development-only architecture plus capacity comparison; shared prior and one seed pair are not independent replication"}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/neural.example.json")
    parser.add_argument("--plan", type=Path, default=root / "configs/mastery-C-plan.json")
    parser.add_argument("--language", choices=LANGUAGES_ORDER)
    parser.add_argument("--prior-bundle", type=Path)
    parser.add_argument("--prior-condition", choices=("B0", "B1"), default="B1")
    parser.add_argument("--policy", choices=("same_pair", "attributes"), default="same_pair")
    parser.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.aggregate_dir is not None:
        paths = sorted(args.aggregate_dir.rglob("decoder-study.json"))
        if not paths:
            raise ValueError("No decoder-study.json manifests found")
        result = aggregate_decoder_studies([json.loads(path.read_text()) for path in paths])
        destination = args.output or Path("runs/decoder-comparison.json")
        destination.parent.mkdir(parents=True, exist_ok=True); safe_json(destination, result)
        print(json.dumps(result), flush=True); return 0
    if args.language is None or args.prior_bundle is None:
        parser.error("--language and --prior-bundle are required for explicit D execution")
    from .config import ExperimentConfig
    from .spatial_prior import load_prior_bundle
    path = run_decoder_study(ExperimentConfig.load(args.config), args.language, MasteryPlan.load(args.plan),
        args.output or "runs/decoder-study", policy=args.policy, prior_bundle=load_prior_bundle(args.prior_bundle),
        prior_condition=args.prior_condition)
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
