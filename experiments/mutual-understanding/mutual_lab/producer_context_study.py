"""Conditional H discovery: whole producer-global conditioning, three seed blocks.

H0 retains CNN conditioning; H1 uses fixed PNG-object position moments. Initial
NN tensors/capacity, G1 understanding, object attention and native streams match.
Only the global producer input bundle changes, both initialization and every
token's concatenated input. This is development engineering, not qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import mutual_lab
from mutual_lab.diagnostics import safe_json
from mutual_lab.frozen_features import FROZEN_FEATURE_POLICY
from mutual_lab.grounding_study import CONTRAST_COVERAGE, _valid_rate
from mutual_lab.mastery_lab import MasteryPlan, run_trial
from mutual_lab.native_preparation import LANGUAGES_ORDER

VERSION = "producer-context-study-v1"
CONDITIONS = {"H0": "cnn", "H1": "object_moments"}
DISCOVERY_SEEDS = ((79007, 79009), (79017, 79019), (79027, 79029))
CONFIRMATION_SEEDS = ((99007, 99009), (99017, 99019), (99027, 99029))
SOURCE_RULE = "all_discovery_cells_minimum_final_production_tie_h0_v1"
ROOT = Path(mutual_lab.__file__).resolve().parents[1]
INTERPRETATION = ("Same NN parameter count; whole P global-conditioning representation/basis/scaling changes, "
    "including GRU initial hidden and every token input. G1 U and object attention stay fixed. Engineered two-object "
    "perception; no isolated-initial-state, general-vision, word-alignment or research-promotion claim. "
    "Three paired L1 seed blocks share one visual prior; failed690xx cohort remains failed, never retried.")


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)


def _base_plan():
    return MasteryPlan.load(ROOT / "configs/mastery-C-plan.json")


def _validate_plan(plan):
    actual, base = asdict(plan), asdict(_base_plan())
    for value in (actual, base):
        for key in ("seed_pairs", "b_seed_pair", "metadata"):
            value.pop(key)
    if actual != base or plan.seed_pairs != DISCOVERY_SEEDS or plan.b_seed_pair != DISCOVERY_SEEDS[0]:
        raise ValueError("H changes only the predeclared three discovery L1 seed pairs and descriptive metadata")
    if plan.model_profile.hidden_dim != 128:
        raise ValueError("H requires the unchanged H128 architecture")


def _provider():
    name, path = "mutual_lab.producer_context", Path(__file__).with_name("producer_context.py").resolve()
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path:
            raise ValueError("A different producer-context provider is already loaded")
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _sources():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("producer_context.py", "producer_context_study.py")}


class RecordingFactory:
    def __init__(self, condition, pngs=()):
        self.__module__ = "mutual_lab.producer_context_study"
        if condition not in CONDITIONS:
            raise ValueError("Unknown H condition")
        self.__qualname__ = "RecordingFactory." + condition
        self.condition, self.pngs, self.initial_state = condition, tuple(pngs), None

    def __call__(self, profile, initialization_seed, bundle, condition="B1"):
        if condition != "B1" or self.initial_state is not None:
            raise ValueError("Construct exactly one fresh H trial with the common B1 prior")
        from mutual_lab.frozen_features import enforce_singleton_features
        from mutual_lab.neural import digest, torch
        from mutual_lab.spatial_prior import spatial_prior_hash
        provider = _provider()
        model = provider.build_producer_context_model(profile, initialization_seed, bundle, condition=condition,
                                                      context=CONDITIONS[self.condition])
        enforce_singleton_features(model)
        before, rng = digest(model.state_dict()), torch.random.get_rng_state().clone()
        self.initial_state = {"full_state_sha256": before, "initialization_seed": initialization_seed,
            "understanding_sha256": digest(model.understanding.state_dict()),
            "producer_sha256": digest(model.producer.state_dict()), "prior_sha256": spatial_prior_hash(model),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_flags": [(name, parameter.requires_grad) for name, parameter in model.named_parameters()],
            "capacity_matched": True, "frozen_feature_policy": model.frozen_feature_policy,
            "context_specification": model.producer_context_metadata,
            "producer_memory_specification": model.object_memory_metadata,
            "understanding_specification": model.understanding_memory_metadata}
        hashes = sorted(hashlib.sha256(png).hexdigest() for png in self.pngs)
        audit = {"input_pngs": len(hashes), "unique_pngs": len(set(hashes)), "successful_pngs": 0,
            "png_pool_sha256": hashlib.sha256(_canonical(hashes).encode()).hexdigest(),
            "fallback_count": 0, "fallback_policy": "raise; no fallback", "status": "running"}
        self.initial_state["memory_preflight"] = audit
        try:
            if not hashes or len(hashes) != len(set(hashes)):
                raise ValueError("Preflight requires a nonempty unique fit/dev PNG pool")
            table = provider.preflight_context(model, self.pngs)
            audit.update(status="completed", successful_pngs=len(hashes), feature_tables=table)
        except Exception as exc:
            audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            audit["model_state_unchanged"] = digest(model.state_dict()) == before
            audit["rng_unchanged"] = bool(torch.equal(rng, torch.random.get_rng_state()))
            if not audit["model_state_unchanged"] or not audit["rng_unchanged"]:
                raise RuntimeError("H preflight changed NN state or RNG")
        return model


def _checkpoint_understanding(directory, raw):
    """Hash actual U tensors at every available budget, never infer from metrics."""
    from mutual_lab.neural import digest, torch
    result = {}
    for item in raw.get("checkpoints", []):
        name = item["path"]
        if Path(name).name != name or "\\" in name:
            raise ValueError("Unsafe checkpoint path")
        path = Path(directory) / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Checkpoint hash differs from the engine manifest")
        sidecar = json.loads(path.with_suffix(".json").read_text())
        if sidecar != item:
            raise ValueError("Checkpoint sidecar differs from the report")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("format") != "mutual-mastery-development-v1" or payload.get("kind") != "development_only"
                or payload.get("seed_pair") != raw["seed_pair"] or payload.get("step") != item["step"]):
            raise ValueError("Unexpected H checkpoint provenance")
        understanding = {name.removeprefix("understanding."): value for name, value in payload["model"].items()
                         if name.startswith("understanding.")}
        if not understanding or str(item["step"]) in result:
            raise ValueError("Missing or duplicate checkpoint understanding state")
        result[str(item["step"])] = digest(understanding)
    return result


def context_pair_audit(reports):
    names = [r["context_condition"] for r in reports]
    if len(names) != len(set(names)) or set(names) - CONDITIONS.keys():
        raise ValueError("Duplicate or unknown H condition")
    pair = {r["context_condition"]: r for r in reports}
    if any(name not in pair or pair[name]["status"] != "completed" for name in CONDITIONS):
        return {"status": "incomplete", "checks": {}}
    left, right = pair["H0"], pair["H1"]
    a, b = left["engine_report"], right["engine_report"]
    ia, ib = left["initial_state"], right["initial_state"]
    keys = ("language", "replica", "seed_pair", "plan_sha256", "split", "data_sha256", "config_sha256",
        "source_sha256", "model_profile", "parameter_count", "trainable_parameter_count", "batch_size", "learning_rate",
        "policy", "negative_sampling_seed", "negative_exposures", "optimization", "clip_norm_per_branch", "choice_count",
        "initial_understanding_sha256", "final_understanding_sha256", "initial_producer_sha256", "initial_prior_sha256",
        "final_prior_sha256", "meaning_stream_sha256", "comprehension_caption_stream_sha256", "environment_sha256",
        "frozen_feature_policy", "frozen_feature_tables")
    checks = {key: key in a and key in b and a[key] == b[key] for key in keys}
    checkpoints = left.get("understanding_checkpoints", {})
    expected_steps = {str(step) for step in (0, *a["plan"]["budgets"])}
    checks.update({
        "all_initial_nn_exact": bool(ia.get("full_state_sha256")) and all(ia.get(key) == ib.get(key) for key in
            ("full_state_sha256", "understanding_sha256", "producer_sha256", "prior_sha256", "parameter_count", "trainable_parameter_count", "initialization_seed", "parameter_flags")),
        "understanding_exact_at_all_budgets": set(checkpoints) == expected_steps
            and checkpoints == right.get("understanding_checkpoints")
            and checkpoints.get("0") == a.get("initial_understanding_sha256")
            and checkpoints.get(str(a["plan"]["budgets"][-1])) == a.get("final_understanding_sha256"),
        "captured_initialization": all(init.get("understanding_sha256") == raw.get("initial_understanding_sha256")
            and init.get("producer_sha256") == raw.get("initial_producer_sha256")
            and init.get("parameter_count") == raw.get("parameter_count")
            and init.get("trainable_parameter_count") == raw.get("trainable_parameter_count")
            and init.get("initialization_seed") == raw.get("seed_pair", [None])[0]
            and init.get("prior_sha256") == raw.get("initial_prior_sha256") == raw.get("final_prior_sha256")
            for init, raw in ((ia, a), (ib, b))),
        "fixed_g1_and_object_attention": ia.get("understanding_specification") == ib.get("understanding_specification")
            and ia.get("understanding_specification", {}).get("understanding") == "objects"
            and ia.get("producer_memory_specification") == ib.get("producer_memory_specification")
            and ia.get("producer_memory_specification", {}).get("memory_source") == "objects",
        "context_axis": all(record.get("producer_context") == CONDITIONS[name]
            and record["initial_state"].get("context_specification", {}).get("context") == CONDITIONS[name]
            and record["initial_state"]["context_specification"].get("nn_parameters_changed") is False
            and record["engine_report"].get("model_factory") == "mutual_lab.producer_context_study.RecordingFactory." + name
            for name, record in pair.items()),
        "canonical_immutable_tables": all(raw.get("frozen_feature_policy") == FROZEN_FEATURE_POLICY
            and bool(raw.get("frozen_feature_tables", {}).get("step_0"))
            and raw["frozen_feature_tables"]["step_0"] == raw["frozen_feature_tables"].get("final") for raw in (a, b)),
        "preflight": all(init.get("memory_preflight", {}).get("status") == "completed"
            and init["memory_preflight"]["input_pngs"] == init["memory_preflight"]["successful_pngs"] == init["memory_preflight"]["unique_pngs"] > 0
            and init["memory_preflight"].get("fallback_count") == 0
            and init["memory_preflight"].get("model_state_unchanged") is True and init["memory_preflight"].get("rng_unchanged") is True for init in (ia, ib))
            and ia["memory_preflight"]["png_pool_sha256"] == ib["memory_preflight"]["png_pool_sha256"],
        "all_budgets": all([p["step"] for p in raw.get("curves", [])] == [0, *raw["plan"]["budgets"]]
            and raw.get("training_steps_completed") == raw["plan"]["budgets"][-1] for raw in (a, b)),
        "fixed_native_engine": all(raw.get("phase") == "C" and raw.get("condition") == "C1" and raw.get("status") == "completed"
            and raw.get("policy") == "mixed_blocks" and raw.get("choice_count") == 4
            and raw.get("optimization") == "independent-branch-clipping-adam-v1" and raw.get("clip_norm_per_branch") == 1
            and raw.get("training_languages") == [raw.get("language")] and raw.get("foreign_training_languages") == []
            and raw.get("protected_splits_used") == [] and raw.get("excluded_from_research") is True
            and raw.get("eligible_for_fresh_experiment") is False for raw in (a, b)),
        "candidate_development_only": all(set(p["candidate"]["categories"]) == set(CONTRAST_COVERAGE)
            and p["candidate"].get("excluded_from_research") is True and p["candidate"].get("eligible_for_fresh_experiment") is False
            for raw in (a, b) for p in raw["curves"]),
    })
    return {"status": "matched" if all(checks.values()) else "mismatch", "checks": checks}


def run_context_study(config, language, replica, plan, output="runs/producer-context-study", progress=print, *, prior_bundle=None, study_request=None):
    plan.validate(config); _validate_plan(plan)
    if language not in LANGUAGES_ORDER or type(replica) is not int or not 0 <= replica < 3 or prior_bundle is None:
        raise ValueError("Supply one declared language/replica and the shared B1 prior")
    from mutual_lab import development_world as world
    splits = world.development_splits(config, plan.data_seed)
    pngs = tuple(world.render_scene(scene) for key in ("train", "dev") for scene in splits[key])
    pair = plan.seed_pairs[replica]; engine_plan = replace(plan, b_seed_pair=pair)
    directory = Path(output).resolve() / f"producer-context-{language}-r{replica}-{uuid.uuid4().hex[:8]}"; directory.mkdir(parents=True)
    suite = {"schema_version": 1, "kind": VERSION, "phase": "H", "axis": "producer_global_conditioning",
        "language": language, "replica": replica, "seed_pair": list(pair), "plan": asdict(plan), "engine_plan": asdict(engine_plan),
        "source_rule": SOURCE_RULE, "confirmation_seed_pairs": [list(p) for p in CONFIRMATION_SEEDS],
        "model": "attentive", "understanding": "objects", "producer_memory": "objects", "fixed_prior_condition": "B1", "policy": "mixed_blocks",
        "frozen_feature_policy": FROZEN_FEATURE_POLICY, "memory_normalization": "l2_first32_raw_yx_v1",
        "capacity_matched": True, "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "study_request": study_request, "implementation_source_sha256": _sources(), "interpretation": INTERPRETATION,
        "status": "running", "reports": [], "pair_audit": {"status": "incomplete", "checks": {}}}
    begun = time.monotonic()

    def snapshot():
        suite["pair_audit"] = context_pair_audit(suite["reports"])
        suite["elapsed_seconds"] = time.monotonic() - begun
        safe_json(directory / "producer-context-study.json", suite)

    snapshot()
    for condition, context in CONDITIONS.items():
        factory = RecordingFactory(condition, pngs)
        record = {"context_condition": condition, "producer_context": context, "status": "running", "initial_state": {}, "understanding_checkpoints": {}}
        try:
            remaining = plan.max_suite_seconds - (time.monotonic() - begun)
            if remaining <= 0:
                record["status"] = "suite_time_limit"
            else:
                path = run_trial(config, language, "C", "C1", replica, pair, engine_plan, directory / "raw-engine", progress,
                    policy="mixed_blocks", prior_bundle=prior_bundle, remaining_seconds=remaining, model_factory=factory)
                raw = json.loads((path / "report.json").read_text())
                record.update(status=raw["status"], engine_report=raw, raw_report_path=str((path / "report.json").relative_to(directory)))
                record["understanding_checkpoints"] = _checkpoint_understanding(path, raw)
        except KeyboardInterrupt as exc:
            from mutual_lab.composition_suite import ArmCancelled
            record.update(status="cancelled", initial_state=factory.initial_state or {})
            suite["reports"].append(record); suite["status"] = "cancelled"
            try:
                if isinstance(exc, ArmCancelled):
                    raw = getattr(exc, "report", None)
                    if not isinstance(raw, Mapping):
                        try: raw = json.loads(exc.report_path.read_text())
                        except Exception: raw = None
                    if isinstance(raw, Mapping): record["engine_report"] = dict(raw)
                    try: record["raw_report_path"] = str(exc.report_path.relative_to(directory))
                    except Exception: pass
            finally:
                try: snapshot()
                finally: raise exc
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record["initial_state"] = factory.initial_state or {}
        suite["reports"].append(record); snapshot()
    suite["status"] = "completed" if suite["pair_audit"]["status"] == "matched" else "incomplete"
    snapshot(); return directory


def _qualified(row, plan):
    dev, candidate, categories = row["dev"], row["candidate"], row["contrasts"]
    details = candidate.get("categories", {})
    valid = (row.get("status") == "completed" and _valid_rate(dev, "comprehension_hits", "comprehension_items", "comprehension_accuracy", 448)
        and _valid_rate(dev, "valid_generation_hits", "generation_scenes", "valid_generation_accuracy", 112)
        and set(categories) == set(details) == set(CONTRAST_COVERAGE))
    scores = [dev.get("comprehension_accuracy"), dev.get("valid_generation_accuracy")]
    for category, (anchors, pairs) in CONTRAST_COVERAGE.items():
        metrics, gate = categories.get(category, {}), details.get(category, {})
        valid &= (_valid_rate(metrics, "hits", "items", "accuracy", anchors * 4)
            and gate.get("available_anchors") == anchors and gate.get("available_pairs") == pairs
            and anchors >= plan.min_contrast_anchors and pairs >= plan.min_contrast_pairs
            and gate.get("coverage_sufficient") is True and gate.get("score_target_met") is True
            and all(gate.get(k) == metrics.get(k) for k in ("hits", "items", "accuracy")))
        scores.append(metrics.get("accuracy"))
    finite = all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in scores)
    targets = finite and scores[0] >= plan.comprehension_min and scores[1] >= plan.production_min and all(v >= plan.contrast_min for v in scores[2:])
    return bool(valid and targets and candidate.get("status") == "development_candidate" and candidate.get("score_targets_met") is True
        and candidate.get("coverage_sufficient") is True and candidate.get("excluded_from_research") is True
        and candidate.get("eligible_for_fresh_experiment") is False)


def select_context_candidate(rows, plan, *, complete):
    eligible, worst, checks = [], {}, []
    for condition in CONDITIONS:
        final = [r for r in rows if r["condition"] == condition and r["step"] == plan.budgets[-1]]
        identities = [(r["language"], r["replica"]) for r in final]
        valid_cells = (len(identities) == 12 and set(identities) == {(language, replica) for language in LANGUAGES_ORDER for replica in range(3)}
            and all(type(row["replica"]) is int and row.get("seed_pair") == list(DISCOVERY_SEEDS[row["replica"]]) for row in final))
        passes = []
        for row in final:
            passed = _qualified(row, plan); passes.append(passed)
            checks.append({"condition": condition, "language": row["language"], "replica": row["replica"], "qualified": passed})
        worst[condition] = min((r["dev"]["valid_generation_accuracy"] for r in final), default=None) if valid_cells else None
        if valid_cells and all(passes): eligible.append(condition)
    chosen = max(eligible, key=lambda name: (worst[name], name == "H0")) if complete and eligible else None
    return {"rule": SOURCE_RULE, "status": "selected" if chosen else "undetermined", "condition": chosen,
        "eligible_conditions": eligible if complete else [], "worst_final_production": worst, "fixed_final_step": plan.budgets[-1],
        "tie_break": "H0", "reason": "selected_by_predeclared_rule" if chosen else "no_all_discovery_cells_candidate" if complete else "incomplete_or_invalid_study",
        "candidate_checks": checks, "excluded_from_research": True, "eligible_for_fresh_experiment": False}


def aggregate_context_studies(suites, plan):
    _validate_plan(plan)
    seen, rows, failed, protocols, signatures, priors, initial_by_replica = set(), [], [], set(), {}, set(), {}
    for suite in suites:
        identity = (suite.get("language"), suite.get("replica"))
        if (suite.get("kind") != VERSION or suite.get("phase") != "H" or suite.get("axis") != "producer_global_conditioning"
                or identity[0] not in LANGUAGES_ORDER or type(identity[1]) is not int or identity[1] not in range(3) or identity in seen):
            raise ValueError("Unknown, duplicate or non-H discovery cell")
        seen.add(identity); language, replica = identity
        if suite["plan"] != json.loads(_canonical(asdict(plan))) or suite.get("seed_pair") != list(plan.seed_pairs[replica]):
            raise ValueError("H discovery plans or declared replica seeds differ")
        protocols.add(_canonical({key: suite[key] for key in ("study_request", "implementation_source_sha256", "source_rule")}))
        if (suite.get("capacity_matched") is not True or suite.get("excluded_from_research") is not True
                or suite.get("eligible_for_fresh_experiment") is not False or suite.get("source_rule") != SOURCE_RULE):
            failed.append({"language": language, "replica": replica, "status": "invalid_scope"})
        audit = context_pair_audit(suite["reports"])
        if audit["status"] != "matched": failed.append({"language": language, "replica": replica, "status": "pair_" + audit["status"], "checks": audit["checks"]})
        present = {r["context_condition"] for r in suite["reports"]}
        for condition in CONDITIONS.keys() - present:
            failed.append({"language": language, "replica": replica, "condition": condition, "status": "missing"})
        for record in suite["reports"]:
            condition, raw = record["context_condition"], record.get("engine_report", {})
            if record["status"] != "completed": failed.append({"language": language, "replica": replica, "condition": condition, "status": record["status"]})
            if raw:
                if (raw.get("language") != language or raw.get("replica") != replica or raw.get("seed_pair") != list(plan.seed_pairs[replica])
                        or raw.get("plan") != json.loads(_canonical(asdict(replace(plan, b_seed_pair=plan.seed_pairs[replica]))))):
                    raise ValueError("Raw H engine seed/plan metadata differs")
                if record["status"] == "completed":
                    priors.add(raw.get("initial_prior_sha256"))
                    initial_by_replica.setdefault(replica, record["initial_state"].get("full_state_sha256"))
                    signature = {key: raw.get(key) for key in ("source_sha256", "config_sha256", "split", "model_profile", "seed_pair", "policy", "initial_prior_sha256")}
                    signature["initial_state"] = record["initial_state"]
                    key = (replica, condition)
                    if signatures.setdefault(key, signature) != signature:
                        raise ValueError("Matched H replica provenance/initialization differs across languages")
                for point in raw.get("curves", []):
                    rows.append({"language": language, "replica": replica, "seed_pair": list(plan.seed_pairs[replica]),
                        "condition": condition, "producer_context": CONDITIONS[condition], "step": point["step"], "status": record["status"],
                        "dev": point["dev"], "contrasts": point["contrasts"]["categories"], "candidate": point["candidate"]})
    if len(protocols) > 1: raise ValueError("H pinned requests or implementations differ")
    if len(priors) > 1: raise ValueError("All H cells must share one identical frozen B1 prior")
    if len(set(initial_by_replica.values())) != len(initial_by_replica):
        failed.append({"status": "discovery_initializations_not_distinct"})
    missing = [{"language": language, "replica": replica} for language in LANGUAGES_ORDER for replica in range(3) if (language, replica) not in seen]
    complete = not missing and not failed
    counts = [{"condition": condition, "step": step, "expected_cells": 12,
        "reported_cells": sum(r["condition"] == condition and r["step"] == step for r in rows),
        "completed_cells": sum(r["condition"] == condition and r["step"] == step and r["status"] == "completed" for r in rows)}
        for condition in CONDITIONS for step in (0, *plan.budgets)]
    return {"schema_version": 1, "kind": VERSION + "-comparison", "phase": "H", "axis": "producer_global_conditioning",
        "status": "completed" if complete else "incomplete", "plan": asdict(plan), "source_rule": SOURCE_RULE,
        "missing_cells": missing, "failed_trials": failed, "rows": rows, "budget_counts": counts,
        "selection": select_context_candidate(rows, plan, complete=complete), "capacity_matched": True,
        "fixed_prior_condition": "B1", "producer_memory": "objects", "understanding": "objects", "policy": "mixed_blocks",
        "frozen_feature_policy": FROZEN_FEATURE_POLICY, "memory_normalization": "l2_first32_raw_yx_v1",
        "discovery_seed_pairs": [list(pair) for pair in DISCOVERY_SEEDS], "confirmation_seed_pairs": [list(pair) for pair in CONFIRMATION_SEEDS],
        "excluded_from_research": True, "eligible_for_fresh_experiment": False, "interpretation": INTERPRETATION}


def _request(path, plan):
    if path is None: return None
    data = Path(path).read_bytes(); content = json.loads(data)
    expected = {"schema_version": 1, "phase": "H", "model": "attentive", "fixed_prior_condition": "B1", "policy": "mixed_blocks",
        "producer_memory": "objects", "understanding": "objects", "hidden_dim": 128, "data_version": "five-colors-three-shapes-v1",
        "fixed_final_step": plan.budgets[-1], "frozen_feature_policy": FROZEN_FEATURE_POLICY,
        "memory_normalization": "l2_first32_raw_yx_v1", "source_rule": SOURCE_RULE}
    if any(content.get(key) != value for key, value in expected.items()):
        raise ValueError("Pinned H request differs from the preregistered discovery")
    return {"content": content, "sha256": hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/neural.example.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mastery-H-plan.json")
    parser.add_argument("--language", choices=LANGUAGES_ORDER)
    parser.add_argument("--replica", type=int, choices=range(3))
    parser.add_argument("--prior-bundle", type=Path)
    parser.add_argument("--study-request", type=Path)
    parser.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv); plan = MasteryPlan.load(args.plan); _validate_plan(plan)
    request = _request(args.study_request, plan)
    if args.aggregate_dir is not None:
        suites = [json.loads(path.read_text()) for path in sorted(args.aggregate_dir.rglob("producer-context-study.json"))]
        if any(suite.get("study_request") != request for suite in suites): raise ValueError("H manifests differ from the supplied request")
        result = aggregate_context_studies(suites, plan); output = args.output or Path("runs/producer-context-H-summary.json")
        output.parent.mkdir(parents=True, exist_ok=True); safe_json(output, result)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
        return 0 if result["status"] == "completed" else 1
    if args.language is None or args.replica is None or args.prior_bundle is None:
        parser.error("--language, --replica and --prior-bundle are required for H execution")
    from mutual_lab.config import ExperimentConfig
    from mutual_lab.spatial_prior import load_prior_bundle
    path = run_context_study(ExperimentConfig.load(args.config), args.language, args.replica, plan,
        args.output or "runs/producer-context-study", prior_bundle=load_prior_bundle(args.prior_bundle), study_request=request)
    print(path, flush=True)
    return 0 if json.loads((path / "producer-context-study.json").read_text())["status"] == "completed" else 1


if __name__ == "__main__": raise SystemExit(main())
