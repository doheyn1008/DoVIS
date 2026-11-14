"""Unactivated F preparation: two capacity-matched producer memory providers.

F0 is a NEW per-token normalized CNN baseline; F1 is engineered PNG component
memory. The source/partition/descriptors form one representation bundle with
idealized-world assumptions, not one isolated internal feature or general
vision evidence. No research learner is qualified or training started on import.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import mutual_lab
from mutual_lab.composition_diagnostics import CATEGORIES
from mutual_lab.diagnostics import safe_json
from mutual_lab.fit_distractors import NEGATIVE_POLICIES, negative_policy_manifest
from mutual_lab.mastery_lab import MasteryPlan, run_trial
from mutual_lab.native_preparation import LANGUAGES_ORDER
from mutual_lab.storage import now

VERSION = "memory-study-v1"
CONDITIONS = {"F0": "cnn", "F1": "objects"}
ROOT = Path(mutual_lab.__file__).resolve().parents[1]
ADAPTER = {"engine_phase": "C", "engine_condition": "C1", "fixed_prior_condition": "B1",
    "meaning": "Raw C1 identifies the common frozen pretrained prior only; F0/F1 identify producer memory providers",
    "raw_artifacts": "Raw C reports/checkpoints preserved; memory-study.json declares the F axis"}
INTERPRETATION = ("Development-only engineered perception bundle: memory source, partition and descriptors change together. "
    "Both arms normalize first32 channels per token with unchanged xy. F0 is a NEW normalized CNN baseline, "
    "distinct from D/E. F1 assumes opaque backgrounds and two separated unoccluded components. "
    "No human-acquisition, general-vision, isolated-feature, independent-replication or research-qualification claim")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _reference_plan():
    return MasteryPlan.load(ROOT / "configs/mastery-C-plan.json")


def _validate_plan(plan):
    actual, expected = asdict(plan), asdict(_reference_plan())
    actual.pop("metadata"); expected.pop("metadata")
    if _canonical(actual) != _canonical(expected):
        raise ValueError("F must retain the C data/model/seeds/budgets/thresholds and training controls")


def _object_module():
    """Load the adjacent standalone prototype under its future package name."""
    name, path = "mutual_lab.object_memory", Path(__file__).resolve().with_name("object_memory.py")
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path:
            raise ValueError("A different object-memory implementation is already loaded")
        return previous
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _prototype_sources():
    return {name: hashlib.sha256(Path(__file__).resolve().with_name(name).read_bytes()).hexdigest()
            for name in ("memory_study.py", "object_memory.py")}


class RecordingFactory:
    """Capture one real initialization and preflight only the fixed PNG pool."""

    def __init__(self, memory_condition, pngs=()):
        self.__module__ = "mutual_lab.memory_study"
        if memory_condition not in CONDITIONS:
            raise ValueError("Unknown producer-memory condition")
        self.memory_condition, self.pngs = memory_condition, tuple(pngs)
        self.__qualname__ = "RecordingFactory." + memory_condition
        self.initial_state = None

    def __call__(self, profile, initialization_seed, bundle, condition="B1"):
        if condition != "B1" or self.initial_state is not None:
            raise ValueError("Factory constructs exactly one fresh trial with the common B1 prior")
        from mutual_lab.neural import digest, torch
        from mutual_lab.spatial_prior import spatial_prior_hash
        from mutual_lab.frozen_features import enforce_singleton_features
        model = _object_module().build_object_memory_model(profile, initialization_seed, bundle,
            condition=condition, memory=CONDITIONS[self.memory_condition])
        # Bind singleton CNN math before cache warmup; the core engine applies
        # this idempotently again before its benchmark/evaluation path.
        enforce_singleton_features(model)
        before, rng = digest(model.state_dict()), torch.random.get_rng_state().clone()
        self.initial_state = {"full_state_sha256": before,
            "understanding_sha256": digest(model.understanding.state_dict()),
            "producer_sha256": digest(model.producer.state_dict()), "prior_sha256": spatial_prior_hash(model),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "residual_vocabulary_projection_zero": bool((model.producer.attention_output.weight == 0).all()),
            "memory_specification": model.object_memory_metadata, "capacity_matched": True,
            "frozen_feature_policy": model.frozen_feature_policy,
            "input_scope": "PNG pixels and native text only; no Scene metadata, palette, shape labels or fitted statistics"}
        hashes = sorted(hashlib.sha256(png).hexdigest() for png in self.pngs)
        audit = {"input_pngs": len(self.pngs), "unique_pngs": len(set(hashes)),
            "png_pool_sha256": hashlib.sha256(_canonical(hashes).encode()).hexdigest(),
            "successful_pngs": 0, "invalid_pngs": 0, "fallback_count": 0,
            "fallback_policy": "raise_on_invalid_memory; no fallback", "memory_tokens_per_png": 24,
            "memory_channels": 34, "normalized_feature_channels": 32,
            "finite_values": True, "normalization_valid": True, "xy_within_unit_interval": True,
            "zero_feature_tokens": 0, "status": "running"}
        self.initial_state["memory_preflight"] = audit
        try:
            if len(self.pngs) != len(set(hashes)):
                raise ValueError("Preflight PNGs must be unique")
            with torch.no_grad():
                for png in self.pngs:
                    memory = model.visual_grid([png])
                    if memory.shape != (1, 24, 34) or not bool(torch.isfinite(memory).all()):
                        raise ValueError("Expected one finite 24x34 producer-memory grid per PNG")
                    norms = memory[0, :, :32].norm(dim=-1)
                    zeros = norms == 0
                    normalized = bool(torch.logical_or(zeros, torch.isclose(norms, torch.ones_like(norms), atol=1e-6)).all())
                    xy_valid = bool((memory[0, :, 32:].abs() <= 1 + 1e-6).all())
                    audit["normalization_valid"] &= normalized
                    audit["xy_within_unit_interval"] &= xy_valid
                    audit["zero_feature_tokens"] += int(zeros.sum())
                    if not normalized or not xy_valid:
                        raise ValueError("Producer-memory normalization/coordinates failed preflight")
                    audit["successful_pngs"] += 1
            audit["status"] = "completed"
        except Exception as exc:
            audit.update(status="failed", invalid_pngs=1, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            audit["model_state_unchanged"] = digest(model.state_dict()) == before
            audit["rng_unchanged"] = bool(torch.equal(rng, torch.random.get_rng_state()))
            if not audit["model_state_unchanged"] or not audit["rng_unchanged"]:
                raise RuntimeError("PNG memory preflight changed model state or RNG")
        return model


def _coverage(raw):
    return [{"step": point["step"], "categories": {category: {key: values[key]
        for key in ("available_anchors", "available_pairs", "coverage_sufficient")}
        for category, values in point["candidate"]["categories"].items()}}
        for point in raw.get("curves", [])]


def memory_pair_audit(reports, policy):
    if policy not in NEGATIVE_POLICIES:
        raise ValueError("Unknown fixed distractor policy")
    names = [record["memory_condition"] for record in reports]
    if len(names) != len(set(names)) or set(names) - CONDITIONS.keys():
        raise ValueError("Duplicate or unknown memory arm")
    pair = {record["memory_condition"]: record for record in reports}
    if any(name not in pair or pair[name]["status"] != "completed" for name in CONDITIONS):
        return {"status": "incomplete", "checks": {}}
    left, right = (pair[name] for name in CONDITIONS)
    a, b = left["engine_report"], right["engine_report"]
    ia, ib = left["initial_state"], right["initial_state"]
    common = ("language", "replica", "seed_pair", "plan_sha256", "split", "data_sha256", "config_sha256",
        "source_sha256", "model_profile", "parameter_count", "trainable_parameter_count", "batch_size", "learning_rate",
        "policy", "negative_sampling_seed", "negative_exposures", "optimization", "clip_norm_per_branch",
        "choice_count", "training_chance_accuracy", "comprehension_forms", "production_forms", "teacher_forced_evaluation_forms",
        "initial_understanding_sha256", "initial_producer_sha256", "initial_prior_sha256", "final_understanding_sha256",
        "final_prior_sha256", "prior_metadata", "meaning_stream_sha256", "comprehension_caption_stream_sha256", "environment_sha256")
    checks = {key: key in a and key in b and a[key] == b[key] for key in common}
    checks.update({
        "full_initial_state": bool(ia.get("full_state_sha256")) and ia.get("full_state_sha256") == ib.get("full_state_sha256"),
        "captured_initialization": all(init.get("understanding_sha256") == raw.get("initial_understanding_sha256")
            and init.get("producer_sha256") == raw.get("initial_producer_sha256")
            and init.get("parameter_count") == raw.get("parameter_count")
            and init.get("trainable_parameter_count") == raw.get("trainable_parameter_count")
            and init.get("capacity_matched") is True and init.get("residual_vocabulary_projection_zero") is True
            for init, raw in ((ia, a), (ib, b))),
        "frozen_prior": all(bool(init.get("prior_sha256")) and init.get("prior_sha256") == raw.get("initial_prior_sha256")
            == raw.get("final_prior_sha256") for init, raw in ((ia, a), (ib, b))),
        "memory_axis": all(record.get("memory_source") == CONDITIONS[name]
            and record.get("memory_specification") == record["initial_state"].get("memory_specification")
            and record["memory_specification"].get("memory_source") == CONDITIONS[name]
            and record["memory_specification"].get("excluded_from_research") is True
            and raw.get("model_factory") == "mutual_lab.memory_study.RecordingFactory." + name
            for name, record, raw in (("F0", left, a), ("F1", right, b))),
        "same_normalization_rule": bool(ia["memory_specification"].get("normalization"))
            and ia["memory_specification"]["normalization"] == ib["memory_specification"].get("normalization")
            and all(init["memory_specification"].get("coordinate_order") == "y,x"
                and init["memory_specification"].get("memory_normalization") == "l2_first32_raw_yx_v1" for init in (ia, ib)),
        "png_preflight": all(init.get("memory_preflight", {}).get("status") == "completed"
            and init["memory_preflight"]["input_pngs"] == init["memory_preflight"]["unique_pngs"]
                == init["memory_preflight"]["successful_pngs"] > 0
            and all(init["memory_preflight"].get(key) is True for key in
                ("finite_values", "normalization_valid", "xy_within_unit_interval", "model_state_unchanged", "rng_unchanged"))
            and init["memory_preflight"].get("invalid_pngs") == 0 and init["memory_preflight"].get("fallback_count") == 0
            for init in (ia, ib)),
        "same_png_preflight_pool": ia["memory_preflight"].get("png_pool_sha256") == ib["memory_preflight"].get("png_pool_sha256"),
        "fixed_policy": all(raw.get("policy") == policy for raw in (a, b)),
        "canonical_frozen_features": all(raw.get("frozen_feature_policy") == "per_png_singleton_v1"
            and init.get("frozen_feature_policy") == "per_png_singleton_v1" for raw, init in ((a, ia), (b, ib))),
        "feature_tables_immutable": all(bool(raw.get("frozen_feature_tables", {}).get("step_0"))
            and raw["frozen_feature_tables"]["step_0"] == raw["frozen_feature_tables"].get("final") for raw in (a, b)),
        "global_visual_tables_equal": bool(a.get("frozen_feature_tables", {}).get("step_0", {}).get("caches", {}).get("visual_cache"))
            and a.get("frozen_feature_tables", {}).get("step_0", {}).get("caches", {}).get("visual_cache")
                == b.get("frozen_feature_tables", {}).get("step_0", {}).get("caches", {}).get("visual_cache"),
        "engine_adapter": all(raw.get("phase") == "C" and raw.get("condition") == "C1" and raw.get("status") == "completed"
            for raw in (a, b)),
        "independent_branch_clipping": all(raw.get("optimization") == "independent-branch-clipping-adam-v1"
            and raw.get("clip_norm_per_branch") == 1 for raw in (a, b)),
        "native_only_boundary": all(raw.get("training_languages") == [raw.get("language")]
            and raw.get("foreign_training_languages") == [] and raw.get("protected_splits_used") == []
            and raw.get("excluded_from_research") is True and raw.get("eligible_for_fresh_experiment") is False for raw in (a, b)),
        "all_budgets": all([point["step"] for point in raw.get("curves", [])] == [0, *raw["plan"]["budgets"]]
            and raw.get("training_steps_completed") == raw["plan"]["budgets"][-1] for raw in (a, b)),
        "candidate_coverage": _coverage(a) == _coverage(b) and all(set(point["candidate"]["categories"]) == set(CATEGORIES)
            and point["candidate"].get("excluded_from_research") is True
            and point["candidate"].get("eligible_for_fresh_experiment") is False
            for raw in (a, b) for point in raw.get("curves", [])),
    })
    return {"status": "matched" if all(checks.values()) else "mismatch", "checks": checks}


def _request(path, plan, policy=None):
    if path is None:
        return None
    data = Path(path).read_bytes()
    content = json.loads(data)
    expected = {"schema_version": 1, "model": "attentive", "fixed_prior_condition": "B1",
        "fixed_final_step": plan.budgets[-1], "data_version": "five-colors-three-shapes-v1",
        "memory_normalization": "l2_first32_raw_yx_v1"}
    if (any(content.get(key) != value for key, value in expected.items())
            or content.get("policy") not in NEGATIVE_POLICIES
            or (policy is not None and content.get("policy") != policy)):
        raise ValueError("Pinned F request differs from the declared study")
    return {"sha256": hashlib.sha256(data).hexdigest(), "content": content}


def run_memory_study(config, language, plan, output="runs/memory-study", progress=print,
                     *, policy="mixed_blocks", prior_bundle=None, study_request=None):
    plan.validate(config); _validate_plan(plan)
    if language not in LANGUAGES_ORDER or policy not in NEGATIVE_POLICIES or prior_bundle is None:
        raise ValueError("Supply a known language, explicit fixed policy and common frozen B1 prior bundle")
    from mutual_lab import development_world as world
    splits = world.development_splits(config, plan.data_seed)
    pngs = tuple(world.render_scene(scene) for key in ("train", "dev") for scene in splits[key])
    directory = Path(output).resolve() / f"memory-{language}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    suite = {"schema_version": 1, "kind": VERSION, "phase": "F", "axis": "producer_perceptual_memory",
        "language": language, "policy": policy, "negative_policy_specification": negative_policy_manifest(policy),
        "model": "attentive", "fixed_prior_condition": "B1", "plan": asdict(plan), "created_at": now(),
        "study_request": study_request, "prototype_source_sha256": _prototype_sources(), "status": "running",
        "engine_adapter": ADAPTER, "capacity_matched": True, "excluded_from_research": True,
        "source_scope": "Engineered producer memory provider only; understanding retains the identical global frozen CNN path",
        "eligible_for_fresh_experiment": False, "interpretation": INTERPRETATION,
        "shared_visual_prior": True, "independent_replications": False, "reports": [],
        "pair_audit": {"status": "incomplete", "checks": {}}}
    begun = time.monotonic()

    def snapshot():
        suite["pair_audit"] = memory_pair_audit(suite["reports"], policy)
        suite["elapsed_seconds"] = time.monotonic() - begun
        safe_json(directory / "memory-study.json", suite)

    snapshot()
    for condition, source in CONDITIONS.items():
        factory = RecordingFactory(condition, pngs)
        record = {"memory_condition": condition, "memory_source": source, "replica": 0,
            "seed_pair": list(plan.b_seed_pair), "status": "running", "initial_state": {}, "memory_specification": {}}
        remaining = plan.max_suite_seconds - (time.monotonic() - begun)
        try:
            if remaining <= 0:
                record["status"] = "suite_time_limit"
            else:
                path = run_trial(config, language, "C", "C1", 0, plan.b_seed_pair, plan,
                    directory / "raw-engine", progress, policy=policy, prior_bundle=prior_bundle,
                    remaining_seconds=remaining, model_factory=factory)
                raw = json.loads((path / "report.json").read_text())
                record.update(status=raw["status"], engine_report=raw,
                    raw_report_path=str((path / "report.json").relative_to(directory)))
        except KeyboardInterrupt as exc:
            from mutual_lab.composition_suite import ArmCancelled
            record.update(status="cancelled", initial_state=factory.initial_state or {})
            record["memory_specification"] = record["initial_state"].get("memory_specification", {})
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
                    raise exc
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record["initial_state"] = factory.initial_state or {}
        record["memory_specification"] = record["initial_state"].get("memory_specification", {})
        suite["reports"].append(record); snapshot()
    suite["status"] = "completed" if suite["pair_audit"]["status"] == "matched" else "incomplete"
    snapshot()
    return directory


def aggregate_memory_studies(suites, plan=None):
    """Keep every fixed-budget result and invariant; never choose a winner."""
    seen, protocols, rows, failed, deltas, signatures = set(), set(), [], [], [], {}
    shared_prior = None
    for suite in suites:
        if suite.get("kind") != VERSION or suite.get("phase") != "F" or suite.get("axis") != "producer_perceptual_memory":
            raise ValueError("Expected memory-study.json; C/D/E artifacts are not F studies")
        language = suite["language"]
        if language not in LANGUAGES_ORDER or language in seen:
            raise ValueError("Unknown or duplicate memory-study language")
        seen.add(language)
        local_plan = MasteryPlan.from_dict(suite["plan"]); _validate_plan(local_plan)
        if plan is not None and _canonical(asdict(local_plan)) != _canonical(asdict(plan)):
            raise ValueError("F plans differ")
        protocols.add(_canonical({key: suite[key] for key in ("plan", "policy", "negative_policy_specification",
            "model", "fixed_prior_condition", "prototype_source_sha256", "study_request")}))
        reports, policy = suite["reports"], suite["policy"]
        audit = memory_pair_audit(reports, policy)
        if audit["status"] != "matched":
            failed.append({"language": language, "status": "pair_" + audit["status"], "checks": audit["checks"]})
        if suite.get("capacity_matched") is not True or suite.get("excluded_from_research") is not True or suite.get("eligible_for_fresh_experiment") is not False:
            failed.append({"language": language, "status": "invalid_study_scope"})
        for condition in sorted(CONDITIONS.keys() - {record["memory_condition"] for record in reports}):
            failed.append({"language": language, "condition": condition, "status": "missing"})
        paired = {}
        for record in reports:
            condition, status, raw = record["memory_condition"], record["status"], record.get("engine_report", {})
            if raw and (raw.get("language") != language or raw.get("replica") != 0
                    or raw.get("seed_pair") != list(local_plan.b_seed_pair) or raw.get("plan") != suite["plan"]):
                raise ValueError("Raw engine metadata differs from its F manifest")
            if status != "completed":
                failed.append({"language": language, "condition": condition, "status": status, "error": record.get("error")})
            else:
                signature = {key: raw.get(key) for key in ("split", "config_sha256", "source_sha256", "model_profile",
                    "seed_pair", "policy", "optimization", "clip_norm_per_branch")}
                signature["initial_state"] = record["initial_state"]
                if signatures.setdefault(condition, signature) != signature:
                    raise ValueError("F provenance or initialization differs across languages")
                if shared_prior is None:
                    shared_prior = raw.get("initial_prior_sha256")
                if raw.get("initial_prior_sha256") != shared_prior:
                    raise ValueError("F languages/arms must share one frozen visual prior")
            for point in raw.get("curves", []):
                row = {"language": language, "condition": condition, "memory_source": CONDITIONS[condition],
                    "step": point["step"], "status": status, "dev": point["dev"],
                    "contrasts": point["contrasts"]["categories"], "candidate": point["candidate"]}
                rows.append(row); paired[(condition, point["step"])] = row
        if audit["status"] == "matched":
            for step in (0, *local_plan.budgets):
                a, b = (paired[(condition, step)] for condition in CONDITIONS)
                deltas.append({"language": language, "step": step,
                    "comprehension_delta": b["dev"]["comprehension_accuracy"] - a["dev"]["comprehension_accuracy"],
                    "generation_delta": b["dev"]["valid_generation_accuracy"] - a["dev"]["valid_generation_accuracy"]})
    if len(protocols) > 1:
        raise ValueError("F plans, policy, prototype implementations or pinned request differ")
    plan = plan or (MasteryPlan.from_dict(suites[0]["plan"]) if suites else _reference_plan())
    _validate_plan(plan)
    counts = []
    for condition in CONDITIONS:
        for step in (0, *plan.budgets):
            available = [row for row in rows if row["condition"] == condition and row["step"] == step]
            completed = [row for row in available if row["status"] == "completed"]
            counts.append({"condition": condition, "step": step, "expected_languages": 4,
                "reported_rows": len(available), "completed_rows": len(completed),
                **{key: sum(row["dev"][key] for row in completed) for key in
                    ("comprehension_hits", "comprehension_items", "valid_generation_hits", "generation_scenes")}})
    missing = [language for language in LANGUAGES_ORDER if language not in seen]
    return {"schema_version": 1, "kind": VERSION + "-comparison", "phase": "F", "axis": "producer_perceptual_memory",
        "status": "incomplete" if failed or missing else "completed", "missing_languages": missing,
        "failed_trials": failed, "rows": rows, "budget_counts": counts, "paired_deltas": deltas,
        "capacity_matched": True, "fixed_prior_condition": "B1", "policy": suites[0]["policy"] if suites else None,
        "source_scope": "Engineered producer memory provider only; understanding retains the identical global frozen CNN path",
        "selection": "undetermined; no automatic winner, extension, retry or research promotion",
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "shared_visual_prior": True, "independent_replications": False, "interpretation": INTERPRETATION}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/neural.example.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mastery-C-plan.json")
    parser.add_argument("--language", choices=LANGUAGES_ORDER)
    parser.add_argument("--prior-bundle", type=Path)
    parser.add_argument("--policy", choices=NEGATIVE_POLICIES)
    parser.add_argument("--study-request", type=Path)
    parser.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = MasteryPlan.load(args.plan)
    request = _request(args.study_request, plan, args.policy)
    if args.aggregate_dir is not None:
        paths = sorted(args.aggregate_dir.rglob("memory-study.json"))
        suites = [json.loads(path.read_text()) for path in paths]
        if any(suite.get("study_request") != request for suite in suites):
            raise ValueError("F manifests differ from the supplied pinned request")
        result = aggregate_memory_studies(suites, plan)
        destination = args.output or Path("runs/memory-comparison.json")
        destination.parent.mkdir(parents=True, exist_ok=True); safe_json(destination, result)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
        return 0 if result["status"] == "completed" else 1
    if args.language is None or args.prior_bundle is None or args.policy is None:
        parser.error("--language, --prior-bundle and --policy are required for explicit F execution")
    from mutual_lab.config import ExperimentConfig
    from mutual_lab.spatial_prior import load_prior_bundle
    path = run_memory_study(ExperimentConfig.load(args.config), args.language, plan,
        args.output or "runs/memory-study", prior_bundle=load_prior_bundle(args.prior_bundle),
        policy=args.policy, study_request=request)
    print(path, flush=True)
    return 0 if json.loads((path / "memory-study.json").read_text())["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
