"""Conditional G preparation, outside the frozen experiment package.

G0 retains base understanding; G1 adds text-conditioned access to two raw PNG
component tokens and trainable readout parameters. Producer memory is fixed
before either arm. This compares an engineered information/readout/capacity
bundle, not one isolated mechanism or human/general-language acquisition.
Importing this prototype starts no training and selects no model.
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
from dataclasses import asdict
from pathlib import Path

import mutual_lab
from mutual_lab.composition_diagnostics import CATEGORIES
from mutual_lab.diagnostics import safe_json
from mutual_lab.fit_distractors import negative_policy_manifest
from mutual_lab.frozen_features import FROZEN_FEATURE_POLICY
from mutual_lab.mastery_lab import MasteryPlan, run_trial
from mutual_lab.native_preparation import LANGUAGES_ORDER
from mutual_lab.storage import now

VERSION = "grounding-study-v1"
CONDITIONS = {"G0": "global", "G1": "objects"}
SELECTION_RULE = "all_languages_minimum_six_final_scores_tie_g0_v1"
CONTRAST_COVERAGE = {"role_reversal": (112, 28), "color_only": (112, 28),
    "shape_only": (108, 27), "binding_swap": (60, 15)}
PRODUCER_CACHES = ("visual_cache", "spatial_grid_cache", "object_memory_cache")
ROOT = Path(mutual_lab.__file__).resolve().parents[1]
ADAPTER = {"engine_phase": "C", "engine_condition": "C1", "fixed_prior_condition": "B1",
    "meaning": "C1 identifies the inherited frozen prior; G0/G1 identify understanding readout bundles"}
INTERPRETATION = ("Development-only understanding readout/information/capacity bundle. G1 gains two raw PNG component "
    "tokens and additional trainable parameters; capacity is NOT matched. Producer memory, weights, native stream and "
    "canonical frozen CNN features are fixed across arms. Both depend on engineered two-component world assumptions. "
    "One shared prior and one seed pair are not independent replication, human acquisition or research qualification")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _reference_plan():
    return MasteryPlan.load(ROOT / "configs/mastery-C-plan.json")


def _validate_plan(plan):
    actual, expected = asdict(plan), asdict(_reference_plan())
    actual.pop("metadata"); expected.pop("metadata")
    if _canonical(actual) != _canonical(expected):
        raise ValueError("G must retain the C data/model/seeds/budgets/thresholds and optimization controls")


def _understanding_module():
    name = "mutual_lab.understanding_memory"
    path = Path(__file__).resolve().with_name("understanding_memory.py")
    previous = sys.modules.get(name)
    if previous is not None:
        if Path(previous.__file__).resolve() != path:
            raise ValueError("A different understanding-memory implementation is already loaded")
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


def _sources():
    local = Path(__file__).resolve().parent
    package = ROOT / "mutual_lab"
    paths = [local / "grounding_study.py", local / "understanding_memory.py",
             package / "object_memory.py", package / "frozen_features.py"]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class RecordingFactory:
    """Capture one actual initialization and preflight PNG-only memory paths."""

    def __init__(self, grounding_condition, producer_memory, pngs=()):
        self.__module__ = "mutual_lab.grounding_study"
        if grounding_condition not in CONDITIONS or producer_memory not in ("cnn", "objects"):
            raise ValueError("Supply a known understanding arm and one fixed producer memory")
        self.grounding_condition, self.producer_memory = grounding_condition, producer_memory
        self.pngs, self.initial_state, self.model = tuple(pngs), None, None
        # Producer memory is behavior, not a distinguishable tensor layout.
        # Encode both axes so checkpoint restoration never guesses a provider.
        self.__qualname__ = "RecordingFactory." + grounding_condition + "_" + producer_memory

    def __call__(self, profile, initialization_seed, bundle, condition="B1"):
        if condition != "B1" or self.initial_state is not None:
            raise ValueError("Factory constructs exactly one fresh trial with the common B1 prior")
        from mutual_lab.frozen_features import enforce_singleton_features
        from mutual_lab.neural import digest, torch
        from mutual_lab.spatial_prior import spatial_prior_hash
        model = _understanding_module().build_understanding_memory_model(profile, initialization_seed, bundle,
            condition=condition, understanding=CONDITIONS[self.grounding_condition], producer_memory=self.producer_memory)
        enforce_singleton_features(model)
        self.model = model
        before, rng = digest(model.state_dict()), torch.random.get_rng_state().clone()
        base = model.understanding.base
        extra = sum(p.numel() for p in model.understanding.parameters()) - sum(p.numel() for p in base.parameters())
        self.initial_state = {"full_state_sha256": before, "base_understanding_sha256": digest(base.state_dict()),
            "understanding_sha256": digest(model.understanding.state_dict()),
            "producer_sha256": digest(model.producer.state_dict()), "prior_sha256": spatial_prior_hash(model),
            "base_understanding_parameters": sum(p.numel() for p in base.parameters()),
            "extra_understanding_parameters": extra, "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_feature_policy": model.frozen_feature_policy, "capacity_matched": False,
            "understanding_specification": model.understanding_memory_metadata,
            "producer_memory_specification": model.object_memory_metadata,
            "input_scope": "Native text and PNG pixels only; two raw component tokens, never repeated-grid row indices or Scene labels"}
        hashes = sorted(hashlib.sha256(png).hexdigest() for png in self.pngs)
        audit = {"input_pngs": len(self.pngs), "unique_pngs": len(set(hashes)), "successful_pngs": 0,
            "png_pool_sha256": hashlib.sha256(_canonical(hashes).encode()).hexdigest(),
            "invalid_pngs": 0, "fallback_count": 0, "fallback_policy": "raise; no fallback",
            "producer_tokens": 24, "understanding_tokens": 2 if self.grounding_condition == "G1" else 0,
            "channels": 34, "status": "running", "finite_values": True, "normalization_valid": True,
            "coordinate_order": "y,x", "memory_normalization": "l2_first32_raw_yx_v1"}
        self.initial_state["memory_preflight"] = audit
        try:
            if len(self.pngs) != len(set(hashes)):
                raise ValueError("Preflight PNGs must be unique")
            with torch.no_grad():
                for png in self.pngs:
                    visual = model.visual([png])
                    memories = [(model.visual_grid([png]), 24)]
                    if self.grounding_condition == "G1":
                        memories.append((model.understanding_objects([png]), 2))
                    if visual.shape != (1, profile.hidden_dim) or not bool(torch.isfinite(visual).all()):
                        raise ValueError("Invalid global CNN feature vector")
                    for memory, tokens in memories:
                        if memory.shape != (1, tokens, 34) or not bool(torch.isfinite(memory).all()):
                            raise ValueError("Expected finite PNG memory with the declared raw token count")
                        norms = memory[:, :, :32].norm(dim=-1)
                        valid = bool(torch.logical_or(norms == 0, torch.isclose(norms, torch.ones_like(norms), atol=1e-6)).all())
                        if not valid or not bool((memory[:, :, 32:].abs() <= 1 + 1e-6).all()):
                            raise ValueError("PNG token normalization/coordinates failed preflight")
                    audit["successful_pngs"] += 1
            audit["status"] = "completed"
        except Exception as exc:
            audit.update(status="failed", invalid_pngs=1, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            audit["model_state_unchanged"] = digest(model.state_dict()) == before
            audit["rng_unchanged"] = bool(torch.equal(rng, torch.random.get_rng_state()))
            if not audit["model_state_unchanged"] or not audit["rng_unchanged"]:
                raise RuntimeError("PNG preflight changed model tensors or RNG")
        return model


def _coverage(raw):
    return [{"step": point["step"], "categories": {category: {key: metrics[key]
        for key in ("available_anchors", "available_pairs", "coverage_sufficient")}
        for category, metrics in point["candidate"]["categories"].items()}}
        for point in raw.get("curves", [])]


def _producer_tables(raw):
    caches = raw.get("frozen_feature_tables", {}).get("step_0", {}).get("caches", {})
    return {name: caches.get(name) for name in PRODUCER_CACHES}


def grounding_pair_audit(reports, producer_memory):
    if producer_memory not in ("cnn", "objects"):
        raise ValueError("Unknown fixed producer memory")
    names = [record["grounding_condition"] for record in reports]
    if len(names) != len(set(names)) or set(names) - CONDITIONS.keys():
        raise ValueError("Duplicate or unknown grounding arm")
    pair = {record["grounding_condition"]: record for record in reports}
    if any(name not in pair or pair[name]["status"] != "completed" for name in CONDITIONS):
        return {"status": "incomplete", "checks": {}}
    left, right = (pair[name] for name in CONDITIONS)
    a, b = left["engine_report"], right["engine_report"]
    ia, ib = left["initial_state"], right["initial_state"]
    common = ("language", "replica", "seed_pair", "plan_sha256", "split", "data_sha256", "config_sha256", "source_sha256",
        "model_profile", "batch_size", "learning_rate", "policy", "negative_sampling_seed", "negative_exposures",
        "optimization", "clip_norm_per_branch", "choice_count", "training_chance_accuracy", "comprehension_forms",
        "production_forms", "teacher_forced_evaluation_forms", "initial_producer_sha256", "final_producer_sha256",
        "initial_prior_sha256", "final_prior_sha256", "prior_metadata", "meaning_stream_sha256",
        "comprehension_caption_stream_sha256", "environment_sha256", "frozen_feature_policy")
    checks = {key: key in a and key in b and a[key] == b[key] for key in common}
    checks.update({
        "base_understanding_initialization": bool(ia.get("base_understanding_sha256"))
            and ia.get("base_understanding_sha256") == ib.get("base_understanding_sha256"),
        "captured_initialization": all(init.get("understanding_sha256") == raw.get("initial_understanding_sha256")
            and init.get("producer_sha256") == raw.get("initial_producer_sha256")
            and init.get("parameter_count") == raw.get("parameter_count")
            and init.get("trainable_parameter_count") == raw.get("trainable_parameter_count")
            and init.get("capacity_matched") is False for init, raw in ((ia, a), (ib, b))),
        "explicit_extra_capacity": ia.get("extra_understanding_parameters") == 0
            and ib.get("extra_understanding_parameters", 0) > 0
            and ib.get("parameter_count", 0) - ia.get("parameter_count", 0) == ib.get("extra_understanding_parameters")
            and ib.get("trainable_parameter_count", 0) - ia.get("trainable_parameter_count", 0) == ib.get("extra_understanding_parameters"),
        "frozen_prior": all(bool(init.get("prior_sha256")) and init.get("prior_sha256") == raw.get("initial_prior_sha256")
            == raw.get("final_prior_sha256") for init, raw in ((ia, a), (ib, b))),
        "understanding_axis": all(record.get("understanding_mode") == CONDITIONS[name]
            and raw.get("model_factory") == "mutual_lab.grounding_study.RecordingFactory." + name + "_" + producer_memory
            and record.get("understanding_specification") == record["initial_state"].get("understanding_specification")
            for name, record, raw in (("G0", left, a), ("G1", right, b))),
        "fixed_producer_provider": all(record.get("producer_memory") == producer_memory for record in (left, right))
            and ia.get("producer_memory_specification") == ib.get("producer_memory_specification"),
        "canonical_features": all(init.get("frozen_feature_policy") == FROZEN_FEATURE_POLICY
            and raw.get("frozen_feature_policy") == FROZEN_FEATURE_POLICY for init, raw in ((ia, a), (ib, b))),
        "producer_feature_tables_equal": all(_producer_tables(a).values()) and _producer_tables(a) == _producer_tables(b),
        "each_full_feature_table_immutable": all(bool(raw.get("frozen_feature_tables", {}).get("step_0"))
            and raw["frozen_feature_tables"]["step_0"] == raw["frozen_feature_tables"].get("final") for raw in (a, b)),
        "png_preflight": all(init.get("memory_preflight", {}).get("status") == "completed"
            and init["memory_preflight"]["input_pngs"] == init["memory_preflight"]["successful_pngs"]
                == init["memory_preflight"]["unique_pngs"] > 0
            and init["memory_preflight"].get("fallback_count") == 0 and init["memory_preflight"].get("invalid_pngs") == 0
            and init["memory_preflight"].get("model_state_unchanged") is True
            and init["memory_preflight"].get("rng_unchanged") is True for init in (ia, ib)),
        "same_preflight_pool": ia["memory_preflight"].get("png_pool_sha256") == ib["memory_preflight"].get("png_pool_sha256"),
        "raw_two_understanding_tokens": ia["memory_preflight"].get("understanding_tokens") == 0
            and ib["memory_preflight"].get("understanding_tokens") == 2,
        "fixed_mixed_policy": all(raw.get("policy") == "mixed_blocks" for raw in (a, b)),
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


def readout_diagnostics(model, config, splits, language):
    """Separate final readout/counter-caption observations, never target scores."""
    from mutual_lab import development_world as world
    from mutual_lab.frozen_features import feature_cache_manifest
    from mutual_lab.neural import digest, torch
    before, rng, tables = digest(model.state_dict()), torch.random.get_rng_state().clone(), feature_cache_manifest(model)
    counts = {category: {"anchor_items": 0, "counter_caption_pairs": 0, "both_correct": 0,
        "base_absolute_sum": 0., "residual_absolute_sum": 0., "base_margin_sum": 0., "residual_margin_sum": 0.}
        for category in CATEGORIES}
    probes = world.build_development_contrasts(config, splits)
    with torch.no_grad():
        for probe in probes["probes"]:
            choices = [world.render_scene(scene) for scene in probe.choices]
            for form, text in enumerate(world.captions(probe.scene, language)):
                opposite = 1 - probe.correct_choice
                counter = world.captions(probe.choices[opposite], language)[form]
                parts = model.comprehension_logit_parts([text, counter], [choices, choices])
                base, residual, total = (parts[key] for key in ("base", "residual", "total"))
                if any(value.shape != (2, 2) or not bool(torch.isfinite(value).all()) for value in (base, residual, total)):
                    raise ValueError("Invalid auxiliary understanding logit decomposition")
                row = counts[probe.category]
                row["anchor_items"] += 1
                row["counter_caption_pairs"] += 1
                row["both_correct"] += int(int(total[0].argmax()) == probe.correct_choice and int(total[1].argmax()) == opposite)
                row["base_absolute_sum"] += float(base[0].abs().mean())
                row["residual_absolute_sum"] += float(residual[0].abs().mean())
                row["base_margin_sum"] += float(base[0, probe.correct_choice] - base[0, opposite])
                row["residual_margin_sum"] += float(residual[0, probe.correct_choice] - residual[0, opposite])
    for row in counts.values():
        items = row["anchor_items"]
        row.update({"mean_" + key.removesuffix("_sum"): row[key] / items if items else None
                    for key in ("base_absolute_sum", "residual_absolute_sum", "base_margin_sum", "residual_margin_sum")})
        row["counter_caption_consistency"] = row["both_correct"] / row["counter_caption_pairs"] if row["counter_caption_pairs"] else None
    readonly = (digest(model.state_dict()) == before and bool(torch.equal(rng, torch.random.get_rng_state()))
                and feature_cache_manifest(model) == tables)
    if not readonly:
        raise RuntimeError("Auxiliary diagnostics changed model state, RNG or frozen feature tables")
    return {"status": "completed", "qualification_gate": False, "step": "fixed_final_only", "categories": counts,
        "read_only_verified": True, "scope": "Auxiliary score magnitudes/margins and two-caption consistency; original candidate counts remain unchanged"}


def run_grounding_study(config, language, plan, output="runs/grounding-study", progress=print,
                        *, producer_memory=None, prior_bundle=None, study_request=None, diagnostics=False):
    plan.validate(config); _validate_plan(plan)
    if language not in LANGUAGES_ORDER or producer_memory not in ("cnn", "objects") or prior_bundle is None:
        raise ValueError("Supply a known language, one fixed producer memory and the common B1 prior bundle")
    from mutual_lab import development_world as world
    splits = world.development_splits(config, plan.data_seed)
    pngs = tuple(world.render_scene(scene) for key in ("train", "dev") for scene in splits[key])
    directory = Path(output).resolve() / f"grounding-{language}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    suite = {"schema_version": 1, "kind": VERSION, "phase": "G", "axis": "understanding_readout",
        "language": language, "producer_memory": producer_memory, "fixed_prior_condition": "B1", "policy": "mixed_blocks",
        "negative_policy_specification": negative_policy_manifest("mixed_blocks"), "plan": asdict(plan), "created_at": now(),
        "study_request": study_request, "implementation_source_sha256": _sources(), "status": "running",
        "engine_adapter": ADAPTER, "capacity_matched": False, "excluded_from_research": True,
        "eligible_for_fresh_experiment": False, "frozen_feature_policy": FROZEN_FEATURE_POLICY,
        "source_scope": "Understanding-only readout with extra parameters and raw object-information access; fixed producer provider",
        "interpretation": INTERPRETATION, "shared_visual_prior": True, "independent_replications": False,
        "reports": [], "pair_audit": {"status": "incomplete", "checks": {}}}
    begun = time.monotonic()

    def snapshot():
        suite["pair_audit"] = grounding_pair_audit(suite["reports"], producer_memory)
        suite["elapsed_seconds"] = time.monotonic() - begun
        safe_json(directory / "grounding-study.json", suite)

    snapshot()
    for condition, mode in CONDITIONS.items():
        factory = RecordingFactory(condition, producer_memory, pngs)
        record = {"grounding_condition": condition, "understanding_mode": mode, "producer_memory": producer_memory,
            "replica": 0, "seed_pair": list(plan.b_seed_pair), "status": "running", "initial_state": {},
            "understanding_specification": {}, "auxiliary_diagnostics": {"status": "not_requested", "qualification_gate": False}}
        remaining = plan.max_suite_seconds - (time.monotonic() - begun)
        try:
            if remaining <= 0:
                record["status"] = "suite_time_limit"
            else:
                path = run_trial(config, language, "C", "C1", 0, plan.b_seed_pair, plan, directory / "raw-engine",
                    progress, policy="mixed_blocks", prior_bundle=prior_bundle, remaining_seconds=remaining, model_factory=factory)
                raw = json.loads((path / "report.json").read_text())
                record.update(status=raw["status"], engine_report=raw, raw_report_path=str((path / "report.json").relative_to(directory)))
                if diagnostics and raw["status"] == "completed":
                    try:
                        record["auxiliary_diagnostics"] = readout_diagnostics(factory.model, config, splits, language)
                    except Exception as exc:
                        record["auxiliary_diagnostics"] = {"status": "failed", "qualification_gate": False,
                            "error": f"{type(exc).__name__}: {exc}"}
        except KeyboardInterrupt as exc:
            from mutual_lab.composition_suite import ArmCancelled
            record.update(status="cancelled", initial_state=factory.initial_state or {})
            record["understanding_specification"] = record["initial_state"].get("understanding_specification", {})
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
        record["understanding_specification"] = record["initial_state"].get("understanding_specification", {})
        suite["reports"].append(record); snapshot()
    suite["status"] = "completed" if suite["pair_audit"]["status"] == "matched" else "incomplete"
    snapshot()
    return directory


def _valid_rate(metrics, hits_key, items_key, accuracy_key, expected_items):
    hits, items, accuracy = (metrics.get(key) for key in (hits_key, items_key, accuracy_key))
    return (type(hits) is int and type(items) is int and items == expected_items and 0 <= hits <= items
        and type(accuracy) in (int, float) and math.isfinite(accuracy)
        and math.isclose(accuracy, hits / items, abs_tol=1e-12))


def select_grounding_candidate(rows, plan, *, complete):
    """Apply only the predeclared four-language, six-score fixed-final rule."""
    eligible, worst, checks = [], {}, []
    for condition in CONDITIONS:
        final = [row for row in rows if row["condition"] == condition and row["step"] == plan.budgets[-1]]
        values, all_pass = [], len(final) == 4 and {row["language"] for row in final} == set(LANGUAGES_ORDER)
        for row in final:
            dev, candidate, categories = row["dev"], row["candidate"], row["contrasts"]
            details = candidate.get("categories", {})
            valid = (row.get("status") == "completed"
                and _valid_rate(dev, "comprehension_hits", "comprehension_items", "comprehension_accuracy", 448)
                and _valid_rate(dev, "valid_generation_hits", "generation_scenes", "valid_generation_accuracy", 112)
                and set(categories) == set(CATEGORIES) == set(details))
            scores = [dev.get("comprehension_accuracy"), dev.get("valid_generation_accuracy")]
            thresholds = [plan.comprehension_min, plan.production_min]
            for category in CATEGORIES:
                metrics, declared = categories.get(category, {}), details.get(category, {})
                anchors, pairs = CONTRAST_COVERAGE[category]
                valid &= (_valid_rate(metrics, "hits", "items", "accuracy", anchors * 4)
                    and declared.get("available_anchors") == anchors and declared.get("available_pairs") == pairs
                    and anchors >= plan.min_contrast_anchors and pairs >= plan.min_contrast_pairs
                    and declared.get("coverage_sufficient") is True and declared.get("score_target_met") is True
                    and all(declared.get(key) == metrics.get(key) for key in ("hits", "items", "accuracy")))
                scores.append(metrics.get("accuracy")); thresholds.append(plan.contrast_min)
            finite = all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in scores)
            score_targets = finite and all(value >= threshold for value, threshold in zip(scores, thresholds))
            passed = bool(valid and score_targets and candidate.get("status") == "development_candidate"
                and candidate.get("score_targets_met") is True and candidate.get("coverage_sufficient") is True
                and candidate.get("excluded_from_research") is True and candidate.get("eligible_for_fresh_experiment") is False)
            all_pass &= passed
            if finite:
                values.extend(scores)
            checks.append({"condition": condition, "language": row["language"], "final_candidate": passed,
                           "counts_and_coverage_valid": bool(valid), "score_targets_met": bool(score_targets)})
        worst[condition] = min(values) if len(values) == 24 else None
        if all_pass and worst[condition] is not None:
            eligible.append(condition)
    selected = max(eligible, key=lambda condition: (worst[condition], condition == "G0")) if complete and eligible else None
    return {"rule": SELECTION_RULE, "status": "selected" if selected else "undetermined",
        "reason": "selected_by_predeclared_rule" if selected else "no_all_language_candidate" if complete else "incomplete_or_invalid_study",
        "condition": selected, "eligible_conditions": eligible if complete else [], "worst_scores": worst,
        "fixed_final_step": plan.budgets[-1], "tie_break": "G0", "candidate_checks": checks,
        "metric": "Minimum of six raw final accuracies across four languages; only all-language candidates are eligible",
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "interpretation": "Development selection only; no retry, extension, best-step choice or research promotion"}


def aggregate_grounding_studies(suites, plan=None):
    seen, protocols, rows, failed, deltas, signatures = set(), set(), [], [], [], {}
    shared_prior = None
    for suite in suites:
        if suite.get("kind") != VERSION or suite.get("phase") != "G" or suite.get("axis") != "understanding_readout":
            raise ValueError("Expected grounding-study.json; raw C/D/E/F reports are not G studies")
        language = suite["language"]
        if language not in LANGUAGES_ORDER or language in seen:
            raise ValueError("Unknown or duplicate grounding-study language")
        seen.add(language)
        local_plan = MasteryPlan.from_dict(suite["plan"]); _validate_plan(local_plan)
        if plan is not None and _canonical(asdict(local_plan)) != _canonical(asdict(plan)):
            raise ValueError("G plans differ")
        protocols.add(_canonical({key: suite[key] for key in ("plan", "producer_memory", "fixed_prior_condition", "policy",
            "negative_policy_specification", "implementation_source_sha256", "study_request")}))
        reports = suite["reports"]
        audit = grounding_pair_audit(reports, suite["producer_memory"])
        if audit["status"] != "matched":
            failed.append({"language": language, "status": "pair_" + audit["status"], "checks": audit["checks"]})
        if suite.get("capacity_matched") is not False or suite.get("excluded_from_research") is not True or suite.get("eligible_for_fresh_experiment") is not False:
            failed.append({"language": language, "status": "invalid_study_scope"})
        for condition in sorted(CONDITIONS.keys() - {record["grounding_condition"] for record in reports}):
            failed.append({"language": language, "condition": condition, "status": "missing"})
        paired = {}
        for record in reports:
            condition, status, raw = record["grounding_condition"], record["status"], record.get("engine_report", {})
            if raw and (raw.get("language") != language or raw.get("replica") != 0
                    or raw.get("seed_pair") != list(local_plan.b_seed_pair) or raw.get("plan") != suite["plan"]):
                raise ValueError("Raw engine metadata differs from its G manifest")
            if status != "completed":
                failed.append({"language": language, "condition": condition, "status": status, "error": record.get("error")})
            else:
                signature = {key: raw.get(key) for key in ("split", "config_sha256", "source_sha256", "model_profile",
                    "seed_pair", "policy", "optimization", "clip_norm_per_branch")}
                signature["initial_state"] = record["initial_state"]
                if signatures.setdefault(condition, signature) != signature:
                    raise ValueError("G provenance or initialization differs across languages")
                if shared_prior is None:
                    shared_prior = raw.get("initial_prior_sha256")
                if raw.get("initial_prior_sha256") != shared_prior:
                    raise ValueError("G arms/languages must share one frozen prior")
            for point in raw.get("curves", []):
                row = {"language": language, "condition": condition, "step": point["step"], "status": status,
                    "dev": point["dev"], "contrasts": point["contrasts"]["categories"], "candidate": point["candidate"]}
                rows.append(row); paired[(condition, point["step"])] = row
        if audit["status"] == "matched":
            for step in (0, *local_plan.budgets):
                a, b = (paired[(condition, step)] for condition in CONDITIONS)
                deltas.append({"language": language, "step": step,
                    "comprehension_delta": b["dev"]["comprehension_accuracy"] - a["dev"]["comprehension_accuracy"],
                    "generation_delta": b["dev"]["valid_generation_accuracy"] - a["dev"]["valid_generation_accuracy"]})
    if len(protocols) > 1:
        raise ValueError("G plans, fixed producer, sources or pinned requests differ")
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
    completed = not failed and not missing
    return {"schema_version": 1, "kind": VERSION + "-comparison", "phase": "G", "axis": "understanding_readout",
        "status": "completed" if completed else "incomplete", "missing_languages": missing,
        "failed_trials": failed, "rows": rows, "budget_counts": counts, "paired_deltas": deltas,
        "producer_memory": suites[0]["producer_memory"] if suites else None, "fixed_prior_condition": "B1", "policy": "mixed_blocks",
        "capacity_matched": False, "selection": select_grounding_candidate(rows, plan, complete=completed),
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "shared_visual_prior": True, "independent_replications": False, "interpretation": INTERPRETATION,
        "auxiliary_diagnostics_gate": False}


def _request(path, plan, producer_memory=None):
    if path is None:
        return None
    data = Path(path).read_bytes()
    content = json.loads(data)
    expected = {"schema_version": 1, "phase": "G", "fixed_prior_condition": "B1", "policy": "mixed_blocks",
        "fixed_final_step": plan.budgets[-1], "data_version": "five-colors-three-shapes-v1",
        "frozen_feature_policy": FROZEN_FEATURE_POLICY}
    if (any(content.get(key) != value for key, value in expected.items()) or content.get("producer_memory") not in ("cnn", "objects")
            or (producer_memory is not None and content.get("producer_memory") != producer_memory)
            or content.get("selection_rule", SELECTION_RULE) != SELECTION_RULE):
        raise ValueError("Pinned G request differs from the fixed grounding study")
    return {"sha256": hashlib.sha256(data).hexdigest(), "content": content}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/neural.example.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mastery-C-plan.json")
    parser.add_argument("--language", choices=LANGUAGES_ORDER)
    parser.add_argument("--producer-memory", choices=("cnn", "objects"))
    parser.add_argument("--prior-bundle", type=Path)
    parser.add_argument("--study-request", type=Path)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    plan = MasteryPlan.load(args.plan)
    request = _request(args.study_request, plan, args.producer_memory)
    if args.aggregate_dir is not None:
        suites = [json.loads(path.read_text()) for path in sorted(args.aggregate_dir.rglob("grounding-study.json"))]
        if any(suite.get("study_request") != request for suite in suites):
            raise ValueError("G manifests differ from the supplied pinned request")
        result = aggregate_grounding_studies(suites, plan)
        destination = args.output or Path("runs/grounding-comparison.json")
        destination.parent.mkdir(parents=True, exist_ok=True); safe_json(destination, result)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
        return 0 if result["status"] == "completed" else 1
    if args.language is None or args.producer_memory is None or args.prior_bundle is None:
        parser.error("--language, --producer-memory and --prior-bundle are required for explicit G execution")
    from mutual_lab.config import ExperimentConfig
    from mutual_lab.spatial_prior import load_prior_bundle
    path = run_grounding_study(ExperimentConfig.load(args.config), args.language, plan,
        args.output or "runs/grounding-study", producer_memory=args.producer_memory,
        prior_bundle=load_prior_bundle(args.prior_bundle), study_request=request, diagnostics=args.diagnostics)
    print(path, flush=True)
    return 0 if json.loads((path / "grounding-study.json").read_text())["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
