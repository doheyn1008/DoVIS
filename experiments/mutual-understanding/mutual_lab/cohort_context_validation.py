"""Separate H confirmation-v2 for ONE winner of the complete 790xx discovery grid.

One fixed condition is selected before twelve fresh native-language models are
constructed. Only the shared image prior is loaded; no source L1 checkpoint is
accepted. This is seed stability on selected development data, not fresh-data
or independent visual replication, language-distance evidence, or qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import mutual_lab
from mutual_lab import candidate_validation as base
from mutual_lab.diagnostics import safe_json
from mutual_lab.fit_distractors import negative_policy_manifest
from mutual_lab.frozen_features import FROZEN_FEATURE_POLICY
from mutual_lab.mastery_lab import MasteryPlan, run_trial
from mutual_lab.native_preparation import LANGUAGES_ORDER
from mutual_lab.storage import now

VERSION = "cohort-context-validation-v2"
SOURCE_SEED_PAIRS = ((79007, 79009), (79017, 79019), (79027, 79029))
SELECTION_RULE = "all_discovery_cells_minimum_final_production_tie_h0_v1"
SEED_PAIRS = ((99007, 99009), (99017, 99019), (99027, 99029))
ROOT = Path(mutual_lab.__file__).resolve().parents[1]
SCOPE = ("Development-only seed stability of one selected architecture on the same selected development population. "
         "Fresh L1 weights/Adam, one shared frozen visual prior; no independent visual replication, fresh-data "
         "generalization, model selection inside the cohort, human acquisition or research qualification. "
         "This is a separate H intervention family; the failed 690xx v1 cohort remains unchanged and is not replaced")
CATALOG = {
    "H0": ("producer_context", "objects", "attentive", "producer-context-study-v1-comparison",
           "mutual_lab.producer_context_study.RecordingFactory.H0"),
    "H1": ("producer_context", "objects", "attentive", "producer-context-study-v1-comparison",
           "mutual_lab.producer_context_study.RecordingFactory.H1"),
}
CONTEXTS = {"H0": "cnn", "H1": "object_moments"}
BASE_SOURCES = ("neural.py", "development_models.py", "spatial_prior.py", "attentive_spatial.py",
                "frozen_features.py", "fit_distractors.py", "mastery_lab.py", "development_world.py", "candidate_validation.py")
_canonical = base._canonical
_reference_plan = base._reference_plan
_point_checks = base._point_checks


def _validate_plan(plan):
    if (plan.seed_pairs != SEED_PAIRS or plan.b_seed_pair != SEED_PAIRS[0]
            or any(type(seed) is not int for pair in (*plan.seed_pairs, plan.b_seed_pair) for seed in pair)):
        raise ValueError("The cohort requires exactly the three predeclared fresh seed pairs")
    actual, expected = asdict(plan), asdict(_reference_plan())
    for key in ("seed_pairs", "b_seed_pair", "metadata"):
        actual.pop(key); expected.pop(key)
    if _canonical(actual) != _canonical(expected):
        raise ValueError("The cohort must retain C numeric model/data/budgets/criteria and training controls")


def _matches_pair(value, expected):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(seed) is int for seed in value) and tuple(value) == expected)


def _h_module():
    """Only the statically known H implementation is importable."""
    from mutual_lab import producer_context_study
    return producer_context_study


def _selection(content):
    condition, spec = content.get("source_condition"), content.get("candidate_spec")
    if condition not in CATALOG or not isinstance(spec, dict) or set(spec) != {
            "candidate_kind", "producer_memory", "understanding", "producer_context"}:
        raise ValueError("Supply exactly one known H0/H1 candidate specification")
    kind, memory, understanding, summary_kind, factory = CATALOG[condition]
    expected = {"candidate_kind": kind, "producer_memory": memory, "understanding": understanding,
                "producer_context": CONTEXTS[condition]}
    if spec != expected or content.get("source_summary_kind") != summary_kind:
        raise ValueError("H candidate condition, architecture and summary kind disagree")
    return condition, spec, factory


def _source_paths(content):
    _selection(content)
    paths = {name: ROOT / "mutual_lab" / name for name in BASE_SOURCES}
    for name in ("object_memory.py", "memory_study.py", "grounding_study.py", "understanding_memory.py"):
        paths[name] = ROOT / "mutual_lab" / name
    local = Path(_h_module().__file__).resolve().parent
    for name in ("producer_context.py", "producer_context_study.py"):
        paths[name] = local / name
    return paths


def _implementation_hashes(content):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in _source_paths(content).items()}


def _validate_source_plan(plan):
    if (plan.seed_pairs != SOURCE_SEED_PAIRS or plan.b_seed_pair != SOURCE_SEED_PAIRS[0]
            or any(type(seed) is not int for pair in (*plan.seed_pairs, plan.b_seed_pair) for seed in pair)):
        raise ValueError("H discovery requires exactly the separate 790xx seed grid")
    actual, expected = asdict(plan), asdict(_reference_plan())
    for key in ("seed_pairs", "b_seed_pair", "metadata"):
        actual.pop(key); expected.pop(key)
    if _canonical(actual) != _canonical(expected):
        raise ValueError("H discovery must retain C numeric model/data/budgets/criteria")


def _source_checks(summary, content, plan):
    condition, spec, _ = _selection(content)
    if (not isinstance(summary, dict) or summary.get("kind") != content["source_summary_kind"]
            or summary.get("status") != "completed" or summary.get("failed_trials")
            or any(summary.get(key) for key in ("missing_languages", "missing_trials", "missing_cells"))
            or summary.get("excluded_from_research") is not True
            or summary.get("eligible_for_fresh_experiment") is not False
            or summary.get("fixed_prior_condition") != "B1" or summary.get("policy") != "mixed_blocks"
            or summary.get("source_rule") != SELECTION_RULE or summary.get("capacity_matched") is not True
            or summary.get("producer_memory") != "objects" or summary.get("understanding") != "objects"
            or summary.get("frozen_feature_policy") != FROZEN_FEATURE_POLICY
            or summary.get("discovery_seed_pairs") != [list(pair) for pair in SOURCE_SEED_PAIRS]
            or summary.get("confirmation_seed_pairs") != [list(pair) for pair in SEED_PAIRS]):
        raise ValueError("H source comparison must be complete, invariant-valid and development-only")
    source_plan = MasteryPlan.from_dict(summary.get("plan", {}))
    _validate_source_plan(source_plan)
    indexed = {}
    for row in summary.get("rows", []):
        if row.get("step") != plan.budgets[-1]:
            continue
        key = (row.get("condition"), row.get("replica"), row.get("language"))
        if (key[0] not in CATALOG or type(key[1]) is not int or key[1] not in range(3)
                or key[2] not in LANGUAGES_ORDER or key in indexed
                or row.get("seed_pair") != list(SOURCE_SEED_PAIRS[key[1]])
                or any(type(seed) is not int for seed in row.get("seed_pair", []))
                or row.get("status") != "completed"):
            raise ValueError("H source needs exactly 24 unique, completed, correctly seeded final cells")
        point = {"dev": row.get("dev", {}), "candidate": row.get("candidate", {}),
                 "contrasts": {"categories": row.get("contrasts", {})}}
        checks, passes = _point_checks(point, plan)
        if not all(checks.values()):
            raise ValueError("H source final metrics, counts, coverage or candidate flags are inconsistent")
        indexed[key] = (row, passes)
    expected = {(arm, replica, language) for arm in CATALOG for replica in range(3) for language in LANGUAGES_ORDER}
    if set(indexed) != expected:
        raise ValueError("H discovery must retain every cell from both full 12-model conditions")
    eligible, minima = [], {}
    for arm in CATALOG:
        cells = [indexed[(arm, replica, language)] for replica in range(3) for language in LANGUAGES_ORDER]
        minima[arm] = min(row["dev"]["valid_generation_accuracy"] for row, _ in cells)
        if all(passes for _, passes in cells):
            eligible.append(arm)
    winner = max(eligible, key=lambda arm: (minima[arm], arm == "H0")) if eligible else None
    selection = summary.get("selection", {})
    if (winner is None or condition != winner or selection.get("status") != "selected"
            or selection.get("condition") != winner or selection.get("rule") != SELECTION_RULE
            or selection.get("fixed_final_step") != plan.budgets[-1] or selection.get("tie_break") != "H0"):
        raise ValueError("Confirmation must use the one prespecified all-12-cell H source winner")
    return {"status": "verified", "selected_condition": winner, "rule": SELECTION_RULE,
            "fixed_final_step": plan.budgets[-1], "verified_source_cells": 24,
            "eligible_conditions": eligible, "minimum_final_production": minima}


def _validate_request(record, plan):
    if not isinstance(record, dict) or set(record) != {"sha256", "content", "source_summary", "prior_evidence"}:
        raise ValueError("A fully pinned candidate request with source/prior evidence is required")
    content = record["content"]
    if not isinstance(content, dict) or type(content.get("schema_version")) is not int or content["schema_version"] != 1:
        raise ValueError("Invalid cohort request schema")
    _selection(content)
    for key in ("source_revision",):
        if not isinstance(content.get(key), str) or not re.fullmatch("[0-9a-f]{40}", content[key]):
            raise ValueError("Invalid source revision")
    for key in ("source_summary_sha256", "source_plan_sha256", "prior_manifest_sha256", "prior_checkpoint_sha256", "validation_plan_sha256", "confirmation_implementation_sha256"):
        if not isinstance(content.get(key), str) or not re.fullmatch("[0-9a-f]{64}", content[key]):
            raise ValueError("Invalid pinned cohort hash: " + key)
    if content["confirmation_implementation_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("The pinned H confirmation implementation differs")
    if not isinstance(record["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", record["sha256"]):
        raise ValueError("Invalid candidate-request byte hash")
    if any(type(content.get(key)) is not int or content[key] <= 0 for key in ("source_run_id", "source_run_attempt")):
        raise ValueError("Pin the source workflow run and attempt")
    if content["source_run_id"] == 43634532997:
        raise ValueError("The original E run failed its producer invariant and is never a candidate source")
    expected = {"phase": "H-confirmation", "fixed_prior_condition": "B1", "validation_run_attempt": 1, "fixed_final_step": plan.budgets[-1],
                "data_version": base.DATA_VERSION, "frozen_feature_policy": FROZEN_FEATURE_POLICY}
    if type(content.get("validation_run_attempt")) is not int or any(content.get(key) != value for key, value in expected.items()) or content.get("policy") != "mixed_blocks":
        raise ValueError("Request differs from the fixed cohort controls")
    if not isinstance(content.get("rationale"), str) or not content["rationale"].strip():
        raise ValueError("Record why this one source candidate was selected")
    hashes = content.get("candidate_source_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_source_paths(content)) or any(
            not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value) for value in hashes.values()):
        raise ValueError("Pin exactly the known architecture and training source files")
    evidence = record["prior_evidence"]
    if (not isinstance(evidence, dict) or evidence.get("manifest_sha256") != content["prior_manifest_sha256"]
            or evidence.get("checkpoint_sha256") != content["prior_checkpoint_sha256"]
            or not isinstance(evidence.get("trained_prior_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", evidence["trained_prior_sha256"])):
        raise ValueError("The pinned image-prior evidence differs")
    _source_checks(record["source_summary"], content, plan)


def load_candidate_request(path, plan, *, plan_path, prior_path, source_summary_path, source_plan_path=None):
    raw = Path(path).read_bytes();content = json.loads(raw)
    if not isinstance(content, dict):raise ValueError("Candidate request must be an object")
    files = {"validation_plan_sha256": Path(plan_path), "prior_manifest_sha256": Path(prior_path).with_suffix(".json"),
             "prior_checkpoint_sha256": Path(prior_path), "source_summary_sha256": Path(source_summary_path),
             "source_plan_sha256": Path(source_plan_path) if source_plan_path else ROOT / "configs/mastery-H-plan.json"}
    for key, source in files.items():
        if content.get(key) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError("Pinned file differs: " + key)
    source_plan = MasteryPlan.load(files["source_plan_sha256"])
    _validate_source_plan(source_plan)
    _validate_plan(plan)
    if _canonical(asdict(plan)) != _canonical(asdict(MasteryPlan.load(files["validation_plan_sha256"]))):
        raise ValueError("The supplied confirmation plan differs from its pinned file")
    if content.get("confirmation_implementation_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("The pinned H confirmation implementation differs")
    manifest = json.loads(files["prior_manifest_sha256"].read_text())
    prior_report = manifest.get("report", {})
    if (manifest.get("format") != "mutual-spatial-prior-development-v1"
            or manifest.get("kind") != "development_only" or manifest.get("excluded_from_research") is not True
            or manifest.get("sha256") != content["prior_checkpoint_sha256"]
            or prior_report.get("status") != "completed" or prior_report.get("dimension") != 128
            or prior_report.get("image_count") != 220 or prior_report.get("excluded_from_research") is not True
            or prior_report.get("eligible_for_fresh_experiment") is not False):
        raise ValueError("Only the pinned completed C image prior is accepted")
    summary = json.loads(files["source_summary_sha256"].read_text())
    if _canonical(summary.get("plan")) != _canonical(asdict(source_plan)):
        raise ValueError("The source summary differs from its pinned discovery plan")
    record = {"sha256": hashlib.sha256(raw).hexdigest(), "content": content,
              "source_summary": summary,
              "prior_evidence": {"manifest_sha256": content["prior_manifest_sha256"],
                  "checkpoint_sha256": content["prior_checkpoint_sha256"],
                  "trained_prior_sha256": manifest["report"]["trained_prior_sha256"]}}
    _validate_request(record, plan)
    if _implementation_hashes(content) != content["candidate_source_sha256"]:
        raise ValueError("The selected architecture/training implementation changed after source selection")
    return record


def _factory(content, pngs):
    condition, _, _ = _selection(content)
    return _h_module().RecordingFactory(condition, pngs)


def _representation(initial):
    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if not key.startswith("initial_")}
        if isinstance(value, list):return [strip(item) for item in value]
        return value
    return strip({key: initial[key] for key in ("memory_specification", "producer_memory_specification",
                 "understanding_specification", "context_specification", "extra_understanding_parameters", "base_understanding_parameters") if key in initial})


def _checkpoint_evidence(path, raw, pair):
    from mutual_lab.neural import digest, torch
    manifest = next(item for item in raw["checkpoints"] if item["step"] == 0)
    checkpoint_path = path / manifest["path"]
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Initial checkpoint SHA256 mismatch")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = state["model"]
    branches = {branch: digest({name[len(branch)+1:]: value for name, value in model.items()
                               if name.startswith(branch + ".")}) for branch in ("understanding", "producer", "visual_encoder")}
    return {"sha256": manifest["sha256"], "step_zero": state["step"] == 0,
        "optimizer_empty": not state["optimizer"]["state"],
        "main_rng_fresh": state["training_rng_state"] == random.Random(pair[1]).getstate(),
        "negative_rng_fresh": state["negative_rng_state"] == random.Random(raw["negative_sampling_seed"]).getstate(),
        "initial_weights_match": branches["understanding"] == raw["initial_understanding_sha256"]
            and branches["producer"] == raw["initial_producer_sha256"] and branches["visual_encoder"] == raw["initial_prior_sha256"],
        "feature_policy_matches": state.get("frozen_feature_policy") == FROZEN_FEATURE_POLICY}


def run_cohort_validation(config, language, replica, plan, output="runs/cohort-context-validation", progress=print,
                          *, prior_bundle=None, candidate_request=None):
    plan.validate(config);_validate_plan(plan);_validate_request(candidate_request, plan)
    if language not in LANGUAGES_ORDER or type(replica) is not int or replica not in range(3):
        raise ValueError("Supply one known language and replica 0, 1 or 2")
    content = candidate_request["content"]
    if prior_bundle is None or prior_bundle.get("_bundle_sha256") != content["prior_checkpoint_sha256"]:
        raise ValueError("Supply the same pinned image-only prior bundle")
    if _implementation_hashes(content) != content["candidate_source_sha256"]:
        raise ValueError("The fixed candidate implementation changed")
    from mutual_lab import development_world as world
    splits = world.development_splits(config, plan.data_seed)
    pngs = [world.render_scene(scene) for name in ("train", "dev") for scene in splits[name]]
    factory = _factory(content, pngs);pair = SEED_PAIRS[replica]
    trial_plan = replace(plan, b_seed_pair=pair)
    directory = Path(output).resolve() / f"cohort-{language}-r{replica}-{uuid.uuid4().hex[:8]}";directory.mkdir(parents=True)
    suite = {"schema_version": 1, "kind": VERSION, "created_at": now(), "status": "running",
        "language": language, "replica": replica, "seed_pair": list(pair), "plan": asdict(plan),
        "candidate_request": candidate_request, "candidate_spec": content["candidate_spec"],
        "source_condition": content["source_condition"], "fixed_prior_condition": "B1", "policy": content["policy"],
        "negative_policy_specification": negative_policy_manifest(content["policy"]),
        "fresh_initialization": True, "resumed": False, "scratch_language_training": True,
        "shared_visual_prior": True, "independent_visual_replications": False,
        "excluded_from_research": True, "eligible_for_fresh_experiment": False, "interpretation": SCOPE,
        "implementation_source_sha256": _implementation_hashes(content), "initial_state": {},
        "cohort_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution": {"workflow_run_id": os.getenv("GITHUB_RUN_ID"), "workflow_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
                      "code_revision": os.getenv("GITHUB_SHA"), "trial_id": directory.name}}
    begun = time.monotonic()
    def snapshot():
        suite.update(initial_state=factory.initial_state or {}, elapsed_seconds=time.monotonic()-begun)
        safe_json(directory/"cohort-context-validation.json",suite)
    snapshot()
    try:
        path = run_trial(config,language,"C","C1",replica,pair,trial_plan,directory/"raw-engine",progress,
            policy=content["policy"],prior_bundle=prior_bundle,remaining_seconds=plan.max_trial_seconds,model_factory=factory)
        raw=json.loads((path/"report.json").read_text())
        suite.update(status=raw["status"],engine_report=raw,raw_report_path=str((path/"report.json").relative_to(directory)))
        if raw["status"]=="completed":suite["initial_checkpoint_evidence"]=_checkpoint_evidence(path,raw,pair)
    except KeyboardInterrupt as exc:
        from mutual_lab.composition_suite import ArmCancelled
        suite["status"]="cancelled"
        try:
            if isinstance(exc,ArmCancelled):
                raw=getattr(exc,"report",None)
                if not isinstance(raw,Mapping):
                    try:raw=json.loads(exc.report_path.read_text())
                    except Exception:raw=None
                if isinstance(raw,Mapping):suite["engine_report"]=dict(raw)
                try:suite["raw_report_path"]=str(exc.report_path.relative_to(directory))
                except Exception:pass
        finally:
            try:snapshot()
            finally:raise exc
    except Exception as exc:suite.update(status="failed",error=f"{type(exc).__name__}: {exc}")
    snapshot();return directory


def aggregate_cohort_validations(suites,plan,*,candidate_request):
    _validate_plan(plan);_validate_request(candidate_request,plan)
    content=candidate_request["content"];_,spec,factory_name=_selection(content)
    indexed={};rows=[];failures=[];trial_checks=[];signatures=[];feature_hashes=[]
    for suite in suites:
        if suite.get("kind")!=VERSION:raise ValueError("Expected cohort-validation.json, not discovery or raw reports")
        language,replica=suite.get("language"),suite.get("replica")
        if language not in LANGUAGES_ORDER or type(replica) is not int or replica not in range(3):raise ValueError("Unknown cohort cell")
        if (replica,language) in indexed:raise ValueError("Duplicate cohort attempt; do not pick a successful retry")
        indexed[(replica,language)]=suite
        if suite.get("candidate_request")!=candidate_request:raise ValueError("Pinned candidate provenance differs")
        if _canonical(suite.get("plan"))!=_canonical(asdict(plan)):raise ValueError("Cohort plans differ")
        raw=suite.get("engine_report",{});initial=suite.get("initial_state",{});pair=SEED_PAIRS[replica]
        points=raw.get("curves",[]);steps=[0,*plan.budgets];tables=raw.get("frozen_feature_tables",{})
        evidence=suite.get("initial_checkpoint_evidence",{});benchmark=raw.get("benchmark")
        checks={
            "fixed_candidate":suite.get("candidate_spec")==spec and suite.get("source_condition")==content["source_condition"]
                and suite.get("fixed_prior_condition")=="B1" and suite.get("policy")==content["policy"]
                and raw.get("model_factory")==factory_name and raw.get("policy")==content["policy"]
                and suite.get("negative_policy_specification")==negative_policy_manifest(content["policy"]),
            "fresh_language_training":all(suite.get(key) is True for key in ("fresh_initialization","scratch_language_training","shared_visual_prior"))
                and suite.get("resumed") is False and suite.get("independent_visual_replications") is False,
            "fresh_optimizer_and_rng":all(evidence.get(key) is True for key in ("step_zero","optimizer_empty","main_rng_fresh","negative_rng_fresh","initial_weights_match","feature_policy_matches")),
            "declared_seed":_matches_pair(suite.get("seed_pair"),pair) and _matches_pair(raw.get("seed_pair"),pair),
            "engine_adapter":raw.get("phase")=="C" and raw.get("condition")=="C1" and raw.get("language")==language and type(raw.get("replica")) is int and raw.get("replica")==replica,
            "trial_plan":_canonical(raw.get("plan"))==_canonical(asdict(replace(plan,b_seed_pair=pair))),
            "completed":suite.get("status")=="completed" and raw.get("status")=="completed",
            "all_budgets":[point.get("step") for point in points]==steps and raw.get("training_steps_completed")==plan.budgets[-1],
            "all_checkpoint_manifests":[item.get("step") for item in raw.get("checkpoints",[])]==steps
                and all(item.get("format")=="mutual-mastery-development-v1" and item.get("kind")=="development_only"
                    and item.get("excluded_from_research") is True and item.get("eligible_for_fresh_experiment") is False
                    and type(item.get("step")) is int and item.get("path")==f"step-{item.get('step'):06d}.pt"
                    and isinstance(item.get("sha256"),str) and re.fullmatch('[0-9a-f]{64}',item['sha256']) for item in raw.get("checkpoints",[])),
            "same_development_population":raw.get("data_version")==base.DATA_VERSION
                and raw.get("split",{}).get("version")==base.DATA_VERSION and raw.get("split",{}).get("pair_sha256")==base.PAIR_SHA256
                and raw.get("split",{}).get("scene_counts")=={"train":220,"dev":112} and raw.get("split",{}).get("png_pool_sha256")==base.PNG_SHA256
                and raw.get("split",{}).get("protected_pairs_excluded")==22,
            "initialization_captured":bool(initial) and initial.get("understanding_sha256")==raw.get("initial_understanding_sha256")
                and initial.get("producer_sha256")==raw.get("initial_producer_sha256") and initial.get("prior_sha256")==raw.get("initial_prior_sha256")
                and initial.get("parameter_count")==raw.get("parameter_count") and initial.get("trainable_parameter_count")==raw.get("trainable_parameter_count"),
            "frozen_prior":raw.get("initial_prior_sha256")==candidate_request["prior_evidence"]["trained_prior_sha256"]==raw.get("final_prior_sha256")
                and raw.get("prior_metadata",{}).get("condition")=="B1" and raw.get("prior_metadata",{}).get("prior_sha256")==raw.get("initial_prior_sha256")
                and raw.get("prior_metadata",{}).get("source_bundle_sha256")==content["prior_checkpoint_sha256"],
            "canonical_immutable_features":raw.get("frozen_feature_policy")==FROZEN_FEATURE_POLICY and bool(tables.get("step_0"))
                and tables.get("step_0")==tables.get("final") and tables.get("final",{}).get("policy")==FROZEN_FEATURE_POLICY
                and bool(tables.get("final",{}).get("caches"))
                and all(point.get("frozen_feature_table_sha256")==tables.get("final",{}).get("sha256") for point in points),
            "benchmark_reset":plan.benchmark_steps==0 or(isinstance(benchmark,Mapping) and benchmark.get("steps")==plan.benchmark_steps
                and benchmark.get("initial_weights_restored") is True
                and bool(benchmark.get("image_caches_cleared"))),
            "native_only_input":raw.get("training_languages")==[language] and raw.get("foreign_training_languages")==[] and raw.get("protected_splits_used")==[],
            "development_only":all(suite.get(key) is value and raw.get(key) is value for key,value in (("excluded_from_research",True),("eligible_for_fresh_experiment",False))),
            "execution_identity":suite.get("execution",{}).get("workflow_run_attempt") in (None,"1")
                and (suite.get("execution",{}).get("code_revision") is None or
                     (raw.get("revision",{}).get("commit")==suite["execution"]["code_revision"] and raw.get("revision",{}).get("working_tree_dirty") is False)),
            "fixed_training_controls":raw.get("optimization")=="independent-branch-clipping-adam-v1" and raw.get("clip_norm_per_branch")==1
                and raw.get("choice_count")==4 and raw.get("training_chance_accuracy")==.25 and raw.get("comprehension_forms")==4
                and raw.get("production_forms")==1 and raw.get("teacher_forced_evaluation_forms")==4
                and raw.get("negative_sampling_seed")==pair[1]+plan.negative_seed_offset and raw.get("model_profile")==asdict(plan.model_profile)
                and type(raw.get("parameter_count")) is int and raw["parameter_count"]>0 and type(raw.get("trainable_parameter_count")) is int
                and 0<raw["trainable_parameter_count"]<raw["parameter_count"],
            "cohort_wrapper_source":suite.get("cohort_source_sha256")==content["confirmation_implementation_sha256"],
            "pinned_implementation":suite.get("implementation_source_sha256")==content["candidate_source_sha256"]
                and all(raw.get("source_sha256",{}).get(name)==digest for name,digest in content["candidate_source_sha256"].items()),
        }
        if spec['candidate_kind']!='e_attentive':
            memory=initial.get('memory_specification',initial.get('producer_memory_specification',{}))
            preflight=initial.get('memory_preflight',{})
            checks['fixed_memory']=memory.get('memory_source')==spec['producer_memory'] and memory.get('memory_normalization')=='l2_first32_raw_yx_v1'
            checks['memory_preflight']=preflight.get('status')=='completed' and preflight.get('input_pngs')==332 and preflight.get('unique_pngs')==332 and preflight.get('successful_pngs')==332 and preflight.get('invalid_pngs')==0 and preflight.get('fallback_count')==0
        if spec['candidate_kind']=='producer_context':
            u=initial.get('understanding_specification',{})
            checks['fixed_understanding']=u.get('understanding')=='objects' and u.get('producer_memory')=='objects'
            checks['fixed_producer_context']=initial.get('context_specification',{}).get('context')==spec['producer_context']
        exposures=raw.get('negative_exposures',{});total=plan.budgets[-1]*raw.get('batch_size',0) if type(raw.get('batch_size')) is int else 0
        checks['negative_exposure_counts']=total>0 and exposures.get('examples')==total and exposures.get('distractor_slots')==3*total and exposures.get('semantic_duplicate_choices')==0 and exposures.get('png_duplicate_choices')==0
        if content['policy']=='mixed_blocks':checks['mixed_schedule']=all(exposures.get('branch_'+branch+'_examples')==count for branch,count in base._mixed_branch_counts(total).items())
        final=False
        for point in points:
            point_checks,passes=_point_checks(point,plan)
            if not all(point_checks.values()):checks['point_'+str(point.get('step'))]=False
            if point.get('step')==plan.budgets[-1]:final=passes
            rows.append({'language':language,'replica':replica,'seed_pair':list(pair),'step':point.get('step'),'status':suite.get('status'),
                         'dev':point.get('dev',{}),'contrasts':point.get('contrasts',{}).get('categories',{}),'candidate':point.get('candidate',{}),'point_checks':point_checks})
        trial_checks.append({'language':language,'replica':replica,'checks':checks,'final_candidate':final and all(checks.values())})
        if not all(checks.values()):failures.append({'language':language,'replica':replica,'status':suite.get('status'),'failed_checks':[key for key,value in checks.items() if not value],'error':suite.get('error',raw.get('error'))})
        execution={key:suite.get('execution',{}).get(key) for key in ('workflow_run_id','workflow_run_attempt','code_revision')}
        signatures.append({'raw':{key:raw.get(key) for key in ('config_sha256','source_sha256','model_profile','model_factory','parameter_count','trainable_parameter_count','batch_size','learning_rate','split','prior_metadata')},
                           'representation':_representation(initial),'execution':execution,'cohort_source_sha256':suite.get('cohort_source_sha256')})
        feature_hashes.append({'language':language,'replica':replica,'sha256':tables.get('final',{}).get('sha256'),'environment':raw.get('environment')})
    missing=[{'replica':replica,'language':language} for replica in range(3) for language in LANGUAGES_ORDER if (replica,language) not in indexed]
    blocks=[];u=[];p=[];priors=[]
    for replica,pair in enumerate(SEED_PAIRS):
        raws=[indexed[(replica,language)].get('engine_report',{}) for language in LANGUAGES_ORDER if (replica,language) in indexed]
        block_checks={key:len(raws)==4 and all(raw.get(key) for raw in raws) and len({_canonical(raw.get(key)) for raw in raws})==1 for key in ('initial_understanding_sha256','initial_producer_sha256','initial_prior_sha256','meaning_stream_sha256')}
        if raws:u.append(raws[0].get('initial_understanding_sha256'));p.append(raws[0].get('initial_producer_sha256'))
        priors.extend(raw.get('initial_prior_sha256') for raw in raws)
        block_trials=[row for row in trial_checks if row['replica']==replica]
        blocks.append({'replica':replica,'seed_pair':list(pair),'checks':block_checks,'final_candidate_count':sum(row['final_candidate'] for row in block_trials)})
    checks={'all_twelve_jobs':len(indexed)==12 and not missing,'all_trials_valid':len(trial_checks)==12 and all(all(row['checks'].values()) for row in trial_checks),
            'all_seed_blocks_matched':all(all(block['checks'].values()) for block in blocks),
            'distinct_language_initializations_across_seed_blocks':len(u)==len(p)==3 and all(u) and all(p) and len(set(u))==len(set(p))==3,
            'one_shared_frozen_visual_prior':len(priors)==12 and all(priors) and len(set(priors))==1,
            'same_protocol_provenance':len(signatures)==12 and bool(signatures[0]['raw'].get('source_sha256')) and bool(signatures[0]['raw'].get('config_sha256')) and len({_canonical(value) for value in signatures})==1,
            'all_twelve_fixed_final_candidates':len(trial_checks)==12 and all(row['final_candidate'] for row in trial_checks)}
    completed=not missing and not failures and all(value for key,value in checks.items() if key!='all_twelve_fixed_final_candidates')
    execution_completed=len(indexed)==12 and all(suite.get('status')=='completed' and suite.get('engine_report',{}).get('status')=='completed' for suite in indexed.values())
    robust=all(checks.values())
    return {'schema_version':1,'kind':VERSION+'-comparison','status':'completed' if completed else 'incomplete',
        'execution_completed':execution_completed,'provenance_valid':completed,'assessment':'robust_development_candidate' if robust else 'targets_not_met' if completed else 'indeterminate_incomplete',
        'robust_development_candidate':robust,'candidate_spec':spec,'source_condition':content['source_condition'],'candidate_request':candidate_request,'plan':asdict(plan),
        'expected_jobs':12,'reported_jobs':len(indexed),'missing_trials':missing,'failed_trials':failures,'checks':checks,'seed_blocks':blocks,'trial_checks':trial_checks,'rows':rows,
        'fixed_final_step':plan.budgets[-1],'final_candidate_count':sum(row['final_candidate'] for row in trial_checks),
        'noncandidate_final_trials':[{'language':row['language'],'replica':row['replica']} for row in trial_checks if not row['final_candidate']],
        'observed_feature_tables':feature_hashes,'cross_host_feature_table_bit_equality_required':False,
        'shared_visual_prior':True,'independent_visual_replications':False,'excluded_from_research':True,'eligible_for_fresh_experiment':False,'interpretation':SCOPE}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/neural.example.json')
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/mastery-H-cohort-plan.json')
    parser.add_argument('--candidate-request',type=Path,required=True);parser.add_argument('--source-summary',type=Path,required=True)
    parser.add_argument('--source-plan',type=Path,required=True);parser.add_argument('--prior-bundle',type=Path,required=True)
    parser.add_argument('--language',choices=LANGUAGES_ORDER);parser.add_argument('--replica',type=int,choices=range(3));parser.add_argument('--aggregate-dir',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv);plan=MasteryPlan.load(args.plan);_validate_plan(plan)
    request=load_candidate_request(args.candidate_request,plan,plan_path=args.plan,prior_path=args.prior_bundle,source_summary_path=args.source_summary,source_plan_path=args.source_plan)
    if args.aggregate_dir is not None:
        suites=[json.loads(path.read_text()) for path in sorted(args.aggregate_dir.rglob('cohort-context-validation.json'))]
        result=aggregate_cohort_validations(suites,plan,candidate_request=request);destination=args.output or Path('runs/cohort-context-validation-comparison.json')
        destination.parent.mkdir(parents=True,exist_ok=True);safe_json(destination,result);print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if result['status']=='completed' else 1
    if args.language is None or args.replica is None:parser.error('--language and --replica are required for execution')
    from mutual_lab.config import ExperimentConfig
    from mutual_lab.spatial_prior import load_prior_bundle
    path=run_cohort_validation(ExperimentConfig.load(args.config),args.language,args.replica,plan,args.output or 'runs/cohort-context-validation',prior_bundle=load_prior_bundle(args.prior_bundle),candidate_request=request)
    print(path,flush=True);return 0 if json.loads((path/'cohort-context-validation.json').read_text())['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
