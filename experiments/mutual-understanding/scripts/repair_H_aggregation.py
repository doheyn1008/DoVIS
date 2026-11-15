"""Audit and re-aggregate immutable H results without rerunning any learner.

This run-specific post-hoc repair changes two host-cache HASH METADATA fields
in ephemeral initial-state copies only. Every original ZIP, extracted byte,
checkpoint tensor, pair control, metric and selection rule remains unchanged.
Use PYTHONPATH/cwd from the original frozen scientific checkout, not a new model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

SOURCE_RUN = 43652561578
SOURCE_REVISION = "736097e5298fb71037a420ec8bd635c73a0154dd"
SOURCE_STUDY_SHA = "1edff52e0a044d240bed6a1b570ea8cd08310ae7212cdf55d7a05bd6062ce26f"
PLAN_SHA = "a2622ee9bada8ce5f79365fa790d90bf2902a806c8af4d4a389b5081009a962c"
EVIDENCE_SHA = "9572bbcc5e68a13265665b803e1d279f92c9644726b4d5c070516efdbbb5f47f"
VISUAL_GROUPS = {"e8666a8dcfa27ff9b51ef9dc7c3409806c1c53323820eafc2810f5d1c76a349b",
                 "f08f116ae2a9489f206551e073f9f43000a79ce256884faf8608830c8ffad8b0"}
CANONICAL_VISUAL = "f08f116ae2a9489f206551e073f9f43000a79ce256884faf8608830c8ffad8b0"
ARTIFACTS = {
    'l1-H-input-43652561578': (14135954751, 'adce33e1a2fdd7001dfbde4395579aa8858d54baca96640d2ae5f93db0473f97'),
    'l1-H-trials-en-r0-43652561578': (14137617578, 'ea8e75c05bc67146cb9f0620f4ce5ee1007328bf3f3b9d9760773b5ed27df79b'),
    'l1-H-trials-en-r1-43652561578': (14136913402, '0bc24d1d5487cd77f7cdcde90fa29cc9a0ff319691ea3ea92e60e0879a26474c'),
    'l1-H-trials-en-r2-43652561578': (14137671798, '7a27c2905538c4698ad0b4b9fd8e2c550da7c9c29cd217118a16fe53c6715095'),
    'l1-H-trials-es-r0-43652561578': (14137905033, '50d2a1a23f61f00e3e1bccabb53b6f2170a0c742f31de9623fccd8ae4e1cece6'),
    'l1-H-trials-es-r1-43652561578': (14137338945, '0dc908f97de2041eac91a7feff54cc4bb0a6ad027e725419919cbcfcc334d516'),
    'l1-H-trials-es-r2-43652561578': (14137826711, '18b71bc72307ea738cab257ce177b3892ce74f42ee5c8d859c52ac34cedd399c'),
    'l1-H-trials-ja-r0-43652561578': (14137231603, '8822ad580981452cdc728fea4b554945a7f4147ac0076ba4659a24ad25ce9a6a'),
    'l1-H-trials-ja-r1-43652561578': (14137880089, '593053a1be3bee2e71a39c6cbc21f57846ce4c08d4781c931ec31a006e7833a0'),
    'l1-H-trials-ja-r2-43652561578': (14137776507, '342fe8cd754392bce39ce5b03dfbc97163f35bf2814ed13a690fe08e6e61e15c'),
    'l1-H-trials-ko-r0-43652561578': (14137104172, 'ae6c1bf3de678ef9bd5675dfd4bfddb80824c5e4e30bbc392ed2a042e9f6fa0c'),
    'l1-H-trials-ko-r1-43652561578': (14137786814, 'e89e2c32421cf283cc757b087b5a3b165c54f87c57755dcfcee2d454818a1f09'),
    'l1-H-trials-ko-r2-43652561578': (14137282533, 'ccea4a34333d592cc9ea981730c15b230e1864bcf81b29e5db87a8073fe47b5f'),
}
ALLOWED_PATHS = {("memory_preflight", "feature_tables", "caches", "visual_cache", "sha256"),
                 ("memory_preflight", "feature_tables", "sha256")}
EXPECTED_CACHES = {"object_memory_cache", "producer_context_cache", "spatial_grid_cache", "understanding_object_cache", "visual_cache"}
NONVISUAL = {"object_memory_cache": "f4271f87168d043e77a5792ef8dfcdab217dc6c3c78a9eca6ca313cbfc5779d6",
    "producer_context_cache": "cc91a31d6716dc11b3e3508c45f26e9d1f691c29a2f2b69c025d51829fff35c3",
    "spatial_grid_cache": "e4f60d0aa6d7f3d3b6a6494b1c861b99f649c6f9ec51abaf201b20f297327c95",
    "understanding_object_cache": "354d558488890f935ad429241c05bbf136cba617f98d10a5c7d23abefe14218f"}


def _json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _safe(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Unsafe artifact member")
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or any(p in (".", "..") or ":" in p for p in parts):
        raise ValueError("Unsafe artifact member")
    path = Path(root)
    for part in parts:
        path /= part
        if path.is_symlink(): raise ValueError("Symlinked artifact")
    if not path.is_file(): raise ValueError("Missing original artifact member: " + str(path))
    return path


def _authority(root):
    """Bind all extracted data, including metrics/predictions, to original ZIPs."""
    files, pins = {}, []
    for name, (identity, expected) in sorted(ARTIFACTS.items()):
        path = _safe(root, name + ".zip")
        if _sha(path) != expected: raise ValueError("Original artifact ZIP digest changed: " + name)
        pins.append({"name": name, "id": identity, "sha256": expected})
        with zipfile.ZipFile(path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if len(infos) != len({info.filename for info in infos}): raise ValueError("Duplicate original ZIP members")
            for info in infos:
                extracted = _safe(Path(root) / name, info.filename)
                digest = hashlib.sha256()
                with archive.open(info) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
                value = digest.hexdigest()
                if _sha(extracted) != value: raise ValueError("Extracted artifact bytes changed: " + str(extracted))
                files[extracted.resolve()] = value
        actual = {p.resolve() for p in (Path(root) / name).rglob("*") if p.is_file()}
        if actual != {p for p in files if p.is_relative_to((Path(root) / name).resolve())}:
            raise ValueError("Extra extracted files outside the authenticated original ZIP")
    return files, pins


def _table(table, digest, *, h=True):
    if not isinstance(table, dict) or set(table) != {"policy", "caches", "sha256"} or table["policy"] != "per_png_singleton_v1":
        raise ValueError("Invalid original frozen feature table")
    if h and set(table["caches"]) != EXPECTED_CACHES: raise ValueError("Unknown H cache fields")
    for name, cache in table["caches"].items():
        if set(cache) != {"entries", "sha256"} or type(cache["entries"]) is not int or not re.fullmatch("[0-9a-f]{64}", cache["sha256"]):
            raise ValueError("Invalid original cache metadata")
        if h and cache["entries"] != (0 if name == "spatial_grid_cache" else 332): raise ValueError("Original H cache coverage changed")
        if h and name != "visual_cache" and cache["sha256"] != NONVISUAL[name]: raise ValueError("A non-visual-cache hash changed")
    if table["caches"]["visual_cache"]["sha256"] not in VISUAL_GROUPS: raise ValueError("Unknown host feature group")
    if table["sha256"] != digest({"policy": table["policy"], "caches": table["caches"]}):
        raise ValueError("Original feature-table derived hash does not verify")


def _prior_evidence(path, digest):
    path = Path(path)
    if _sha(path) != EVIDENCE_SHA: raise ValueError("The prior-690 host evidence differs from its immutable pin")
    evidence = json.loads(path.read_text())
    if (evidence.get("schema_version") != 1 or evidence.get("kind") != "dovis-known-frozen-feature-host-groups"
            or evidence.get("source_run_id") != 43647997707 or evidence.get("source_revision") != "c65ab2752c117ddf66c98c4e871f553cf9c8f95a"
            or evidence.get("predates_H_training") is not True or evidence.get("policy") != "per_png_singleton_v1"
            or evidence.get("cache") != "visual_cache" or set(evidence.get("allowed_sha256", [])) != VISUAL_GROUPS
            or len(evidence.get("evidence_files", [])) != 2): raise ValueError("Invalid prior host evidence")
    seen, reports = set(), []
    for item in evidence["evidence_files"]:
        artifact = item["artifact"]
        if artifact["id"] not in (14135346265, 14135337469) or artifact["workflow_run"]["id"] != 43647997707:
            raise ValueError("Unexpected prior evidence artifact")
        report_path = _safe(path.parent / artifact["name"], item["report_relative_path"])
        if _sha(report_path) != item["report_sha256"]: raise ValueError("Prior report evidence changed")
        report = json.loads(report_path.read_text()); table = report["frozen_feature_tables"]["step_0"]
        _table(table, digest, h=False)
        if (report.get("frozen_feature_policy") != evidence["policy"] or report["frozen_feature_tables"].get("final") != table
                or table["sha256"] != item["feature_table_sha256"]
                or table["caches"]["visual_cache"]["sha256"] != item["visual_cache_sha256"]):
            raise ValueError("Prior report does not prove its declared host group")
        seen.add(item["visual_cache_sha256"]); reports.append({"path": str(report_path), "sha256": item["report_sha256"], "artifact": artifact})
    if seen != VISUAL_GROUPS: raise ValueError("Prior evidence does not cover both known groups")
    return {"sha256": EVIDENCE_SHA, "source_run_id": evidence["source_run_id"], "reports": reports}


def _differences(a, b, prefix=()):
    if type(a) is not type(b): return [prefix]
    if isinstance(a, dict):
        if set(a) != set(b): return [prefix]
        return [item for key in sorted(a) for item in _differences(a[key], b[key], (*prefix, key))]
    if isinstance(a, list):
        if len(a) != len(b): return [prefix]
        return [item for i, (x, y) in enumerate(zip(a, b)) for item in _differences(x, y, (*prefix, i))]
    return [] if a == b else [prefix]


def _independent_selection(rows, plan):
    """Separate count/coverage/six-target proof, never infer targets from flags."""
    import math
    coverage = {"role_reversal": (112, 28), "color_only": (112, 28), "shape_only": (108, 27), "binding_swap": (60, 15)}
    eligible, worst, checks = [], {}, []
    def rate(metrics, h, n, a, expected):
        hit, count, value = metrics.get(h), metrics.get(n), metrics.get(a)
        return (type(hit) is int and type(count) is int and count == expected and 0 <= hit <= count
            and type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 and abs(value - hit / count) < 1e-12)
    expected_ids = {(language, replica) for language in ("ko", "ja", "en", "es") for replica in range(3)}
    for condition in ("H0", "H1"):
        final = [r for r in rows if r["condition"] == condition and r["step"] == 9600]
        valid_cells = len(final) == 12 and {(r["language"], r["replica"]) for r in final} == expected_ids
        passed = []
        for row in final:
            dev, gate, categories = row["dev"], row["candidate"], row["contrasts"]
            valid = (row["status"] == "completed" and rate(dev, "comprehension_hits", "comprehension_items", "comprehension_accuracy", 448)
                and rate(dev, "valid_generation_hits", "generation_scenes", "valid_generation_accuracy", 112)
                and set(categories) == set(gate.get("categories", {})) == set(coverage)
                and dev.get("comprehension_accuracy", -1) >= plan.comprehension_min
                and dev.get("valid_generation_accuracy", -1) >= plan.production_min)
            for name, (anchors, pairs) in coverage.items():
                metric, declared = categories.get(name, {}), gate.get("categories", {}).get(name, {})
                valid &= (rate(metric, "hits", "items", "accuracy", anchors * 4)
                    and metric.get("accuracy", -1) >= plan.contrast_min and declared.get("available_anchors") == anchors
                    and declared.get("available_pairs") == pairs and anchors >= plan.min_contrast_anchors and pairs >= plan.min_contrast_pairs
                    and declared.get("coverage_sufficient") is True and declared.get("score_target_met") is True
                    and all(declared.get(key) == metric.get(key) for key in ("hits", "items", "accuracy")))
            valid &= (gate.get("status") == "development_candidate" and gate.get("score_targets_met") is True
                and gate.get("coverage_sufficient") is True and gate.get("excluded_from_research") is True
                and gate.get("eligible_for_fresh_experiment") is False)
            passed.append(bool(valid)); checks.append({"condition": condition, "language": row["language"], "replica": row["replica"], "qualified": bool(valid)})
        worst[condition] = min(r["dev"]["valid_generation_accuracy"] for r in final) if valid_cells else None
        if valid_cells and all(passed): eligible.append(condition)
    selected = max(eligible, key=lambda condition: (worst[condition], condition == "H0")) if eligible else None
    return {"rule": "all_discovery_cells_minimum_final_production_tie_h0_v1", "condition": selected,
        "eligible_conditions": eligible, "worst_final_production": worst, "fixed_final_step": 9600, "tie_break": "H0", "candidate_checks": checks}


def repair_aggregation(input_root, plan_path, host_evidence):
    """Return repaired summary and audit manifest; write no original/result files."""
    from mutual_lab import producer_context_study as original
    from mutual_lab.mastery_lab import MasteryPlan
    from mutual_lab.neural import digest, torch
    root = Path(input_root).resolve(); plan_path = Path(plan_path)
    study_path = Path(original.__file__).resolve()
    if _sha(study_path) != SOURCE_STUDY_SHA: raise ValueError("The imported H scientific source is not the original frozen implementation")
    if _sha(plan_path) != PLAN_SHA: raise ValueError("The original H plan changed")
    plan = MasteryPlan.load(plan_path); original._validate_plan(plan)
    files, artifact_pins = _authority(root)
    context_path = root / 'l1-H-input-43652561578/provenance/run-context.json'
    if context_path.resolve() not in files: raise ValueError("Missing authenticated original H context")
    context = json.loads(context_path.read_text())
    if (context.get("code_revision") != SOURCE_REVISION or str(context.get("run_id")) != str(SOURCE_RUN)
            or str(context.get("run_attempt")) != "1" or context.get("repository") != "doheyn1008/DoVIS"):
        raise ValueError("Unexpected original H workflow provenance")
    package = study_path.parent
    for name, sha in context["source_sha256"].items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*[.]py", name) or _sha(package / name) != sha:
            raise ValueError("An original scientific package source changed: " + str(name))
    evidence = _prior_evidence(host_evidence, digest)
    suites, original_rows, checkpoints, pair_audits, seen, init_hashes = [], [], [], [], set(), {}
    source_suites, source_reports, observed_tables = {}, {}, []
    for language in original.LANGUAGES_ORDER:
        for replica in range(3):
            artifact = root / f"l1-H-trials-{language}-r{replica}-{SOURCE_RUN}"
            paths = list(artifact.rglob("producer-context-study.json"))
            if len(paths) != 1: raise ValueError("Each original H cell must have exactly one suite")
            path = paths[0]; suite = json.loads(path.read_text())
            source_suites[str(path.relative_to(root))] = files[path.resolve()]
            if (suite.get("kind") != original.VERSION or suite.get("phase") != "H" or suite.get("axis") != "producer_global_conditioning"
                    or suite.get("language") != language or suite.get("replica") != replica or suite.get("status") != "completed"
                    or len(suite.get("reports", [])) != 2): raise ValueError("Original H cells are incomplete or mislabeled")
            audit = original.context_pair_audit(suite["reports"])
            if audit["status"] != "matched" or audit != suite.get("pair_audit"):
                raise ValueError("An original H pair control failed")
            pair_audits.append({"language": language, "replica": replica, **audit})
            for record in suite["reports"]:
                condition, initial, raw = record["context_condition"], record["initial_state"], record["engine_report"]
                identity = (language, replica, condition)
                if identity in seen: raise ValueError("Duplicate original H trial")
                seen.add(identity)
                report_path = _safe(path.parent, record["raw_report_path"])
                if report_path.resolve() not in files or json.loads(report_path.read_text()) != raw:
                    raise ValueError("Embedded H report differs from authenticated original bytes")
                source_reports[str(report_path.relative_to(root))] = files[report_path.resolve()]
                if raw.get("source_sha256") != context["source_sha256"]: raise ValueError("Trial scientific sources differ from the original input")
                _table(initial["memory_preflight"]["feature_tables"], digest)
                table = raw["frozen_feature_tables"]["step_0"]; _table(table, digest)
                _table(raw["frozen_feature_tables"]["final"], digest)
                if initial["memory_preflight"]["feature_tables"] != table or raw["frozen_feature_tables"]["final"] != table:
                    raise ValueError("Original own-trial preflight/step0/final cache drift is not a host-metadata repair")
                observed_tables.append({"language": language, "replica": replica, "condition": condition,
                    "initial_preflight": initial["memory_preflight"]["feature_tables"], "raw_tables": raw["frozen_feature_tables"]})
                if any(point.get("frozen_feature_table_sha256") != table["sha256"] for point in raw["curves"]):
                    raise ValueError("An original intermediate feature table drifted")
                actual_u = {}
                if len(raw["checkpoints"]) != 4: raise ValueError("All four actual original H checkpoints are required")
                for item in raw["checkpoints"]:
                    checkpoint = _safe(report_path.parent, item["path"])
                    sidecar = checkpoint.with_suffix(".json")
                    if (files.get(checkpoint.resolve()) != item["sha256"] or sidecar.resolve() not in files
                            or json.loads(sidecar.read_text()) != item): raise ValueError("Original checkpoint SHA/sidecar mismatch")
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    if (payload.get("format") != "mutual-mastery-development-v1" or payload.get("kind") != "development_only"
                            or payload.get("excluded_from_research") is not True or payload.get("eligible_for_fresh_experiment") is not False
                            or payload.get("step") != item["step"] or payload.get("seed_pair") != raw["seed_pair"]
                            or _json(payload.get("plan")) != _json(raw["plan"]) or payload.get("model_factory") != raw["model_factory"]
                            or any(payload.get(key) != raw.get(key) for key in ("language", "replica", "phase", "condition", "revision", "data_sha256", "split", "frozen_feature_policy"))
                            or payload.get("source_sha256") != raw["source_sha256"]
                            or payload.get("frozen_feature_tables") != raw["frozen_feature_tables"]):
                        raise ValueError("Original actual checkpoint provenance/control differs")
                    model = payload["model"]
                    subset = lambda prefix: {name.removeprefix(prefix): value for name, value in model.items() if name.startswith(prefix)}
                    u, p, prior = digest(subset("understanding.")), digest(subset("producer.")), digest(subset("visual_encoder."))
                    if prior != raw["initial_prior_sha256"]: raise ValueError("Frozen prior weights changed")
                    actual_u[str(item["step"])] = u
                    if item["step"] == 0:
                        if digest(model) != initial["full_state_sha256"] or u != raw["initial_understanding_sha256"] or p != raw["initial_producer_sha256"]:
                            raise ValueError("Actual initial NN weights differ from captured H initialization")
                        if init_hashes.setdefault(replica, digest(model)) != digest(model):
                            raise ValueError("Initial NN state differs across languages/conditions within a seed block")
                    if item["step"] == 9600 and (u != raw["final_understanding_sha256"] or p != raw["final_producer_sha256"]):
                        raise ValueError("Actual final NN weights differ from original report")
                    checkpoints.append({"path": str(checkpoint.relative_to(root)), "sha256": item["sha256"],
                        "step": item["step"], "understanding_sha256": u, "producer_sha256": p, "prior_sha256": prior})
                if actual_u != record["understanding_checkpoints"] or set(actual_u) != {"0", "2400", "4800", "9600"}:
                    raise ValueError("Actual U checkpoint weights differ at an original budget")
                for point in raw["curves"]:
                    original_rows.append({"language": language, "replica": replica, "seed_pair": list(plan.seed_pairs[replica]),
                        "condition": condition, "producer_context": original.CONDITIONS[condition], "step": point["step"],
                        "status": record["status"], "dev": point["dev"], "contrasts": point["contrasts"]["categories"], "candidate": point["candidate"]})
            suites.append(suite)
    if len(seen) != 24 or len(checkpoints) != 96 or len(set(init_hashes.values())) != 3:
        raise ValueError("Incomplete or non-distinct original H discovery blocks")
    normalized, changes = copy.deepcopy(suites), []
    references = {}
    for suite in suites:
        for record in suite["reports"]:
            key = (suite["replica"], record["context_condition"])
            initial = record["initial_state"]
            reference = references.setdefault(key, initial)
            if any(path not in ALLOWED_PATHS for path in _differences(reference, initial)):
                raise ValueError("Cross-language initial metadata differs beyond the two authorized host hash paths")
    for suite in normalized:
        for record in suite["reports"]:
            table = record["initial_state"]["memory_preflight"]["feature_tables"]
            old_visual, old_table = table["caches"]["visual_cache"]["sha256"], table["sha256"]
            table["caches"]["visual_cache"]["sha256"] = CANONICAL_VISUAL
            table["sha256"] = digest({"policy": table["policy"], "caches": table["caches"]})
            if (old_visual, old_table) != (CANONICAL_VISUAL, table["sha256"]):
                changes.append({"language": suite["language"], "replica": suite["replica"], "condition": record["context_condition"],
                    "old_visual_cache_sha256": old_visual, "new_visual_cache_sha256": CANONICAL_VISUAL,
                    "old_derived_table_sha256": old_table, "new_derived_table_sha256": table["sha256"]})
    for raw_suite, normalized_suite in zip(suites, normalized):
        for path in _differences(raw_suite, normalized_suite):
            if len(path) < 4 or path[0] != "reports" or path[1] not in (0, 1) or path[2] != "initial_state" or path[3:] not in ALLOWED_PATHS:
                raise ValueError("Normalization changed an unauthorized original leaf")
    summary = original.aggregate_context_studies(normalized, plan)
    if summary["status"] != "completed" or summary["rows"] != original_rows:
        raise ValueError("Repair changed original metric rows or failed original aggregate controls")
    if summary["selection"] != original.select_context_candidate(original_rows, plan, complete=True):
        raise ValueError("Repair changed the original all-12/minimum-P/tie-H0 selection rule")
    independent = _independent_selection(original_rows, plan)
    if any(summary["selection"].get(key) != value for key, value in independent.items()):
        raise ValueError("Independent raw-count/coverage/minimum-P selection proof failed")
    if Counter(_json(row) for row in summary["rows"]) != Counter(_json(row) for row in original_rows):
        raise ValueError("Original metric row multiset changed")
    after = {str(path.relative_to(root)): _sha(path) for path in files}
    before = {str(path.relative_to(root)): value for path, value in files.items()}
    if after != before: raise ValueError("Original H artifact files changed during audit/repair")
    summary["posthoc_metadata_repair"] = {"applied": True, "source_original_aggregation_failed": True,
        "scope": "Two initial preflight host-cache hash metadata paths in ephemeral aggregation copies only; no model, cache values, raw report or metric changes"}
    repair_run = {"run_id": os.environ.get("GITHUB_RUN_ID"), "revision": os.environ.get("GITHUB_SHA"), "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}
    summary["aggregation_repair"] = {"original_run_id": SOURCE_RUN, "original_revision": SOURCE_REVISION, "original_run_attempt": 1,
        "original_aggregation_failed": True, "original_failure_job_id": 132631927039, "repair_run": repair_run,
        "script_sha256": _sha(__file__), "source_study_sha256": SOURCE_STUDY_SHA}
    manifest = {"schema_version": 1, "kind": "dovis-H-posthoc-metadata-repair", "posthoc_metadata_repair": True,
        "source_run_id": SOURCE_RUN, "source_revision": SOURCE_REVISION, "source_run_attempt": 1,
        "source_original_aggregation_failed": True, "source_failure_job_id": 132631927039,
        "repair_run": repair_run,
        "script_sha256": _sha(__file__), "source_study_sha256": SOURCE_STUDY_SHA, "plan_sha256": PLAN_SHA,
        "summary_sha256": hashlib.sha256(_json(summary)).hexdigest(), "artifact_pins": artifact_pins,
        "known_prior_host_evidence": evidence, "authorized_paths": ["initial_state." + ".".join(path) for path in sorted(ALLOWED_PATHS)],
        "normalizations": changes, "raw_pair_audits": pair_audits, "actual_checkpoint_audits": checkpoints,
        "observed_original_feature_tables": observed_tables, "independent_raw_selection": independent,
        "original_suite_sha256": source_suites, "original_report_sha256": source_reports,
        "original_files_sha256_before": before, "original_files_sha256_after": after,
        "original_rows_sha256": hashlib.sha256(_json(original_rows)).hexdigest(), "metric_rows_unchanged": True,
        "original_selection_rule_unchanged": True, "nn_training_rerun": False, "original_failed_run_preserved": True,
        "excluded_from_research": True, "eligible_for_fresh_experiment": False}
    return summary, manifest


def _publish(path, data):
    path = Path(path).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream: stream.write(data)
        os.link(temporary, path)
    finally: Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--host-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(path.resolve().is_relative_to(args.input.resolve()) for path in (args.output, args.manifest)):
        raise ValueError("Repair outputs must not be written inside the original input tree")
    if args.output.resolve() == args.manifest.resolve() or args.output.exists() or args.manifest.exists():
        raise ValueError("Repair output files must be distinct and new")
    summary, manifest = repair_aggregation(args.input, args.plan, args.host_evidence)
    _publish(args.output, _json(summary)); _publish(args.manifest, _json(manifest))
    print(json.dumps({"summary": str(args.output), "manifest": str(args.manifest), "summary_sha256": manifest["summary_sha256"]}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
