"""Independent, read-only H1 confirmation artifact audit.

Inputs must be original GitHub artifact ZIPs plus authenticated API metadata and
their extracted bytes. No native weights are used to construct references.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import stat
import sys
import time
import zipfile

PAIRS = ((99007, 99009), (99017, 99019), (99027, 99029))
LANGUAGES = ("ko", "ja", "en", "es")
STEPS = (0, 2400, 4800, 9600)
REQUEST_SHA = "eb78a12365c62da6ccc89236a65aff4d5b6d4bc99664402702d568a0e1992b3d"
PRIOR_SHA = "862eed3ec93a4567019075adcce59faf75295987a5cf6b816cace607da926e6d"
PRIOR_MANIFEST_SHA = "3ef0e0ce056c4c553468a0572f42098b09d3b3d7f9d68fe9021a5832a7e272a6"
SOURCE_SUMMARY_SHA = "50075f5b62e8a623476dd0556d405ac3958ed277c1d5ed75e8814cfad519d88a"
COHORT_SHA = "7ede9cd0a40a585d11fc94d08738d8a74abc2607b252c3ccc6e4b4614ef6eba4"
FACTORY = "mutual_lab.producer_context_study.RecordingFactory.H1"
REPAIR = {
    "aggregation_repair_run_id": 43657319705,
    "aggregation_repair_run_attempt": 1,
    "aggregation_repair_revision": "da1630dd3d8fc09aeba97cf427e7f466e2b9a1f9",
    "aggregation_repair_request_sha256": "0ab9dc3575d5c9c590dfaed361b4e46428a93fd681482b5f839118c2f2125d57",
    "aggregation_repair_script_sha256": "0e4ce27de62094c6cdb524ea45c344d87b532dd9a675fd5339abf664ddbf0259",
    "aggregation_repair_manifest_sha256": "6874c721243b9a0eaab740b426e7d527e92b8365d068a0077ed72e961c30d048",
    "aggregation_repair_provenance_sha256": "e39f6564a458f26eab7acf8ecfefaa6402f61701da6b755f2e5fc0837028d011",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def same(a, b, name):
    require(canon(a) == canon(b), name)


def local(root, relative):
    require(isinstance(relative, str) and "\\" not in relative, "Invalid relative path")
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and all(p not in (".", "..", "") for p in parts.parts), "Unsafe relative path")
    root, path = Path(root).resolve(), Path(root) / relative
    require(path.resolve().is_relative_to(root), "Artifact path escaped its root")
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent), "Symlinked artifact path")
    require(path.is_file(), "Missing artifact file: " + str(path))
    return path


def authenticate(artifact, name, run_id, revision):
    artifact = Path(artifact).resolve()
    require(artifact.name == name, "Unexpected artifact directory")
    meta_path, archive = artifact.parent / (name + ".artifact.json"), artifact.parent / (name + ".zip")
    meta = load(meta_path)
    require(meta.get("name") == name and meta.get("expired") is False, "Wrong or expired artifact metadata")
    require(meta.get("workflow_run", {}).get("id") == run_id and meta["workflow_run"].get("head_sha") == revision,
            "Artifact belongs to another source run/revision")
    require(meta.get("size_in_bytes") == archive.stat().st_size and meta.get("digest") == "sha256:" + sha(archive),
            "Authenticated original ZIP differs")
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())), "Corrupt/duplicate ZIP entries")
        wanted = set()
        for entry in z.infolist():
            require(not stat.S_ISLNK(entry.external_attr >> 16), "ZIP symlink")
            if entry.is_dir():
                continue
            target = local(artifact, entry.filename)
            with z.open(entry) as stream:
                h, count = hashlib.sha256(), 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk); count += len(chunk)
            require(count == target.stat().st_size and h.hexdigest() == sha(target), "Extracted original bytes changed")
            wanted.add(target.relative_to(artifact).as_posix())
        actual = {p.relative_to(artifact).as_posix() for p in artifact.rglob("*") if p.is_file()}
        require(wanted == actual, "Artifact extracted file set differs from original ZIP")
    return {"name": name, "id": meta["id"], "zip_sha256": sha(archive), "files": len(wanted),
            "api_metadata_sha256": sha(meta_path), "verified": True}


def request_checks(request, path):
    require(sha(path) == REQUEST_SHA, "The original preregistered request changed")
    require(request.get("source_condition") == "H1" and request.get("source_run_id") == 43652561578
            and request.get("source_run_attempt") == 1 and request.get("source_revision") == "736097e5298fb71037a420ec8bd635c73a0154dd",
            "Wrong original H source identity")
    require(request.get("source_summary_sha256") == SOURCE_SUMMARY_SHA and request.get("prior_checkpoint_sha256") == PRIOR_SHA
            and request.get("prior_manifest_sha256") == PRIOR_MANIFEST_SHA and request.get("confirmation_implementation_sha256") == COHORT_SHA,
            "Source/visual-prior/confirmation implementation pins differ")
    require(all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in REPAIR.items()), "Repair provenance pins differ")
    require(request.get("candidate_spec") == {"candidate_kind": "producer_context", "producer_memory": "objects",
            "understanding": "attentive", "producer_context": "object_moments"}, "Not the one fixed H1 factory")


def official_rows_index(official, source_record, plan_dict, spec):
    require(official.get("kind") == "cohort-context-validation-v2-comparison", "Wrong official aggregate kind")
    same(official.get("candidate_request"), source_record, "Official aggregate names another source request")
    same(official.get("plan"), plan_dict, "Official aggregate plan differs")
    same(official.get("candidate_spec"), spec, "Official aggregate model specification differs")
    require(official.get("source_condition") == "H1" and type(official.get("fixed_final_step")) is int and official["fixed_final_step"] == 9600
            and official.get("shared_visual_prior") is True and official.get("independent_visual_replications") is False
            and official.get("excluded_from_research") is True and official.get("eligible_for_fresh_experiment") is False,
            "Official aggregate scope/condition/budget differs")
    for row in official.get("rows", []):
        require(row.get("language") in LANGUAGES and type(row.get("replica")) is int and row["replica"] in range(3)
                and type(row.get("step")) is int and row["step"] in STEPS and row.get("status") == "completed",
                "Malformed official row identity/status")
        same(row.get("seed_pair"), list(PAIRS[row["replica"]]), "Official row seed labels differ")
    official_rows = {(r["language"], r["replica"], r["step"]): r for r in official.get("rows", [])}
    require(len(official_rows) == len(official.get("rows", [])) == 48
            and set(official_rows) == {(l, r, s) for l in LANGUAGES for r in range(3) for s in STEPS}, "Official summary row identities differ")
    return official_rows


def native_reference(package, prior, plan, config, splits):
    from mutual_lab.neural import configure_cpu, torch, digest
    from mutual_lab.spatial_prior import load_prior_bundle, fit_pool_hash
    from mutual_lab.producer_context_study import RecordingFactory
    from mutual_lab.development_world import render_scene
    configure_cpu()
    require(sha(prior) == PRIOR_SHA and sha(Path(prior).with_suffix(".json")) == PRIOR_MANIFEST_SHA, "Prior bytes changed")
    pngs = [render_scene(scene) for split in ("train", "dev") for scene in splits[split]]
    bundle = load_prior_bundle(prior, expected_fit_pool_sha256=fit_pool_hash(pngs[:220]))
    result = {}
    for replica, pair in enumerate(PAIRS):
        with torch.random.fork_rng():
            torch.manual_seed(pair[1])
            factory = RecordingFactory("H1", pngs)
            model = factory(plan.model_profile, pair[0], bundle, condition="B1")
            optimizer = torch.optim.Adam(model.parameters(), lr=config.native_training_profile.learning_rate)
            result[replica] = {"full_state_sha256": digest(model.state_dict()),
                "understanding_sha256": digest(model.understanding.state_dict()), "producer_sha256": digest(model.producer.state_dict()),
                "prior_sha256": digest(model.visual_encoder.state_dict()),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "flags": [(n, p.requires_grad) for n, p in model.named_parameters()],
                "shapes": {n: (list(v.shape), str(v.dtype)) for n, v in model.state_dict().items()},
                "optimizer_groups": optimizer.state_dict()["param_groups"],
                "torch_rng_sha256": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
                "torch_rng_shape": list(torch.random.get_rng_state().shape),
                "initialization_seed": pair[0], "training_seed": pair[1], "reference_uses_prior_only": True}
            require(not optimizer.state_dict()["state"], "Fresh reference optimizer is not empty")
        del model, optimizer, factory
        gc.collect()
    require(len({r["full_state_sha256"] for r in result.values()}) == 3, "Independent reference seed blocks are not distinct")
    return result


def replay_streams(plan, config, splits):
    """Replay meanings, captions and negative choices; never evaluate/train NN."""
    from mutual_lab.development_world import captions, render_scene
    from mutual_lab.world import variants
    from mutual_lab.fit_distractors import build_fit_catalog, make_training_examples
    catalog = build_fit_catalog(splits["train"], render_fn=render_scene)
    rendered = {s: render_scene(s) for s in splits["train"]}
    examples, all_captions = [], {language: [] for language in LANGUAGES}
    for scene in splits["train"]:
        choices = variants(scene.group)
        for index in range(4):
            examples.append({"image": rendered[scene], "choices": [rendered[s] for s in choices],
                "answer": choices.index(scene), "_scene_key": scene.key})
            for language in LANGUAGES:
                all_captions[language].append(captions(scene, language)[index])
    result = {}
    for replica, pair in enumerate(PAIRS):
        rng, negative = random.Random(pair[1]), random.Random(pair[1] + plan.negative_seed_offset)
        meanings, caption_hashes, counts = hashlib.sha256(), {l: hashlib.sha256() for l in LANGUAGES}, Counter()
        snapshots = {0: {"training_rng_sha256": hashlib.sha256(canon(rng.getstate()).encode()).hexdigest(),
            "negative_rng_sha256": hashlib.sha256(canon(negative.getstate()).encode()).hexdigest()}}
        for step in range(1, STEPS[-1] + 1):
            indices = rng.choices(range(len(examples)), k=config.native_training_profile.batch_size)
            batch = [examples[i] for i in indices]
            meanings.update(json.dumps([row["_scene_key"] for row in batch]).encode() + b"\n")
            for language in LANGUAGES:
                caption_hashes[language].update(json.dumps([all_captions[language][i] for i in indices], ensure_ascii=False).encode() + b"\n")
            _, audit = make_training_examples(batch, catalog, negative, exposure_offset=(step-1)*len(batch), policy="mixed_blocks")
            counts.update(audit)
            if step in STEPS:
                snapshots[step] = {"training_rng_sha256": hashlib.sha256(canon(rng.getstate()).encode()).hexdigest(),
                    "negative_rng_sha256": hashlib.sha256(canon(negative.getstate()).encode()).hexdigest()}
        result[replica] = {"snapshots": snapshots, "meaning_stream_sha256": meanings.hexdigest(),
            "caption_stream_sha256": {l: h.hexdigest() for l, h in caption_hashes.items()}, "negative_exposures": dict(counts),
            "exposures": STEPS[-1] * config.native_training_profile.batch_size}
    return result


def validate_cache(tables, initial, points, digest):
    require(tables.get("step_0") and tables.get("step_0") == tables.get("final"), "Own frozen features changed")
    table = tables["step_0"]
    require(table.get("policy") == "per_png_singleton_v1", "Not per-PNG singleton features")
    require(table.get("sha256") == digest({k: table[k] for k in ("policy", "caches")}), "Feature table composite digest differs")
    expected = {"object_memory_cache": 332, "producer_context_cache": 332, "spatial_grid_cache": 0,
                "understanding_object_cache": 332, "visual_cache": 332}
    require(set(table.get("caches", {})) == set(expected), "Frozen feature cache schema differs")
    for name, entries in expected.items():
        v = table["caches"][name]
        require(type(v.get("entries")) is int and v["entries"] == entries and re.fullmatch("[0-9a-f]{64}", v.get("sha256", "")), "Invalid frozen feature cache")
    pre = initial.get("memory_preflight", {})
    require(pre.get("status") == "completed" and all(pre.get(k) == 332 and type(pre.get(k)) is int for k in ("input_pngs", "unique_pngs", "successful_pngs")), "Preflight coverage differs")
    require(pre.get("fallback_count") == 0 and type(pre.get("fallback_count")) is int and pre.get("fallback_policy") == "raise; no fallback"
            and pre.get("model_state_unchanged") is True and pre.get("rng_unchanged") is True, "Preflight changed state/RNG or used fallback")
    require("invalid_pngs" not in pre or (type(pre["invalid_pngs"]) is int and pre["invalid_pngs"] == 0), "Invalid preflight PNGs")
    same(pre.get("feature_tables"), table, "Preflight and step0 features differ")
    require(all(p.get("frozen_feature_table_sha256") == table["sha256"] for p in points), "Evaluation feature table hash changed")
    return table["sha256"]


def validate_repair_context(ctx, input_directory):
    evidence = ctx.get("aggregation_repair_evidence", {})
    require(evidence.get("native_cells_rerun") is False and evidence.get("source_original_conclusion_preserved") == "failure",
            "Repair erased original failure or reran source native cells")
    source = evidence.get("source_original_run", {})
    require(source.get("id") == 43652561578 and source.get("run_attempt") == 1
            and source.get("head_sha") == "736097e5298fb71037a420ec8bd635c73a0154dd"
            and source.get("status") == "completed" and source.get("conclusion") == "failure", "Original failed source changed")
    jobs = evidence.get("source_first_attempt_jobs", [])
    names = {"verify", "aggregate_h"} | {f"phase_h ({l}, {r})" for l in LANGUAGES for r in range(3)}
    require(isinstance(jobs, list) and len(jobs) == 14 and {j.get("name") for j in jobs} == names
            and len({j.get("id") for j in jobs}) == 14, "Original source job grid changed")
    for job in jobs:
        require(job.get("status") == "completed" and job.get("conclusion") == ("failure" if job["name"] == "aggregate_h" else "success"),
                "Another original source job failed")
        if job["name"] == "aggregate_h":
            require(job.get("id") == 132631927039, "Another original failed aggregate")
    run = evidence.get("repair_run", {})
    require(run.get("id") == REPAIR["aggregation_repair_run_id"] and run.get("run_attempt") == 1
            and run.get("head_sha") == REPAIR["aggregation_repair_revision"] and run.get("status") == "completed"
            and run.get("conclusion") == "success" and run.get("path") == ".github/workflows/l1-H-aggregation-repair.yml", "Separate successful repair changed")
    definitions = (("repair-manifest.json", "aggregation_repair_manifest_sha256", "manifest"),
        ("aggregation-repair-provenance.json", "aggregation_repair_provenance_sha256", "repair_provenance"),
        ("repair-request.json", "aggregation_repair_request_sha256", "repair_request"))
    values = {}
    for name, pin, field in definitions:
        paths = list(Path(input_directory).rglob(name))
        require(len(paths) == 1 and sha(paths[0]) == REPAIR[pin], "Repair input bytes differ: " + name)
        values[field] = load(paths[0])
        same(evidence.get(field), values[field], "Embedded repair evidence differs from authenticated file")
    manifest, provenance = values["manifest"], values["repair_provenance"]
    require(manifest.get("nn_training_rerun") is False and manifest.get("posthoc_metadata_repair") is True
            and manifest.get("original_failed_run_preserved") is True and manifest.get("metric_rows_unchanged") is True
            and manifest.get("original_selection_rule_unchanged") is True and len(manifest.get("actual_checkpoint_audits", [])) == 96
            and len(manifest.get("raw_pair_audits", [])) == 12 and len(manifest.get("normalizations", [])) == 2,
            "Repair manifest changed scientific results or native states")
    same(manifest.get("original_files_sha256_before"), manifest.get("original_files_sha256_after"), "Repair changed original source bytes")
    require(provenance.get("run_id") == REPAIR["aggregation_repair_run_id"] and provenance.get("source_original_conclusion") == "failure"
            and provenance.get("original_H_workflow_overall_success") is False and provenance.get("no_native_training_or_checkpoint_replacement") is True,
            "Repair provenance mislabels original workflow")
    return {"source_original_conclusion": "failure", "repair_run_id": run["id"], "repair_revision": run["head_sha"],
            "verified_original_jobs": 14, "verified_original_checkpoint_records": 96, "repair_pins": REPAIR,
            "original_native_cells_rerun": False}


def check_payload(payload, manifest, raw, step, pair, reference, stream, source_map, revision, digest):
    require(payload.get("format") == "mutual-mastery-development-v1" and payload.get("kind") == "development_only"
            and payload.get("excluded_from_research") is True and payload.get("eligible_for_fresh_experiment") is False, "Wrong tensor artifact scope")
    require(type(payload.get("step")) is int and type(manifest.get("step")) is int
            and payload["step"] == step == manifest.get("step"), "Payload/manifest actual step mismatch")
    require(manifest.get("format") == "mutual-mastery-development-v1" and manifest.get("kind") == "development_only"
            and manifest.get("excluded_from_research") is True and manifest.get("eligible_for_fresh_experiment") is False,
            "Checkpoint manifest scope differs")
    for key in ("language", "phase", "condition", "replica", "seed_pair", "model_factory", "source_sha256", "data_sha256", "split", "revision", "data_version", "prior_metadata"):
        same(payload.get(key), raw.get(key), "Payload report mismatch: " + key)
    require(payload.get("phase") == "C" and payload.get("condition") == "C1" and payload.get("model_factory") == FACTORY, "Wrong actual model factory")
    same(payload.get("seed_pair"), list(pair), "Actual tensor uses another seed")
    same(payload.get("plan"), raw.get("plan"), "Actual tensor plan differs")
    same(payload.get("source_sha256"), source_map, "Actual tensor implementation differs")
    require(payload.get("revision", {}).get("commit") == revision and payload["revision"].get("working_tree_dirty") is False, "Actual tensor revision differs")
    require(payload.get("negative_exposure_offset") == step * raw["batch_size"], "Actual exposure offset differs")
    weights = payload.get("model", {})
    shapes = {n: (list(v.shape), str(v.dtype)) for n, v in weights.items()}
    same(shapes, reference["shapes"], "Actual NN tensor names/shapes/dtypes differ")
    import torch
    require(all(torch.isfinite(v).all().item() for v in weights.values() if v.is_floating_point()), "Nonfinite NN tensor")
    hashes = {b: digest({n[len(b)+1:]: v for n, v in weights.items() if n.startswith(b + ".")})
              for b in ("understanding", "producer", "visual_encoder")}
    require(hashes["visual_encoder"] == reference["prior_sha256"] == raw["initial_prior_sha256"] == raw["final_prior_sha256"], "Frozen prior actual tensors changed")
    if step == 0:
        require(digest(weights) == reference["full_state_sha256"], "Actual initial NN is not independently reconstructed fresh NN")
        require(not payload["optimizer"]["state"], "Actual initial Adam contains state")
    else:
        require(payload["optimizer"]["state"], "Actual trained Adam is empty")
        require(all(float(state["step"]) == step for state in payload["optimizer"]["state"].values()), "Adam step differs from fixed checkpoint")
    same(payload["optimizer"]["param_groups"], reference["optimizer_groups"], "Actual Adam hyperparameters differ")
    state = stream["snapshots"][step]
    require(hashlib.sha256(canon(payload["training_rng_state"]).encode()).hexdigest() == state["training_rng_sha256"], "Actual meaning/caption RNG differs")
    require(hashlib.sha256(canon(payload["negative_rng_state"]).encode()).hexdigest() == state["negative_rng_sha256"], "Actual negative RNG differs")
    require(payload["torch_rng_state"].dtype == torch.uint8 and list(payload["torch_rng_state"].shape) == reference["torch_rng_shape"]
            and hashlib.sha256(payload["torch_rng_state"].numpy().tobytes()).hexdigest() == reference["torch_rng_sha256"], "Private Torch RNG differs from fresh controlled stream")
    require(payload.get("frozen_feature_policy") == "per_png_singleton_v1", "Actual tensor missing singleton policy")
    same(payload.get("frozen_feature_tables"), raw.get("frozen_feature_tables"), "Actual tensor feature tables differ")
    return {"step": step, "sha256": manifest["sha256"], "full_state_sha256": digest(weights), "branches": hashes,
            "actual_rng_and_optimizer_verified": True}


def fixed_inference(checkpoint, prior, directory, raw, splits):
    from mutual_lab.mastery_infer import load_development_model, infer_development
    from mutual_lab.development_world import captions, render_scene, grade_generation
    from mutual_lab.world import variants
    model = load_development_model(checkpoint, prior)
    require(model.mastery_inference_metadata.get("model_factory") == FACTORY, "Read-only loader selected wrong factory")
    generations = load(Path(directory) / "generations.json")
    predictions = load(Path(directory) / "comprehension_predictions.json")
    rows = []
    for scene in splits["dev"][:3]:
        key = canon(scene.to_dict())
        saved = [r for r in generations if r["step"] == 9600 and r["split"] == "dev" and canon(r["scene"]) == key]
        require(len(saved) == 1, "Bounded restored generation identity differs")
        answer = infer_development(model, render_scene(scene))
        grade = grade_generation(answer["generation"], scene, raw["language"])
        same(grade, {k: v for k, v in saved[0].items() if k not in ("step", "split")}, "Actual restored generation differs")
        checks = []
        choices = [render_scene(s) for s in variants(scene.group)]
        for index, text in enumerate(captions(scene, raw["language"])):
            expected = [r for r in predictions if r["step"] == 9600 and r["split"] == "dev" and r["template_index"] == index and canon(r["scene"]) == key]
            require(len(expected) == 1, "Bounded restored comprehension identity differs")
            got = infer_development(model, text=text, choices_png=choices)
            require(got["comprehension"]["prediction"] == expected[0]["prediction"] and got["read_only_state_verified"] is True, "Actual restored understanding differs")
            checks.append(True)
        rows.append({"scene": scene.to_dict(), "generation_matches": True, "four_choice_matches": checks,
                     "read_only_state_verified": answer["read_only_state_verified"], "text": grade["text"],
                     "terminated": grade["terminated"], "invalid_encoding": grade["invalid_encoding"]})
    return {"status": "pass", "sampling_rule": "First three original dev rows; all four caption forms", "examples": rows,
            "generation_queries": 3, "four_choice_queries": 12, "restored_model_sha256": model.mastery_inference_metadata["restored_model_sha256"]}


def run(args):
    began = time.monotonic()
    package = Path(args.package).resolve()
    sys.path.insert(0, str(package))
    from mutual_lab.config import ExperimentConfig
    from mutual_lab.mastery_lab import MasteryPlan
    from mutual_lab import development_world as world
    from mutual_lab.neural import digest, torch
    from mutual_lab.cohort_context_validation import load_candidate_request
    from audit_rows import audit_rows
    request = load(args.request); request_checks(request, args.request)
    source_map = {p.name: sha(p) for p in (package / "mutual_lab").glob("*.py")}
    require(len(source_map) == 42 and source_map["cohort_context_validation.py"] == COHORT_SHA, "Unexpected trained package source set")
    require(all(source_map.get(k) == v for k, v in request["candidate_source_sha256"].items()), "NN15 source implementation changed")
    config = ExperimentConfig.load(package / "configs/neural.example.json")
    plan = MasteryPlan.load(package / "configs/mastery-H-cohort-plan.json").validate(config)
    require(tuple(plan.seed_pairs) == PAIRS and tuple(plan.budgets) == STEPS[1:], "Predeclared confirmation grid/budgets differ")
    splits = world.development_splits(config, plan.data_seed)
    require(len(splits["train"]) == 220 and len(splits["dev"]) == 112, "Native scene split changed")
    prior = Path(args.prior) if args.prior else Path(args.input) / "prior/prior.pt"
    refs = native_reference(package, prior, plan, config, splits)
    streams = replay_streams(plan, config, splits)
    prepared = {"kind": "independent-H-confirmation-reference-v1", "request_sha256": REQUEST_SHA, "source_sha256": source_map,
        "seed_pairs": PAIRS, "steps": STEPS, "fresh_reference_initializations": refs, "reference_streams": streams,
        "no_native_training_performed": True, "only_image_prior_loaded": True, "audit_script_sha256": sha(__file__)}
    if args.prepare_only:
        return {"status": "prepared_before_outcomes", **prepared, "elapsed_seconds": time.monotonic()-began}
    require(type(args.run_id) is int and args.run_id > 0 and re.fullmatch("[0-9a-f]{40}", args.revision or ""), "Pin actual confirmation run/revision")
    input_proof = authenticate(args.input, f"l1-H-cohort-input-{args.run_id}", args.run_id, args.revision)
    ctx = load(Path(args.input) / "provenance/run-context.json")
    require(str(ctx.get("run_id")) == str(args.run_id) and str(ctx.get("run_attempt")) == "1" and ctx.get("code_revision") == args.revision, "New confirmation context identity differs")
    same(ctx.get("source_sha256"), source_map, "Confirmation context NN package changed")
    require(ctx.get("request_sha256") == REQUEST_SHA, "Context admitted another request")
    same(ctx.get("request"), request, "Context request content changed")
    require(ctx.get("source_language_checkpoints_downloaded_or_resumed") is False, "Source language states were reused")
    repair_binding = validate_repair_context(ctx, args.input)
    require(sha(Path(args.input) / "source-summary.json") == SOURCE_SUMMARY_SHA, "Source repaired summary differs")
    source_record = load_candidate_request(args.request, plan, plan_path=package / "configs/mastery-H-cohort-plan.json",
        prior_path=prior, source_summary_path=Path(args.input) / "source-summary.json", source_plan_path=package / "configs/mastery-H-plan.json")
    expected = {(l, r) for l in LANGUAGES for r in range(3)}
    reports, errors, auth, summary_rows = [], [], [input_proof], []
    for language, replica in sorted(expected):
        try:
            name = f"l1-H-cohort-trial-{language}-r{replica}-{args.run_id}"
            artifact = Path(args.artifacts_root) / name
            auth.append(authenticate(artifact, name, args.run_id, args.revision))
            paths = list(artifact.rglob("cohort-context-validation.json")); require(len(paths) == 1, "Missing/duplicate original model attempt")
            suite = load(paths[0]); directory = paths[0].parent
            require(suite.get("kind") == "cohort-context-validation-v2" and type(suite.get("schema_version")) is int
                    and suite["schema_version"] == 1, "Wrong wrapper schema")
            require(suite.get("language") == language and type(suite.get("replica")) is int and suite["replica"] == replica, "Wrong/duplicate model identity")
            same(suite.get("candidate_request"), source_record, "Model admitted another candidate request")
            require(suite.get("status") == "completed" and suite.get("source_condition") == "H1" and suite.get("cohort_source_sha256") == COHORT_SHA, "Incomplete or changed model wrapper")
            execution = suite.get("execution", {})
            require(str(execution.get("workflow_run_id")) == str(args.run_id) and str(execution.get("workflow_run_attempt")) == "1" and execution.get("code_revision") == args.revision, "Model execution was replaced")
            raw_path = local(directory, suite["raw_report_path"]); raw = load(raw_path)
            same(raw, suite.get("engine_report"), "Wrapper embedded a different raw report")
            require(raw.get("kind") == "mastery-lab-v1-trial" and type(raw.get("schema_version")) is int
                    and raw["schema_version"] == 1 and raw.get("data_version") == world.VERSION, "Wrong raw trial schema/data version")
            require(all(suite.get(k) is value and raw.get(k) is value for k, value in (
                    ("excluded_from_research", True), ("eligible_for_fresh_experiment", False))), "Raw model/suite development scope differs")
            require(raw.get("status") == "completed" and type(raw.get("training_steps_completed")) is int and raw["training_steps_completed"] == 9600
                    and raw.get("language") == language and type(raw.get("replica")) is int and raw["replica"] == replica, "Actual raw model did not complete")
            same(raw.get("seed_pair"), list(PAIRS[replica]), "Raw model seed differs")
            same(raw.get("source_sha256"), source_map, "Raw model source differs")
            same(raw.get("plan"), asdict(replace(plan, b_seed_pair=PAIRS[replica])), "Raw plan changed")
            same(raw.get("split"), world.development_manifest(config, plan.data_seed), "Data split/protected exclusion differs")
            expected_data = {split: [{"scene": scene.to_dict(), "captions": world.captions(scene, language)} for scene in scenes] for split, scenes in splits.items()}
            same(load(raw_path.parent / "dataset.json"), expected_data, "Actual native dataset differs")
            require(raw.get("data_sha256") == digest(expected_data) and raw.get("config_sha256") == digest(config.to_dict()), "Native data/numerical hashes differ")
            require(raw.get("training_languages") == [language] and raw.get("foreign_training_languages") == [] and raw.get("protected_splits_used") == [], "Foreign/protected training input")
            require(raw.get("optimization") == "independent-branch-clipping-adam-v1" and raw.get("clip_norm_per_branch") == 1 and raw.get("policy") == "mixed_blocks", "Training control changed")
            require(raw.get("batch_size") == config.native_training_profile.batch_size and raw.get("learning_rate") == config.native_training_profile.learning_rate
                    and raw.get("choice_count") == 4 and raw.get("training_chance_accuracy") == .25 and raw.get("comprehension_forms") == 4
                    and raw.get("production_forms") == 1 and raw.get("teacher_forced_evaluation_forms") == 4, "Numerical/exposure controls changed")
            same(suite.get("plan"), asdict(plan), "Wrapper plan differs")
            same(suite.get("candidate_spec"), request["candidate_spec"], "Wrapper names another architecture")
            same(suite.get("implementation_source_sha256"), request["candidate_source_sha256"], "Wrapper architecture source differs")
            require(suite.get("fixed_prior_condition") == "B1" and suite.get("policy") == "mixed_blocks", "Wrapper prior/policy differs")
            require(all(suite.get(k) is True for k in ("fresh_initialization", "scratch_language_training", "shared_visual_prior"))
                    and suite.get("resumed") is False and suite.get("independent_visual_replications") is False, "Wrapper did not declare controlled fresh models")
            initial, ref = suite.get("initial_state", {}), refs[replica]
            for k in ("full_state_sha256", "understanding_sha256", "producer_sha256", "prior_sha256", "parameter_count", "trainable_parameter_count", "initialization_seed"):
                require(initial.get(k) == ref[k], "Independent fresh constructor mismatch: " + k)
            same(initial.get("parameter_flags"), ref["flags"], "Frozen/trainable parameter flags differ")
            require(raw.get("initial_understanding_sha256") == ref["understanding_sha256"] and raw.get("initial_producer_sha256") == ref["producer_sha256"],
                    "Raw initial branch hashes differ from independently reconstructed NN")
            require(raw.get("parameter_count") == 477443 == ref["parameter_count"] and raw.get("trainable_parameter_count") == 364675 == ref["trainable_parameter_count"], "NN capacity differs")
            same(raw.get("negative_exposures"), streams[replica]["negative_exposures"], "Negative sampling stream/counts differ")
            require(raw.get("meaning_stream_sha256") == streams[replica]["meaning_stream_sha256"] and raw.get("comprehension_caption_stream_sha256") == streams[replica]["caption_stream_sha256"][language], "Meaning/caption stream changed")
            points = raw.get("curves", []); require([p.get("step") for p in points] == list(STEPS), "Missing/duplicate budget evaluation")
            features = validate_cache(raw.get("frozen_feature_tables", {}), initial, points, digest)
            checks = raw.get("checkpoints", []); require([v.get("step") for v in checks] == list(STEPS), "Missing/duplicate checkpoint budget")
            checkpoints = []
            for step, manifest in zip(STEPS, checks):
                require(manifest.get("path") == f"step-{step:06d}.pt", "Unexpected checkpoint basename")
                file = local(raw_path.parent, manifest["path"])
                require(sha(file) == manifest.get("sha256"), "Actual checkpoint SHA differs")
                same(load(file.with_suffix(".json")), manifest, "Checkpoint sidecar differs")
                payload = torch.load(file, map_location="cpu", weights_only=True)
                checkpoints.append(check_payload(payload, manifest, raw, step, PAIRS[replica], ref, streams[replica], source_map, args.revision, digest))
            evidence = suite.get("initial_checkpoint_evidence", {})
            require(evidence.get("sha256") == checkpoints[0]["sha256"] and all(evidence.get(k) is True for k in (
                    "step_zero", "optimizer_empty", "main_rng_fresh", "negative_rng_fresh", "initial_weights_match", "feature_policy_matches")),
                    "Declared initial checkpoint evidence differs")
            require(checkpoints[-1]["branches"]["understanding"] == raw.get("final_understanding_sha256") and checkpoints[-1]["branches"]["producer"] == raw.get("final_producer_sha256"), "Actual final branch hashes differ")
            regraded = audit_rows(raw_path.parent, raw, splits, language, config=config)
            summary_rows.extend({"language": language, "replica": replica, "step": p["step"], "dev": p["dev"], "contrasts": p["contrasts"]["categories"], "candidate": p["candidate"]} for p in points)
            inferred = fixed_inference(raw_path.parent / "step-009600.pt", prior, raw_path.parent, raw, splits)
            reports.append({"language": language, "replica": replica, "status": "pass", "checkpoints": checkpoints,
                "rows": regraded, "final_qualified": regraded["fixed_final_qualified"], "inference": inferred,
                "feature_table_sha256": features, "environment": raw.get("environment"), "report_sha256": sha(raw_path)})
        except Exception as error:
            errors.append({"language": language, "replica": replica, "error": f"{type(error).__name__}: {error}"})
    official = load(args.summary)
    summary_proof = authenticate(Path(args.summary).parent, f"l1-H-cohort-summary-{args.run_id}", args.run_id, args.revision)
    official_rows = official_rows_index(official, source_record, asdict(plan), request["candidate_spec"])
    for row in summary_rows:
        other = official_rows.get((row["language"], row["replica"], row["step"]), {})
        for key in ("dev", "contrasts", "candidate"):
            same(row[key], other.get(key), "Official summary does not match authenticated raw rows")
    count = sum(r["final_qualified"] for r in reports)
    complete = len(reports) == 12 and not errors
    robust = complete and count == 12
    if not (type(official.get("final_candidate_count")) is int and official["final_candidate_count"] == count
            and official.get("robust_development_candidate") is robust):
        errors.append({"scope": "official_summary", "error": "Official final qualification disagrees with independently valid/regraded models"})
    if complete:
        require(official.get("status") == "completed" and official.get("provenance_valid") is True and official.get("execution_completed") is True, "Official completion/provenance differs")
    complete = len(reports) == 12 and not errors
    robust = complete and count == 12
    return {"kind": "independent-H-confirmation-48-state-audit-v1", "status": "pass" if complete else "failed",
        "audit_complete": complete, "run_id": args.run_id, "revision": args.revision, "request_sha256": REQUEST_SHA,
        "actual_checkpoint_tensors_verified": sum(len(r["checkpoints"]) for r in reports), "cells": reports, "errors": errors,
        "all_twelve_final_candidates": robust, "final_candidate_count": count, "robust_development_candidate": robust,
        "assessment": "robust_development_candidate" if robust else "targets_not_met" if complete else "indeterminate_incomplete",
        "original_source_aggregation_failure_retained": True, "aggregation_repair": repair_binding,
        "authenticated_artifacts": [*auth, summary_proof], "official_summary_sha256": sha(args.summary),
        "official48rows_match_authenticated_raws": complete, "fresh_seed_references": refs, "cross_host_feature_table_bit_equality_required": False,
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "source_sha256": source_map, "audit_script_sha256": sha(__file__), "rows_audit_script_sha256": sha(Path(__file__).with_name("audit_rows.py")),
        "elapsed_seconds": time.monotonic()-began, "reference_criteria": prepared}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    require(args.prepare_only or all((args.artifacts_root, args.input, args.summary, args.run_id, args.revision)), "Supply actual complete cohort inputs")
    require(args.prior or args.input, "Supply image-only prior")
    try:
        report = run(args)
    except Exception as error:
        report = {"status": "audit_error", "audit_complete": False, "error": f"{type(error).__name__}: {error}",
            "run_id": args.run_id, "revision": args.revision, "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "audit_script_sha256": sha(__file__)}
    require(not args.output.exists(), "Do not overwrite a previous audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "audit_complete", "actual_checkpoint_tensors_verified", "final_candidate_count", "robust_development_candidate", "error")}))
    raise SystemExit(0 if report["status"] in ("pass", "prepared_before_outcomes") else 1)


if __name__ == "__main__":
    main()
