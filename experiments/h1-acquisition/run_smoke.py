"""Development-only H1 cross-situational acquisition and state-integrity pilot.

Actual frozen-producer utterances and shared PNGs are the only L2 inputs.
Authored captions and Scene objects stay on the researcher's evaluation side.
This runner never opens the protected research evaluation or promotes a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import sys
import time
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "mutual-understanding"
sys.path.insert(0, str(LAB))

ARCHIVE_SHA256 = "78bee0bc13b8f91b414c42125020bf88f0d6a52362825ef0463a93afdb36fd16"
CONDITIONS = ("aligned", "no_update", "shuffled", "reset_understanding")
FORMAT = "h1-development-pair-acquisition-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_core_sources(manifest):
    expected = json.loads(Path(manifest).read_text())
    files = expected["files"]
    root = LAB / "mutual_lab"
    require(set(files) == {p.name for p in root.glob("*.py")}, "Frozen core source file set changed")
    for name, sha in files.items():
        require(file_sha(root / name) == sha, "Frozen core source changed: " + name)
    return {"manifest_sha256": file_sha(manifest), "files": len(files), "all_unchanged": True}


def derangement(size, seed):
    require(type(size) is int and size >= 2, "Need at least two distinct exposures")
    rng = random.Random(seed)
    for _ in range(10000):
        order = list(range(size))
        rng.shuffle(order)
        if all(i != j for i, j in enumerate(order)):
            return order
    raise ValueError("No derangement within the fixed search bound")


def prepare_artifacts(archive, directory, expected_learners):
    """Extract only the two predeclared originals, preserving their exact bytes."""
    require(file_sha(archive) == ARCHIVE_SHA256, "Wrong original model archive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP member")
        for n in names:
            p = PurePosixPath(n)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in n, "Unsafe archive member")
            require((z.getinfo(n).external_attr >> 16) & 0o170000 != 0o120000, "Archive symlink")
        index = json.loads(z.read("index.json"))
        require(index.get("development_only") is True and index.get("excluded_from_research") is True
                and index.get("eligible_for_fresh_experiment") is False, "Archive scope changed")
        selected = {}
        for language, expected in expected_learners.items():
            trials = [t for t in index["trials"] if t["language"] == language and t["replica"] == expected["replica"]]
            require(len(trials) == 1, "Missing or duplicated fixed learner")
            t = trials[0]
            checkpoint = t["checkpoint"]
            require(checkpoint["step"] == 9600 and checkpoint["sha256"] == expected["checkpoint_sha256"], "Wrong fixed final learner")
            prior_sha = t["prior_sha256"]
            prior = index["priors"][prior_sha]
            paths = {}
            for key, member in (("checkpoint", checkpoint["member"]), ("sidecar", checkpoint["sidecar_member"]),
                                ("prior", prior["member"]), ("prior_sidecar", prior["sidecar_member"]),
                                ("config", t["context"]["config.json"])):
                # The paths have already been checked above; retain same basename/sidecar relationship.
                out = directory / language / PurePosixPath(member).name
                if key.startswith("prior"):
                    out = directory / "prior" / PurePosixPath(member).name
                out.parent.mkdir(parents=True, exist_ok=True)
                raw = z.read(member)
                if out.exists():
                    require(out.read_bytes() == raw, "Conflicting duplicate extraction")
                else:
                    out.write_bytes(raw)
                paths[key] = out
            require(file_sha(paths["checkpoint"]) == checkpoint["sha256"] and file_sha(paths["prior"]) == prior_sha,
                    "Extracted tensor hash differs")
            selected[language] = paths
    return selected


def make_semantic_plan(config, exposures, stream_seed, shuffle_seed, evaluation_seed):
    from mutual_lab.development_world import development_splits, development_manifest, render_scene
    from mutual_lab.world import variants
    splits = development_splits(config, 59003)
    require(len(splits["train"]) == 220 and len(splits["dev"]) == 112, "Fixed development population changed")
    require(not ({s.group for s in splits["train"]} & {s.group for s in splits["dev"]}), "L2 exposure/evaluation group overlap")
    require(type(exposures) is int and 2 <= exposures <= len(splits["train"]), "Invalid exposure budget")
    scenes = random.Random(stream_seed).sample(list(splits["train"]), exposures)
    permutation = derangement(exposures, shuffle_seed)
    rng = random.Random(evaluation_seed)
    evaluation = []
    for scene in splits["dev"]:
        choices = list(variants(scene.group))
        rng.shuffle(choices)
        evaluation.append((scene, choices, choices.index(scene)))
    pngs = {s.key: render_scene(s) for split in splits.values() for s in split}
    public = {"scope": "development_only", "data_seed": 59003,
              "development_manifest": development_manifest(config, 59003),
              "shared_scene_keys": [s.key for s in scenes], "shuffled_text_indices": permutation,
              "evaluation": [{"scene_key": s.key, "choices": [c.key for c in choices], "answer": answer}
                             for s, choices, answer in evaluation],
              "png_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in pngs.items()},
              "protected_images_rendered_or_scored": False,
              "evaluation_scope": "L2-unexposed development pairs, already used in L1 development; not a new research holdout"}
    return splits, scenes, permutation, evaluation, pngs, public


def evaluate_pair(learners, evaluation, pngs, frozen_dev, exchange, batch_size=32):
    from mutual_lab.development_world import captions
    points, raw_rows = [], []
    languages = tuple(learners)
    for language, learner in learners.items():
        peer = next(l for l in languages if l != language)
        before = learner.state_digest()
        caches_before = learner.cache_digest()
        queries = []
        for scene, choices, answer in evaluation:
            for kind, texts in (("native_authored", captions(scene, language)),
                                ("foreign_authored", captions(scene, peer)),
                                ("foreign_peer_generated", [frozen_dev[peer][scene.key]["text"]])):
                for template, text in enumerate(texts):
                    queries.append({"kind": kind, "scene_key": scene.key, "text": text, "template": template,
                                    "answer": answer, "choices": [s.key for s in choices]})
        for offset in range(0, len(queries), batch_size):
            rows = queries[offset:offset + batch_size]
            guesses = learner.predict_many([r["text"] for r in rows], [[pngs[k] for k in r["choices"]] for r in rows])
            require(len(guesses) == len(rows), "Evaluation query count mismatch")
            for row, guess in zip(rows, guesses):
                require(type(guess) is int and 0 <= guess < 4, "Malformed prediction")
                raw_rows.append({**row, "learner": language, "peer": peer, "exchange": exchange,
                                 "prediction": guess, "correct": guess == row["answer"]})
        for kind in ("native_authored", "foreign_authored", "foreign_peer_generated"):
            rows = [r for r in raw_rows if r["learner"] == language and r["kind"] == kind]
            hits = sum(r["correct"] for r in rows)
            accuracy = hits / len(rows)
            points.append({"learner": language, "target_language": language if kind == "native_authored" else peer,
                           "exchange": exchange, "metric": kind, "hits": hits, "items": len(rows),
                           "raw_accuracy": accuracy, "chance_corrected_accuracy": (accuracy - .25) / .75,
                           "choice_count": 4, "baseline_forced_to_zero": False})
        require(learner.state_digest() == before and learner.cache_digest() == caches_before, "Evaluation updated learning state/cache")
    return points, raw_rows


def save_pair(path, learners, metadata):
    from mutual_lab.neural import torch, digest
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.with_suffix(".json").exists(), "Refusing checkpoint overwrite")
    payload = {"format": FORMAT, "kind": "development_acquisition_pair", "excluded_from_research": True,
               "eligible_for_fresh_experiment": False, "eligible_for_l2_research": False, "eligible_for_l2": False,
               "metadata": metadata, "learners": {k: a.state_dict() for k, a in learners.items()}}
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
        write_json(path.with_suffix(".json"), {"format": FORMAT, "kind": payload["kind"],
                   "excluded_from_research": True, "eligible_for_fresh_experiment": False,
                   "eligible_for_l2_research": False, "eligible_for_l2": False,
                   "sha256": file_sha(path), "payload_sha256": digest(payload),
                   "metadata": metadata, "learner_state_sha256": {k: a.state_digest() for k, a in learners.items()},
                   "learner_snapshot_sha256": {k: digest(v) for k, v in payload["learners"].items()}})
    finally:
        temporary.unlink(missing_ok=True)
    return path


def restore_pair(path, learners, expected):
    from mutual_lab.neural import torch, digest
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    require(manifest.get("format") == FORMAT and manifest.get("kind") == "development_acquisition_pair"
            and manifest.get("excluded_from_research") is True and manifest.get("eligible_for_fresh_experiment") is False
            and manifest.get("eligible_for_l2_research") is False and manifest.get("eligible_for_l2") is False
            and manifest.get("sha256") == file_sha(path), "Invalid pair checkpoint scope/hash")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("format") == FORMAT and payload.get("kind") == manifest["kind"]
            and payload.get("excluded_from_research") is True and payload.get("eligible_for_fresh_experiment") is False
            and payload.get("eligible_for_l2_research") is False and payload.get("eligible_for_l2") is False
            and digest(payload) == manifest["payload_sha256"], "Invalid pair payload")
    require(payload["metadata"] == expected == manifest["metadata"] and set(payload["learners"]) == set(learners), "Pair plan/stream/condition differs")
    exchange, condition = expected.get("exchange"), expected.get("condition")
    require(type(exchange) is int and exchange >= 0 and condition in CONDITIONS, "Invalid outer acquisition cursor")
    for language in learners:
        state = payload["learners"][language]
        counts = state.get("counters", {})
        require(counts.get("exposures") == exchange
                and counts.get("updates") == (0 if condition == "no_update" else exchange)
                and counts.get("native_replay_exposures") == 0
                and digest(state) == manifest["learner_snapshot_sha256"][language],
                "Pair checkpoint cursor/learner digest differs")
    for language, learner in learners.items():
        learner.load_state_dict(payload["learners"][language])
        require(learner.state_digest() == manifest["learner_state_sha256"][language], "Restored learner digest differs")
    return payload


def verify_exchange_counters(learners, stream, permutation, condition, exchange):
    """Bind each private learner to the outer matched exposure prefix."""
    for language, learner in learners.items():
        peer = next(l for l in learners if l != language)
        texts = [stream[permutation[i] if condition == "shuffled" else i]["utterances"][peer]["text"]
                 for i in range(exchange)]
        updates = 0 if condition == "no_update" else exchange
        expected = {"exposures": exchange, "updates": updates, "characters": sum(map(len, texts)),
                    "bytes_received": sum(len(t.encode("utf-8")) for t in texts),
                    "input_units": sum(len(learner.model.tokenizer.encode(t)) - 2 for t in texts),
                    "training_sequence_units": sum(len(learner.model.tokenizer.encode(t)) for t in texts) if updates else 0,
                    "native_replay_exposures": 0}
        require(learner.counters() == expected, "Learner counters differ from the fixed exchange prefix")


def _learning_inputs(scene, text, pngs):
    from mutual_lab.world import variants
    return text, pngs[scene.key], [pngs[s.key] for s in variants(scene.group) if s != scene]


def advance(learners, scenes, permutation, pngs, stream, condition, start, end):
    languages = tuple(learners)
    logs = []
    for i in range(start, end):
        # Both utterances were emitted from frozen L1 producers before any condition started.
        source_index = permutation[i] if condition == "shuffled" else i
        actual = stream[source_index]
        event = {"exchange": i + 1, "shared_scene_key": scenes[i].key,
                 "utterance_source_index": source_index, "updates": {}}
        for language, learner in learners.items():
            peer = next(l for l in languages if l != language)
            inputs = _learning_inputs(scenes[i], actual["utterances"][peer]["text"], pngs)
            event["updates"][language] = learner.update(*inputs, updates=0 if condition == "no_update" else 1)
        logs.append(event)
    return logs


def run(archive, plan_path, output, revision):
    from h1_adapter import H1DevelopmentLearner
    from mutual_lab.config import ExperimentConfig
    from mutual_lab.development_world import grade_generation
    from mutual_lab.neural import configure_cpu
    configure_cpu()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    summary = {"kind": "h1-development-acquisition-procedural-smoke-v1", "status": "running",
               "excluded_from_research": True, "eligible_for_fresh_experiment": False,
               "eligible_for_l2_research": False, "eligible_for_l2": False,
               "protected_images_rendered_or_scored": False, "revision": revision, "conditions": {}, "error": None}
    write_json(output / "summary.json", summary)
    try:
        plan = json.loads(Path(plan_path).read_text())
        # The frozen plan is deliberately bounded. A larger experiment needs a different declared plan.
        require(plan["exposures"] == 32 and plan["evaluate_at"] == [0, 8, 16, 32], "Unexpected fixed development budget")
        require(tuple(plan["conditions"]) == CONDITIONS and plan["languages"] == ["ko", "ja"], "Unexpected learner/condition grid")
        require(plan["optimizer"]["learning_rate"] == .001 and plan["optimizer"]["updates_per_exposure"] == 1
                and plan["optimizer"]["native_replay_per_update"] == plan["optimizer"]["foreign_replay_per_update"] == 0,
                "Unexpected learning rule")
        require(plan["excluded_from_research"] is True and plan["eligible_for_fresh_experiment"] is False, "Plan must remain development-only")
        summary["plan_sha256"] = file_sha(plan_path)
        summary["source_verification"] = verify_core_sources(HERE / "configs/frozen-core-source-sha256.json")
        summary["stage_source_sha256"] = {n: file_sha(HERE / n) for n in ("run_smoke.py", "h1_adapter.py")}
        expected_learners = {l: {"replica": plan["replica"], "checkpoint_sha256": p["sha256"]}
                             for l, p in plan["checkpoints"].items()}
        paths = prepare_artifacts(archive, output / "inputs", expected_learners)
        original_input_hashes = {str(p.relative_to(output)): file_sha(p) for p in set(
            p for values in paths.values() for p in values.values())}
        for language, p in paths.items():
            require(file_sha(p["sidecar"]) == plan["checkpoints"][language]["sidecar_sha256"]
                    and file_sha(p["prior"]) == plan["prior"]["sha256"]
                    and file_sha(p["prior_sidecar"]) == plan["prior"]["sidecar_sha256"]
                    and file_sha(p["config"]) == plan["source"]["blueprint_sha256"], "Fixed artifact sidecar/prior/blueprint differs")
        require(paths["ko"]["config"].read_bytes() == paths["ja"]["config"].read_bytes(), "Learners have different data blueprints")
        config = ExperimentConfig.load(paths["ko"]["config"])
        splits, scenes, permutation, evaluation, pngs, semantic = make_semantic_plan(
            config, plan["exposures"], plan["seeds"]["stream"], plan["seeds"]["shuffle"], plan["seeds"]["evaluation"])
        require(semantic["development_manifest"]["pair_sha256"] == plan["data_pair_sha256"], "Development split differs from frozen plan")
        write_json(output / "semantic-plan.json", semantic)
        def build(condition):
            learners = {l: H1DevelopmentLearner.from_checkpoint(paths[l]["checkpoint"], paths[l]["prior"],
                         allow_development_training=True, language=l, learning_rate=plan["optimizer"]["learning_rate"],
                         seed=plan["seeds"]["learner"], reset_understanding=condition == "reset_understanding") for l in plan["languages"]}
            for learner in learners.values():
                learner.warm(list(pngs.values()))
            return learners
        originals = build("aligned")
        frozen_dev = {l: {} for l in plan["languages"]}
        stream = []
        # This fixed stream is reused by every condition; no output is filtered/repaired by its score.
        for scene in scenes:
            utterances = {l: a.emit(pngs[scene.key]) for l, a in originals.items()}
            stream.append({"scene_key": scene.key, "utterances": utterances,
                           "researcher_only_semantic_grades": {l: grade_generation(t, scene, l) for l, t in utterances.items()}})
        for scene, _, _ in evaluation:
            for language, learner in originals.items():
                frozen_dev[language][scene.key] = learner.emit(pngs[scene.key])
        write_json(output / "frozen-peer-stream.json", stream)
        write_json(output / "frozen-peer-evaluation.json", frozen_dev)
        collisions = {}
        for language in plan["languages"]:
            collisions[language] = {"items": len(scenes), "same_surface_as_aligned": sum(
                stream[permutation[i]]["utterances"][language]["text"] == stream[i]["utterances"][language]["text"]
                for i in range(len(scenes))), "recognized_meaning_matches_shared_scene": sum(
                stream[permutation[i]]["researcher_only_semantic_grades"][language]["recognized_meaning"] == scenes[i].to_dict()
                for i in range(len(scenes))), "rows_removed_or_repaired": 0}
        write_json(output / "shuffled-association-collisions.json", collisions)
        baseline_generation = {}
        for language in plan["languages"]:
            rows = [grade_generation(frozen_dev[language][scene.key], scene, language) for scene, _, _ in evaluation]
            hits = sum(r["exact_match"] and r["terminated"] and not r["invalid_encoding"]
                       and not r["special_tokens_in_body"] for r in rows)
            baseline_generation[language] = {"hits": hits, "items": len(rows), "accuracy": hits / len(rows), "rows": rows}
        write_json(output / "native-generation-baseline.json", baseline_generation)
        summary["stream_sha256"] = value_sha(stream)
        summary["semantic_plan_sha256"] = value_sha(semantic)
        summary["frozen_peer_eval_sha256"] = value_sha(frozen_dev)
        del originals
        all_points = []
        for condition in CONDITIONS:
            learners = build(condition)
            frozen = {l: a.branch_digests() for l, a in learners.items()}
            from mutual_lab.neural import digest
            initial_optimizer = {l: digest(a.optimizer.state_dict()) for l, a in learners.items()}
            initial_caches = {l: a.cache_digest() for l, a in learners.items()}
            points, raw_rows, events = [], [], []
            previous = 0
            resume_ref = None
            for stop in plan["evaluate_at"]:
                events.extend(advance(learners, scenes, permutation, pngs, stream, condition, previous, stop))
                verify_exchange_counters(learners, stream, permutation, condition, stop)
                p, rows = evaluate_pair(learners, evaluation, pngs, frozen_dev, stop)
                points.extend(p)
                raw_rows.extend(rows)
                metadata = {"condition": condition, "exchange": stop, "plan_sha256": summary["plan_sha256"],
                            "stream_sha256": summary["stream_sha256"], "semantic_plan_sha256": summary["semantic_plan_sha256"],
                            "frozen_peer_eval_sha256": summary["frozen_peer_eval_sha256"], "revision": revision}
                if stop in (16, 32):
                    ref = save_pair(output / "checkpoints" / f"{condition}-e{stop:06d}.pt", learners, metadata)
                    if stop == 16 and condition == "aligned":
                        resume_ref = (ref, metadata)
                current = {l: a.branch_digests() for l, a in learners.items()}
                for l in learners:
                    require(learners[l].cache_digest() == initial_caches[l], "Prewarmed feature cache changed")
                    for branch, sha in frozen[l].items():
                        if branch != "understanding":
                            require(current[l][branch] == sha, "Frozen model branch changed")
                    if condition == "no_update":
                        require(current[l] == frozen[l], "No-update condition changed weights")
                        require(digest(learners[l].optimizer.state_dict()) == initial_optimizer[l], "No-update Adam changed")
                previous = stop
                print(f"{condition}: exposure {stop}/32", flush=True)
            final_hashes = {l: a.state_digest() for l, a in learners.items()}
            # Regenerate a fixed three items, without changing any saved learning state.
            producer_checks = {}
            for language, learner in learners.items():
                producer_checks[language] = all(learner.emit(pngs[s.key]) == frozen_dev[language][s.key]
                                                for s, _, _ in evaluation[:3])
                require(producer_checks[language], "Frozen producer output changed")
                require(learner.state_digest() == final_hashes[language], "Producer evaluation changed learning state")
            changed = {l: learners[l].branch_digests()["understanding"] != frozen[l]["understanding"] for l in learners}
            require(all(v == (condition != "no_update") for v in changed.values()), "Expected understanding update/no-update mismatch")
            resume = None
            if condition == "aligned":
                resumed = build(condition)
                restore_pair(resume_ref[0], resumed, resume_ref[1])
                verify_exchange_counters(resumed, stream, permutation, condition, 16)
                advance(resumed, scenes, permutation, pngs, stream, condition, 16, 32)
                verify_exchange_counters(resumed, stream, permutation, condition, 32)
                require({l: a.state_digest() for l, a in resumed.items()} == final_hashes, "Continuous and restored learning diverged")
                require({l: a.cache_digest() for l, a in resumed.items()} == {l: a.cache_digest() for l, a in learners.items()},
                        "Continuous and restored feature caches diverged")
                resume = {"split_exchange": 16, "final_exchange": 32, "exact_learning_state_equal": True,
                          "note": "Planned replay of the same prefix for state-integrity audit; not an independent experiment"}
            condition_summary = {"status": "completed", "development_only": True, "understanding_weights_changed": changed,
                                 "frozen_producer_outputs_equal": producer_checks, "initial_branch_sha256": frozen,
                                 "final_branch_sha256": {l: a.branch_digests() for l, a in learners.items()},
                                 "final_state_sha256": final_hashes, "counters": {l: a.counters() for l, a in learners.items()},
                                 "resume_check": resume, "points": points, "all_evaluations_read_only": True}
            write_json(output / f"{condition}-summary.json", condition_summary)
            write_json(output / f"{condition}-predictions.json", raw_rows)
            write_json(output / f"{condition}-events.json", events)
            summary["conditions"][condition] = condition_summary
            all_points.extend({**p, "condition": condition} for p in points)
            write_json(output / "summary.json", summary)
            del learners
        require(file_sha(archive) == ARCHIVE_SHA256 and all(file_sha(output / p) == h for p, h in original_input_hashes.items()),
                "Original model archive/checkpoint/prior/blueprint changed")
        summary.update(status="completed", all_mechanistic_checks_passed=True,
                       original_input_files_sha256=original_input_hashes, original_input_bytes_unchanged=True,
                       model_quality_gain_required_for_smoke_pass=False,
                       models_archive_sha256=ARCHIVE_SHA256,
                       limitations=["One fixed Korean/Japanese initialization block; no language-distance inference.",
                           "Fixed native speakers and contrastive PNG association; no adaptive clarification/accommodation.",
                           "L2-unexposed evaluation pairs were already used in L1 development.",
                           "Authored captions are evaluation-only; actual imperfect peer outputs are never repaired/filtered.",
                           "Raw four-choice chance is .25; zero initial understanding is not assumed or established.",
                           "No protected evaluation, native qualification, natural-corpus training or formal L2 admission."])
        write_json(output / "learning-curves.json", all_points)
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}", all_mechanistic_checks_passed=False)
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-zip", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=HERE / "configs/development-smoke.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Public frozen preparation commit (40 hex)")
    args = parser.parse_args(argv)
    require(len(args.revision) == 40 and all(c in "0123456789abcdef" for c in args.revision), "Require a concrete preparation revision")
    run(args.models_zip, args.plan, args.output, args.revision)


if __name__ == "__main__":
    main()
