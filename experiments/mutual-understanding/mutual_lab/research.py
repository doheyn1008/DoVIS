"""Persistent first-contact protocol, independent screening and restartable checkpoints."""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from .agents import observe_persistent, speak
from .metrics import classify_initial, evaluate, raw_threshold, summarize_points
from .neural import NeuralAgent
from .storage import RunStore, now, write_json
from .world import make_research_dataset, native_captions


def dialogue_plan(dataset, config):
    rng = random.Random(config.effective_seed("dialogue"))
    order = list(dataset.train)
    plan = []
    while len(plan) < config.max_exchanges:
        rng.shuffle(order)
        plan.extend(order)
    return plan[:config.max_exchanges]


def run_neural_experiment(config, output="runs", progress=None, *, resume_from=None):
    from .runner import ExperimentFailed
    config.validate()
    data = make_research_dataset(config)
    old = Path(resume_from).resolve() if resume_from is not None else None
    resume = json.loads((old / "learning_checkpoint.json").read_text()) if old else None
    if resume and resume["dataset_sha256"] != data.fingerprint:
        raise ValueError("Resume dataset differs from the checkpoint")
    agents = ([NeuralAgent.restore(old / resume["agents"][a.name], a, config) for a in config.agents] if old else
              [NeuralAgent(a, config) for a in config.agents])
    store = RunStore(output, config)
    points = resume["points"] if resume else []
    exchange = resume["exchange"] if resume else 0
    pretest = resume["pretest"] if resume else []
    inclusion = resume["inclusion"] if resume else None
    captions = resume["captions"] if resume else {}
    final = []
    caption_audit = resume.get("caption_audit", {}) if resume else {}
    reason = "max_exchanges"
    begun, cpu_begun = time.monotonic(), time.process_time()
    e = config.evaluation_profile
    horizon = e.common_horizon if e.common_horizon is not None else config.max_exchanges
    native = [{"name": a.config.name, **a.metadata} for a in agents]
    initial_hashes = [a.state_digest() for a in agents]
    limitations = ["Bounded authored natural-language corpus; L1 qualification is not general language mastery.",
                  "Frozen autoregressive native producer; no interactive clarification or speaker accommodation.",
                  "Fixed pixel pooling/random projection is not learned human visual perception.",
                  "Wilson intervals describe correlated items, not model-replica uncertainty.",
                  "Chance-band equivalence in this small pilot is not proof of zero understanding."]
    if config.acquisition_profile.reset_l1_understanding:
        limitations.append("Control: receiver L1 understanding reset; native producer remains trained.")

    def directions(status="completed"):
        return summarize_points(points, config.threshold, config.sustained_checks,
                                e.threshold_scale, horizon, status) if points else []

    def snapshot(status, error=None):
        summary = {"schema_version": 2, "run_id": store.run_id, "status": status, "mode": "neural",
                   "learning_mode": "persistent_weights", "protocol": "frozen-native-producer-v1",
                   "seed": config.seed, "started_at": store.started_at, "updated_at": now(),
                   "agents": [asdict(a.config) for a in agents], "native_models": native,
                   "model_profile": asdict(config.model_profile),
                   "native_training_profile": asdict(config.native_training_profile),
                   "acquisition_profile": asdict(config.acquisition_profile),
                   "evaluation_profile": asdict(config.evaluation_profile),
                   "model_replica_id": config.model_replica_id, "experience_replica_id": config.experience_replica_id,
                   "pair_id": config.pair_id, "scene_split_seed": config.effective_seed("scene_split"),
                   "dialogue_seed": config.effective_seed("dialogue"), "evaluation_seed": config.effective_seed("evaluation"),
                   "exchanges_completed": exchange, "utterances_completed": 2 * exchange,
                   "semantic_scene_presentations": exchange,
                   "unique_semantic_scenes": len({s.key for s in plan[:exchange]}),
                   "primary_exposure_unit": "one shared scene and one received partner utterance per learner",
                   "threshold": config.threshold, "threshold_scale": e.threshold_scale,
                   "threshold_raw_equivalent": raw_threshold(config.threshold, e.threshold_scale),
                   "sustained_checks": config.sustained_checks, "common_horizon": horizon,
                   "stop_reason": reason if status == "completed" else None,
                   "transport": {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0},
                   "exposure": {a.config.name: {**a.counters(),
                       "training_compute_proxy_parameter_units": 6 * a.metadata["understanding_parameter_count"] * a.training_sequence_units,
                       "compute_proxy_note": "parameter-unit proxy; not measured FLOPs"} for a in agents},
                   "wall_seconds_this_segment": time.monotonic() - begun,
                   "cpu_seconds_this_segment": time.process_time() - cpu_begun,
                   "directions": directions(status), "pretest": pretest, "inclusion": inclusion,
                   "caption_audit": caption_audit,
                   "final_evaluation": final, "dataset_sha256": data.fingerprint,
                   "initial_learner_state_sha256": initial_hashes,
                   "resume_from": str(old) if old else None, "limitations": limitations, "error": error}
        write_json(store.path / "summary.json", summary)
        return summary

    def persist_learning_checkpoint():
        folder = store.path / "checkpoints" / f"e{exchange:06d}"
        refs = {a.config.name: str(a.save(folder / f"{a.config.name}.pt").relative_to(store.path)) for a in agents}
        # Publish only after both learners and sidecars are durable; a partial save is never resumable.
        write_json(store.path / "learning_checkpoint.json", {"schema_version": 1, "exchange": exchange,
                   "agents": refs, "points": points, "pretest": pretest, "inclusion": inclusion,
                   "caption_audit": caption_audit,
                   "captions": captions, "dataset_sha256": data.fingerprint,
                   "state_sha256": [a.state_digest() for a in agents]})

    def checkpoint():
        current = evaluate(agents, data.view("curve"), captions["curve"], exchange,
                           config.effective_seed("evaluation"))
        points.extend(current)
        store.metrics(points)
        store.event({"type": "evaluation", "exchange": exchange, "points": current})
        persist_learning_checkpoint()
        snapshot("running")
        if progress:
            progress({"run_id": store.run_id, "exchange": exchange, "points": current})

    def should_stop():
        ds = directions()
        return config.stop_on_threshold and len(ds) == 2 and all(d["threshold_outcome"] == "reached" for d in ds)

    plan = dialogue_plan(data, config)
    association_rng = random.Random(config.effective_seed("dialogue") + 7919)
    shuffled_scene_plan = [association_rng.choice([s for s in data.train if s != actual]) for actual in plan]
    snapshot("running")
    try:
        (store.path / "native").mkdir()
        for a in agents:
            saved_native = old / "native" / f"{a.config.name}.pt" if old else None
            source = saved_native if saved_native and saved_native.exists() else Path(a.config.native_checkpoint)
            for original, destination in ((source, store.path / "native" / f"{a.config.name}.pt"),
                    (source.with_suffix(".json"), store.path / "native" / f"{a.config.name}.json")):
                shutil.copy2(original, destination)
        write_json(store.path / "dataset.json", data.to_dict())
        write_json(store.path / "native_qualification.json", native)
        write_json(store.path / "dialogue_plan.json", {"scene_keys": [s.key for s in plan],
                   "note": "fixed across language pairs at the same dialogue seed; both speakers generate before updates"})
        store.event({"type": "run_started", "mode": "neural", "resume_from": str(old) if old else None,
                     "model_replica_id": config.model_replica_id, "experience_replica_id": config.experience_replica_id})
        if old:
            if resume["state_sha256"] != [a.state_digest() for a in agents]:
                raise ValueError("Restored learner state does not match checkpoint digest")
            for image in (old / "assets").glob("*.png"):
                shutil.copy2(image, store.path / "assets" / image.name)
            with (old / "events.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["type"] == "dialogue" and event["exchange"] <= exchange:
                        store.event({**event, "restored_from": old.name})
            store.metrics(points)
            persist_learning_checkpoint()
        else:
            # All evaluation captions are generated by original L1 models before any L2 exposure.
            for split in ("pretest", "curve", "final"):
                captions[split] = {a.config.name: [speak(a, p.scene, frozen=True) for p in data.view(split).probes]
                                   for a in agents}
                caption_audit[split] = {a.config.name: [text in native_captions(p.scene, a.config.language)
                    for text, p in zip(captions[split][a.config.name], data.view(split).probes)] for a in agents}
            pretest = evaluate(agents, data.view("pretest"), captions["pretest"], 0,
                               config.effective_seed("evaluation"))
            inclusion = classify_initial(pretest, [a.metadata["native_eligible"] for a in agents],
                                         e.chance_margin, verified_foreign_exposure=True)
            if not all(all(flags) for split in caption_audit.values() for flags in split.values()):
                inclusion = {**inclusion, "analysis_group": "ineligible",
                             "reason": "one or more frozen captions failed the preregistered native-caption semantic audit"}
            controls = []
            if config.acquisition_profile.reset_l1_understanding:
                controls.append("L1 understanding reset")
            if config.acquisition_profile.association == "shuffled":
                controls.append("utterance-scene shuffled")
            if config.acquisition_profile.updates_per_exposure == 0:
                controls.append("zero optimizer updates")
            if controls:
                inclusion = {**inclusion, "cohort_without_control": inclusion["analysis_group"],
                             "analysis_group": "control", "reason": "; ".join(controls)}
            checkpoint()
        write_json(store.path / "pretest.json", {"points": pretest, "inclusion": inclusion})
        write_json(store.path / "caption_audit.json", caption_audit)
        write_json(store.path / "frozen_captions.json", captions["curve"])
        write_json(store.path / "all_frozen_captions.json", captions)
        if should_stop():
            reason = "sustained_threshold_both_directions"
        else:
            for next_exchange in range(exchange + 1, config.max_exchanges + 1):
                scene = plan[next_exchange - 1]
                # Generate BOTH utterances from frozen native producers before either learner updates.
                texts = [speak(a, scene) for a in agents]
                for index, agent in enumerate(agents):
                    shared = scene
                    if config.acquisition_profile.association == "shuffled":
                        # Uncorrelated wrong scenes, sampled per exposure and matched across language pairs.
                        shared = shuffled_scene_plan[next_exchange - 1]
                    observe_persistent(agent, shared, texts[1 - index])
                exchange = next_exchange
                store.event({"type": "dialogue", "exchange": exchange, "image": store.image(scene),
                             "learning_image": store.image(shared),
                             "association": config.acquisition_profile.association,
                             "messages": [{"speaker": a.config.name, "language": a.config.language, "text": t}
                                          for a, t in zip(agents, texts)]})
                if exchange % config.eval_every == 0 or exchange == config.max_exchanges:
                    checkpoint()
                    if should_stop():
                        reason = "sustained_threshold_both_directions"
                        break
        final = evaluate(agents, data.view("final"), captions["final"], exchange,
                         config.effective_seed("evaluation"))
        write_json(store.path / "final_evaluations.json", final)
        summary = snapshot("completed")
        store.event({"type": "run_completed", "exchange": exchange, "stop_reason": reason})
        if progress:
            progress({"run_id": store.run_id, "summary": summary})
        return store.path
    except (Exception, KeyboardInterrupt) as exc:
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        message = f"{type(exc).__name__}: {exc}"
        snapshot(status, message)
        store.event({"type": "run_failed", "exchange": exchange, "error": message})
        raise ExperimentFailed(message, store.path) from exc
