"""Fit-only negative controls and frozen spatial-prior L1 development trials.

No output from this module can qualify a first-contact research learner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .composition_diagnostics import balanced_splits, build_contrast_probes, contrast_manifest, score_contrasts, split_manifest
from .composition_suite import ArmCancelled, environment_metadata
from .config import ExperimentConfig, integer
from .development_models import build_development_model
from .diagnostics import evaluate_diagnostic, safe_json, semantic_index
from .images import scene_png
from .native_preparation import LANGUAGES_ORDER, PreparationPlan
from .profiles import ModelProfile
from .storage import now, revision
from .world import native_captions, variants

VERSION = "mastery-lab-v1"
FORMAT = "mutual-mastery-development-v1"


@dataclass(frozen=True)
class MasteryPlan:
    schema_version: int = 1
    budgets: tuple[int, ...] = (1200, 2400)
    seed_pairs: tuple[tuple[int, int], ...] = ((39007, 39009), (39017, 39019), (39027, 39029))
    b_seed_pair: tuple[int, int] = (39007, 39009)
    data_seed: int = 29003
    negative_seed_offset: int = 100003
    max_trial_seconds: int = 900
    max_suite_seconds: int = 5400
    benchmark_steps: int = 20
    comprehension_min: float = .9
    production_min: float = .9
    contrast_min: float = .9
    min_contrast_anchors: int = 8
    min_contrast_pairs: int = 2
    model_profile: ModelProfile = field(default_factory=lambda: ModelProfile(
        "mastery-medium-v1", embedding_dim=64, hidden_dim=128))
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key in ("budgets", "b_seed_pair"):
            if key in values:
                values[key] = tuple(values[key])
        if "seed_pairs" in values:
            values["seed_pairs"] = tuple(tuple(pair) for pair in values["seed_pairs"])
        if "model_profile" in values:
            values["model_profile"] = ModelProfile(**values["model_profile"])
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self, config):
        integer(self.schema_version, "mastery schema version", 1, 1)
        if not self.seed_pairs or len(set(self.seed_pairs)) != len(self.seed_pairs):
            raise ValueError("Supply unique, nonempty development seed pairs")
        for pair in (*self.seed_pairs, self.b_seed_pair):
            if len(pair) != 2:
                raise ValueError("Each initialization/training seed pair must contain two seeds")
            for seed in pair:
                integer(seed, "mastery development seed", 0, 2**32 - 1)
            PreparationPlan(budgets=self.budgets, data_seed=self.data_seed,
                            initialization_seed=pair[0], training_seed=pair[1]).validate(config)
        for key, low, high in (("negative_seed_offset", 1, 2**31), ("max_trial_seconds", 1, 7200),
                               ("max_suite_seconds", 1, 21600), ("benchmark_steps", 0, 100),
                               ("min_contrast_anchors", 1, 144), ("min_contrast_pairs", 1, 36)):
            integer(getattr(self, key), key, low, high)
        if any(pair[1] + self.negative_seed_offset > 2**32 - 1 for pair in (*self.seed_pairs, self.b_seed_pair)):
            raise ValueError("Negative sampling seed exceeds its supported range")
        for key in ("comprehension_min", "production_min", "contrast_min"):
            value = getattr(self, key)
            if type(value) not in {float, int} or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be a finite probability")
        if not isinstance(self.metadata, dict):
            raise ValueError("Plan metadata must be a dictionary")
        self.model_profile.validate()
        return self


def candidate_assessment(point, probes, plan):
    """Keep performance and insufficient evaluation coverage as separate facts."""
    details = {}
    for category, metrics in point["contrasts"]["categories"].items():
        matching = [probe for probe in probes["probes"] if probe.category == category]
        anchors, pairs = len({probe.scene for probe in matching}), len({probe.scene.group for probe in matching})
        details[category] = {"hits": metrics["hits"], "items": metrics["items"],
            "accuracy": metrics["accuracy"], "available_anchors": anchors, "available_pairs": pairs,
            "coverage_sufficient": anchors >= plan.min_contrast_anchors and pairs >= plan.min_contrast_pairs,
            "score_target_met": metrics["accuracy"] is not None and metrics["accuracy"] >= plan.contrast_min}
    coverage = bool(details) and all(item["coverage_sufficient"] for item in details.values())
    score = (point["dev"]["comprehension_accuracy"] >= plan.comprehension_min
             and point["dev"]["valid_generation_accuracy"] >= plan.production_min
             and bool(details) and all(item["score_target_met"] for item in details.values()))
    return {"status": "indeterminate_coverage" if not coverage else "development_candidate" if score else "targets_not_met",
            "score_targets_met": score, "coverage_sufficient": coverage, "categories": details,
            "eligible_for_fresh_experiment": False, "excluded_from_research": True}


class _Deadline(Exception):
    pass


def run_trial(config, language, phase, condition, replica, seed_pair, plan=MasteryPlan(),
              output="runs/mastery-lab", progress=print, *, policy="same_pair", prior_bundle=None,
              remaining_seconds=None):
    begun = time.monotonic()
    plan.validate(config)
    if language not in LANGUAGES_ORDER or phase not in ("A", "B"):
        raise ValueError("Unknown mastery phase or language")
    expected = ("A0", "A1") if phase == "A" else ("B0", "B1")
    if condition not in expected or policy not in ("same_pair", "attributes"):
        raise ValueError("Unknown condition or distractor policy")
    if phase == "B" and prior_bundle is None:
        raise ValueError("Phase B requires the frozen common prior bundle")
    allowance = min(plan.max_trial_seconds, remaining_seconds if remaining_seconds is not None else plan.max_trial_seconds)
    deadline = begun + allowance
    initial_seed, training_seed = seed_pair
    prep = PreparationPlan(budgets=plan.budgets, data_seed=plan.data_seed,
                           initialization_seed=initial_seed, training_seed=training_seed)
    splits = balanced_splits(config, prep)
    from .fit_distractors import build_fit_catalog, make_training_examples
    from .neural import configure_cpu, digest, native_examples, nn, torch
    configure_cpu()
    torch.manual_seed(training_seed)
    catalog = build_fit_catalog(splits["train"])
    effective_policy = ("same_pair" if condition == "A0" else "attributes") if phase == "A" else policy
    if phase == "A":
        model = build_development_model(plan.model_profile, initial_seed)
        prior_hash = lambda: digest(model.visual_projection)
        prior_metadata = {"kind": "fixed_pixel_random_projection"}
    else:
        from .spatial_prior import build_spatial_model, spatial_prior_hash
        expected_hashes = sorted(hashlib.sha256(scene_png(scene)).hexdigest() for scene in splits["train"])
        if sorted(prior_bundle["report"]["fit_png_sha256"]) != expected_hashes:
            raise ValueError("Common prior was not trained exclusively on this trial's fit PNGs")
        model = build_spatial_model(plan.model_profile, initial_seed, prior_bundle, condition=condition)
        prior_hash = lambda: spatial_prior_hash(model)
        prior_metadata = model.spatial_prior_metadata
    examples = {key: native_examples(model, scenes, language) for key, scenes in splits.items()}
    for key, pool in examples.items():
        for example, scene in zip(pool, (scene for scene in splits[key] for _ in range(4))):
            example.update(_scene_key=scene.key, _first_caption=native_captions(scene, language)[0])
    probes = build_contrast_probes(config, splits["dev"])
    environment = environment_metadata(torch)
    directory = Path(output).resolve() / f"{condition}-r{replica}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    data = {key: [{"scene": scene.to_dict(), "captions": native_captions(scene, language)} for scene in scenes]
            for key, scenes in splits.items()}
    safe_json(directory / "dataset.json", data)
    safe_json(directory / "config.json", config.to_dict())
    safe_json(directory / "contrast_probes.json", contrast_manifest(probes))
    safe_json(directory / "training_catalog.json", catalog.manifest())
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path(__file__).resolve().parent.glob("*.py"))}
    report = {"schema_version": 1, "kind": VERSION + "-trial", "excluded_from_research": True,
        "eligible_for_fresh_experiment": False, "created_at": now(), "revision": revision(),
        "phase": phase, "language": language, "condition": condition, "replica": replica,
        "seed_pair": list(seed_pair), "plan": asdict(plan), "plan_sha256": digest(asdict(plan)),
        "status": "running", "policy": effective_policy, "data_sha256": digest(data),
        "split": split_manifest(splits), "source_sha256": sources, "config_sha256": digest(config.to_dict()),
        "model_profile": asdict(plan.model_profile), "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "environment": environment, "environment_sha256": digest(environment),
        "training_languages": [language], "foreign_training_languages": [], "protected_splits_used": [],
        "initial_understanding_sha256": digest(model.understanding.state_dict()),
        "initial_producer_sha256": digest(model.producer.state_dict()),
        "initial_prior_sha256": prior_hash(), "prior_metadata": prior_metadata,
        "optimization": "independent-branch-clipping-adam-v1", "clip_norm_per_branch": 1,
        "batch_size": config.native_training_profile.batch_size,
        "learning_rate": config.native_training_profile.learning_rate,
        "choice_count": 4, "training_chance_accuracy": .25,
        "comprehension_forms": 4, "production_forms": 1, "teacher_forced_evaluation_forms": 4,
        "negative_sampling_seed": training_seed + plan.negative_seed_offset,
        "training_steps_completed": 0, "curves": [], "checkpoints": [], "negative_exposures": {},
        "runtime_allowance_seconds": allowance, "benchmark": None}
    meaning_stream, caption_stream = hashlib.sha256(), hashlib.sha256()
    generations, predictions, step = [], [], 0
    index = semantic_index(language)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.native_training_profile.learning_rate)
    rng, negative_rng = random.Random(training_seed), random.Random(training_seed + plan.negative_seed_offset)
    audit = Counter()

    def check_deadline():
        if time.monotonic() >= deadline:
            raise _Deadline()

    def state_hash():
        return digest({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": rng.getstate(), "negative_rng": negative_rng.getstate(),
                       "torch_rng": torch.random.get_rng_state()})

    def snapshot():
        report.update(elapsed_seconds=time.monotonic() - begun,
                      meaning_stream_sha256=meaning_stream.hexdigest(),
                      comprehension_caption_stream_sha256=caption_stream.hexdigest(),
                      negative_exposures=dict(audit), final_prior_sha256=prior_hash(),
                      final_understanding_sha256=digest(model.understanding.state_dict()),
                      final_producer_sha256=digest(model.producer.state_dict()))
        if report["initial_prior_sha256"] != report["final_prior_sha256"]:
            raise RuntimeError("Frozen visual prior parameters or buffers changed")
        safe_json(directory / "report.json", report)
        safe_json(directory / "generations.json", generations)
        safe_json(directory / "comprehension_predictions.json", predictions)

    def checkpoint():
        path = directory / f"step-{step:06d}.pt"
        payload = {"format": FORMAT, "kind": "development_only", "excluded_from_research": True,
            "eligible_for_fresh_experiment": False, "language": language, "phase": phase,
            "condition": condition, "replica": replica, "seed_pair": list(seed_pair), "step": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "training_rng_state": rng.getstate(), "negative_rng_state": negative_rng.getstate(),
            "torch_rng_state": torch.random.get_rng_state(), "plan": asdict(plan),
            "source_sha256": sources, "data_sha256": report["data_sha256"], "split": report["split"],
            "environment": environment, "prior_metadata": prior_metadata, "revision": report["revision"]}
        payload["negative_exposure_offset"] = step * config.native_training_profile.batch_size
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary); temporary.replace(path)
        manifest = {"format": FORMAT, "kind": "development_only", "excluded_from_research": True,
                    "eligible_for_fresh_experiment": False, "step": step,
                    "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        safe_json(path.with_suffix(".json"), manifest)
        report["checkpoints"] = [item for item in report["checkpoints"] if item["path"] != path.name] + [manifest]

    def update(batch, selected):
        optimizer.zero_grad(set_to_none=True)
        production = [dict(example, text=example["_first_caption"]) for example in batch]
        loss = model.grounding_loss(selected) + model.production_loss(production)
        if not torch.isfinite(loss).item():
            raise ValueError("Non-finite mastery loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.understanding.parameters(), 1, error_if_nonfinite=True)
        nn.utils.clip_grad_norm_(model.producer.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()

    def benchmark():
        nonlocal optimizer
        if not plan.benchmark_steps:
            return
        weights, torch_rng = copy.deepcopy(model.state_dict()), torch.random.get_rng_state()
        bench_rng, bench_negative = random.Random(training_seed), random.Random(training_seed + plan.negative_seed_offset)
        started = time.monotonic()
        try:
            for index in range(plan.benchmark_steps):
                batch = bench_rng.choices(examples["train"], k=config.native_training_profile.batch_size)
                selected, _ = make_training_examples(batch, catalog, bench_negative,
                    exposure_offset=index * len(batch), policy=effective_policy)
                update(batch, selected)
        finally:
            model.load_state_dict(weights)
            torch.random.set_rng_state(torch_rng)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.native_training_profile.learning_rate)
        seconds = time.monotonic() - started
        report["benchmark"] = {"steps": plan.benchmark_steps, "training_seconds": seconds,
            "steps_per_second": plan.benchmark_steps / max(seconds, 1e-12),
            "initial_weights_restored": digest(model.state_dict()) == digest(weights),
            "scope": "Training-only throughput; evaluation/setup overhead excluded; budget not adapted"}

    def evaluate():
        check_deadline()
        before, point = state_hash(), {"step": step}
        for key, scenes in splits.items():
            metrics, outputs = evaluate_diagnostic(model, examples[key], scenes, language, index,
                                                   config.native_training_profile.batch_size)
            valid = sum(item["exact_match"] and item["terminated"] and not item["invalid_encoding"]
                        and not item["special_tokens_in_body"] for item in outputs)
            metrics.update(valid_generation_hits=valid, valid_generation_accuracy=valid / len(outputs))
            point[key] = metrics
            generations.extend({"step": step, "split": key, **item} for item in outputs)
            for i, example in enumerate(examples[key]):
                predicted = model.predict(example["text"], example["choices"])
                predictions.append({"step": step, "split": key, "scene": scenes[i // 4].to_dict(),
                    "template_index": i % 4, "text": example["text"], "answer": example["answer"],
                    "prediction": predicted, "correct": predicted == example["answer"],
                    "predicted_scene": variants(scenes[i // 4].group)[predicted].to_dict()})
        point["contrasts"] = score_contrasts(model, probes, language)
        point["candidate"] = candidate_assessment(point, probes, plan)
        if before != state_hash():
            raise RuntimeError("Evaluation changed model/optimizer/training RNG state")
        report["curves"].append(point)
        checkpoint(); snapshot(); check_deadline()
        if progress:
            progress(f"{phase} {language} r{replica} {condition} step={step} "
                     f"U={point['dev']['comprehension_accuracy']:.1%} "
                     f"P={point['dev']['valid_generation_accuracy']:.1%}", flush=True)

    try:
        snapshot(); check_deadline(); benchmark(); snapshot(); check_deadline(); evaluate()
        for next_step in range(1, plan.budgets[-1] + 1):
            check_deadline()
            batch = rng.choices(examples["train"], k=config.native_training_profile.batch_size)
            meaning_stream.update(json.dumps([item["_scene_key"] for item in batch]).encode() + b"\n")
            caption_stream.update(json.dumps([item["text"] for item in batch], ensure_ascii=False).encode() + b"\n")
            selected, counts = make_training_examples(batch, catalog, negative_rng,
                exposure_offset=(next_step - 1) * len(batch), policy=effective_policy)
            audit.update(counts)
            update(batch, selected)
            step = next_step; report["training_steps_completed"] = step
            if step in plan.budgets:
                evaluate()
        check_deadline(); report["status"] = "completed"; snapshot(); check_deadline()
        return directory
    except _Deadline:
        report["status"] = "time_limit"; checkpoint(); snapshot(); return directory
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        checkpoint(); snapshot()
        if isinstance(exc, KeyboardInterrupt):
            raise ArmCancelled(directory / "report.json") from exc
        return directory


def pair_audit(reports, phase):
    results = []
    replicas = sorted({report["replica"] for report in reports})
    for replica in replicas:
        pair = {report["condition"]: report for report in reports if report["replica"] == replica}
        names = ("A0", "A1") if phase == "A" else ("B0", "B1")
        if any(name not in pair or pair[name]["status"] != "completed" for name in names):
            results.append({"replica": replica, "status": "incomplete"}); continue
        left, right = (pair[name] for name in names)
        keys = ("seed_pair", "split", "data_sha256", "config_sha256", "source_sha256", "model_profile",
                "parameter_count", "batch_size", "learning_rate",
                "initial_understanding_sha256", "initial_producer_sha256",
                "meaning_stream_sha256", "comprehension_caption_stream_sha256", "environment_sha256")
        matching = {key: left[key] == right[key] for key in keys}
        if phase == "A":
            matching["initial_prior_sha256"] = left["initial_prior_sha256"] == right["initial_prior_sha256"]
            matching["final_producer_sha256"] = left["final_producer_sha256"] == right["final_producer_sha256"]
        results.append({"replica": replica, "status": "matched" if all(matching.values()) else "mismatch",
                        "checks": matching})
    return results


def run_phase(config, language, phase, plan=MasteryPlan(), output="runs/mastery-lab", progress=print,
              *, policy="same_pair", prior_bundle=None):
    plan.validate(config)
    if phase not in ("A", "B") or language not in LANGUAGES_ORDER:
        raise ValueError("Unknown mastery phase or language")
    if phase == "B" and prior_bundle is None:
        raise ValueError("Phase B requires a common prior bundle")
    directory = Path(output).resolve() / f"phase-{phase}-{language}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    suite = {"schema_version": 1, "kind": VERSION + "-suite", "phase": phase, "language": language,
        "plan": asdict(plan), "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "status": "running", "created_at": now(), "reports": [], "pair_audit": []}
    begun = time.monotonic()
    seeds = plan.seed_pairs if phase == "A" else (plan.b_seed_pair,)
    conditions = ("A0", "A1") if phase == "A" else ("B0", "B1")
    safe_json(directory / "mastery-suite.json", suite)
    for replica, pair in enumerate(seeds):
        for condition in conditions:
            remaining = plan.max_suite_seconds - (time.monotonic() - begun)
            if remaining <= 0:
                report = {"replica": replica, "condition": condition, "seed_pair": list(pair),
                          "status": "suite_time_limit", "curves": []}
            else:
                try:
                    path = run_trial(config, language, phase, condition, replica, pair, plan,
                        directory / "trials", progress, policy=policy, prior_bundle=prior_bundle,
                        remaining_seconds=remaining)
                    report = json.loads((path / "report.json").read_text())
                except KeyboardInterrupt as exc:
                    report = (json.loads(exc.report_path.read_text()) if isinstance(exc, ArmCancelled) else
                              {"replica": replica, "condition": condition, "status": "cancelled", "curves": []})
                    suite["reports"].append(report); suite["status"] = "cancelled"
                    suite["pair_audit"] = pair_audit(suite["reports"], phase)
                    safe_json(directory / "mastery-suite.json", suite); raise
                except Exception as exc:
                    report = {"replica": replica, "condition": condition, "seed_pair": list(pair),
                              "status": "failed", "curves": [], "error": f"{type(exc).__name__}: {exc}"}
            suite["reports"].append(report)
            suite["pair_audit"] = pair_audit(suite["reports"], phase)
            suite["elapsed_seconds"] = time.monotonic() - begun
            safe_json(directory / "mastery-suite.json", suite)
    suite["status"] = "completed" if all(report["status"] == "completed" for report in suite["reports"]) and all(
        audit["status"] == "matched" for audit in suite["pair_audit"]) else "incomplete"
    safe_json(directory / "mastery-suite.json", suite)
    return directory


def aggregate_reports(suites, phase):
    seen, plans, rows, failed, signatures = set(), set(), [], [], {}
    paired_deltas, prior_hashes = [], {}
    for suite in suites:
        language = suite["language"]
        if suite["phase"] != phase or language not in LANGUAGES_ORDER or language in seen:
            raise ValueError("Duplicate/unknown language or incompatible phase")
        seen.add(language); plans.add(json.dumps(suite["plan"], sort_keys=True))
        plan = MasteryPlan.from_dict(suite["plan"])
        expected = {(replica, condition) for replica in range(len(plan.seed_pairs) if phase == "A" else 1)
                    for condition in (("A0", "A1") if phase == "A" else ("B0", "B1"))}
        actual = [(r["replica"], r["condition"]) for r in suite["reports"]]
        if len(actual) != len(set(actual)) or set(actual) - expected:
            raise ValueError("Duplicate or undeclared trial")
        for replica, condition in sorted(expected - set(actual)):
            failed.append({"language": language, "replica": replica, "condition": condition, "status": "missing"})
        by_pair = {}
        for report in suite["reports"]:
            key = (report["replica"], report["condition"])
            if report["status"] != "completed":
                failed.append({"language": language, "replica": key[0], "condition": key[1], "status": report["status"]})
            else:
                signature = json.dumps({name: report[name] for name in
                    ("source_sha256", "config_sha256", "split", "model_profile", "seed_pair", "policy")}, sort_keys=True)
                if key in signatures and signatures[key] != signature:
                    raise ValueError("Matched trial metadata differs across languages")
                signatures[key] = signature
                if {point["step"] for point in report["curves"]} != {0, *plan.budgets}:
                    failed.append({"language": language, "replica": key[0], "condition": key[1], "status": "missing_budget"})
                if phase == "B":
                    previous = prior_hashes.setdefault(key[1], report["initial_prior_sha256"])
                    if previous != report["initial_prior_sha256"]:
                        raise ValueError("Languages did not share the same frozen visual prior")
            for point in report.get("curves", []):
                row = {"language": language, "replica": report["replica"], "condition": report["condition"],
                       "step": point["step"], "status": report["status"], "dev": point["dev"],
                       "contrasts": point["contrasts"]["categories"], "candidate": point["candidate"]}
                rows.append(row); by_pair[(key[0], key[1], point["step"])] = row
        for audit in pair_audit(suite["reports"], phase):
            if audit["status"] != "matched":
                failed.append({"language": language, "replica": audit["replica"], "status": "pair_" + audit["status"]})
        if phase == "A":
            for replica in range(len(plan.seed_pairs)):
                for step in plan.budgets:
                    left, right = (by_pair.get((replica, condition, step)) for condition in ("A0", "A1"))
                    if left and right:
                        paired_deltas.append({"language": language, "replica": replica, "step": step,
                            "comprehension_delta": right["dev"]["comprehension_accuracy"] - left["dev"]["comprehension_accuracy"],
                            "generation_delta": right["dev"]["valid_generation_accuracy"] - left["dev"]["valid_generation_accuracy"],
                            "contrast_deltas": {category: right["contrasts"][category]["accuracy"] - left["contrasts"][category]["accuracy"]
                                if left["contrasts"][category]["accuracy"] is not None and right["contrasts"][category]["accuracy"] is not None else None
                                for category in left["contrasts"]}})
    if len(plans) > 1:
        raise ValueError("Mastery plans differ")
    missing = [language for language in LANGUAGES_ORDER if language not in seen]
    selection = {"policy": "undetermined", "reason": "Incomplete or unmatched A comparison"}
    if phase == "A" and not missing and not failed and suites:
        final_step = MasteryPlan.from_dict(suites[0]["plan"]).budgets[-1]
        final = [row for row in paired_deltas if row["step"] == final_step]
        n = len(MasteryPlan.from_dict(suites[0]["plan"]).seed_pairs)
        if len(final) == len(LANGUAGES_ORDER) * n and all(
                len(row["contrast_deltas"]) == 4 and all(v is not None for v in row["contrast_deltas"].values())
                for row in final):
            blocks = [sum(sum(row["contrast_deltas"].values()) / 4 for row in final
                          if row["replica"] == replica) / len(LANGUAGES_ORDER) for replica in range(n)]
            language_deltas = {language: sum(row["comprehension_delta"] for row in final
                                if row["language"] == language) / n for language in LANGUAGES_ORDER}
            grand = sum(blocks) / n
            attributes = grand > 0 and sum(value > 0 for value in blocks) >= 2 and all(
                value >= -.05 for value in language_deltas.values())
            selection = {"policy": "attributes" if attributes else "same_pair", "final_step": final_step,
                         "seed_block_macro_deltas": blocks, "grand_macro_delta": grand,
                         "language_mean_four_choice_deltas": language_deltas,
                         "rule": "Positive grand category macro delta; at least two positive seed blocks; every language mean four-choice delta >= -0.05",
                         "interpretation": "Predeclared development policy rule, not a statistical significance claim"}
    return {"schema_version": 1, "kind": VERSION + "-comparison", "phase": phase,
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "status": "incomplete" if missing or failed else "completed",
            "missing_languages": missing, "failed_trials": failed, "rows": rows,
            "paired_deltas": paired_deltas, "b_policy_selection": selection if phase == "A" else None,
            "interpretation": "Development observations only; shared prior is not independent visual replication"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/neural.example.json"))
    parser.add_argument("--plan", default=str(Path(__file__).resolve().parents[1] / "configs/mastery-plan.json"))
    parser.add_argument("--phase", choices=("A", "B"), required=True)
    parser.add_argument("--language", choices=LANGUAGES_ORDER)
    parser.add_argument("--policy", choices=("same_pair", "attributes"), default="same_pair")
    parser.add_argument("--prior-bundle", type=Path)
    parser.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.aggregate_dir is not None:
        paths = sorted(args.aggregate_dir.rglob("mastery-suite.json"))
        if not paths:
            raise ValueError("No mastery suite reports found")
        result = aggregate_reports([json.loads(path.read_text()) for path in paths], args.phase)
        path = Path(args.output or "runs/mastery-summary.json"); path.parent.mkdir(parents=True, exist_ok=True)
        safe_json(path, result); print(json.dumps(result), flush=True); return 0
    if args.language is None:
        parser.error("--language is required")
    plan = MasteryPlan.load(args.plan)
    bundle = None
    if args.prior_bundle is not None:
        from .spatial_prior import load_prior_bundle
        bundle = load_prior_bundle(args.prior_bundle)
    path = run_phase(ExperimentConfig.load(args.config), args.language, args.phase, plan,
                     args.output or "runs/mastery-lab", policy=args.policy, prior_bundle=bundle)
    print(f"Saved development-only mastery suite: {path}", flush=True)
    return 0 if json.loads((path / "mastery-suite.json").read_text())["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
