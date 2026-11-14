"""Development-only, controlled composition comparisons on one runner per language."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .composition_diagnostics import (
    balanced_splits, build_contrast_probes, contrast_manifest, score_contrasts, split_manifest,
)
from .config import ExperimentConfig, integer
from .development_models import build_development_model
from .diagnostics import evaluate_diagnostic, safe_json, semantic_index
from .native_preparation import LANGUAGES_ORDER, PreparationPlan, build_preparation_splits
from .profiles import ModelProfile
from .storage import now, revision
from .world import native_captions, variants

VERSION = "composition-suite-v1"
CHECKPOINT_FORMAT = "mutual-composition-development-v1"
OPTIMIZATION = "independent-branch-clipping-adam-v1"
DEFAULT_ARMS = (
    "tiny_original_four", "tiny_balanced_four", "medium_balanced_four",
    "medium_balanced_one", "medium_balanced_bi_one", "medium_balanced_wide_one",
)
ARM_DEFINITIONS = {
    "tiny_original_four": ("tiny", "original", "last", 4),
    "tiny_balanced_four": ("tiny", "balanced", "last", 4),
    "tiny_balanced_one": ("tiny", "balanced", "last", 1),  # Explicit small isolation check.
    "medium_balanced_four": ("medium", "balanced", "last", 4),
    "medium_balanced_one": ("medium", "balanced", "last", 1),
    "medium_balanced_bi_one": ("medium", "balanced", "bidirectional", 1),
    "medium_balanced_wide_one": ("medium", "balanced", "wide_last", 1),
}


def arm_spec(name):
    if name not in ARM_DEFINITIONS:
        raise ValueError(f"Unknown development arm: {name}")
    size, split, encoder, forms = ARM_DEFINITIONS[name]
    return {"name": name, "size": size, "split_policy": split,
            "encoder": encoder, "production_forms": forms}


@dataclass(frozen=True)
class SuitePlan:
    budgets: tuple[int, ...] = (1200, 2400)
    arms: tuple[str, ...] = DEFAULT_ARMS
    max_seconds_per_arm: int = 600
    max_seconds: int = 1800
    comprehension_min: float = .9
    production_min: float = .9

    def validate(self, config):
        PreparationPlan(budgets=self.budgets, comprehension_min=self.comprehension_min,
                        production_min=self.production_min).validate(config)
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("Supply unique development arms")
        for name in self.arms:
            arm_spec(name)
        integer(self.max_seconds_per_arm, "per-arm runtime limit", 1, 1800)
        integer(self.max_seconds, "suite runtime limit", 1, 7200)
        return self


def model_profile(config, arm):
    return (config.model_profile if arm_spec(arm)["size"] == "tiny" else
            ModelProfile("development-medium-gru-v1", embedding_dim=64, hidden_dim=128))


def environment_metadata(torch):
    import numpy
    import PIL
    try:
        cpu = next(line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu, "torch": str(torch.__version__), "numpy": numpy.__version__, "pillow": PIL.__version__,
            "torch_config": torch.__config__.show(), "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
            "environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED",
                "MKL_CBWR", "ONEDNN_MAX_CPU_ISA")}}


def run_arm(config, language, arm, plan=SuitePlan(), output="runs/composition-suite",
            progress=print, *, remaining_seconds=None):
    """Train one declared arm; the understanding stream always retains four forms."""
    plan.validate(config)
    if language not in LANGUAGES_ORDER:
        raise ValueError("Unknown suite language")
    spec = arm_spec(arm)
    preparation = PreparationPlan(budgets=plan.budgets)
    splits = (balanced_splits(config, preparation) if spec["split_policy"] == "balanced"
              else build_preparation_splits(config, preparation))
    profile = model_profile(config, arm)
    profile.validate()
    from .neural import configure_cpu, digest, native_examples, torch, nn
    configure_cpu()
    torch.manual_seed(preparation.training_seed)
    model = build_development_model(profile, preparation.initialization_seed, spec["encoder"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.native_training_profile.learning_rate)
    rng = random.Random(preparation.training_seed)
    examples = {name: native_examples(model, scenes, language) for name, scenes in splits.items()}
    for name, pool in examples.items():
        for item, scene in zip(pool, (s for s in splits[name] for _ in range(4))):
            item["_scene_key"] = scene.key
            item["_first_caption"] = native_captions(scene, language)[0]
    probes = build_contrast_probes(config, splits["dev"])
    metadata = environment_metadata(torch)
    root = Path(__file__).resolve().parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    directory = Path(output).resolve() / (language + "-" + arm + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    data = {name: [{"scene": s.to_dict(), "captions": native_captions(s, language)} for s in scenes]
            for name, scenes in splits.items()}
    safe_json(directory / "dataset.json", data)
    safe_json(directory / "contrast_probes.json", contrast_manifest(probes))
    safe_json(directory / "config.json", config.to_dict())
    report = {"schema_version": 1, "kind": VERSION, "excluded_from_research": True,
        "eligible_for_fresh_experiment": False, "language": language, "arm": spec,
        "plan": asdict(plan), "created_at": now(), "revision": revision(), "status": "running",
        "training_languages": [language], "foreign_training_languages": [], "protected_splits_used": [],
        "split": split_manifest(splits), "data_sha256": digest(data),
        "config_sha256": digest(config.to_dict()), "source_sha256": sources,
        "environment": metadata, "environment_sha256": digest(metadata),
        "model_profile": asdict(profile), "model_spec": model.development_spec,
        "optimization_protocol": OPTIMIZATION, "gradient_clip_norm_per_branch": 1,
        "learning_rate": config.native_training_profile.learning_rate,
        "batch_size": config.native_training_profile.batch_size,
        "comprehension_training_forms": "all_four_authored_forms",
        "production_training_forms": "first_authored_form" if spec["production_forms"] == 1 else "all_four_authored_forms",
        "teacher_forced_evaluation_forms": "all_four_authored_forms",
        "initial_understanding_sha256": digest(model.understanding.state_dict()),
        "initial_producer_sha256": digest(model.producer.state_dict()),
        "initial_visual_sha256": digest(model.visual_projection),
        "training_steps_completed": 0, "curves": [], "checkpoints": [],
        "interpretation": [
            "Development-only comparisons; no native qualification or foreign acquisition.",
            "All arms use separate branch clipping, so the original split arm is a new baseline, not a numerical replay of old global clipping.",
            "Balanced-vs-original changes both train and dev pairs; it compares split policies, not an isolated causal partner-count effect.",
            "Binary contrast chance is 0.5; original four-choice chance is 0.25. Availability and denominators differ.",
            "One-form production retains all comprehension forms and exactly the same sampled meaning stream.",
            "Teacher-forced evaluation still uses four forms and is not a comparable training-objective loss across production policies.",
            "No automatic research eligibility or best-checkpoint selection is performed."]}
    meaning_stream, caption_stream = hashlib.sha256(), hashlib.sha256()
    generations, predictions, step = [], [], 0
    index = semantic_index(language)
    begun = time.monotonic()
    allowance = min(plan.max_seconds_per_arm, remaining_seconds if remaining_seconds is not None
                    else plan.max_seconds_per_arm)

    def state_hash():
        return digest({"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "rng": rng.getstate(), "torch_rng": torch.random.get_rng_state()})

    def snapshot():
        report.update(elapsed_seconds=time.monotonic() - begun,
                      meaning_stream_sha256=meaning_stream.hexdigest(),
                      comprehension_caption_stream_sha256=caption_stream.hexdigest(),
                      final_understanding_sha256=digest(model.understanding.state_dict()),
                      final_producer_sha256=digest(model.producer.state_dict()))
        safe_json(directory / "report.json", report)
        safe_json(directory / "generations.json", generations)
        safe_json(directory / "comprehension_predictions.json", predictions)

    def save_checkpoint():
        path = directory / f"step-{step:06d}.pt"
        payload = {"format": CHECKPOINT_FORMAT, "kind": "development_only", "excluded_from_research": True,
            "eligible_for_fresh_experiment": False, "step": step, "language": language, "arm": spec,
            "plan": asdict(plan), "model_profile": asdict(profile), "model_spec": model.development_spec,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "training_rng_state": rng.getstate(), "torch_rng_state": torch.random.get_rng_state(),
            "source_sha256": sources, "config_sha256": report["config_sha256"],
            "split": report["split"], "environment": metadata, "revision": report["revision"],
            "meaning_stream_sha256": meaning_stream.hexdigest(),
            "comprehension_caption_stream_sha256": caption_stream.hexdigest()}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        manifest = {"format": CHECKPOINT_FORMAT, "kind": "development_only",
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "step": step, "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        safe_json(path.with_suffix(".json"), manifest)
        report["checkpoints"] = [c for c in report["checkpoints"] if c["path"] != path.name] + [manifest]

    def evaluate():
        before = state_hash()
        row = {"step": step}
        for name, scenes in splits.items():
            metrics, outputs = evaluate_diagnostic(model, examples[name], scenes, language, index,
                                                   config.native_training_profile.batch_size)
            valid = sum(item["exact_match"] and item["terminated"] and not item["invalid_encoding"]
                        and not item["special_tokens_in_body"] for item in outputs)
            metrics.update(valid_generation_hits=valid, valid_generation_accuracy=valid / len(outputs))
            row[name] = metrics
            generations.extend({"step": step, "split": name, **item} for item in outputs)
            for i, item in enumerate(examples[name]):
                predicted = model.predict(item["text"], item["choices"])
                scene = scenes[i // 4]
                predictions.append({"step": step, "split": name, "scene": scene.to_dict(),
                    "template_index": i % 4, "text": item["text"], "answer": item["answer"],
                    "prediction": predicted, "correct": predicted == item["answer"],
                    "predicted_scene": variants(scene.group)[predicted].to_dict()})
        row["contrasts"] = score_contrasts(model, probes, language)
        if before != state_hash():
            raise RuntimeError("Composition evaluation changed learning state")
        report["curves"].append(row)
        save_checkpoint()
        snapshot()
        if progress:
            progress(f"{language} {arm} step={step} understand={row['dev']['comprehension_accuracy']:.1%} "
                     f"generate={row['dev']['valid_generation_accuracy']:.1%}", flush=True)

    snapshot()
    try:
        evaluate()
        for next_step in range(1, plan.budgets[-1] + 1):
            if time.monotonic() - begun >= allowance:
                report["status"] = "time_limit"
                save_checkpoint(); snapshot()
                return directory
            batch = rng.choices(examples["train"], k=config.native_training_profile.batch_size)
            meaning_stream.update(json.dumps([e["_scene_key"] for e in batch]).encode() + b"\n")
            caption_stream.update(json.dumps([e["text"] for e in batch], ensure_ascii=False).encode() + b"\n")
            production = ([dict(e, text=e["_first_caption"]) for e in batch]
                          if spec["production_forms"] == 1 else batch)
            optimizer.zero_grad(set_to_none=True)
            comprehension_loss = model.grounding_loss(batch)
            production_loss = model.production_loss(production)
            loss = comprehension_loss + production_loss
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite suite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.understanding.parameters(), 1, error_if_nonfinite=True)
            nn.utils.clip_grad_norm_(model.producer.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            step = next_step
            report["training_steps_completed"] = step
            if step in plan.budgets:
                evaluate()
        report["status"] = "completed"
        snapshot()
        return directory
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_checkpoint(); snapshot()
        if isinstance(exc, KeyboardInterrupt):
            raise
        return directory


def run_suite(config, language, plan=SuitePlan(), output="runs/composition-suite", progress=print):
    plan.validate(config)
    if language not in LANGUAGES_ORDER:
        raise ValueError("Unknown suite language")
    directory = Path(output).resolve() / (language + "-suite-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    suite = {"schema_version": 1, "kind": VERSION, "excluded_from_research": True,
             "eligible_for_fresh_experiment": False, "language": language, "plan": asdict(plan),
             "status": "running", "created_at": now(), "reports": []}
    begun = time.monotonic()
    safe_json(directory / "suite.json", suite)
    for arm in plan.arms:
        remaining = plan.max_seconds - (time.monotonic() - begun)
        if remaining <= 0:
            suite["reports"].append({"arm": arm_spec(arm), "status": "suite_time_limit",
                                     "language": language, "curves": []})
        else:
            try:
                path = run_arm(config, language, arm, plan, directory / "arms", progress,
                               remaining_seconds=remaining)
                suite["reports"].append(json.loads((path / "report.json").read_text()))
            except Exception as exc:
                suite["reports"].append({"arm": arm_spec(arm), "status": "failed",
                    "language": language, "curves": [], "error": f"{type(exc).__name__}: {exc}"})
            except KeyboardInterrupt:
                suite["status"] = "cancelled"
                safe_json(directory / "suite.json", suite)
                raise
        suite["elapsed_seconds"] = time.monotonic() - begun
        safe_json(directory / "suite.json", suite)
    suite["status"] = "completed" if all(r["status"] == "completed" for r in suite["reports"]) else "incomplete"
    safe_json(directory / "suite.json", suite)
    return directory


def aggregate_suites(suites):
    """Report every declared arm/language/budget; never promote a selected result."""
    seen, reference, rows, failed, signatures = set(), None, [], [], {}
    for suite in suites:
        language = suite["language"]
        if language not in LANGUAGES_ORDER or language in seen:
            raise ValueError("Unknown or duplicate suite language")
        seen.add(language)
        encoded = json.dumps(suite["plan"], sort_keys=True)
        if reference is not None and reference != encoded:
            raise ValueError("Suite plans differ")
        reference = encoded
        declared = set(suite["plan"]["arms"])
        available = [report["arm"]["name"] for report in suite["reports"]]
        if len(set(available)) != len(available) or set(available) - declared:
            raise ValueError("Duplicate or undeclared arm in suite")
        for missing_arm in sorted(declared - set(available)):
            failed.append({"language": language, "arm": missing_arm, "status": "missing"})
        for report in suite["reports"]:
            if report["status"] != "completed":
                failed.append({"language": language, "arm": report["arm"]["name"], "status": report["status"]})
            if report["status"] == "completed":
                if {point["step"] for point in report["curves"]} != {0, *suite["plan"]["budgets"]}:
                    failed.append({"language": language, "arm": report["arm"]["name"], "status": "missing_budget"})
                # Environment may differ between languages; expose it, do not call it identical.
                keys = ("config_sha256", "source_sha256", "model_profile", "arm", "split")
                signature = json.dumps({key: report[key] for key in keys}, sort_keys=True)
                arm = report["arm"]["name"]
                if arm in signatures and signatures[arm] != signature:
                    raise ValueError("Matched arm metadata differs across languages")
                signatures[arm] = signature
            for point in report.get("curves", []):
                rows.append({"language": language, "arm": report["arm"]["name"], "step": point["step"],
                    "status": report["status"],
                    "comprehension_accuracy": point["dev"]["comprehension_accuracy"],
                    "valid_generation_accuracy": point["dev"]["valid_generation_accuracy"],
                    **{key: point["dev"].get(key) for key in
                       ("comprehension_hits", "comprehension_items", "valid_generation_hits", "generation_scenes")},
                    "meets_original_dev_targets": (
                        point["dev"]["comprehension_accuracy"] >= suite["plan"]["comprehension_min"]
                        and point["dev"]["valid_generation_accuracy"] >= suite["plan"]["production_min"]),
                    "contrasts": point.get("contrasts", {}).get("categories", {})})
    missing = [language for language in LANGUAGES_ORDER if language not in seen]
    return {"schema_version": 1, "kind": VERSION + "-comparison", "excluded_from_research": True,
            "eligible_for_fresh_experiment": False, "status": "incomplete" if missing or failed else "completed",
            "missing_languages": missing, "failed_arms": failed, "rows": rows,
            "interpretation": "Raw development comparisons only; no automatic winning arm or native qualification."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/neural.example.json"))
    parser.add_argument("--language", choices=LANGUAGES_ORDER)
    parser.add_argument("--budgets", default="1200,2400")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--max-seconds-per-arm", type=int, default=600)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--aggregate-dir", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.aggregate_dir is not None:
        paths = sorted(args.aggregate_dir.rglob("suite.json"))
        if not paths:
            raise ValueError("No suite reports found")
        result = aggregate_suites([json.loads(p.read_text()) for p in paths])
        path = Path(args.output or "runs/composition-suite-summary.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_json(path, result)
        print(json.dumps(result), flush=True)
        return 0
    if args.language is None:
        parser.error("--language is required")
    plan = SuitePlan(budgets=tuple(int(n) for n in args.budgets.split(",")),
                     arms=tuple(args.arms.split(",")), max_seconds_per_arm=args.max_seconds_per_arm,
                     max_seconds=args.max_seconds)
    path = run_suite(ExperimentConfig.load(args.config), args.language, plan,
                     args.output or "runs/composition-suite")
    print(f"Saved development-only suite: {path}", flush=True)
    return 0 if json.loads((path / "suite.json").read_text())["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
