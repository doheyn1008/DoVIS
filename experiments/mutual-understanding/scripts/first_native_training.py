"""Bounded first L1 understanding run; every output is development-only.

Run from the experiment directory: python scripts/first_native_training.py
The producer is frozen at random initialization. These checkpoints cannot qualify
an L1 learner or be supplied to a fresh cross-language experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mutual_lab.config import ExperimentConfig
from mutual_lab.diagnostics import DiagnosticOptions, diagnostic_splits, safe_json
from mutual_lab.languages import LANGUAGES
from mutual_lab.storage import now, revision
from mutual_lab.world import native_captions


@dataclass(frozen=True)
class TrainingOptions:
    steps_per_stage: int = 1200
    eval_every: int = 200
    max_seconds: int = 480
    data_seed: int = 19003
    initialization_seed: int = 19007
    training_seed: int = 19009

    def diagnostic_options(self):
        return DiagnosticOptions(steps=self.steps_per_stage, eval_every=self.eval_every,
            train_pairs=1, dev_pairs=1, data_seed=self.data_seed,
            initialization_seed=self.initialization_seed, training_seed=self.training_seed)

    def validate(self, config):
        self.diagnostic_options().validate(config)
        for name, upper in (("steps_per_stage", 2000), ("max_seconds", 540)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in [1, {upper}]")
        return self


def development_data(config, options):
    """Expose the exact tiny pools without importing optional tensor dependencies."""
    options.validate(config)
    return diagnostic_splits(config, options.diagnostic_options())


def finite_norm(tensors, torch):
    value = math.sqrt(sum(float(t.detach().double().square().sum()) for t in tensors))
    if not math.isfinite(value):
        raise ValueError("Non-finite gradient, parameter or update norm")
    return value


def evaluate(model, examples, torch):
    with torch.no_grad():
        ids, lengths = model.tokenizer.batch([e["text"] for e in examples])
        representations = model.understanding(ids, lengths)
        similarities = representations @ representations.T
        other = ~torch.eye(len(examples), dtype=torch.bool)
        predictions = [model.predict(e["text"], e["choices"]) for e in examples]
        confusion = [[0] * 4 for _ in range(4)]
        rows = []
        for example, predicted in zip(examples, predictions):
            confusion[example["answer"]][predicted] += 1
            rows.append({"text": example["text"], "scene_key": example["scene_key"],
                "template_index": example["template_index"], "answer": example["answer"],
                "prediction": predicted, "correct": predicted == example["answer"]})
        result = {"loss": float(model.grounding_loss(examples)),
            "accuracy": sum(r["correct"] for r in rows) / len(rows), "items": len(rows),
            "representation_coordinate_variance": float(representations.var(dim=0, unbiased=False).mean()),
            "minimum_text_cosine": float(similarities[other].min()),
            "maximum_text_cosine": float(similarities[other].max()),
            "confusion_answer_by_prediction": confusion, "predictions": rows}
    if not all(math.isfinite(result[key]) for key in ("loss", "accuracy",
            "representation_coordinate_variance", "minimum_text_cosine", "maximum_text_cosine")):
        raise ValueError("Non-finite understanding evaluation")
    return result


def run_training(config, language="ko", options=TrainingOptions(),
                 output="runs/first-native-training", progress=print):
    options.validate(config)
    if language not in LANGUAGES:
        raise ValueError("Unknown language")
    splits = development_data(config, options)
    from mutual_lab.neural import NativeModel, configure_cpu, digest, native_examples, torch, nn
    configure_cpu()
    model = NativeModel(config.model_profile, options.initialization_seed)
    for parameter in model.producer.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.understanding.parameters(),
                                 lr=config.native_training_profile.learning_rate)
    rng = random.Random(options.training_seed)
    initial_producer = digest(model.producer.state_dict())
    initial_understanding = {k: v.detach().clone() for k, v in model.understanding.state_dict().items()}
    examples = {}
    for name, scenes in splits.items():
        examples[name] = native_examples(model, scenes, language)
        for i, example in enumerate(examples[name]):
            example.update(scene_key=scenes[i // 4].key, template_index=i % 4)
    data = {name: [{"scene": s.to_dict(), "captions": native_captions(s, language)} for s in scenes]
            for name, scenes in splits.items()}
    directory = Path(output).resolve() / (language + "-" + uuid.uuid4().hex[:10])
    directory.mkdir(parents=True)
    safe_json(directory / "dataset.json", data)
    safe_json(directory / "config.json", config.to_dict())
    sources = (Path(__file__), *(ROOT / "mutual_lab" / name for name in
               ("neural.py", "world.py", "diagnostics.py", "profiles.py", "config.py")))
    report = {"schema_version": 1, "kind": "first_native_understanding_training",
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "status": "running", "created_at": now(), "revision": revision(),
        "language": language, "options": asdict(options), "device": "cpu", "torch_threads": 1,
        "torch_version": str(torch.__version__), "model_profile": asdict(config.model_profile),
        "native_training_profile": asdict(config.native_training_profile),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_understanding_parameters": sum(p.numel() for p in model.understanding.parameters()),
        "producer_status": "random initialization; frozen; no production training or qualification",
        "producer_initial_sha256": initial_producer, "producer_unchanged": True,
        "initial_understanding_sha256": digest(initial_understanding), "data_sha256": digest(data),
        "config_sha256": digest(config.to_dict()),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "data_source": "One train pair and one disjoint dev pair from research TRAIN only",
        "protected_splits_used": [], "loss": "understanding grounding only; temperature 0.1",
        "batch_size": config.native_training_profile.batch_size,
        "gradient_clip_norm": 1, "stages_continue_same_weights_optimizer_and_rng": True,
        "warning": "Tiny-pool memorization diagnostics do not establish L1 fluency or language-pair acquisition results.",
        "training_steps_completed": 0, "curves": [], "checkpoints": []}
    begun = time.monotonic()
    stage_name, stage_step = "not_started", 0
    modules = {"embedding": model.understanding.embedding,
               "gru": model.understanding.gru, "head": model.understanding.head}
    latest_update = None

    def save_checkpoint():
        payload = {"format": "mutual-development-v1", "kind": "development_only",
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "stage": stage_name, "stage_step": stage_step,
            "training_steps_completed": report["training_steps_completed"],
            "language": language, "options": asdict(options), "data_sha256": report["data_sha256"],
            "source_sha256": report["source_sha256"], "revision": report["revision"],
            "model_profile": asdict(config.model_profile), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "training_rng_state": rng.getstate(),
            "torch_rng_state": torch.random.get_rng_state(),
            "producer_status": report["producer_status"], "producer_initial_sha256": initial_producer}
        path = directory / (stage_name + ".pt")
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        manifest = {"format": payload["format"], "kind": payload["kind"],
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": path.name, "stage_step": stage_step,
            "training_steps_completed": report["training_steps_completed"]}
        safe_json(path.with_suffix(".json"), manifest)
        report["checkpoints"] = [r for r in report["checkpoints"] if r["path"] != path.name] + [manifest]

    def snapshot():
        report["elapsed_seconds"] = time.monotonic() - begun
        report["producer_unchanged"] = digest(model.producer.state_dict()) == initial_producer
        report["final_understanding_sha256"] = digest(model.understanding.state_dict())
        report["understanding_update_from_initial_l2"] = finite_norm(
            [v - initial_understanding[k] for k, v in model.understanding.state_dict().items()], torch)
        if not report["producer_unchanged"]:
            raise RuntimeError("Frozen producer changed")
        safe_json(directory / "report.json", report)

    def evaluate_stage(stage_examples):
        before = digest({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng.getstate()})
        row = {"stage": stage_name, "stage_step": stage_step,
            "total_step": report["training_steps_completed"], "latest_sampled_update": latest_update,
            **{name: evaluate(model, pool, torch) for name, pool in stage_examples.items()}}
        if before != digest({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng.getstate()}):
            raise RuntimeError("Evaluation changed learning state")
        report["curves"].append(row)
        save_checkpoint()
        snapshot()
        if progress:
            progress(f"{language} {stage_name} step={stage_step} "
                     f"train={row['train']['accuracy']:.1%} dev={row['dev']['accuracy']:.1%} "
                     f"loss={row['train']['loss']:.4f}", flush=True)

    snapshot()
    try:
        for stage_name, templates in (("one_template", (0,)), ("four_templates", (0, 1, 2, 3))):
            stage_step, latest_update = 0, None
            stage_examples = {name: [e for e in pool if e["template_index"] in templates]
                              for name, pool in examples.items()}
            evaluate_stage(stage_examples)
            for next_step in range(1, options.steps_per_stage + 1):
                if time.monotonic() - begun >= options.max_seconds:
                    report["status"] = "time_limit"
                    save_checkpoint(); snapshot()
                    return directory
                batch = rng.choices(stage_examples["train"], k=config.native_training_profile.batch_size)
                measured = next_step % options.eval_every == 0 or next_step == options.steps_per_stage
                previous = {name: [p.detach().clone() for p in module.parameters()]
                            for name, module in modules.items()} if measured else None
                optimizer.zero_grad(set_to_none=True)
                loss = model.grounding_loss(batch)
                if not math.isfinite(float(loss.detach())):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                if measured:
                    latest_update = {"stage_step": next_step,
                        "gradient_l2_before_clipping": {name: finite_norm(
                            [p.grad for p in module.parameters() if p.grad is not None], torch)
                            for name, module in modules.items()}}
                nn.utils.clip_grad_norm_(model.understanding.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                stage_step = next_step
                report["training_steps_completed"] += 1
                if measured:
                    latest_update["parameter_update_l2"] = {name: finite_norm(
                        [p.detach() - old for p, old in zip(module.parameters(), previous[name])], torch)
                        for name, module in modules.items()}
                    evaluate_stage(stage_examples)
        report["status"] = "completed"
        report["full_small_train_understanding_fit_observed"] = report["curves"][-1]["train"]["accuracy"] == 1
        snapshot()
        return directory
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_checkpoint(); snapshot()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/neural.example.json"))
    parser.add_argument("--language", choices=tuple(LANGUAGES), default="ko")
    parser.add_argument("--steps-per-stage", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--max-seconds", type=int, default=480)
    parser.add_argument("--output", default="runs/first-native-training")
    args = parser.parse_args(argv)
    options = TrainingOptions(steps_per_stage=args.steps_per_stage,
                              eval_every=args.eval_every, max_seconds=args.max_seconds)
    path = run_training(ExperimentConfig.load(args.config), args.language, options, args.output)
    print(f"Saved development-only training: {path}", flush=True)
    report = json.loads((path / "report.json").read_text())
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
