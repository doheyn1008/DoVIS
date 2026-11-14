"""Coverage-controlled, development-only joint L1 training and budget comparison.

Only the existing research TRAIN pool is partitioned here. These weights never
qualify a research learner; qualification requires frozen rules and fresh seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ExperimentConfig, integer
from .diagnostics import evaluate_diagnostic, safe_json, semantic_index
from .languages import LANGUAGES
from .storage import now, revision
from .world import COLORS, SHAPES, Object, make_research_dataset, native_captions, variants

LANGUAGES_ORDER = ("ko", "ja", "en", "es")
CHECKPOINT_FORMAT = "mutual-native-preparation-development-v1"
SPLIT_RULE = "seeded-pair-order-first-coverage-valid-dev-subset-v1"


@dataclass(frozen=True)
class PreparationPlan:
    budgets: tuple[int, ...] = (1200, 2400, 4800)
    comprehension_min: float = .9
    production_min: float = .9
    train_pairs: int = 10
    dev_pairs: int = 4
    data_seed: int = 29003
    initialization_seed: int = 29007
    training_seed: int = 29009
    max_seconds: int = 900

    def validate(self, config):
        config.validate()
        if config.mode != "neural":
            raise ValueError("Native preparation requires a neural experiment blueprint")
        if not isinstance(self.budgets, (tuple, list)) or not self.budgets:
            raise ValueError("Supply at least one development budget")
        for budget in self.budgets:
            integer(budget, "development budget", 1, 10000)
        if list(self.budgets) != sorted(set(self.budgets)):
            raise ValueError("Budgets must be unique and strictly increasing")
        for key in ("train_pairs", "dev_pairs"):
            integer(getattr(self, key), key, 1, 35)
        if self.train_pairs * 2 < len(COLORS) * len(SHAPES) or self.dev_pairs < 2:
            raise ValueError("Pair counts cannot satisfy the required object/color/shape coverage")
        integer(self.max_seconds, "development runtime limit", 1, 900)
        seeds = [self.data_seed, self.initialization_seed, self.training_seed]
        for seed in seeds:
            integer(seed, "development seed", 0, 2**32 - 1)
        if len(set(seeds)) != 3 or set(seeds) & {config.seed, *(config.effective_seed(k)
                for k in ("scene_split", "dialogue", "evaluation"))}:
            raise ValueError("Development seeds must be distinct and separate from research seeds")
        for key in ("comprehension_min", "production_min"):
            value = getattr(self, key)
            if type(value) not in {int, float} or not math.isfinite(value) or not .25 <= value <= 1:
                raise ValueError(f"{key} must be a finite development target in [.25, 1]")
        return self


def feature_coverage(scenes):
    objects = {o for scene in scenes for o in (scene.subject, scene.object)}
    return {"objects": [asdict(o) for o in sorted(objects)],
            "colors": sorted({o.color for o in objects}),
            "shapes": sorted({o.shape for o in objects}),
            "relations": sorted({scene.relation for scene in scenes})}


def build_preparation_splits(config, plan=PreparationPlan()):
    """Choose once by semantic coverage, never by model scores or language."""
    plan.validate(config)
    pairs = sorted({scene.group for scene in make_research_dataset(config).train})
    if plan.train_pairs + plan.dev_pairs > len(pairs):
        raise ValueError("Preparation subsets exceed the existing research TRAIN pool")
    random.Random(plan.data_seed).shuffle(pairs)
    required = {Object(color, shape) for color in COLORS for shape in SHAPES}
    for dev in itertools.combinations(pairs, plan.dev_pairs):
        train = tuple(pair for pair in pairs if pair not in dev)[:plan.train_pairs]
        seen = set(itertools.chain.from_iterable(train))
        dev_objects = set(itertools.chain.from_iterable(dev))
        if (seen == required and {o.color for o in dev_objects} == set(COLORS)
                and {o.shape for o in dev_objects} == set(SHAPES)):
            return {name: tuple(scene for pair in pool for scene in variants(pair))
                    for name, pool in (("train", train), ("dev", dev))}
    raise ValueError("No coverage-valid split exists in the research TRAIN pool")


def classify_all_languages(reports):
    """A development budget passes only when all four matched runs pass.

    Missing/failed runs are retained as incomplete, never silently excluded.
    Caption hashes differ across languages; semantic pool and code must match.
    """
    by_language = {}
    keys = ("plan", "model_profile", "config_sha256", "semantic_data_sha256", "source_sha256")
    reference = None
    for report in reports:
        language = report["language"]
        if language not in LANGUAGES_ORDER or language in by_language:
            raise ValueError("Unknown or duplicate language in preparation comparison")
        signature = json.dumps({key: report[key] for key in keys}, sort_keys=True)
        if reference is not None and signature != reference:
            raise ValueError("Preparation comparison metadata mismatch")
        reference = signature
        by_language[language] = report
    missing = [language for language in LANGUAGES_ORDER if language not in by_language]
    failed = [language for language, report in by_language.items() if report["status"] != "completed"]
    plan = next(iter(by_language.values()))["plan"] if by_language else asdict(PreparationPlan())
    rows = []
    for step in plan["budgets"]:
        languages = {}
        for language in LANGUAGES_ORDER:
            report = by_language.get(language)
            matches = [row for row in report["curves"] if row["step"] == step] if report else []
            metrics = matches[0]["dev"] if len(matches) == 1 else None
            understanding = metrics.get("comprehension_accuracy") if metrics else None
            production = metrics.get("valid_generation_accuracy") if metrics else None
            finite = all(type(value) in {float, int} and math.isfinite(value)
                         and 0 <= value <= 1 for value in (understanding, production))
            languages[language] = {"comprehension_accuracy": understanding,
                "valid_generation_accuracy": production,
                **{key: metrics.get(key) if metrics else None for key in
                   ("comprehension_hits", "comprehension_items", "valid_generation_hits", "generation_scenes")},
                "passes": bool(finite and understanding >= plan["comprehension_min"]
                               and production >= plan["production_min"])}
        rows.append({"step": step, "languages": languages,
                     "all_languages_pass": all(row["passes"] for row in languages.values())})
    complete = not missing and not failed
    common = next((row["step"] for row in rows if row["all_languages_pass"]), None) if complete else None
    return {"schema_version": 1, "kind": "native_preparation_comparison",
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "status": ("incomplete" if not complete else "development_targets_met" if common is not None
                       else "development_targets_not_met"), "common_budget": common,
            "missing_languages": missing, "failed_languages": failed, "plan": plan, "per_budget": rows,
            "interpretation": "Candidate development targets only; not native qualification or L2 acquisition evidence"}


def run_preparation(config, language, plan=PreparationPlan(), output="runs/native-preparation", progress=print):
    plan.validate(config)
    if language not in LANGUAGES_ORDER or LANGUAGES[language]["corpus"] != config.native_training_profile.corpus_version:
        raise ValueError("Language corpus is not ready for native preparation")
    splits = build_preparation_splits(config, plan)
    from .neural import NativeModel, configure_cpu, digest, native_examples, torch, nn
    configure_cpu()
    torch.manual_seed(plan.training_seed)
    model = NativeModel(config.model_profile, plan.initialization_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.native_training_profile.learning_rate)
    rng = random.Random(plan.training_seed)
    examples = {name: native_examples(model, scenes, language) for name, scenes in splits.items()}
    semantic_data = {name: [scene.to_dict() for scene in scenes] for name, scenes in splits.items()}
    data = {name: [{"scene": scene.to_dict(), "captions": native_captions(scene, language)}
                   for scene in scenes] for name, scenes in splits.items()}
    directory = Path(output).resolve() / (language + "-" + uuid.uuid4().hex[:10])
    directory.mkdir(parents=True)
    safe_json(directory / "dataset.json", data)
    safe_json(directory / "config.json", config.to_dict())
    root = Path(__file__).resolve().parent
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob("*.py"))}
    report = {"schema_version": 1, "kind": "native_preparation_development",
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "status": "running", "created_at": now(), "revision": revision(), "language": language,
        "training_languages": [language], "foreign_training_languages": [],
        "plan": asdict(plan), "model_profile": asdict(config.model_profile),
        "native_training_profile": asdict(config.native_training_profile),
        "config_sha256": digest(config.to_dict()), "source_sha256": sources,
        "semantic_data_sha256": digest(semantic_data), "data_sha256": digest(data),
        "split_rule": SPLIT_RULE, "feature_coverage": {k: feature_coverage(v) for k, v in splits.items()},
        "pair_keys": {k: [[asdict(o) for o in pair] for pair in sorted({s.group for s in v})]
                      for k, v in splits.items()}, "protected_splits_used": [],
        "data_source": "Coverage-selected pair-disjoint subsets of the research TRAIN partition only",
        "device": "cpu", "torch_threads": 1, "torch_version": str(torch.__version__),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "producer_status": "Jointly trained from random initialization with understanding; no frozen native generator yet",
        "initial_understanding_sha256": digest(model.understanding.state_dict()),
        "initial_producer_sha256": digest(model.producer.state_dict()),
        "effective_training_budget": {"batch_size": config.native_training_profile.batch_size,
            "learning_rate": config.native_training_profile.learning_rate,
            "maximum_steps": plan.budgets[-1],
            "maximum_sampled_native_examples": plan.budgets[-1] * config.native_training_profile.batch_size,
            "loss_weights": {"comprehension": 1, "production": 1}, "gradient_clip_norm": 1},
        "training_steps_completed": 0, "curves": [], "checkpoints": [],
        "warning": "Development target is fixed before this run. Protected qualification is untouched; fresh independent L1 models are still required."}
    generations, comprehension_predictions, step = [], [], 0
    begun = time.monotonic()
    index = semantic_index(language)

    def snapshot():
        report["elapsed_seconds"] = time.monotonic() - begun
        report["final_understanding_sha256"] = digest(model.understanding.state_dict())
        report["final_producer_sha256"] = digest(model.producer.state_dict())
        safe_json(directory / "report.json", report)
        safe_json(directory / "generations.json", generations)
        safe_json(directory / "comprehension_predictions.json", comprehension_predictions)

    def checkpoint():
        name = f"step-{step:06d}.pt"
        path = directory / name
        payload = {"format": CHECKPOINT_FORMAT, "kind": "development_only",
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "step": step, "training_steps_completed": step,
            "language": language, "plan": asdict(plan),
            "training_languages": [language], "foreign_training_languages": [],
            "model_profile": asdict(config.model_profile), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "training_rng_state": rng.getstate(),
            "torch_rng_state": torch.random.get_rng_state(), "source_sha256": sources,
            "semantic_data_sha256": report["semantic_data_sha256"], "data_sha256": report["data_sha256"],
            "config_sha256": report["config_sha256"], "revision": report["revision"]}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        manifest = {"format": CHECKPOINT_FORMAT, "kind": "development_only", "excluded_from_research": True,
            "eligible_for_fresh_experiment": False, "step": step, "path": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        safe_json(path.with_suffix(".json"), manifest)
        report["checkpoints"] = [item for item in report["checkpoints"] if item["path"] != name] + [manifest]

    def evaluate():
        before = digest({"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "rng": rng.getstate(), "torch_rng": torch.random.get_rng_state()})
        row = {"step": step}
        for name, scenes in splits.items():
            metrics, outputs = evaluate_diagnostic(model, examples[name], scenes, language, index,
                                                   config.native_training_profile.batch_size)
            valid = sum(item["exact_match"] and item["terminated"] and not item["invalid_encoding"]
                        and not item["special_tokens_in_body"] for item in outputs)
            metrics.update(valid_generation_hits=valid, valid_generation_accuracy=valid / len(outputs))
            row[name] = metrics
            generations.extend({"step": step, "split": name, **item} for item in outputs)
            for item_index, example in enumerate(examples[name]):
                scene = scenes[item_index // 4]
                predicted = model.predict(example["text"], example["choices"])
                meaning = variants(scene.group)[predicted]
                errors = []
                if meaning.subject != scene.subject:
                    errors.append("subject_object_order")
                if meaning.relation != scene.relation:
                    errors.append("relation_error")
                comprehension_predictions.append({"step": step, "split": name,
                    "scene": scene.to_dict(), "text": example["text"], "template_index": item_index % 4,
                    "answer": example["answer"], "prediction": predicted,
                    "correct": predicted == example["answer"],
                    "predicted_scene": meaning.to_dict(), "error_flags": errors})
        if before != digest({"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "rng": rng.getstate(), "torch_rng": torch.random.get_rng_state()}):
            raise RuntimeError("Preparation evaluation changed learning state")
        report["curves"].append(row)
        checkpoint(); snapshot()
        if progress:
            progress(f"{language} step={step} train-understand={row['train']['comprehension_accuracy']:.1%} "
                     f"dev-understand={row['dev']['comprehension_accuracy']:.1%} "
                     f"dev-generate={row['dev']['valid_generation_accuracy']:.1%}", flush=True)

    snapshot()
    try:
        evaluate()
        for next_step in range(1, plan.budgets[-1] + 1):
            if time.monotonic() - begun >= plan.max_seconds:
                report["status"] = "time_limit"
                checkpoint(); snapshot()
                return directory
            batch = rng.choices(examples["train"], k=config.native_training_profile.batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss = model.grounding_loss(batch) + model.production_loss(batch)
            if not math.isfinite(float(loss.detach())):
                raise ValueError("Non-finite native preparation loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
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
        checkpoint(); snapshot()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/neural.example.json"))
    parser.add_argument("--language", choices=LANGUAGES_ORDER)
    parser.add_argument("--budgets", default="1200,2400,4800")
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--aggregate", nargs="*", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.aggregate is not None:
        result = classify_all_languages([json.loads(path.read_text()) for path in args.aggregate])
        path = Path(args.output or "runs/native-preparation/comparison.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_json(path, result)
        print(json.dumps(result), flush=True)
        return 0
    if args.language is None:
        parser.error("--language is required for training")
    plan = PreparationPlan(budgets=tuple(int(value) for value in args.budgets.split(",")), max_seconds=args.max_seconds)
    path = run_preparation(ExperimentConfig.load(args.config), args.language, plan,
                           args.output or "runs/native-preparation")
    print(f"Saved development-only native preparation: {path}", flush=True)
    return 0 if json.loads((path / "report.json").read_text())["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
