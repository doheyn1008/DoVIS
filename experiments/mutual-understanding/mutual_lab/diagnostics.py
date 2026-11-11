"""Development-only native learning diagnosis. Never creates qualified L1 models.

Only the research TRAIN partition is split into small diagnostic train/dev pools.
Protected qualification/screening/curve/final pairs are never scored or trained.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import random
import re
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ExperimentConfig, integer
from .images import scene_png
from .languages import LANGUAGES
from .storage import atomic_text, now, revision
from .world import COLORS, SHAPES, Object, Scene, make_research_dataset, native_captions, variants


@dataclass(frozen=True)
class DiagnosticOptions:
    steps: int = 600
    eval_every: int = 100
    train_pairs: int = 2
    dev_pairs: int = 2
    data_seed: int = 19003
    initialization_seed: int = 19007
    training_seed: int = 19009

    def validate(self, experiment: ExperimentConfig):
        experiment.validate()
        if experiment.mode != "neural":
            raise ValueError("Native diagnosis requires a schema-2 neural experiment blueprint")
        integer(self.steps, "diagnostic steps", 0, 100000)
        integer(self.eval_every, "diagnostic evaluation interval", 1, 100000)
        for key in ("train_pairs", "dev_pairs"):
            integer(getattr(self, key), key, 1, 35)
        seeds = [self.data_seed, self.initialization_seed, self.training_seed]
        for value in seeds:
            integer(value, "diagnostic seed", 0, 2**32 - 1)
        if len(set(seeds)) != 3 or set(seeds) & {experiment.seed, *(experiment.effective_seed(k)
                for k in ("scene_split", "dialogue", "evaluation"))}:
            raise ValueError("Use distinct development seeds, separate from the experiment seeds")
        return self


def diagnostic_splits(experiment: ExperimentConfig, options: DiagnosticOptions):
    options.validate(experiment)
    research = make_research_dataset(experiment)
    pairs = sorted({s.group for s in research.train})
    if options.train_pairs + options.dev_pairs > len(pairs):
        raise ValueError("Diagnostic train/dev pairs exceed the existing research training pool")
    random.Random(options.data_seed).shuffle(pairs)
    return {"train": tuple(s for p in pairs[:options.train_pairs] for s in variants(p)),
            "dev": tuple(s for p in pairs[options.train_pairs:options.train_pairs + options.dev_pairs] for s in variants(p))}


def normalized_surface(text: str) -> str:
    """Only case, Unicode NFC, whitespace and terminal punctuation are ignored."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split()).rstrip(".!?。！？")


def semantic_index(language: str):
    """Researcher-side finite grammar lookup, never passed to the learner.

    Counterfactual meanings identify recognized object/relation errors. Unknown
    paraphrases remain unscored; this is not a general natural-language parser.
    """
    index = {}
    objects = [Object(c, s) for c in COLORS for s in SHAPES]
    for pair in itertools.combinations(objects, 2):
        for scene in variants(pair):
            for text in native_captions(scene, language):
                index.setdefault(normalized_surface(text), set()).add(scene)
    return index


def grade_generation(trace: dict, expected: Scene, language: str, index) -> dict:
    text = trace["text"]
    exact = text in native_captions(expected, language)
    matches = index.get(normalized_surface(text), set())
    parsed = next(iter(matches)) if len(matches) == 1 else None
    semantic_correct = parsed == expected if parsed is not None else None
    errors = []
    if parsed is None:
        errors.append("ambiguous_surface" if matches else "unrecognized_surface")
    elif semantic_correct:
        if not exact:
            errors.append("surface_only")
    else:
        if parsed.subject == expected.object and parsed.object == expected.subject:
            errors.append("subject_object_order")
        if parsed.subject.color != expected.subject.color or parsed.object.color != expected.object.color:
            errors.append("color_error")
        if parsed.subject.shape != expected.subject.shape or parsed.object.shape != expected.object.shape:
            errors.append("shape_error")
        if parsed.group != expected.group:
            errors.append("object_identity_error")
        if parsed.relation != expected.relation:
            errors.append("relation_error")
    if not trace["terminated"]:
        errors.append("no_eos")
    if trace["invalid_encoding"]:
        errors.append("invalid_encoding")
    if trace["special_tokens_in_body"]:
        errors.append("special_tokens_in_body")
    if re.search(r"(.{2,16})\1\1", text):
        errors.append("repetition_heuristic")
    return {"scene": expected.to_dict(), **trace, "exact_match": exact,
            "recognized_semantic_correct": semantic_correct,
            "recognized_meaning": parsed.to_dict() if parsed else None, "error_flags": errors}


def visual_diagnostic(model, splits) -> dict:
    import torch
    rows = []
    for name, scenes in splits.items():
        groups = sorted({s.group for s in scenes})
        minimum_distance, minimum_margin, correct, collisions = float("inf"), float("inf"), 0, 0
        with torch.no_grad():
            for group in groups:
                vectors = model.visual([scene_png(s) for s in variants(group)])
                similarities = vectors @ vectors.T
                for i in range(4):
                    others = [j for j in range(4) if j != i]
                    minimum_distance = min(minimum_distance, float(torch.linalg.vector_norm(vectors[i] - vectors[others], dim=1).min()))
                    minimum_margin = min(minimum_margin, float(similarities[i, i] - similarities[i, others].max()))
                    correct += int(similarities[i].argmax()) == i
                collisions += sum(float(torch.linalg.vector_norm(vectors[i] - vectors[j])) <= 1e-6
                                  for i in range(4) for j in range(i))
        rows.append({"split": name, "pairs": len(groups), "images": 4 * len(groups),
                     "minimum_within_pair_projection_distance": minimum_distance,
                     "minimum_self_retrieval_cosine_margin": minimum_margin,
                     "near_collisions_at_1e_6": collisions, "image_identity_retrieval_accuracy": correct / (4 * len(groups))})
    return {"rows": rows, "interpretation": "Pixel-only noncollapse check, not evidence of semantic vision or language learning"}


def evaluate_diagnostic(model, examples, scenes, language, index, batch_size):
    import torch
    import torch.nn.functional as F
    comprehension_loss, production_loss, hits, token_hits, token_count = 0.0, 0.0, 0, 0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            comprehension_loss += float(model.grounding_loss(batch)) * len(batch)
            ids, _ = model.tokenizer.batch([e["text"] for e in batch])
            targets = ids[:, 1:]
            logits, _ = model.producer(ids[:, :-1], model.visual([e["image"] for e in batch]))
            production_loss += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                                    ignore_index=0, reduction="sum"))
            mask = targets != 0
            token_hits += int(((logits.argmax(-1) == targets) & mask).sum())
            token_count += int(mask.sum())
            hits += sum(model.predict(e["text"], e["choices"]) == e["answer"] for e in batch)
        generated = [grade_generation(model.generate_with_trace(scene_png(s)), s, language, index) for s in scenes]
    n = len(generated)
    flags = Counter(flag for row in generated for flag in row["error_flags"])
    metrics = {"comprehension_loss": comprehension_loss / len(examples), "production_loss": production_loss / token_count,
               "teacher_forced_token_accuracy": token_hits / token_count, "comprehension_hits": hits,
               "comprehension_items": len(examples), "comprehension_accuracy": hits / len(examples),
               "generation_scenes": n, "generation_exact_hits": sum(r["exact_match"] for r in generated),
               "generation_exact_accuracy": sum(r["exact_match"] for r in generated) / n,
               "recognized_semantic_correct": sum(r["recognized_semantic_correct"] is True for r in generated),
               "semantic_success_lower_bound": sum(r["recognized_semantic_correct"] is True for r in generated) / n,
               "semantic_unknown_count": sum(r["recognized_semantic_correct"] is None for r in generated),
               "generation_error_counts": dict(flags)}
    if not all(math.isfinite(metrics[key]) for key in ("comprehension_loss", "production_loss", "teacher_forced_token_accuracy")):
        raise ValueError("Diagnostic evaluation produced non-finite metrics")
    return metrics, generated


class DiagnosticFailed(RuntimeError):
    def __init__(self, message, report_path):
        super().__init__(message)
        self.report_path = report_path


def safe_json(path: Path, data):
    # Keep raw token IDs and escaped malformed BMP output inspectable too.
    atomic_text(path, json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False) + "\n")


def run_native_diagnostic(experiment: ExperimentConfig, language: str, options: DiagnosticOptions,
                          output="runs/diagnostics", progress=None) -> Path:
    options.validate(experiment)
    if language not in LANGUAGES or LANGUAGES[language]["corpus"] != experiment.native_training_profile.corpus_version:
        raise ValueError("Diagnostic corpus is not ready for this language/profile")
    splits = diagnostic_splits(experiment, options)
    # All tensor dependencies are optional; importing the CLI/planner remains dependency-free.
    from .neural import NativeModel, configure_cpu, digest, native_examples, torch, nn
    configure_cpu()
    model = NativeModel(experiment.model_profile, options.initialization_seed)
    examples = {name: native_examples(model, scenes, language) for name, scenes in splits.items()}
    index = semantic_index(language)
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment.native_training_profile.learning_rate)
    rng = random.Random(options.training_seed)
    initial_weights = digest(model.state_dict())
    directory = Path(output).resolve() / ("native-diagnostic-" + uuid.uuid4().hex[:10])
    directory.mkdir(parents=True)
    data = {name: [{"scene": s.to_dict(), "captions": native_captions(s, language)} for s in scenes]
            for name, scenes in splits.items()}
    coverage = {name: {"colors": sorted({o.color for s in scenes for o in (s.subject, s.object)}),
                       "shapes": sorted({o.shape for s in scenes for o in (s.subject, s.object)}),
                       "relations": sorted({s.relation for s in scenes})} for name, scenes in splits.items()}
    safe_json(directory / "dataset.json", data)
    report = {"schema_version": 1, "kind": "native_learning_diagnostic", "excluded_from_research": True,
              "status": "running", "language": language, "created_at": now(), "revision": revision(),
              "options": asdict(options), "model_profile": asdict(experiment.model_profile),
              "native_training_profile": asdict(experiment.native_training_profile),
              "effective_training_budget": {"steps": options.steps,
                  "batch_size": experiment.native_training_profile.batch_size,
                  "learning_rate": experiment.native_training_profile.learning_rate,
                  "sampled_native_examples": options.steps * experiment.native_training_profile.batch_size,
                  "loss_weights": {"comprehension": 1, "production": 1}, "gradient_clip_norm": 1},
              "data_sha256": digest(data), "initial_weights_sha256": initial_weights,
              "feature_coverage": coverage,
              "unseen_dev_features": {key: sorted(set(coverage["dev"][key]) - set(coverage["train"][key]))
                                       for key in ("colors", "shapes", "relations")},
              "parameter_count": sum(p.numel() for p in model.parameters()), "torch_version": str(torch.__version__),
              "data_source": "Small pair-disjoint train/dev subsets of the research TRAIN partition only",
              "assumptions": ["Development seeds must be excluded from subsequent independent research model replicas.",
                 "Dev performance includes unseen feature/word learning if unseen_dev_features is nonempty; inspect coverage first.",
                 "No native qualification, first-contact screening, curve or final evaluation data is scored or trained.",
                 "Semantic grading recognizes four authored forms plus NFC/case/whitespace/terminal punctuation normalization only.",
                 "Unrecognized text is unknown, not automatically a semantic error. Semantic success is a lower bound.",
                 "Error flags overlap; repetition detection is a heuristic. Raw generation tokens and EOS/encoding flags are retained.",
                 "No native/acquisition checkpoint or research summary is produced. This report cannot qualify a research learner."],
              "curves": [], "visual": None, "training_steps_completed": 0}
    generations = []
    begun = time.monotonic()

    def snapshot():
        report["elapsed_seconds"] = time.monotonic() - begun
        safe_json(directory / "report.json", report)
        safe_json(directory / "generations.json", generations)
        fields = ("step", "split", "comprehension_loss", "production_loss", "teacher_forced_token_accuracy",
                  "comprehension_accuracy", "generation_exact_accuracy", "semantic_success_lower_bound", "semantic_unknown_count")
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(report["curves"])
        atomic_text(directory / "learning_curve.csv", stream.getvalue())

    def state_digest():
        return digest({"weights": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng.getstate()})

    def checkpoint(step):
        before = state_digest()
        current = {}
        for name in ("train", "dev"):
            metrics, generated = evaluate_diagnostic(model, examples[name], splits[name], language, index,
                                                      experiment.native_training_profile.batch_size)
            row = {"step": step, "split": name, **metrics}
            report["curves"].append(row)
            generations.extend({"step": step, "split": name, **r} for r in generated)
            current[name] = metrics
        if before != state_digest():
            raise RuntimeError("Diagnostic evaluation changed weights, optimizer or training RNG")
        snapshot()
        if progress:
            progress({"diagnostic_step": step, **current})

    snapshot()
    try:
        report["visual"] = visual_diagnostic(model, splits)
        checkpoint(0)
        for step in range(1, options.steps + 1):
            batch = rng.choices(examples["train"], k=experiment.native_training_profile.batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss = model.grounding_loss(batch) + model.production_loss(batch)
            if not math.isfinite(float(loss.detach())):
                raise ValueError("Diagnostic training produced a non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            report["training_steps_completed"] = step
            if step % options.eval_every == 0 or step == options.steps:
                checkpoint(step)
        report["final_weights_sha256"] = digest(model.state_dict())
        report["status"] = "completed"
        final_train = next(r for r in reversed(report["curves"]) if r["split"] == "train")
        report["full_small_train_fit_observed"] = (final_train["comprehension_accuracy"] == 1 and
            final_train["generation_exact_accuracy"] == 1 and not final_train["generation_error_counts"])
        snapshot()
        return directory
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        snapshot()
        raise DiagnosticFailed(report["error"], directory) from exc
