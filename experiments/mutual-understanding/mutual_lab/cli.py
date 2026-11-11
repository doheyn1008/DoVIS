"""Small dependency-free CLI for demos, endpoint runs, sweeps, and local inspection."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import uuid
from dataclasses import replace
from pathlib import Path

from .config import AgentConfig, ExperimentConfig
from .languages import LANGUAGES
from .runner import ExperimentFailed, run_experiment
from .storage import write_json
from .world import make_dataset


def progress(event: dict) -> None:
    if "diagnostic_step" in event:
        values = " | ".join(f"{name}: understand {event[name]['comprehension_accuracy']:.1%}, "
                             f"generate {event[name]['generation_exact_accuracy']:.1%}"
                             for name in ("train", "dev"))
        print(f"Diagnostic step {event['diagnostic_step']} | {values}", flush=True)
    if "native_step" in event:
        print(f"L1 {event['language']} step {event['native_step']} | loss {event['loss']:.4f}", flush=True)
    if "points" in event:
        scores = " | ".join(f"{p['source']}->{p['listener']}: {p['accuracy']:.1%} ({p['hits']}/{p['n']})"
                            for p in event["points"])
        print(f"{event['exchange']:>5} exchanges | {scores}", flush=True)


def sweep(config: ExperimentConfig, seeds: list[int], output: str | Path, quiet: bool) -> Path:
    if config.mode == "neural":
        raise ValueError("Use grid for neural replicas; sweep seeds are not independent L1 models")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Supply at least one unique seed")
    for seed in seeds:
        replace(config, seed=seed).validate()
    paths = []
    rows = []
    for seed in seeds:
        path = run_experiment(replace(config, seed=seed), output, None if quiet else progress)
        paths.append(str(path))
        result = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for direction in result["directions"]:
            rows.append({"seed": seed, "run_id": path.name, **direction})
    aggregate = []
    for source, listener in dict.fromkeys((row["source"], row["listener"]) for row in rows):
        group = [row for row in rows if (row["source"], row["listener"]) == (source, listener)]
        reached = [row["sustained_threshold_started_exchange"] for row in group if not row["right_censored"]]
        aggregate.append({"source": source, "listener": listener, "n_runs": len(group),
                          "n_reached": len(reached), "n_right_censored": len(group) - len(reached),
                          "median_threshold_exchange_among_reached_only": statistics.median(reached) if reached else None,
                          "mean_last_accuracy": statistics.mean(row["last_accuracy"] for row in group)})
    directory = Path(output).resolve() / "sweeps" / ("sweep-" + uuid.uuid4().hex[:10])
    directory.mkdir(parents=True)
    write_json(directory / "aggregate.json", {"runs": paths, "directions": aggregate,
                                              "note": "Censored runs are counted; their caps are not threshold times."})
    write_json(directory / "directions.json", rows)
    with (directory / "directions.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = ("seed", "run_id", "source", "listener", "baseline_accuracy", "last_accuracy",
                "sustained_threshold_started_exchange", "sustained_threshold_confirmed_exchange", "right_censored",
                "censoring_exchange", "normalized_auc_observed")
        writer = csv.DictWriter(stream, keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return directory


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Grounded dialogue and bidirectional comprehension lab")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a synthetic pipeline check without any model/API calls")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--lang-a", choices=tuple(LANGUAGES), default="en")
    demo.add_argument("--lang-b", choices=tuple(LANGUAGES), default="ko")
    demo.add_argument("--max-exchanges", type=int, default=80)
    demo.add_argument("--eval-every", type=int, default=5)
    demo.add_argument("--eval-items", type=int, default=24)
    demo.add_argument("--no-early-stop", action="store_true")
    demo.add_argument("--output", default="runs")
    demo.add_argument("--quiet", action="store_true")
    run = commands.add_parser("run", help="Run a validated JSON configuration")
    run.add_argument("config")
    run.add_argument("--output", default="runs")
    run.add_argument("--quiet", action="store_true")
    validate = commands.add_parser("validate", help="Check a configuration and estimate request count without sending requests")
    validate.add_argument("config")
    repeated = commands.add_parser("sweep", help="Repeat a JSON configuration over seeds and aggregate censored outcomes")
    repeated.add_argument("config")
    repeated.add_argument("--seeds", required=True, help="Comma-separated unique integers, e.g. 7,11,23")
    repeated.add_argument("--output", default="runs")
    repeated.add_argument("--quiet", action="store_true")
    serve = commands.add_parser("serve", help="Open the local dashboard; its launch button runs synthetic demos only")
    serve.add_argument("--runs-dir", default="runs")
    serve.add_argument("--port", type=int, default=8765)
    native = commands.add_parser("train-native", help="Train one monolingual GRU and record native qualification")
    native.add_argument("config")
    native.add_argument("--language", required=True, choices=tuple(LANGUAGES))
    native.add_argument("--output", required=True, help="Native .pt checkpoint path")
    native.add_argument("--initialization-seed", type=int, default=7)
    native.add_argument("--native-training-seed", type=int, default=11)
    native.add_argument("--quiet", action="store_true")
    for name in ("grid-plan", "grid"):
        batch = commands.add_parser(name, help="Plan or execute the hierarchical all-pairs grid")
        batch.add_argument("config")
        if name == "grid":
            batch.add_argument("--output", default="runs")
            batch.add_argument("--resume-dir")
            batch.add_argument("--limit", type=int, help="Run only this many pending tasks")
            batch.add_argument("--quiet", action="store_true")
    resume = commands.add_parser("resume", help="Restore both persistent learners from their last complete checkpoint")
    resume.add_argument("run_path")
    resume.add_argument("--output", default="runs")
    resume.add_argument("--quiet", action="store_true")
    diagnostic = commands.add_parser("diagnose-native", help="Development-only small train/dev learning diagnosis; excluded from research")
    diagnostic.add_argument("config")
    diagnostic.add_argument("--language", choices=tuple(LANGUAGES), required=True)
    diagnostic.add_argument("--steps", type=int, default=600)
    diagnostic.add_argument("--eval-every", type=int, default=100)
    diagnostic.add_argument("--train-pairs", type=int, default=2)
    diagnostic.add_argument("--dev-pairs", type=int, default=2)
    diagnostic.add_argument("--data-seed", type=int, default=19003)
    diagnostic.add_argument("--initialization-seed", type=int, default=19007)
    diagnostic.add_argument("--training-seed", type=int, default=19009)
    diagnostic.add_argument("--output", default="runs/diagnostics")
    diagnostic.add_argument("--quiet", action="store_true")
    screening = commands.add_parser("plan-screening", help="Simulate inclusion probabilities under explicit pair-cluster assumptions")
    screening.add_argument("plan", help="Planning JSON; does not change experiment settings")
    screening.add_argument("--output", default="runs/diagnostics")
    screening.add_argument("--quiet", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "serve":
            from .server import serve
            serve(args.runs_dir, args.port)
            return 0
        if args.command == "plan-screening":
            from .screening_plan import ScreeningPlan, write_screening_plan
            plan = ScreeningPlan.from_dict(json.loads(Path(args.plan).read_text(encoding="utf-8")))
            def show_case(row):
                print(f"{row['scenario_id']} | {row['pairs']} pairs | both equivalent {row['primary_probability']:.1%}", flush=True)
            path = write_screening_plan(plan, args.output, None if args.quiet else show_case)
            print(f"Saved development-only plan: {path}")
            return 0
        if args.command == "diagnose-native":
            from .diagnostics import DiagnosticFailed, DiagnosticOptions, run_native_diagnostic
            config = ExperimentConfig.load(args.config)
            options = DiagnosticOptions(steps=args.steps, eval_every=args.eval_every,
                train_pairs=args.train_pairs, dev_pairs=args.dev_pairs, data_seed=args.data_seed,
                initialization_seed=args.initialization_seed, training_seed=args.training_seed)
            try:
                path = run_native_diagnostic(config, args.language, options, args.output,
                                              None if args.quiet else progress)
            except DiagnosticFailed as exc:
                print(f"Diagnostic failed: {exc}\nPartial report: {exc.report_path}", file=sys.stderr)
                return 1
            print(f"Saved development-only diagnosis: {path}")
            return 0
        if args.command in {"grid-plan", "grid"}:
            from .grid import GridConfig, plan_grid, run_grid
            grid = GridConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
            if args.command == "grid-plan":
                print(json.dumps(plan_grid(grid), ensure_ascii=False, indent=2))
            else:
                if args.limit is not None and args.limit < 1:
                    raise ValueError("--limit must be positive")
                path = run_grid(grid, args.output, resume_dir=args.resume_dir, limit=args.limit,
                                progress=None if args.quiet else progress)
                print(f"Saved: {path}")
                outcomes = json.loads((path / "outcomes.json").read_text())
                return 1 if outcomes["failed"] or outcomes["cancelled"] else 0
            return 0
        if args.command == "resume":
            from .research import run_neural_experiment
            config = ExperimentConfig.load(Path(args.run_path) / "config.json")
            if config.mode != "neural":
                raise ValueError("Only neural runs have persistent restart checkpoints")
            path = run_neural_experiment(config, args.output, None if args.quiet else progress,
                                         resume_from=args.run_path)
            print(f"Saved: {path}")
            return 0
        if args.command == "demo":
            config = replace(ExperimentConfig(), seed=args.seed, max_exchanges=args.max_exchanges,
                             eval_every=args.eval_every, eval_items=args.eval_items,
                             stop_on_threshold=not args.no_early_stop,
                             agents=(AgentConfig("A", args.lang_a), AgentConfig("B", args.lang_b))).validate()
        else:
            config = ExperimentConfig.load(args.config)
        if args.command == "train-native":
            from .neural import train_native
            path = train_native(config, args.language, args.output, initialization_seed=args.initialization_seed,
                                native_training_seed=args.native_training_seed,
                                model_replica_id=config.model_replica_id if config.model_replica_id != "unassigned" else "r0",
                                progress=None if args.quiet else progress)
            print(f"Saved: {path}")
            return 0
        if args.command == "validate":
            from .world import make_research_dataset
            dataset = make_research_dataset(config) if config.mode == "neural" else make_dataset(config)
            print(json.dumps({"mode": config.mode, "train_scenes": len(dataset.train),
                              "heldout_scenes": sum(len(dataset.view(s).probes) for s in ("curve", "pretest", "final", "native"))
                                  if config.mode == "neural" else len(dataset.holdout), "eval_items_per_direction": len(dataset.probes),
                              "request_upper_bound": config.request_upper_bound(),
                              "request_budget": config.max_requests,
                              "budget_sufficient": config.request_upper_bound() <= config.max_requests,
                              "dataset_sha256": dataset.fingerprint}, indent=2))
            return 0 if config.request_upper_bound() <= config.max_requests else 1
        if args.command == "sweep":
            path = sweep(config, [int(seed.strip()) for seed in args.seeds.split(",")], args.output, args.quiet)
        else:
            path = run_experiment(config, args.output, None if args.quiet else progress)
        print(f"Saved: {path}")
        return 0
    except ExperimentFailed as exc:
        print(f"Run failed: {exc}\nPartial logs: {exc.run_path}", file=sys.stderr)
        return 1
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
