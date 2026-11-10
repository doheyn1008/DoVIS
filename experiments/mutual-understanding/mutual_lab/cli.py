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
from .runner import ExperimentFailed, run_experiment
from .storage import write_json
from .world import make_dataset


def progress(event: dict) -> None:
    if "points" in event:
        scores = " | ".join(f"{p['source']}->{p['listener']}: {p['accuracy']:.1%} ({p['hits']}/{p['n']})"
                            for p in event["points"])
        print(f"{event['exchange']:>5} exchanges | {scores}", flush=True)


def sweep(config: ExperimentConfig, seeds: list[int], output: str | Path, quiet: bool) -> Path:
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
    demo.add_argument("--lang-a", choices=("en", "ko", "ja"), default="en")
    demo.add_argument("--lang-b", choices=("en", "ko", "ja"), default="ko")
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
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "serve":
            from .server import serve
            serve(args.runs_dir, args.port)
            return 0
        if args.command == "demo":
            config = replace(ExperimentConfig(), seed=args.seed, max_exchanges=args.max_exchanges,
                             eval_every=args.eval_every, eval_items=args.eval_items,
                             stop_on_threshold=not args.no_early_stop,
                             agents=(AgentConfig("A", args.lang_a), AgentConfig("B", args.lang_b))).validate()
        else:
            config = ExperimentConfig.load(args.config)
        if args.command == "validate":
            dataset = make_dataset(config)
            print(json.dumps({"mode": config.mode, "train_scenes": len(dataset.train),
                              "heldout_scenes": len(dataset.holdout), "eval_items_per_direction": len(dataset.probes),
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
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
