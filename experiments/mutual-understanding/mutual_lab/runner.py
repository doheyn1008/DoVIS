"""The complete frozen-probe -> baseline -> dialogue -> evaluation lifecycle."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

from .agents import EndpointAgent, SymbolicAgent, observe, speak
from .backend import ChatTransport
from .config import ExperimentConfig
from .images import choice_sheet
from .metrics import evaluate, summarize_points
from .storage import RunStore, now, write_json
from .world import make_dataset


class ExperimentFailed(RuntimeError):
    def __init__(self, message: str, run_path: Path):
        super().__init__(message)
        self.run_path = run_path


def run_experiment(config: ExperimentConfig, output: str | Path = "runs",
                   progress: Callable[[dict], None] | None = None) -> Path:
    config.validate()
    dataset = make_dataset(config)
    if config.request_upper_bound() > config.max_requests:
        raise ValueError(f"Worst-case {config.request_upper_bound()} requests exceeds max_requests="
                         f"{config.max_requests}; reduce the run or raise the explicit budget")
    store = RunStore(output, config)
    transport = ChatTransport(config, store.event)
    agents = ([SymbolicAgent(agent, config) for agent in config.agents] if config.mode == "demo" else
              [EndpointAgent(agent, config, transport) for agent in config.agents])
    points: list[dict] = []
    exchange = 0
    reason = "max_exchanges"
    limitations = (["Synthetic demonstration: template speakers, perfect visual features, naive Bayes listeners.",
                    "Demo scores do not measure human or neural-model language learning."] if config.mode == "demo" else
                   ["Fixed weights; only bounded conversation history changes (in-context adaptation).",
                    "Initial foreign-language knowledge is unverified; native-language prompting does not remove it.",
                    "Model servers must be stateless between requests and isolated from external tools."])
    limitations += ["A restricted colored-shape task is not general language comprehension.",
                    "Checkpoint intervals locate threshold times only to sampled exchanges.",
                    "Repeated use of fixed probes supports comparison but is not an independent final test.",
                    "Wilson intervals are descriptive item-level intervals; probes within object pairs are correlated."]

    def snapshot(status: str, error: str | None = None) -> dict:
        summary = {"schema_version": 1, "run_id": store.run_id, "status": status, "mode": config.mode,
                   "seed": config.seed, "started_at": store.started_at, "updated_at": now(),
                   "exchanges_completed": exchange, "utterances_completed": 2 * exchange,
                   "threshold": config.threshold, "sustained_checks": config.sustained_checks,
                   "stop_reason": reason if status == "completed" else None,
                   "dataset_sha256": dataset.fingerprint, "transport": transport.budget.to_dict(),
                   "agents": [agent.config.__dict__ for agent in agents],
                   "directions": summarize_points(points, config.threshold, config.sustained_checks) if points else [],
                   "limitations": limitations, "error": error}
        write_json(store.path / "summary.json", summary)
        return summary

    def checkpoint() -> None:
        current = evaluate(agents, dataset, captions, exchange, config.seed)
        points.extend(current)
        store.metrics(points)
        store.event({"type": "evaluation", "exchange": exchange, "points": current})
        snapshot("running")
        if progress:
            progress({"run_id": store.run_id, "exchange": exchange, "points": current})

    def should_stop() -> bool:
        if not config.stop_on_threshold:
            return False
        directions = summarize_points(points, config.threshold, config.sustained_checks)
        return len(directions) == 2 and all(not item["right_censored"] for item in directions)

    snapshot("running")
    try:
        dataset_data = dataset.to_dict()
        dataset_data["fingerprint"] = dataset.fingerprint
        dataset_data["train_images"] = [store.image(scene) for scene in dataset.train]
        for index, probe in enumerate(dataset.probes):
            image_name = f"probe-{index:04d}.png"
            (store.path / "assets" / image_name).write_bytes(choice_sheet(probe.choices))
            dataset_data["probes"][index]["choice_image"] = image_name
        write_json(store.path / "dataset.json", dataset_data)
        store.event({"type": "run_started", "mode": config.mode, "seed": config.seed,
                     "planned_request_upper_bound": config.request_upper_bound()})
        # Both speakers are frozen at initialization, before any partner exposure.
        captions = {agent.config.name: [speak(agent, probe.scene, frozen=True) for probe in dataset.probes]
                    for agent in agents}
        write_json(store.path / "frozen_captions.json", captions)
        checkpoint()  # Measure multilingual pretraining / chance before any adaptation.
        if should_stop():
            reason = "sustained_threshold_both_directions"
        else:
            rng = random.Random(config.seed + 104729)
            order = list(dataset.train)
            rng.shuffle(order)
            for next_exchange in range(1, config.max_exchanges + 1):
                offset = (next_exchange - 1) % len(order)
                if offset == 0 and next_exchange > 1:
                    rng.shuffle(order)
                scene = order[offset]
                first, second = (agents[0], agents[1]) if next_exchange % 2 else (agents[1], agents[0])
                first_text = speak(first, scene)
                second_text = speak(second, scene, first_text)
                # Commit the shared exchange only after both utterances complete.
                observe(second, scene, first_text)
                observe(first, scene, second_text)
                exchange = next_exchange
                store.event({"type": "dialogue", "exchange": exchange, "image": store.image(scene),
                             "messages": [{"speaker": first.config.name, "language": first.config.language, "text": first_text},
                                          {"speaker": second.config.name, "language": second.config.language, "text": second_text}]})
                if exchange % config.eval_every == 0 or exchange == config.max_exchanges:
                    checkpoint()
                    if should_stop():
                        reason = "sustained_threshold_both_directions"
                        break
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
