"""Read-only evaluations, uncertainty, and explicitly right-censored threshold times."""

from __future__ import annotations

import math
import random

from .agents import choose
from .world import Dataset


def wilson(hits: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = hits / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate(agents: list, dataset: Dataset, captions: dict[str, list[str]], exchange: int, seed: int) -> list[dict]:
    states = [agent.state_digest() for agent in agents]
    points = []
    for source, listener in ((agents[0], agents[1]), (agents[1], agents[0])):
        trials = []
        for index, probe in enumerate(dataset.probes):
            # Separate random streams keep evaluation from changing dialogue sampling.
            rng = random.Random(f"{seed}:{listener.config.name}:{index}")
            caption = captions[source.config.name][index]
            selected, raw = choose(listener, caption, probe.choices, rng)
            trials.append({"probe": index, "choice": selected, "correct_choice": probe.correct_choice,
                           "correct": selected == probe.correct_choice, "valid": selected is not None,
                           "raw_answer": raw})
        hits = sum(trial["correct"] for trial in trials)
        n = len(trials)
        low, high = wilson(hits, n)
        points.append({"exchange": exchange, "source": source.config.name,
                       "listener": listener.config.name, "source_language": source.config.language,
                       "listener_language": listener.config.language, "listener_exposures": listener.exposures,
                       "accuracy": hits / n, "hits": hits, "n": n,
                       "invalid_answers": sum(not trial["valid"] for trial in trials),
                       "wilson_low": low, "wilson_high": high, "trials": trials})
    if states != [agent.state_digest() for agent in agents]:
        raise RuntimeError("Evaluation mutated learner state; results are invalid")
    return points


def summarize_direction(points: list[dict], threshold: float, sustained_checks: int) -> dict:
    streak = 0
    started = confirmed = first = None
    for index, point in enumerate(points):
        if point["accuracy"] >= threshold:
            if first is None:
                first = point["exchange"]
            streak += 1
            if streak >= sustained_checks and confirmed is None:
                confirmed = point["exchange"]
                started = points[index - sustained_checks + 1]["exchange"]
        else:
            streak = 0
    end = points[-1]["exchange"]
    auc = None
    if end:
        auc = sum((b["exchange"] - a["exchange"]) * (a["accuracy"] + b["accuracy"]) / 2
                  for a, b in zip(points, points[1:])) / end
    last = points[-1]
    return {"source": last["source"], "listener": last["listener"],
            "source_language": last["source_language"], "listener_language": last["listener_language"],
            "baseline_accuracy": points[0]["accuracy"], "last_accuracy": last["accuracy"],
            "last_wilson_interval": [last["wilson_low"], last["wilson_high"]],
            "first_observed_threshold_exchange": first,
            "sustained_threshold_started_exchange": started,
            "sustained_threshold_confirmed_exchange": confirmed,
            "right_censored": confirmed is None,
            "censoring_exchange": end if confirmed is None else None,
            "normalized_auc_observed": auc, "eval_items": last["n"]}


def summarize_points(points: list[dict], threshold: float, sustained_checks: int) -> list[dict]:
    pairs = list(dict.fromkeys((point["source"], point["listener"]) for point in points))
    return [summarize_direction([point for point in points if (point["source"], point["listener"]) == pair],
                                threshold, sustained_checks) for pair in pairs]
