"""Read-only evaluations, uncertainty, and explicitly right-censored threshold times."""

from __future__ import annotations

import math
import random

from .agents import choose
from .world import Dataset


def corrected_accuracy(raw: float, choices: int = 4) -> float:
    """chance-correction-v1: deliberately preserves below-chance negative values."""
    return (raw - 1 / choices) / (1 - 1 / choices)


def raw_threshold(threshold: float, scale: str, choices: int = 4) -> float:
    if scale not in {"raw", "corrected"}:
        raise ValueError("threshold scale must be raw or corrected")
    return threshold if scale == "raw" else 1 / choices + threshold * (1 - 1 / choices)


def wilson(hits: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = hits / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


SCREENING_VERSION = "pair-mean-hoeffding-family-v1"


def screening_interval(mean: float, pairs: int) -> tuple[float, float]:
    """Existing two-direction family bound, shared by real screening and planning."""
    if type(pairs) is not int or pairs < 1 or not 0 <= mean <= 1:
        raise ValueError("Screening requires a mean in [0,1] and at least one pair")
    radius = math.sqrt(math.log(80) / (2 * pairs))
    return max(0, mean - radius), min(1, mean + radius)


def classify_screening_interval(low: float, high: float, margin: float) -> str:
    if .25 - margin <= low and high <= .25 + margin:
        return "chance_equivalent"
    if low > .25 + margin:
        return "positive_baseline"
    if high < .25 - margin:
        return "below_chance"
    return "uncertain"


def initial_analysis_group(labels: list[str]) -> tuple[str, str]:
    if len(labels) == 2 and all(label == "chance_equivalent" for label in labels):
        return "primary", "both independent pretest intervals within preregistered chance band"
    if any(label == "positive_baseline" for label in labels):
        return "positive_baseline_extension", "at least one direction above preregistered chance band"
    return "uncertain", "pretest precision insufficient or outside inclusion band"


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
        # Four probes in an unordered-pair cluster are correlated. Screening uses
        # independent pair means and a bounded Hoeffding interval, with Bonferroni
        # coverage for the two directions. Small pilots will often be inconclusive.
        clusters = {}
        for probe, trial in zip(dataset.probes, trials):
            clusters.setdefault(probe.scene.group, []).append(int(trial["correct"]))
        cluster_mean = sum(sum(v) / len(v) for v in clusters.values()) / len(clusters)
        screening_low, screening_high = screening_interval(cluster_mean, len(clusters))
        points.append({"exchange": exchange, "source": source.config.name,
                       "listener": listener.config.name, "source_language": source.config.language,
                       "listener_language": listener.config.language, "listener_exposures": listener.exposures,
                       "learner_native_language": listener.config.language,
                       "target_language": source.config.language, "speaker_language": source.config.language,
                       "accuracy": hits / n, "raw_accuracy": hits / n,
                       "corrected_accuracy": corrected_accuracy(hits / n),
                       "correction_version": "chance-correction-v1", "choice_count": 4,
                       "corrected_wilson_low": corrected_accuracy(low),
                       "corrected_wilson_high": corrected_accuracy(high), "hits": hits, "n": n,
                       "screening_low": screening_low,
                       "screening_high": screening_high, "screening_clusters": len(clusters),
                       "screening_interval_method": "pair-mean Hoeffding; two-direction family coverage >=95%",
                       "invalid_answers": sum(not trial["valid"] for trial in trials),
                       "wilson_low": low, "wilson_high": high, "trials": trials})
    if states != [agent.state_digest() for agent in agents]:
        raise RuntimeError("Evaluation mutated learner state; results are invalid")
    return points


def common_curve_metrics(points: list[dict], horizon: int | None, scale: str = "raw") -> dict:
    key = "accuracy" if scale == "raw" else "corrected_accuracy"
    result = {"horizon": horizon, "scale": scale, "normalized_auc": None, "fixed_exposure_accuracy": None,
              "available": False, "reason": "common horizon not preregistered"}
    if horizon is None:
        return result
    data = [p for p in points if p["exchange"] <= horizon]
    if not data or data[0]["exchange"] != 0 or data[-1]["exchange"] != horizon:
        return {**result, "reason": "common horizon not fully observed; no extrapolation"}
    area = sum((b["exchange"] - a["exchange"]) * (a[key] + b[key]) / 2 for a, b in zip(data, data[1:]))
    return {**result, "available": True, "reason": None, "normalized_auc": area / horizon,
            "fixed_exposure_accuracy": data[-1][key]}


def compare_common_auc(directions: list[dict]) -> list[float]:
    """Prevent accidentally comparing averages over unequal observed exposure intervals."""
    windows = [d["common_curve"] for d in directions]
    if (not windows or not all(w["available"] for w in windows)
            or len({(w["horizon"], w["scale"]) for w in windows}) != 1):
        raise ValueError("AUC comparisons require the same fully observed preregistered horizon and scale")
    return [w["normalized_auc"] for w in windows]


def summarize_direction(points: list[dict], threshold: float, sustained_checks: int,
                        scale: str = "raw", common_horizon: int | None = None,
                        status: str = "completed") -> dict:
    streak = 0
    started = confirmed = first = None
    for index, point in enumerate(points):
        if point["accuracy"] >= raw_threshold(threshold, scale):
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
            "learner_native_language": last["listener_language"],
            "target_language": last["source_language"], "speaker_language": last["source_language"],
            "threshold_scale": scale, "threshold_raw_equivalent": raw_threshold(threshold, scale),
            "baseline_accuracy": points[0]["accuracy"], "last_accuracy": last["accuracy"],
            "baseline_corrected_accuracy": corrected_accuracy(points[0]["accuracy"]),
            "last_corrected_accuracy": corrected_accuracy(last["accuracy"]),
            "last_wilson_interval": [last["wilson_low"], last["wilson_high"]],
            "first_observed_threshold_exchange": first,
            "sustained_threshold_started_exchange": started,
            "sustained_threshold_confirmed_exchange": confirmed,
            "right_censored": confirmed is None and status == "completed",
            "threshold_outcome": "reached" if confirmed is not None else
                {"completed": "right_censored", "failed": "failed", "cancelled": "cancelled"}.get(status, "in_progress"),
            "censoring_exchange": end if confirmed is None and status == "completed" else None,
            "normalized_auc_observed": auc, "auc_observed_end": end,
            "auc_observed_comparable": False,
            "common_curve": common_curve_metrics(points, common_horizon, scale), "eval_items": last["n"]}


def summarize_points(points: list[dict], threshold: float, sustained_checks: int,
                     scale: str = "raw", common_horizon: int | None = None,
                     status: str = "completed") -> list[dict]:
    pairs = list(dict.fromkeys((point["source"], point["listener"]) for point in points))
    return [summarize_direction([point for point in points if (point["source"], point["listener"]) == pair],
                                threshold, sustained_checks, scale, common_horizon, status) for pair in pairs]


def classify_initial(pretest: list[dict], native_eligibility: list[bool], margin: float,
                     *, verified_foreign_exposure: bool) -> dict:
    """Preregistered equivalence-band inclusion, never 'p > .05 implies zero'."""
    checks = []
    for point in pretest:
        low, high = point["screening_low"], point["screening_high"]
        label = classify_screening_interval(low, high, margin)
        checks.append({"learner_native_language": point["listener_language"],
                       "target_language": point["source_language"], "raw_accuracy": point["accuracy"],
                       "corrected_accuracy": point["corrected_accuracy"],
                       "wilson_interval": [point["wilson_low"], point["wilson_high"]],
                       "screening_interval": [low, high], "classification": label})
    if not verified_foreign_exposure or len(native_eligibility) != 2 or not all(native_eligibility):
        group, reason = "ineligible", "L1 qualification or foreign-training provenance requirement unmet"
    else:
        group, reason = initial_analysis_group([p["classification"] for p in checks])
    return {"analysis_group": group, "reason": reason, "raw_chance_band": [.25 - margin, .25 + margin],
            "interval_method": "pair-mean Hoeffding; two-direction family coverage >=95%; small pilots have low power",
            "foreign_training_exposure_verified_zero": verified_foreign_exposure, "directions": checks}
