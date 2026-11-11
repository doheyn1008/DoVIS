"""Development-only Monte Carlo planning for the EXISTING first-contact gate.

One draw is a pair's joint (0..4, 0..4) correct-answer counts. Clusters are
independent across pairs; within-pair dependence and direction coupling are
explicit assumptions, not estimates from trained learners. No model runs occur.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import integer
from .metrics import (SCREENING_VERSION, classify_screening_interval, initial_analysis_group,
                      screening_interval, wilson)
from .profiles import RESEARCH_EVALUATION_MAX_ITEMS, bounded
from .storage import now, revision, write_json
from .world import COLORS, SHAPES


def hit_distribution(mean: float, correlation: float) -> tuple[float, ...]:
    """Mixture of Binomial(4,p) and all-four-shared Bernoulli(p); ICC equals rho."""
    bounded(mean, "mean accuracy", 0, 1)
    bounded(correlation, "within-pair correlation", 0, 1)
    result = [(1 - correlation) * math.comb(4, k) * mean**k * (1 - mean)**(4 - k) for k in range(5)]
    result[0] += correlation * (1 - mean)
    result[4] += correlation * mean
    return tuple(result)


def distribution_stats(probabilities) -> dict:
    """ICC is implied under exchangeable item outcomes conditional on the count."""
    expected = sum(i * p for i, p in enumerate(probabilities))
    variance = sum(p * (i - expected)**2 for i, p in enumerate(probabilities))
    mean = expected / 4
    icc = (variance / (4 * mean * (1 - mean)) - 1) / 3 if 0 < mean < 1 else None
    return {"raw_mean": mean, "cluster_count_variance": variance, "within_pair_icc": icc}


def joint_hit_distribution(a, b, shared_probability: float) -> list[tuple[int, int, float]]:
    """Mix independent counts and a shared-uniform quantile coupling of counts.

    shared_probability is a mixture probability, NOT Pearson correlation.
    Identical marginals with shared_probability=1 have identical counts.
    """
    bounded(shared_probability, "direction shared probability", 0, 1)
    result = []
    left_a = 0.0
    for i, pa in enumerate(a):
        right_a, left_b = left_a + pa, 0.0
        for j, pb in enumerate(b):
            right_b = left_b + pb
            shared = max(0, min(right_a, right_b) - max(left_a, left_b))
            p = (1 - shared_probability) * pa * pb + shared_probability * shared
            if p > 0:
                result.append((i, j, p))
            left_b = right_b
        left_a = right_a
    return result


@dataclass(frozen=True)
class ScreeningScenario:
    id: str
    means: tuple[float, float] = (.25, .25)
    within_pair_correlation: float = 0
    direction_shared_probability: float = 0
    cluster_model: str = "binomial_shared"
    hits_a: tuple[float, ...] | None = None
    hits_b: tuple[float, ...] | None = None

    def validate(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Scenario id must be a nonempty string")
        bounded(self.direction_shared_probability, "direction shared probability", 0, 1)
        if self.cluster_model == "binomial_shared":
            if len(self.means) != 2 or self.hits_a is not None or self.hits_b is not None:
                raise ValueError("binomial_shared needs two means and no custom hit distributions")
            for mean in self.means:
                bounded(mean, "mean accuracy", 0, 1)
            bounded(self.within_pair_correlation, "within-pair correlation", 0, 1)
        elif self.cluster_model == "custom_hits":
            for probs in (self.hits_a, self.hits_b):
                if probs is None or len(probs) != 5:
                    raise ValueError("Custom hit distributions must have five probabilities for 0..4 hits")
                for p in probs:
                    bounded(p, "hit probability", 0, 1)
                if not math.isclose(sum(probs), 1, abs_tol=1e-12, rel_tol=0):
                    raise ValueError("Hit probabilities must sum to one")
        else:
            raise ValueError("Unsupported cluster model")
        return self

    def distributions(self):
        if self.cluster_model == "custom_hits":
            return self.hits_a, self.hits_b
        return tuple(hit_distribution(p, self.within_pair_correlation) for p in self.means)

    def to_dict(self):
        result = {"id": self.id, "cluster_model": self.cluster_model,
                  "direction_shared_probability": self.direction_shared_probability}
        if self.cluster_model == "custom_hits":
            result.update(hits_a=list(self.hits_a), hits_b=list(self.hits_b))
        else:
            result.update(means=list(self.means), within_pair_correlation=self.within_pair_correlation)
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Scenario must be an object")
        data = dict(value)
        try:
            if data.get("cluster_model") == "custom_hits" and {"means", "within_pair_correlation"} & data.keys():
                raise ValueError("Custom hit probabilities already determine the mean and within-pair ICC")
            for key in ("means", "hits_a", "hits_b"):
                if key in data:
                    if not isinstance(data[key], list):
                        raise ValueError(f"{key} must be an array")
                    data[key] = tuple(data[key])
            return cls(**data).validate()
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid scenario: {exc}") from exc


def default_scenarios():
    return (ScreeningScenario("chance-independent"),
            ScreeningScenario("chance-within-pair-shared", within_pair_correlation=1),
            ScreeningScenario("chance-both-directions-shared", within_pair_correlation=1, direction_shared_probability=1),
            ScreeningScenario("weak-transfer", means=(.30, .25), within_pair_correlation=1),
            ScreeningScenario("above-band", means=(.45, .25), within_pair_correlation=1),
            ScreeningScenario("exactly-one-hit-per-pair", cluster_model="custom_hits",
                              hits_a=(0, 1, 0, 0, 0), hits_b=(0, 1, 0, 0, 0)))


@dataclass(frozen=True)
class ScreeningPlan:
    candidate_pairs: tuple[int, ...] = (98, 200, 320)
    scenarios: tuple[ScreeningScenario, ...] = field(default_factory=default_scenarios)
    chance_margin: float = .15
    replicates: int = 2000
    seed: int = 19001
    max_pair_draws: int = 50_000_000

    def validate(self):
        if not self.candidate_pairs or len(set(self.candidate_pairs)) != len(self.candidate_pairs):
            raise ValueError("Supply unique candidate pair counts")
        for n in self.candidate_pairs:
            integer(n, "candidate pairs", 1, 10000)
        if not self.scenarios or len({s.id for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("Supply scenarios with unique IDs")
        for s in self.scenarios:
            s.validate()
        bounded(self.chance_margin, "chance margin in raw accuracy units", .001, .24)
        integer(self.replicates, "Monte Carlo replicates", 1, 100000)
        integer(self.seed, "planning seed", 0, 2**32 - 1)
        integer(self.max_pair_draws, "pair draw budget", 1, 1_000_000_000)
        if len(self.candidate_pairs) * len(self.scenarios) > 256:
            raise ValueError("At most 256 scenario/size combinations per plan")
        if self.pair_draws > self.max_pair_draws:
            raise ValueError(f"Plan needs {self.pair_draws} cluster draws; exceeds explicit max_pair_draws")
        return self

    @property
    def pair_draws(self):
        return sum(self.candidate_pairs) * len(self.scenarios) * self.replicates

    def to_dict(self):
        return {"schema_version": 1, "candidate_pairs": list(self.candidate_pairs),
                "scenarios": [s.to_dict() for s in self.scenarios], "chance_margin": self.chance_margin,
                "replicates": self.replicates, "seed": self.seed, "max_pair_draws": self.max_pair_draws}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Screening plan must be an object")
        data = dict(value)
        version = data.pop("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported screening plan version")
        try:
            if "candidate_pairs" in data:
                if not isinstance(data["candidate_pairs"], list):
                    raise ValueError("candidate_pairs must be an array")
                data["candidate_pairs"] = tuple(data["candidate_pairs"])
            if "scenarios" in data:
                if not isinstance(data["scenarios"], list):
                    raise ValueError("scenarios must be an array")
                data["scenarios"] = tuple(ScreeningScenario.from_dict(s) for s in data["scenarios"])
            return cls(**data).validate()
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid screening plan: {exc}") from exc


def simulate_screening(plan: ScreeningPlan, progress=None) -> dict:
    plan.validate()
    results = []
    for scenario in plan.scenarios:
        a, b = scenario.distributions()
        stats_a, stats_b = distribution_stats(a), distribution_stats(b)
        joint = joint_hit_distribution(a, b, scenario.direction_shared_probability)
        covariance = sum(p * (i - 4 * stats_a["raw_mean"]) * (j - 4 * stats_b["raw_mean"]) for i, j, p in joint)
        variance_product = stats_a["cluster_count_variance"] * stats_b["cluster_count_variance"]
        for n in plan.candidate_pairs:
            # Case-local streams: adding/reordering other candidate scenarios does not change this result.
            case = {"scenario": scenario.to_dict(), "pairs": n, "margin": plan.chance_margin, "seed": plan.seed}
            seed = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
            rng = random.Random(seed)
            population, weights = [(i, j) for i, j, _ in joint], [p for _, _, p in joint]
            outcomes, labels_a, labels_b = Counter(), Counter(), Counter()
            for _ in range(plan.replicates):
                draws = rng.choices(population, weights, k=n)
                scores = (sum(d[0] for d in draws) / (4 * n), sum(d[1] for d in draws) / (4 * n))
                labels = [classify_screening_interval(*screening_interval(p, n), plan.chance_margin) for p in scores]
                labels_a[labels[0]] += 1; labels_b[labels[1]] += 1
                outcomes[initial_analysis_group(labels)[0]] += 1
            probability = outcomes["primary"] / plan.replicates
            row = {"scenario_id": scenario.id, "pairs": n, "items_per_direction": 4 * n,
                   "raw_chance_band": [.25 - plan.chance_margin, .25 + plan.chance_margin],
                   "interval_radius": math.sqrt(math.log(80) / (2 * n)),
                   "marginal_a": {"hit_probabilities": a, **stats_a}, "marginal_b": {"hit_probabilities": b, **stats_b},
                   "direction_shared_probability": scenario.direction_shared_probability,
                   "implied_direction_count_correlation": covariance / math.sqrt(variance_product) if variance_product > 0 else None,
                   "monte_carlo_replicates": plan.replicates, "primary_count": outcomes["primary"],
                   "primary_probability": probability, "primary_mc_wilson95": wilson(outcomes["primary"], plan.replicates),
                   "primary_mc_standard_error": math.sqrt(probability * (1 - probability) / plan.replicates),
                   "direction_a_equivalent_probability": labels_a["chance_equivalent"] / plan.replicates,
                   "direction_b_equivalent_probability": labels_b["chance_equivalent"] / plan.replicates,
                   "analysis_group_counts": {k: outcomes[k] for k in ("primary", "positive_baseline_extension", "uncertain")},
                   "direction_label_counts": {"a": dict(labels_a), "b": dict(labels_b)},
                   "within_current_world_pair_count": n <= math.comb(len(COLORS) * len(SHAPES), 2),
                   "within_current_pretest_item_limit": 4 * n <= RESEARCH_EVALUATION_MAX_ITEMS}
            results.append(row)
            if progress:
                progress(row)
    return {"schema_version": 1, "kind": "screening_design_simulation", "excluded_from_research": True,
            "conditional_on_other_eligibility_checks_passed": True,
            "screening_version": SCREENING_VERSION, "plan": plan.to_dict(), "pair_draws": plan.pair_draws,
            "results": results, "assumptions": [
                "Hypothetical iid pair clusters; no learner was trained and no actual language performance is inferred.",
                "Inclusion probabilities are conditional on L1 qualification, provenance and caption checks already passing.",
                "Four correlated items per pair; direction coupling is explicit and preserves marginal hit distributions.",
                "Custom hit distributions allow negative ICC, including exactly one hit per pair.",
                "Custom-distribution ICC assumes exchangeable items: uniform correct positions given the cluster count.",
                "Monte Carlo intervals describe simulation error only, not real-world model uncertainty.",
                "Current real screening rule is reused. No margin, world, evaluation policy, or protocol limit is changed.",
                "World/item-limit flags are necessary conditions only; other reserved splits also need space.",
                "No finite-population correction, model-replica variation or empirically fitted cluster model is assumed."]}


def write_screening_plan(plan: ScreeningPlan, output="runs/diagnostics", progress=None) -> Path:
    # Validation and simulation finish before publishing a report; never writes a research summary/config.
    report = simulate_screening(plan, progress)
    directory = Path(output).resolve() / ("screening-" + uuid.uuid4().hex[:10])
    directory.mkdir(parents=True)
    write_json(directory / "report.json", {**report, "created_at": now(), "revision": revision()})
    fields = ("scenario_id", "pairs", "items_per_direction", "interval_radius", "primary_probability",
              "primary_mc_standard_error", "direction_a_equivalent_probability", "direction_b_equivalent_probability",
              "implied_direction_count_correlation", "within_current_world_pair_count", "within_current_pretest_item_limit")
    with (directory / "probabilities.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(report["results"])
    return directory
