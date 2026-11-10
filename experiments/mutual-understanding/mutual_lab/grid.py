"""Capability profiles x independent L1 replicas x repeated experiences x language pairs."""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .config import AgentConfig, ExperimentConfig, integer
from .languages import LANGUAGES
from .profiles import MODEL_PROFILES, ModelProfile
from .storage import write_json
from .world import make_research_dataset


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class GridConfig:
    languages: tuple[str, ...] = ("ko", "ja", "es", "en")
    model_profiles: tuple[ModelProfile, ...] = (ModelProfile(),)
    model_replicas: int = 2
    experience_repeats: int = 2
    seed: int = 7
    experiment: ExperimentConfig = ExperimentConfig(schema_version=2, stop_on_threshold=False)

    def validate(self):
        if len(self.languages) < 2 or len(set(self.languages)) != len(self.languages):
            raise ValueError("Grid needs at least two unique languages")
        if any(l not in LANGUAGES or LANGUAGES[l]["status"] != "controlled-pilot" for l in self.languages):
            raise ValueError("Language corpus not ready")
        if not self.model_profiles or len({p.id for p in self.model_profiles}) != len(self.model_profiles):
            raise ValueError("Model profile IDs must be unique")
        for p in self.model_profiles:
            p.validate()
        integer(self.model_replicas, "model_replicas", 1, 1000)
        integer(self.experience_repeats, "experience_repeats", 1, 1000)
        integer(self.seed, "grid seed", 0, 2**32 - 1)
        self.experiment.validate()
        make_research_dataset(self.experiment)
        return self

    def to_dict(self):
        return {"schema_version": 1, "languages": list(self.languages),
                "model_profiles": [asdict(p) for p in self.model_profiles],
                "model_replicas": self.model_replicas, "experience_repeats": self.experience_repeats,
                "seed": self.seed, "experiment": self.experiment.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Grid must be an object")
        d = dict(value)
        if d.pop("schema_version", 1) != 1:
            raise ValueError("Unsupported grid schema")
        try:
            if "languages" in d:
                if not isinstance(d["languages"], list):
                    raise ValueError("languages must be a list")
                d["languages"] = tuple(d["languages"])
            if "model_profiles" in d:
                d["model_profiles"] = tuple(MODEL_PROFILES[p] if isinstance(p, str) else ModelProfile(**p)
                                            for p in d["model_profiles"])
            if "experiment" in d:
                d["experiment"] = ExperimentConfig.from_dict(d["experiment"])
            return cls(**d).validate()
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"Invalid grid: {exc}") from exc


def plan_grid(grid: GridConfig) -> dict:
    grid.validate()
    tasks = []
    for profile, replica, experience, pair in itertools.product(
            grid.model_profiles, range(grid.model_replicas), range(grid.experience_repeats),
            itertools.combinations(grid.languages, 2)):
        # Same initialization and native sampling rule for all L1s within each independent model replica.
        # Profile repeats also receive paired seeds. Experience seeds do not alter the L1 checkpoint.
        task = {"model_profile": asdict(profile), "model_profile_sha256": fingerprint(asdict(profile)),
                "model_replica_id": f"r{replica}", "experience_replica_id": f"d{experience}",
                "initialization_seed": (grid.seed + 1009 * replica) % 2**32,
                "native_training_seed": (grid.seed + 1013 * replica + 1) % 2**32,
                "scene_split_seed": grid.experiment.effective_seed("scene_split"),
                "dialogue_seed": (grid.seed + 104729 * experience + 3) % 2**32,
                "evaluation_seed": (grid.seed + 13007 * experience + 5) % 2**32,
                "languages": list(pair), "pair_id": "--".join(sorted(pair))}
        task["task_id"] = fingerprint(task)[:20]
        tasks.append(task)
    c, r, d, l = len(grid.model_profiles), grid.model_replicas, grid.experience_repeats, len(grid.languages)
    return {"schema_version": 1, "grid_sha256": fingerprint(grid.to_dict()), "tasks": tasks,
            "budget": {"pair_runs": len(tasks), "direction_curves": 2 * len(tasks),
                "native_checkpoints": c * r * l,
                "native_optimizer_steps": c * r * l * grid.experiment.native_training_profile.steps,
                "acquisition_updates_upper_bound": len(tasks) * 2 * grid.experiment.max_exchanges *
                    grid.experiment.acquisition_profile.updates_per_exposure,
                "formula": "C * R * D * L * (L - 1) / 2",
                "runtime_estimate": "not inferred from model size; measure a pilot on the actual hardware"},
            "inference_unit": "independent native-trained model replica; experience repeats are nested",
            "fixed_target_comparison": "same target producer checkpoint and scene/utterance stream across native learners",
            "split_policy": "one fixed scene partition; experience repeats vary dialogue and evaluation seeds"}


def aggregate_grid(rows: list[dict]) -> dict:
    """Keep curves and nested repeats; never treat D experiences as D independent models."""
    groups = {}
    for row in rows:
        for direction in row.get("directions", []):
            key = (row["model_profile_sha256"], direction["learner_native_language"],
                   direction["target_language"], row["analysis_group"])
            groups.setdefault(key, []).append({"model_replica_id": row["model_replica_id"],
                                               "experience_replica_id": row["experience_replica_id"],
                                               "run_id": row["run_id"], **direction})
    result = []
    for (profile, native, target, inclusion), values in groups.items():
        replicas = []
        for rid in dict.fromkeys(v["model_replica_id"] for v in values):
            group = [v for v in values if v["model_replica_id"] == rid]
            reached = [v["sustained_threshold_started_exchange"] for v in group if v["threshold_outcome"] == "reached"]
            replicas.append({"model_replica_id": rid, "n_experiences": len(group),
                             "outcomes": {s: sum(v["threshold_outcome"] == s for v in group)
                                          for s in ("reached", "right_censored", "failed", "cancelled")},
                             "median_among_reached_only": statistics.median(reached) if reached else None,
                             "unreached_fraction": 1 - len(reached) / len(group),
                             "experiences": group})
        result.append({"model_profile_sha256": profile, "learner_native_language": native,
                       "target_language": target, "analysis_group": inclusion,
                       "n_independent_models": len(replicas), "model_replicas": replicas})
    return {"directions": result, "uncertainty_note": "No CI from pooled dialogue repeats. Resample independent model replicas."}


def run_grid(grid: GridConfig, output="runs", *, resume_dir=None, limit=None, progress=None):
    from .neural import load_checkpoint, train_native
    from .runner import ExperimentFailed, run_experiment
    plan = plan_grid(grid)
    root = Path(resume_dir).resolve() if resume_dir else Path(output).resolve() / "grids" / ("grid-" + uuid.uuid4().hex[:10])
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "grid.json"
    if status_path.exists():
        state = json.loads(status_path.read_text())
        if state["grid_sha256"] != plan["grid_sha256"]:
            raise ValueError("Cannot resume a grid with different conditions")
    else:
        state = {**plan, "statuses": {t["task_id"]: {"status": "pending"} for t in plan["tasks"]}}
        write_json(root / "config.json", grid.to_dict())
        write_json(root / "plan.json", plan)
    write_json(status_path, state)
    executed = 0
    for task in plan["tasks"]:
        record = state["statuses"][task["task_id"]]
        if record["status"] in {"completed", "failed", "cancelled"}:
            continue
        if limit is not None and executed >= limit:
            break
        state["statuses"][task["task_id"]] = {"status": "running"}
        write_json(status_path, state)
        try:
            profile = ModelProfile(**task["model_profile"])
            config = replace(grid.experiment, schema_version=2, model_profile=profile,
                             model_replica_id=task["model_replica_id"], experience_replica_id=task["experience_replica_id"],
                             pair_id=task["pair_id"], scene_split_seed=task["scene_split_seed"],
                             dialogue_seed=task["dialogue_seed"], evaluation_seed=task["evaluation_seed"])
            agents = []
            for index, language in enumerate(task["languages"]):
                path = root / "native" / task["model_profile_sha256"][:12] / task["model_replica_id"] / f"{language}.pt"
                if not path.exists() or not path.with_suffix(".json").exists():
                    train_native(config, language, path, initialization_seed=task["initialization_seed"],
                                 native_training_seed=task["native_training_seed"], model_replica_id=task["model_replica_id"],
                                 progress=progress)
                else:
                    metadata = load_checkpoint(path)["metadata"]
                    if (metadata["initialization_seed"] != task["initialization_seed"] or
                            metadata["native_training_seed"] != task["native_training_seed"]):
                        raise ValueError("Cached native checkpoint seed mismatch")
                agents.append(AgentConfig("A" if index == 0 else "B", language, "neural", native_checkpoint=str(path)))
            config = replace(config, agents=tuple(agents)).validate()
            # A fresh deserialization for EACH pair/experience prevents cross-pair L2 carryover.
            path = run_experiment(config, output, progress)
            state["statuses"][task["task_id"]] = {"status": "completed", "run_path": str(path)}
        except ExperimentFailed as exc:
            status = json.loads((exc.run_path / "summary.json").read_text())["status"]
            state["statuses"][task["task_id"]] = {"status": status, "run_path": str(exc.run_path), "error": str(exc)}
            if status == "cancelled":
                write_json(status_path, state)
                break
        except KeyboardInterrupt:
            state["statuses"][task["task_id"]] = {"status": "cancelled", "error": "interrupted during native preparation"}
            write_json(status_path, state)
            break
        except Exception as exc:
            state["statuses"][task["task_id"]] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        executed += 1
        write_json(status_path, state)
    rows = []
    for task in plan["tasks"]:
        record = state["statuses"][task["task_id"]]
        if record.get("run_path"):
            summary = json.loads((Path(record["run_path"]) / "summary.json").read_text())
            rows.append({**task, **record, "run_id": summary["run_id"], "directions": summary["directions"],
                         "analysis_group": (summary["inclusion"] or {}).get("analysis_group", "unclassified")})
    write_json(root / "aggregate.json", aggregate_grid(rows))
    write_json(root / "outcomes.json", {s: sum(r["status"] == s for r in state["statuses"].values())
                                        for s in ("pending", "running", "completed", "failed", "cancelled")})
    return root
