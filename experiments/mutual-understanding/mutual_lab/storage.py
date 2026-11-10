"""Append-only event logs and atomic snapshots; no evaluation answers enter agent memory."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import ExperimentConfig
from .images import scene_png
from .languages import LANGUAGES
from .prompts import PROMPTS
from .world import Scene


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, data: object) -> None:
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def revision() -> dict:
    try:
        root = Path(__file__).resolve().parent
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=2, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                               text=True, timeout=2, check=True).stdout.strip()
        return {"commit": commit, "working_tree_dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "working_tree_dirty": None}


CSV_FIELDS = ("exchange", "source", "listener", "source_language", "listener_language",
              "learner_native_language", "target_language", "speaker_language", "listener_exposures",
              "accuracy", "raw_accuracy", "corrected_accuracy", "choice_count", "correction_version",
              "hits", "n", "invalid_answers", "wilson_low", "wilson_high", "corrected_wilson_low", "corrected_wilson_high")


class RunStore:
    def __init__(self, root: str | Path, config: ExperimentConfig):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}-s{config.seed}"
        self.path = root / self.run_id
        self.path.mkdir()
        (self.path / "assets").mkdir()
        self.started_at = now()
        write_json(self.path / "config.json", config.to_dict())
        write_json(self.path / "prompts.json", PROMPTS)
        config_hash = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()
        write_json(self.path / "manifest.json", {
            "schema_version": 1, "run_id": self.run_id, "started_at": self.started_at,
            "package_version": __version__, "python": sys.version, "platform": platform.platform(),
            "revision": revision(), "config_sha256": config_hash,
            "prompts_sha256": hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest(),
            "axis_unit": "one shared image + two native-language utterances",
            "evaluation": "frozen native captions, four-choice image comprehension, held-out object pairs",
            "chance_accuracy": .25,
            "languages": LANGUAGES,
        })

    def event(self, event: dict) -> None:
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": now(), **event}, ensure_ascii=False, allow_nan=False) + "\n")

    def image(self, scene: Scene) -> str:
        name = "img-" + hashlib.sha256(scene.key.encode()).hexdigest()[:16] + ".png"
        path = self.path / "assets" / name
        if not path.exists():
            path.write_bytes(scene_png(scene))
        return name

    def metrics(self, points: list[dict]) -> None:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)
        atomic_text(self.path / "metrics.csv", output.getvalue())
        write_json(self.path / "evaluations.json", points)
