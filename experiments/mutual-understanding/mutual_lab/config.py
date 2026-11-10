"""Strict, serializable experiment definitions; secrets stay in environment variables."""

from __future__ import annotations

import ipaddress
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def integer(value: object, name: str, lower: int, upper: int) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


@dataclass(frozen=True)
class AgentConfig:
    name: str
    language: str
    backend: str = "symbolic"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    initialization_note: str = "Unverified: no monolingual initialization evidence supplied."

    def validate(self, allow_remote: bool) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", self.name):
            raise ValueError("Agent name must be a short ASCII identifier")
        if self.language not in {"en", "ko", "ja"}:
            raise ValueError("Supported instruction languages: en, ko, ja")
        if self.backend not in {"symbolic", "endpoint"}:
            raise ValueError("backend must be symbolic or endpoint")
        for key in ("model", "base_url", "api_key_env", "initialization_note"):
            if not isinstance(getattr(self, key), str):
                raise ValueError(f"{key} must be a string")
        if self.backend == "endpoint":
            url = urlsplit(self.base_url)
            if (url.scheme not in {"http", "https"} or not url.hostname
                    or url.username or url.password or url.query or url.fragment):
                raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment")
            try:
                local = ipaddress.ip_address(url.hostname).is_loopback
            except ValueError:
                local = url.hostname == "localhost"
            if not local and not allow_remote:
                raise ValueError("Non-loopback endpoints require allow_remote=true")
            if not self.model.strip():
                raise ValueError("Endpoint agents require a model ID")
            if self.api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env):
                raise ValueError("api_key_env must name an environment variable, not contain a key")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int = 1
    seed: int = 7
    max_exchanges: int = 80
    eval_every: int = 5
    eval_items: int = 24
    holdout_fraction: float = 0.25
    threshold: float = 0.90
    sustained_checks: int = 2
    stop_on_threshold: bool = True
    context_exchanges: int = 24
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_seconds: float = 60.0
    max_requests: int = 1000
    allow_remote: bool = False
    agents: tuple[AgentConfig, AgentConfig] = field(default_factory=lambda: (
        AgentConfig("A", "en"), AgentConfig("B", "ko"),
    ))

    def validate(self) -> "ExperimentConfig":
        integer(self.schema_version, "schema_version", 1, 1)
        integer(self.seed, "seed", 0, 2**32 - 1)
        integer(self.max_exchanges, "max_exchanges", 1, 100_000)
        integer(self.eval_every, "eval_every", 1, self.max_exchanges)
        integer(self.eval_items, "eval_items", 4, 140)
        if self.eval_items % 4:
            raise ValueError("eval_items must be divisible by 4 for balanced answer positions")
        integer(self.sustained_checks, "sustained_checks", 1, 100)
        integer(self.context_exchanges, "context_exchanges", 1, 1000)
        integer(self.max_tokens, "max_tokens", 8, 4096)
        integer(self.max_requests, "max_requests", 1, 10_000_000)
        for key, low, high in (("holdout_fraction", .05, .75), ("threshold", .25, 1.0),
                               ("temperature", 0.0, 2.0), ("timeout_seconds", .01, 600.0)):
            value = getattr(self, key)
            if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be a finite number in [{low}, {high}]")
        for key in ("allow_remote", "stop_on_threshold"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if len(self.agents) != 2:
            raise ValueError("Exactly two agents are required")
        for agent in self.agents:
            agent.validate(self.allow_remote)
        if self.agents[0].name == self.agents[1].name:
            raise ValueError("Agent names must differ")
        if len({agent.backend for agent in self.agents}) != 1:
            raise ValueError("Both agents must use the same backend type for this pilot")
        return self

    @property
    def mode(self) -> str:
        return "demo" if self.agents[0].backend == "symbolic" else "endpoint"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["agents"] = [asdict(agent) for agent in self.agents]
        return data

    def request_upper_bound(self) -> int:
        if self.mode == "demo":
            return 0
        # Frozen captions, checkpoint zero + scheduled/final evaluations, two utterances per exchange.
        checkpoints = 1 + math.ceil(self.max_exchanges / self.eval_every)
        return 2 * self.eval_items * (1 + checkpoints) + 2 * self.max_exchanges

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        data = dict(data)
        try:
            if "agents" in data:
                if not isinstance(data["agents"], list):
                    raise ValueError("agents must be a JSON array")
                data["agents"] = tuple(AgentConfig(**item) for item in data["agents"])
            return cls(**data).validate()
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
