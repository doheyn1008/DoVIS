"""Stateless, bounded Chat Completions transport for locally served vision models."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .config import AgentConfig, ExperimentConfig


class BackendError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


class RequestBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def take(self) -> int:
        if self.used >= self.limit:
            raise BackendError(f"Request budget exhausted ({self.limit}); no automatic retries")
        self.used += 1
        return self.used

    def to_dict(self) -> dict:
        return {"requests": self.used, "request_limit": self.limit,
                "reported_prompt_tokens": self.prompt_tokens,
                "reported_completion_tokens": self.completion_tokens}


class ChatTransport:
    def __init__(self, config: ExperimentConfig, record: Callable[[dict], None]):
        self.config = config
        self.budget = RequestBudget(config.max_requests)
        self.record = record
        # Do not let redirects or ambient proxy settings expand the configured network boundary.
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def complete(self, agent: AgentConfig, purpose: str, messages: list[dict]) -> str:
        key = os.environ.get(agent.api_key_env, "") if agent.api_key_env else ""
        if agent.api_key_env and not key:
            raise BackendError(f"Set the environment variable named in api_key_env for {agent.name}")
        request_number = self.budget.take()
        payload = {"model": agent.model, "messages": messages, "temperature": self.config.temperature,
                   "max_tokens": self.config.max_tokens, "seed": self.config.seed, "stream": False}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        request = Request(agent.base_url.rstrip("/") + "/chat/completions", body, headers, method="POST")
        metadata = {"type": "model_request", "request": request_number, "agent": agent.name,
                    "purpose": purpose, "configured_model": agent.model,
                    "request_sha256": hashlib.sha256(body).hexdigest()}
        started = time.monotonic()
        try:
            with self.opener.open(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read(4_000_001)
            if len(raw) > 4_000_000:
                raise BackendError("Model response exceeds 4 MB")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Chat Completions response must be an object")
            # The server can bill for a failed generation. Account for valid reported
            # counts before inspecting choices/content, without estimating missing values.
            reported_usage = data.get("usage")
            usage = {}
            if isinstance(reported_usage, dict):
                for field in ("prompt_tokens", "completion_tokens"):
                    count = reported_usage.get(field)
                    if type(count) is int and count >= 0:
                        usage[field] = count
                        setattr(self.budget, field, getattr(self.budget, field) + count)
            metadata.update(usage=usage, served_model=data.get("model"),
                            system_fingerprint=data.get("system_fingerprint"))
            choice = data["choices"][0]
            metadata["finish_reason"] = choice.get("finish_reason")
            if choice.get("finish_reason") == "length":
                raise BackendError("Generation reached max_tokens; increase it before rerunning")
            content = choice["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
            if not isinstance(content, str) or not content.strip() or len(content) > 20_000:
                raise BackendError("Model returned empty, non-text, or oversized content")
            metadata["status"] = "ok"
            return content.strip()
        except HTTPError as exc:
            metadata["status"] = f"http_{exc.code}"
            # Do not echo server bodies, which can contain credentials or complete prompts.
            raise BackendError(f"{agent.name} endpoint returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError) as exc:
            metadata["status"] = "connection_error"
            raise BackendError(f"{agent.name} endpoint unavailable or timed out ({type(exc).__name__})") from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            metadata["status"] = "invalid_response"
            raise BackendError(f"{agent.name} endpoint returned an invalid Chat Completions response") from None
        except BackendError:
            metadata["status"] = "invalid_generation"
            raise
        finally:
            metadata["latency_seconds"] = round(time.monotonic() - started, 6)
            self.record(metadata)


def parse_choice(text: str) -> int | None:
    """Malformed answers count as wrong; no fallback guess or answer-aware retries."""
    if text in {"1", "2", "3", "4"}:
        return int(text)
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(value, dict) and set(value) == {"choice"}:
        choice = value["choice"]
        if type(choice) is int and 1 <= choice <= 4:
            return choice
    return None
