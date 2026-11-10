"""Separate demo perception/learning from the actual model's pixel-only interface."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .backend import ChatTransport, parse_choice
from .config import AgentConfig, ExperimentConfig
from .images import choice_sheet, data_url, scene_png
from .prompts import PROMPTS
from .world import Scene, controlled_caption


@dataclass(frozen=True)
class SharedImage:
    """Model-facing input deliberately has no scene, semantic label, or target answer."""
    png: bytes


def tokens(text: str) -> list[str]:
    words = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    return ([f"u:{word}" for word in words]
            + [f"b:{a}/{b}" for a, b in zip(words, words[1:])]
            + [f"p:{i}:{word}" for i, word in enumerate(words)])


class SymbolicAgent:
    """DEMO: perfect visual features + learned cross-situational naive Bayes grounding.

    Only speaking uses the agent's OWN controlled language dictionary. The listener
    never reads the partner's dictionary or a translation table. Its privileged,
    perfect visual perception is a simulation assumption, not a VLM result.
    """

    def __init__(self, config: AgentConfig, experiment: ExperimentConfig):
        self.config = config
        self.experiment = experiment
        self.exposures = 0
        self.counts = [defaultdict(Counter) for _ in range(5)]
        self.documents = [Counter() for _ in range(5)]
        self.vocabulary: set[str] = set()

    def speak(self, scene: Scene, partner_utterance: str | None = None, *, frozen: bool = False) -> str:
        return controlled_caption(scene, self.config.language)

    def observe(self, scene: Scene, utterance: str) -> None:
        features = tokens(utterance)
        self.exposures += 1
        self.vocabulary.update(features)
        for slot, value in enumerate(scene.features):
            self.documents[slot][value] += 1
            self.counts[slot][value].update(features)

    def choose(self, utterance: str, choices: tuple[Scene, ...], rng: random.Random) -> tuple[int | None, str]:
        query = tokens(utterance)
        if not self.exposures:
            answer = rng.randint(1, 4)
        else:
            scores = []
            size = max(1, len(self.vocabulary))
            for scene in choices:
                score = 0.0
                for slot, value in enumerate(scene.features):
                    counts = self.counts[slot].get(value, Counter())
                    total = counts.total() + size
                    score += sum(math.log((counts[token] + 1) / total) for token in query)
                scores.append(score)
            best = max(scores)
            answer = rng.choice([i + 1 for i, score in enumerate(scores) if abs(score - best) < 1e-9])
        return answer, str(answer)

    def state_digest(self) -> str:
        state = {"exposures": self.exposures,
                 "counts": [{key: dict(value) for key, value in slot.items()} for slot in self.counts],
                 "vocabulary": sorted(self.vocabulary), "documents": self.documents}
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


class EndpointAgent:
    """Fixed weights; bounded private history supplies in-context adaptation only."""

    def __init__(self, config: AgentConfig, experiment: ExperimentConfig, transport: ChatTransport):
        self.config = config
        self.experiment = experiment
        self.transport = transport
        self.memory: list[tuple[SharedImage, str]] = []
        self.exposures = 0

    def _messages(self, instruction: str, image: SharedImage, *, frozen: bool = False) -> list[dict]:
        prompts = PROMPTS[self.config.language]
        messages = [{"role": "system", "content": prompts["system"]}]
        # Every request is self-contained. Evaluation never writes into this history.
        memory = [] if frozen else self.memory
        for past_image, past_utterance in memory:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": prompts["memory"] + past_utterance},
                {"type": "image_url", "image_url": {"url": data_url(past_image.png)}},
            ]})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": data_url(image.png)}},
        ]})
        return messages

    def speak(self, image: SharedImage, partner_utterance: str | None = None, *, frozen: bool = False) -> str:
        prompts = PROMPTS[self.config.language]
        instruction = prompts["speak"]
        if partner_utterance is not None:
            instruction += "\n" + prompts["partner"] + partner_utterance
        return self.transport.complete(self.config, "frozen_caption" if frozen else "dialogue",
                                       self._messages(instruction, image, frozen=frozen))

    def observe(self, image: SharedImage, utterance: str) -> None:
        self.exposures += 1
        self.memory.append((image, utterance))
        self.memory = self.memory[-self.experiment.context_exchanges:]

    def choose(self, utterance: str, image: SharedImage, rng: random.Random) -> tuple[int | None, str]:
        instruction = PROMPTS[self.config.language]["choose"] + utterance
        raw = self.transport.complete(self.config, "evaluation", self._messages(instruction, image))
        return parse_choice(raw), raw

    def state_digest(self) -> str:
        digest = hashlib.sha256(str(self.exposures).encode())
        for image, utterance in self.memory:
            digest.update(image.png)
            digest.update(utterance.encode("utf-8"))
        return digest.hexdigest()


def speak(agent, scene: Scene, partner_utterance: str | None = None, *, frozen: bool = False) -> str:
    visible = scene if isinstance(agent, SymbolicAgent) else SharedImage(scene_png(scene))
    return agent.speak(visible, partner_utterance, frozen=frozen)


def observe(agent, scene: Scene, utterance: str) -> None:
    visible = scene if isinstance(agent, SymbolicAgent) else SharedImage(scene_png(scene))
    agent.observe(visible, utterance)


def choose(agent, utterance: str, choices: tuple[Scene, ...], rng: random.Random) -> tuple[int | None, str]:
    visible = choices if isinstance(agent, SymbolicAgent) else SharedImage(choice_sheet(choices))
    return agent.choose(utterance, visible, rng)
