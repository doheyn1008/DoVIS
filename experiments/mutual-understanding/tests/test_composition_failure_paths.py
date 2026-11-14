"""Exercise real arm/suite control flow with a deterministic lightweight runtime."""

import hashlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from mutual_lab import composition_suite as suite
from mutual_lab.config import ExperimentConfig
from mutual_lab.world import native_captions, variants

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    value = 0.0

    def monotonic(self):
        return self.value


class FakeLoss:
    def __add__(self, other):
        return self

    def backward(self):
        pass


class FakeBranch:
    def __init__(self):
        self.weight = 0

    def parameters(self):
        return [self]

    def state_dict(self):
        return {"weight": self.weight}


class FakeModel:
    def __init__(self):
        self.understanding, self.producer = FakeBranch(), FakeBranch()
        self.visual_projection = 0
        self.development_spec = {"parameter_count": 2}

    def parameters(self):
        return [self.understanding, self.producer]

    def state_dict(self):
        return {"understanding": self.understanding.state_dict(),
                "producer": self.producer.state_dict(), "visual_projection": 0}

    def predict(self, text, choices):
        return 0

    def grounding_loss(self, examples):
        return FakeLoss()

    def production_loss(self, examples):
        return FakeLoss()


class FakeOptimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.steps = 0

    def zero_grad(self, **kwargs):
        pass

    def step(self):
        self.steps += 1
        for parameter in self.parameters:
            parameter.weight += 1

    def state_dict(self):
        return {"steps": self.steps}


def examples(model, scenes, language):
    return [{"text": text, "image": "pixels", "choices": [0, 1, 2, 3],
             "answer": variants(scene.group).index(scene)}
            for scene in scenes for text in native_captions(scene, language)]


class CompositionFailurePathTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        self.plan = suite.SuitePlan(budgets=(1,), arms=("tiny_balanced_four",),
                                   max_seconds_per_arm=1, max_seconds=10)
        self.setup_seconds = 0
        self.final_evaluation_seconds = 0
        self.evaluation_calls = 0

    def runtime(self):
        """Replace numerical work while preserving the production file/control flow."""
        fake_torch = types.SimpleNamespace(
            manual_seed=lambda seed: None,
            random=types.SimpleNamespace(get_rng_state=lambda: 0),
            optim=types.SimpleNamespace(Adam=FakeOptimizer),
            isfinite=lambda loss: types.SimpleNamespace(item=lambda: True),
            save=lambda payload, path: Path(path).write_bytes(json.dumps(payload, default=str).encode()),
        )
        fake_neural = types.ModuleType("mutual_lab.neural")
        fake_neural.configure_cpu = lambda: None
        fake_neural.digest = lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
        fake_neural.native_examples = examples
        fake_neural.torch = fake_torch
        fake_neural.nn = types.SimpleNamespace(utils=types.SimpleNamespace(
            clip_grad_norm_=lambda *args, **kwargs: None))

        def build_model(*args, **kwargs):
            self.clock.value += self.setup_seconds
            return FakeModel()

        def evaluate(model, pool, scenes, language, index, batch_size):
            self.evaluation_calls += 1
            # Each snapshot evaluates train and dev; call four ends the step-1 dev evaluation.
            if self.evaluation_calls == 4:
                self.clock.value += self.final_evaluation_seconds
            return ({"comprehension_accuracy": 1.0, "comprehension_hits": len(pool),
                     "comprehension_items": len(pool), "generation_scenes": len(scenes)},
                    [{"exact_match": True, "terminated": True, "invalid_encoding": False,
                      "special_tokens_in_body": 0} for _ in scenes])

        stack = ExitStack()
        stack.enter_context(patch.dict(sys.modules, {"mutual_lab.neural": fake_neural}))
        stack.enter_context(patch.object(suite.time, "monotonic", self.clock.monotonic))
        for name, replacement in {
            "build_development_model": build_model,
            "environment_metadata": lambda torch: {"cpu": "fake"},
            "evaluate_diagnostic": evaluate,
            "semantic_index": lambda language: {},
            "build_contrast_probes": lambda *args: {"probes": []},
            "contrast_manifest": lambda probes: probes,
            "score_contrasts": lambda *args: {"categories": {}},
            "revision": lambda: {"commit": "test"},
        }.items():
            stack.enter_context(patch.object(suite, name, replacement))
        return stack

    def run_arm(self):
        path = suite.run_arm(self.config, "ko", "tiny_balanced_four", self.plan,
                             self.output, progress=None)
        return path, json.loads((path / "report.json").read_text())

    def test_setup_consumes_the_arm_deadline_and_preserves_an_initial_checkpoint(self):
        self.setup_seconds = 3
        with self.runtime():
            path, report = self.run_arm()
        self.assertEqual(report["status"], "time_limit")
        self.assertEqual(report["training_steps_completed"], 0)
        self.assertGreaterEqual(report["elapsed_seconds"], 3)
        self.assertEqual(self.evaluation_calls, 0)
        self.assertTrue((path / "step-000000.pt").exists())

    def test_final_evaluation_overrun_is_not_reported_as_completed(self):
        self.final_evaluation_seconds = 2
        with self.runtime():
            path, report = self.run_arm()
        self.assertEqual(report["status"], "time_limit")
        self.assertEqual(report["training_steps_completed"], 1)
        self.assertEqual([row["step"] for row in report["curves"]], [0, 1])
        self.assertGreaterEqual(report["elapsed_seconds"], 2)
        self.assertTrue((path / "step-000001.pt").exists())

    def test_evaluation_inside_the_deadline_still_completes(self):
        self.final_evaluation_seconds = .75
        with self.runtime():
            _, report = self.run_arm()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["training_steps_completed"], 1)
        self.assertEqual(report["elapsed_seconds"], .75)

    def test_reaching_the_exact_deadline_is_a_time_limit(self):
        self.final_evaluation_seconds = 1
        with self.runtime():
            _, report = self.run_arm()
        self.assertEqual(report["status"], "time_limit")
        self.assertEqual([row["step"] for row in report["curves"]], [0, 1])

    def test_remaining_suite_time_can_be_tighter_than_the_arm_allowance(self):
        self.setup_seconds = .75
        with self.runtime():
            path = suite.run_arm(self.config, "ko", "tiny_balanced_four", self.plan,
                                 self.output, progress=None, remaining_seconds=.5)
        report = json.loads((path / "report.json").read_text())
        self.assertEqual(report["status"], "time_limit")
        self.assertEqual(report["training_steps_completed"], 0)
        self.assertEqual(self.evaluation_calls, 0)
        self.assertEqual(report["elapsed_seconds"], .75)

    def test_interrupt_retains_cancelled_arm_observations_in_suite_and_aggregation(self):
        self.plan = suite.SuitePlan(budgets=(1, 2), arms=("tiny_balanced_four",),
                                   max_seconds_per_arm=10, max_seconds=20)

        def interrupt(event, **kwargs):
            if "step=1 " in event:
                raise KeyboardInterrupt("injected after the first budget")

        with self.runtime(), self.assertRaises(suite.ArmCancelled) as raised:
            suite.run_suite(self.config, "ko", self.plan, self.output, progress=interrupt)
        arm_path = raised.exception.report_path
        self.assertEqual(arm_path.name, "report.json")
        raw_arm = json.loads(arm_path.read_text())
        saved_suite = json.loads(next(self.output.rglob("suite.json")).read_text())
        self.assertEqual(raw_arm["status"], "cancelled")
        self.assertEqual(saved_suite["status"], "cancelled")
        self.assertEqual(len(saved_suite["reports"]), 1)
        self.assertEqual(saved_suite["reports"][0], raw_arm)
        self.assertEqual([row["step"] for row in raw_arm["curves"]], [0, 1])
        comparison = suite.aggregate_suites([saved_suite])
        self.assertEqual(comparison["status"], "incomplete")
        self.assertEqual([row["step"] for row in comparison["rows"]], [0, 1])
        self.assertEqual(comparison["failed_arms"], [{"language": "ko", "arm": "tiny_balanced_four",
                                                     "status": "cancelled"}])


if __name__ == "__main__":
    unittest.main()
