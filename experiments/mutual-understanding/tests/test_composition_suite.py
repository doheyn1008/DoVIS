"""Controlled factor changes, incomplete suites, and actual development learning."""

import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from mutual_lab.composition_suite import (
    DEFAULT_ARMS, SuitePlan, aggregate_suites, arm_spec, model_profile, run_arm, run_suite,
)
from mutual_lab.config import ExperimentConfig
from mutual_lab.profiles import ModelProfile

ROOT = Path(__file__).resolve().parents[1]


def experiment():
    config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
    return replace(config, model_profile=ModelProfile("composition-test", embedding_dim=8, hidden_dim=16,
                                                    max_sequence_length=128),
                   native_training_profile=replace(config.native_training_profile, batch_size=4))


def synthetic_suite(language):
    arm = "tiny_balanced_four"
    return {"language": language, "status": "completed",
            "plan": asdict(SuitePlan(budgets=(1,), arms=(arm,))), "reports": [{
                "status": "completed", "arm": arm_spec(arm), "config_sha256": "same",
                "source_sha256": {"module": "same"}, "model_profile": {"id": "same"},
                "split": {"pair_sha256": "same"},
                "curves": [{"step": step,
                            "dev": {"comprehension_accuracy": 1.0, "valid_generation_accuracy": 0},
                            "contrasts": {"categories": {"role_reversal": {"accuracy": 1.0}}}}
                           for step in (0, 1)]}]}


class SuiteDefinitionTests(unittest.TestCase):
    def test_default_conditions_change_the_declared_factors(self):
        self.assertEqual(len(DEFAULT_ARMS), 6)
        four = arm_spec("medium_balanced_four")
        one = arm_spec("medium_balanced_one")
        self.assertEqual({key: value for key, value in four.items() if key not in ("name", "production_forms")},
                         {key: value for key, value in one.items() if key not in ("name", "production_forms")})
        self.assertEqual(four["production_forms"], 4)
        self.assertEqual(one["production_forms"], 1)
        config = experiment()
        self.assertEqual(model_profile(config, "tiny_balanced_four"), config.model_profile)
        medium = model_profile(config, "medium_balanced_one")
        self.assertEqual((medium.embedding_dim, medium.hidden_dim, medium.layers), (64, 128, 1))

    def test_invalid_plans_fail_before_any_output_is_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "absent"
            for plan in (SuitePlan(budgets=()), SuitePlan(arms=()), SuitePlan(arms=("unknown",)),
                         SuitePlan(arms=("tiny_balanced_four",) * 2), SuitePlan(max_seconds=0)):
                with self.subTest(plan=plan), self.assertRaises(ValueError):
                    run_suite(experiment(), "ko", plan, destination, progress=None)
                self.assertFalse(destination.exists())

    def test_aggregation_retains_missing_failed_and_missing_budget_conditions(self):
        suites = [synthetic_suite(language) for language in ("ko", "ja", "en", "es")]
        self.assertEqual(aggregate_suites(suites)["status"], "completed")
        self.assertEqual(aggregate_suites(suites[:-1])["missing_languages"], ["es"])
        missing_arm = copy.deepcopy(suites)
        missing_arm[-1]["reports"] = []
        self.assertEqual(aggregate_suites(missing_arm)["status"], "incomplete")
        failed = copy.deepcopy(suites)
        failed[-1]["reports"][0]["status"] = "time_limit"
        self.assertEqual(aggregate_suites(failed)["failed_arms"][0]["status"], "time_limit")
        missing_budget = copy.deepcopy(suites)
        missing_budget[-1]["reports"][0]["curves"].pop()
        self.assertEqual(aggregate_suites(missing_budget)["status"], "incomplete")
        mismatched = copy.deepcopy(suites)
        mismatched[-1]["reports"][0]["source_sha256"] = {"module": "changed"}
        with self.assertRaises(ValueError):
            aggregate_suites(mismatched)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class SuiteNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.paths = {}
        cls.reports = {}
        plan = SuitePlan(budgets=(1, 2), arms=("tiny_balanced_four", "tiny_balanced_one"))
        for arm in plan.arms:
            path = run_arm(experiment(), "ko", arm, plan, cls.temporary.name, progress=None)
            cls.paths[arm] = path
            cls.reports[arm] = json.loads((path / "report.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_one_form_changes_only_production_while_understanding_and_meanings_match(self):
        four = self.reports["tiny_balanced_four"]
        one = self.reports["tiny_balanced_one"]
        for key in ("initial_understanding_sha256", "initial_producer_sha256", "initial_visual_sha256",
                    "meaning_stream_sha256", "comprehension_caption_stream_sha256", "final_understanding_sha256"):
            self.assertEqual(four[key], one[key], key)
        self.assertNotEqual(four["final_producer_sha256"], one["final_producer_sha256"])
        self.assertEqual(one["comprehension_training_forms"], "all_four_authored_forms")
        self.assertEqual(one["teacher_forced_evaluation_forms"], "all_four_authored_forms")
        self.assertEqual(one["production_training_forms"], "first_authored_form")

    def test_actual_joint_weights_contrasts_environment_and_checkpoint_boundaries(self):
        from mutual_lab.neural import load_checkpoint, torch
        report = self.reports["tiny_balanced_four"]
        directory = self.paths["tiny_balanced_four"]
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["training_steps_completed"], 2)
        self.assertEqual(report["protected_splits_used"], [])
        self.assertTrue(report["excluded_from_research"])
        self.assertFalse(report["eligible_for_fresh_experiment"])
        for module in ("understanding", "producer"):
            self.assertNotEqual(report[f"initial_{module}_sha256"], report[f"final_{module}_sha256"])
        self.assertEqual(report["training_languages"], ["ko"])
        self.assertEqual(report["foreign_training_languages"], [])
        self.assertEqual(report["curves"][-1]["dev"]["comprehension_items"], 64)
        contrast = report["curves"][-1]["contrasts"]
        self.assertEqual(contrast["chance_accuracy"], .5)
        self.assertEqual(set(contrast["categories"]), {"role_reversal", "color_only", "shape_only", "binding_swap"})
        for key in ("cpu", "torch", "torch_config", "threads", "deterministic_algorithms"):
            self.assertIn(key, report["environment"])
        checkpoint = directory / "step-000002.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(payload["kind"], "development_only")
        self.assertEqual(payload["step"], 2)
        self.assertTrue(payload["optimizer"]["state"])
        self.assertIn("torch_rng_state", payload)
        self.assertIn("training_rng_state", payload)
        with patch("mutual_lab.neural.torch.load") as deserialize:
            for kind in ("native", "acquisition"):
                with self.assertRaises(ValueError):
                    load_checkpoint(checkpoint, expected_kind=kind)
            deserialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
