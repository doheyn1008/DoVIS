"""Safety boundaries and persistent learning state for the first training run."""

import importlib.util
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mutual_lab.config import AgentConfig, ExperimentConfig
from mutual_lab.profiles import EvaluationProfile, ModelProfile, NativeTrainingProfile
from mutual_lab.world import make_research_dataset

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/first_native_training.py"
spec = importlib.util.spec_from_file_location("first_native_training", SCRIPT)
training = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = training
spec.loader.exec_module(training)


def experiment():
    return ExperimentConfig(schema_version=2, eval_items=4, max_exchanges=2, eval_every=1,
        agents=(AgentConfig("A", "ko", "neural"), AgentConfig("B", "ja", "neural")),
        model_profile=ModelProfile("first-training-test", embedding_dim=8, hidden_dim=16),
        native_training_profile=NativeTrainingProfile(steps=3, batch_size=4),
        evaluation_profile=EvaluationProfile(pretest_items=4, final_items=4, native_items=4))


class FirstTrainingDataTests(unittest.TestCase):
    def test_only_disjoint_development_subsets_of_training_are_used(self):
        config = experiment()
        pools = training.development_data(config, training.TrainingOptions())
        source = make_research_dataset(config)
        self.assertEqual([len(pools[name]) for name in ("train", "dev")], [4, 4])
        self.assertFalse({s.group for s in pools["train"]} & {s.group for s in pools["dev"]})
        used = set(pools["train"] + pools["dev"])
        self.assertTrue(used <= set(source.train))
        for split in ("native", "pretest", "curve", "final"):
            self.assertFalse({s.group for s in used} & {p.scene.group for p in source.view(split).probes})

    def test_budget_and_seed_boundaries_are_enforced(self):
        for options in (training.TrainingOptions(steps_per_stage=0),
                        training.TrainingOptions(steps_per_stage=2001),
                        training.TrainingOptions(max_seconds=541),
                        training.TrainingOptions(initialization_seed=7)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                training.development_data(experiment(), options)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class FirstTrainingNeuralTests(unittest.TestCase):
    def test_real_updates_frozen_producer_and_development_checkpoint_exclusion(self):
        from mutual_lab.neural import load_checkpoint, torch
        with tempfile.TemporaryDirectory() as temporary:
            path = training.run_training(experiment(), options=training.TrainingOptions(
                steps_per_stage=2, eval_every=1), output=temporary, progress=None)
            report = json.loads((path / "report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["training_steps_completed"], 4)
            self.assertEqual(report["protected_splits_used"], [])
            self.assertTrue(report["excluded_from_research"])
            self.assertFalse(report["eligible_for_fresh_experiment"])
            self.assertTrue(report["producer_unchanged"])
            self.assertGreater(report["understanding_update_from_initial_l2"], 0)
            self.assertEqual(len(report["checkpoints"]), 2)
            self.assertEqual({row["train"]["items"] for row in report["curves"]}, {4, 16})
            self.assertEqual(report["curves"][3]["total_step"], 2)
            for row in report["curves"]:
                update = row["latest_sampled_update"]
                if update:
                    for measurement in ("gradient_l2_before_clipping", "parameter_update_l2"):
                        self.assertEqual(set(update[measurement]), {"embedding", "gru", "head"})
                        self.assertTrue(all(math.isfinite(v) and v > 0 for v in update[measurement].values()))
            checkpoint = path / "four_templates.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(payload["kind"], "development_only")
            self.assertEqual(payload["training_steps_completed"], 4)
            self.assertTrue(payload["optimizer"]["state"])
            self.assertIn("training_rng_state", payload)
            self.assertIn("torch_rng_state", payload)
            with patch("mutual_lab.neural.torch.load") as deserialize:
                for expected_kind in ("native", "acquisition"):
                    with self.assertRaisesRegex(ValueError, "format"):
                        load_checkpoint(checkpoint, expected_kind=expected_kind)
                deserialize.assert_not_called()

    def test_evaluation_frequency_does_not_change_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            options = training.TrainingOptions(steps_per_stage=3, eval_every=1)
            first = training.run_training(experiment(), options=options, output=temporary, progress=None)
            second = training.run_training(experiment(), options=replace(options, eval_every=3),
                                           output=temporary, progress=None)
            reports = [json.loads((p / "report.json").read_text()) for p in (first, second)]
            self.assertEqual(reports[0]["final_understanding_sha256"], reports[1]["final_understanding_sha256"])
            self.assertEqual(reports[0]["data_sha256"], reports[1]["data_sha256"])


if __name__ == "__main__":
    unittest.main()
