"""Coverage, selection and checkpoint boundaries for development L1 preparation."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from mutual_lab.config import ExperimentConfig
from mutual_lab.native_preparation import (
    PreparationPlan, build_preparation_splits, classify_all_languages, run_preparation,
)
from mutual_lab.profiles import ModelProfile
from mutual_lab.world import COLORS, RELATIONS, SHAPES, Object, make_research_dataset, native_captions, variants


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("ko", "ja", "en", "es")


def experiment():
    config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
    return replace(config, model_profile=ModelProfile("native-preparation-test",
        embedding_dim=8, hidden_dim=16, max_sequence_length=128),
        native_training_profile=replace(config.native_training_profile, batch_size=4))


def synthetic_report(language, *, last_language_late=False):
    return {
        "language": language, "status": "completed", "plan": asdict(PreparationPlan(budgets=(1, 2))),
        "model_profile": {"id": "paired-profile"}, "config_sha256": "same-config",
        "semantic_data_sha256": "same-scenes", "source_sha256": {"source.py": "same-source"},
        # Real text hashes differ across languages and must not prevent a common budget.
        "data_sha256": "native-text-" + language,
        "curves": [
            {"step": 0, "dev": {"comprehension_accuracy": .25, "valid_generation_accuracy": 0}},
            {"step": 1, "dev": {"comprehension_accuracy": 1,
                "valid_generation_accuracy": .5 if last_language_late and language == "es" else 1}},
            {"step": 2, "dev": {"comprehension_accuracy": 1, "valid_generation_accuracy": 1}},
        ],
    }


class PreparationDataTests(unittest.TestCase):
    def test_fixed_split_covers_every_word_and_holds_out_only_combinations(self):
        config = experiment()
        plan = PreparationPlan()
        splits = build_preparation_splits(config, plan)
        self.assertEqual(splits, build_preparation_splits(config, plan))
        self.assertEqual({name: len(pool) for name, pool in splits.items()}, {"train": 40, "dev": 16})
        pairs = {name: {scene.group for scene in pool} for name, pool in splits.items()}
        self.assertEqual(len(pairs["train"]), 10)
        self.assertEqual(len(pairs["dev"]), 4)
        self.assertFalse(pairs["train"] & pairs["dev"])
        source = make_research_dataset(config)
        used = set(splits["train"]) | set(splits["dev"])
        self.assertTrue(used <= set(source.train))
        for name in ("native", "pretest", "curve", "final"):
            self.assertFalse({scene.group for scene in used}
                             & {probe.scene.group for probe in source.view(name).probes})
        train_objects = {obj for scene in splits["train"] for obj in (scene.subject, scene.object)}
        self.assertEqual(train_objects, {Object(c, s) for c in COLORS for s in SHAPES})
        for name, scenes in splits.items():
            objects = {obj for scene in scenes for obj in (scene.subject, scene.object)}
            self.assertEqual({obj.color for obj in objects}, set(COLORS))
            self.assertEqual({obj.shape for obj in objects}, set(SHAPES))
            for pair in pairs[name]:
                actual = {scene for scene in scenes if scene.group == pair}
                self.assertEqual(actual, set(variants(pair)))
                self.assertEqual({scene.relation for scene in actual}, set(RELATIONS))
                self.assertEqual({(s.subject, s.object) for s in actual}, {pair, pair[::-1]})
                for scene in actual:
                    for language in LANGUAGES:
                        self.assertEqual(len(set(native_captions(scene, language))), 4)

    def test_invalid_budget_seeds_and_targets_fail_before_creating_output(self):
        invalid = (
            PreparationPlan(budgets=()), PreparationPlan(budgets=(0,)),
            PreparationPlan(budgets=(2, 1)), PreparationPlan(budgets=(1, 1)),
            PreparationPlan(budgets=(True,)), PreparationPlan(initialization_seed=7),
            PreparationPlan(initialization_seed=29003), PreparationPlan(max_seconds=0),
            PreparationPlan(comprehension_min=float("nan")), PreparationPlan(production_min=1.1),
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "must-not-be-created"
            for plan in invalid:
                with self.subTest(plan=plan), self.assertRaises(ValueError):
                    run_preparation(experiment(), "ko", plan, output=destination, progress=None)
                self.assertFalse(destination.exists())


class CommonBudgetTests(unittest.TestCase):
    def test_budget_is_first_candidate_where_all_four_languages_pass_both_targets(self):
        reports = [synthetic_report(language, last_language_late=True) for language in LANGUAGES]
        result = classify_all_languages(reports)
        self.assertEqual(result["status"], "development_targets_met")
        self.assertEqual(result["common_budget"], 2)
        self.assertFalse(result["per_budget"][0]["all_languages_pass"])
        self.assertTrue(result["per_budget"][1]["all_languages_pass"])
        # Improved mean performance cannot compensate for one language missing a target.
        reports[-1]["curves"][-1]["dev"]["comprehension_accuracy"] = .89
        result = classify_all_languages(reports)
        self.assertEqual(result["status"], "development_targets_not_met")
        self.assertIsNone(result["common_budget"])

    def test_missing_or_failed_languages_never_produce_a_selected_budget(self):
        reports = [synthetic_report(language) for language in LANGUAGES]
        result = classify_all_languages(reports[:-1])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_languages"], ["es"])
        self.assertIsNone(result["common_budget"])
        reports[-1]["status"] = "failed"
        result = classify_all_languages(reports)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["common_budget"])

    def test_comparison_rejects_incompatible_or_duplicate_runs(self):
        reports = [synthetic_report(language) for language in LANGUAGES]
        for key in ("plan", "model_profile", "config_sha256", "semantic_data_sha256", "source_sha256"):
            altered = copy.deepcopy(reports)
            altered[-1][key] = {"different": True} if isinstance(altered[-1][key], dict) else "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                classify_all_languages(altered)
        with self.assertRaises(ValueError):
            classify_all_languages(reports + [copy.deepcopy(reports[0])])


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class PreparationNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.paths = []
        cls.reports = []
        for budgets in ((1, 2), (2,)):
            path = run_preparation(experiment(), "ko", PreparationPlan(budgets=budgets),
                output=cls.temporary.name, progress=None)
            cls.paths.append(path)
            cls.reports.append(json.loads((path / "report.json").read_text(encoding="utf-8")))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_actual_joint_updates_are_saved_but_cannot_qualify_a_research_learner(self):
        from mutual_lab.neural import load_checkpoint, torch
        report, path = self.reports[0], self.paths[0]
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["training_steps_completed"], 2)
        self.assertEqual(report["protected_splits_used"], [])
        self.assertTrue(report["excluded_from_research"])
        self.assertFalse(report["eligible_for_fresh_experiment"])
        self.assertEqual(report["training_languages"], ["ko"])
        self.assertEqual(report["foreign_training_languages"], [])
        for module in ("understanding", "producer"):
            self.assertNotEqual(report[f"initial_{module}_sha256"], report[f"final_{module}_sha256"])
        self.assertEqual([row["step"] for row in report["curves"]], [0, 1, 2])
        self.assertEqual(report["curves"][-1]["train"]["comprehension_items"], 160)
        self.assertEqual(report["curves"][-1]["dev"]["comprehension_items"], 64)
        self.assertEqual(report["curves"][-1]["dev"]["generation_scenes"], 16)
        self.assertIn("valid_generation_accuracy", report["curves"][-1]["dev"])
        checkpoint_row = next(row for row in report["checkpoints"] if row["step"] == 2)
        checkpoint = path / checkpoint_row["path"]
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_row["sha256"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(payload["kind"], "development_only")
        self.assertEqual(payload["training_steps_completed"], 2)
        self.assertTrue(payload["optimizer"]["state"])
        self.assertIn("training_rng_state", payload)
        self.assertIn("torch_rng_state", payload)
        self.assertIn("visual_projection", payload["model"])
        with patch("mutual_lab.neural.torch.load") as deserialize:
            for expected in ("native", "acquisition"):
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    load_checkpoint(checkpoint, expected_kind=expected)
            deserialize.assert_not_called()

    def test_extra_development_evaluation_does_not_change_final_weights(self):
        first, second = self.reports
        for key in ("final_understanding_sha256", "final_producer_sha256", "semantic_data_sha256", "data_sha256"):
            self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
