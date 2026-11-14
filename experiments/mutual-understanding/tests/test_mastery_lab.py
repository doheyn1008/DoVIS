"""Predeclared development gates, paired comparisons and actual training state."""

import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from mutual_lab.composition_diagnostics import balanced_splits, build_contrast_probes
from mutual_lab.config import ExperimentConfig
from mutual_lab.mastery_lab import MasteryPlan, aggregate_reports, candidate_assessment, pair_audit, run_phase, run_trial
from mutual_lab.native_preparation import PreparationPlan
from mutual_lab.profiles import ModelProfile

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("ko", "ja", "en", "es")
CATEGORIES = ("role_reversal", "color_only", "shape_only", "binding_swap")


def experiment():
    config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
    return replace(config, native_training_profile=replace(config.native_training_profile, batch_size=4))


def synthetic_suite(language, phase="A"):
    plan = MasteryPlan(budgets=(1, 2), benchmark_steps=0)
    pairs = plan.seed_pairs if phase == "A" else (plan.b_seed_pair,)
    conditions = ("A0", "A1") if phase == "A" else ("B0", "B1")
    reports = []
    for replica, seeds in enumerate(pairs):
        for condition in conditions:
            improved = condition == "A1"
            reports.append({
                "status": "completed", "phase": phase, "replica": replica, "condition": condition,
                "language": language, "seed_pair": list(seeds), "split": {"pair_sha256": "same"},
                "data_sha256": "text-" + language, "config_sha256": "config", "source_sha256": {"module": "same"},
                "model_profile": {"id": "same"}, "parameter_count": 100, "batch_size": 4,
                "learning_rate": .003, "initial_understanding_sha256": "u" + str(replica),
                "initial_producer_sha256": "p" + str(replica),
                "initial_prior_sha256": "shared-" + (condition if phase == "B" else "projection"),
                "meaning_stream_sha256": "same-meanings", "comprehension_caption_stream_sha256": "same-texts",
                "environment_sha256": "environment-" + language, "final_producer_sha256": "p-final" + str(replica),
                "policy": "attributes" if condition == "A1" else "same_pair",
                "curves": [{"step": step, "dev": {"comprehension_accuracy": .95,
                            "comprehension_hits": 61, "comprehension_items": 64,
                            "valid_generation_accuracy": .5, "valid_generation_hits": 8, "generation_scenes": 16},
                            "contrasts": {"categories": {category: {"hits": 9 if improved else 8,
                                "items": 10, "accuracy": .9 if improved else .8} for category in CATEGORIES}},
                            "candidate": {"status": "indeterminate_coverage"}}
                           for step in (0, 1, 2)],
            })
    return {"kind": "mastery-lab-v1-suite", "phase": phase, "language": language,
            "plan": asdict(plan), "status": "completed", "reports": reports}


class MasteryProtocolTests(unittest.TestCase):
    def test_plan_round_trip_and_invalid_inputs_fail_before_output(self):
        plan = MasteryPlan()
        self.assertEqual(MasteryPlan.from_dict(asdict(plan)), plan)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "absent"
            for invalid in (replace(plan, budgets=()), replace(plan, seed_pairs=()),
                            replace(plan, seed_pairs=((7, 39009),)), replace(plan, benchmark_steps=-1),
                            replace(plan, contrast_min=float("nan"))):
                with self.subTest(plan=invalid), self.assertRaises(ValueError):
                    run_phase(experiment(), "ko", "A", invalid, output=destination, progress=None)
                self.assertFalse(destination.exists())
            with self.assertRaises(ValueError):
                run_phase(experiment(), "ko", "B", plan, output=destination, progress=None)
            self.assertFalse(destination.exists())

    def test_perfect_scores_do_not_override_insufficient_binding_coverage(self):
        config, plan = experiment(), MasteryPlan()
        scenes = balanced_splits(config, PreparationPlan())["dev"]
        probes = build_contrast_probes(config, scenes)
        point = {"dev": {"comprehension_accuracy": 1.0, "valid_generation_accuracy": 1.0},
                 "contrasts": {"categories": {category: {"hits": 4, "items": 4, "accuracy": 1.0}
                                                  for category in CATEGORIES}}}
        result = candidate_assessment(point, probes, plan)
        self.assertTrue(result["score_targets_met"])
        self.assertFalse(result["coverage_sufficient"])
        self.assertEqual(result["status"], "indeterminate_coverage")
        self.assertEqual(result["categories"]["binding_swap"]["available_anchors"], 4)
        self.assertEqual(result["categories"]["binding_swap"]["available_pairs"], 1)
        self.assertFalse(result["eligible_for_fresh_experiment"])

    def test_all_three_seed_pairs_and_four_languages_are_reported(self):
        suites = [synthetic_suite(language) for language in LANGUAGES]
        result = aggregate_reports(suites, "A")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["rows"]), 72)
        self.assertEqual(len(result["paired_deltas"]), 24)
        self.assertEqual(result["b_policy_selection"]["policy"], "attributes")
        self.assertTrue(all(row["generation_delta"] == 0 for row in result["paired_deltas"]))
        self.assertTrue(all(item["status"] == "matched" for item in pair_audit(suites[0]["reports"], "A")))

    def test_policy_selection_is_blocked_by_missing_trials_or_broken_producer_invariance(self):
        suites = [synthetic_suite(language) for language in LANGUAGES]
        missing = aggregate_reports(suites[:-1], "A")
        self.assertEqual(missing["missing_languages"], ["es"])
        self.assertEqual(missing["b_policy_selection"]["policy"], "undetermined")
        broken = copy.deepcopy(suites)
        broken[0]["reports"][1]["final_producer_sha256"] = "different"
        comparison = aggregate_reports(broken, "A")
        self.assertEqual(comparison["status"], "incomplete")
        self.assertEqual(comparison["b_policy_selection"]["policy"], "undetermined")
        self.assertTrue(any(row["status"] == "pair_mismatch" for row in comparison["failed_trials"]))
        mismatched = copy.deepcopy(suites)
        mismatched[-1]["reports"][0]["source_sha256"] = {"module": "different"}
        with self.assertRaises(ValueError):
            aggregate_reports(mismatched, "A")

    def test_policy_rule_requires_old_four_choice_not_to_regress_beyond_tolerance(self):
        suites = [synthetic_suite(language) for language in LANGUAGES]
        for report in suites[0]["reports"]:
            if report["condition"] == "A1":
                report["curves"][-1]["dev"]["comprehension_accuracy"] = .85
        result = aggregate_reports(suites, "A")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["b_policy_selection"]["policy"], "same_pair")

    def test_phase_b_rejects_languages_using_different_visual_priors(self):
        suites = [synthetic_suite(language, "B") for language in LANGUAGES]
        self.assertEqual(aggregate_reports(suites, "B")["status"], "completed")
        suites[-1]["reports"][1]["initial_prior_sha256"] = "not-the-shared-prior"
        with self.assertRaises(ValueError):
            aggregate_reports(suites, "B")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class MasteryNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.plan = MasteryPlan(budgets=(1, 2), benchmark_steps=1,
            model_profile=ModelProfile("mastery-test", embedding_dim=8, hidden_dim=16, max_sequence_length=128))
        cls.paths, cls.reports = {}, {}
        for condition in ("A0", "A1"):
            path = run_trial(experiment(), "ko", "A", condition, 0, cls.plan.seed_pairs[0], cls.plan,
                             output=cls.temporary.name, progress=None)
            cls.paths[condition] = path
            cls.reports[condition] = json.loads((path / "report.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_negative_choices_change_understanding_but_not_generator_or_input_stream(self):
        zero, one = self.reports["A0"], self.reports["A1"]
        self.assertEqual(zero["status"], "completed")
        self.assertEqual(one["status"], "completed")
        for key in ("initial_understanding_sha256", "initial_producer_sha256", "initial_prior_sha256",
                    "meaning_stream_sha256", "comprehension_caption_stream_sha256", "final_producer_sha256"):
            self.assertEqual(zero[key], one[key], key)
        self.assertNotEqual(zero["final_understanding_sha256"], one["final_understanding_sha256"])
        self.assertEqual(one["initial_prior_sha256"], one["final_prior_sha256"])
        self.assertTrue(one["benchmark"]["initial_weights_restored"])
        self.assertEqual(one["negative_exposures"]["examples"], 8)
        self.assertEqual(one["negative_exposures"]["distractor_slots"], 24)
        self.assertEqual(one["negative_exposures"]["semantic_duplicate_choices"], 0)
        self.assertEqual(one["negative_exposures"]["png_duplicate_choices"], 0)
        self.assertEqual(one["protected_splits_used"], [])

    def test_true_optimizer_rng_and_weights_are_saved_but_research_loading_is_rejected(self):
        from mutual_lab.neural import load_checkpoint, torch
        path = self.paths["A1"] / "step-000002.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(state["kind"], "development_only")
        self.assertTrue(state["optimizer"]["state"])
        self.assertIn("training_rng_state", state)
        self.assertIn("negative_rng_state", state)
        self.assertIn("torch_rng_state", state)
        self.assertEqual(state["negative_exposure_offset"], 8)
        with patch("mutual_lab.neural.torch.load") as deserialize:
            for kind in ("native", "acquisition"):
                with self.assertRaises(ValueError):
                    load_checkpoint(path, expected_kind=kind)
            deserialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
