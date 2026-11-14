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
from mutual_lab.composition_suite import ArmCancelled
from mutual_lab.config import ExperimentConfig
from mutual_lab.mastery_lab import MasteryPlan, aggregate_reports, candidate_assessment, pair_audit, phase_conditions, run_phase, run_trial
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
    conditions = phase_conditions(phase)
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
                "initial_prior_sha256": "shared-" + (condition if phase != "A" else "projection"),
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
    def test_cancellation_without_a_readable_trial_report_stops_the_suite_even_if_suite_write_fails(self):
        from mutual_lab.mastery_lab import safe_json
        config = experiment()
        for read_failure, fail_suite_write in ((OSError, False), (OSError, True),
                                               (KeyboardInterrupt, False), (KeyboardInterrupt, True)):
            with self.subTest(read_failure=read_failure, fail_suite_write=fail_suite_write), \
                 tempfile.TemporaryDirectory() as temporary:
                interrupted = ArmCancelled(Path(temporary) / "missing-report.json")
                written_cancelled_suites = []

                def persist(path, value):
                    if Path(path).name == "mastery-suite.json" and value["status"] == "cancelled":
                        written_cancelled_suites.append(copy.deepcopy(value))
                        if fail_suite_write:
                            raise OSError("suite storage unavailable")
                    return safe_json(path, value)

                with patch("mutual_lab.mastery_lab.run_trial", side_effect=interrupted) as trial, \
                     patch("mutual_lab.mastery_lab.safe_json", side_effect=persist), \
                     patch.object(Path, "read_text", side_effect=read_failure("report storage unavailable")), \
                     self.assertRaises(ArmCancelled) as caught:
                    run_phase(config, "ko", "A", output=temporary, progress=None)
                self.assertIs(caught.exception, interrupted)
                self.assertEqual(trial.call_count, 1)
                self.assertEqual(len(written_cancelled_suites), 1)
                cancelled_suite = written_cancelled_suites[0]
                self.assertEqual(cancelled_suite["status"], "cancelled")
                self.assertEqual(len(cancelled_suite["reports"]), 1)
                self.assertEqual(cancelled_suite["reports"][0]["status"], "cancelled")
                if not fail_suite_write:
                    paths = list(Path(temporary).rglob("mastery-suite.json"))
                    self.assertEqual(json.loads(paths[0].read_text()), json.loads(json.dumps(cancelled_suite)))

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

    def test_candidate_gate_rejects_missing_or_unknown_contrast_categories(self):
        config, plan = experiment(), MasteryPlan()
        probes = build_contrast_probes(config, balanced_splits(config, PreparationPlan())["dev"])
        point = {"dev": {"comprehension_accuracy": 1.0, "valid_generation_accuracy": 1.0},
                 "contrasts": {"categories": {category: {"hits": 1, "items": 1, "accuracy": 1.0}
                                                  for category in CATEGORIES}}}
        missing = copy.deepcopy(point)
        del missing["contrasts"]["categories"]["shape_only"]
        unknown = copy.deepcopy(point)
        unknown["contrasts"]["categories"]["other"] = unknown["contrasts"]["categories"].pop("shape_only")
        for malformed in (missing, unknown):
            with self.subTest(categories=list(malformed["contrasts"]["categories"])), self.assertRaises(ValueError):
                candidate_assessment(malformed, probes, plan)

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
        for key in ("source_sha256", "initial_understanding_sha256", "initial_producer_sha256", "initial_prior_sha256"):
            mismatched = copy.deepcopy(suites)
            mismatched[-1]["reports"][0][key] = {"module": "different"} if key == "source_sha256" else "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
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

    def test_spatial_prior_comparison_rejects_changed_distractor_policy_within_a_pair(self):
        suites = [synthetic_suite(language, "B") for language in LANGUAGES]
        # Across-language metadata remains matched; only the within-pair policy differs.
        for one in suites:
            one["reports"][1]["policy"] = "attributes"
        result = aggregate_reports(suites, "B")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["failed_trials"]), 4)
        self.assertTrue(all(row["status"] == "pair_mismatch" for row in result["failed_trials"]))

    def test_phase_c_coverage_is_measured_on_a_separate_development_population(self):
        from mutual_lab import development_world as world
        config, plan = experiment(), MasteryPlan(data_seed=59003)
        splits = world.development_splits(config, plan.data_seed)
        probes = world.build_development_contrasts(config, splits)
        self.assertEqual({key: len(scenes) for key, scenes in splits.items()}, {"train": 220, "dev": 112})
        point = {"dev": {"comprehension_accuracy": 1.0, "valid_generation_accuracy": 1.0},
                 "contrasts": {"categories": {category: {"hits": 4, "items": 4, "accuracy": 1.0}
                                                  for category in CATEGORIES}}}
        result = candidate_assessment(point, probes, plan)
        self.assertTrue(result["coverage_sufficient"])
        self.assertTrue(result["score_targets_met"])
        self.assertEqual(result["status"], "development_candidate")
        self.assertFalse(result["eligible_for_fresh_experiment"])
        for detail in result["categories"].values():
            self.assertGreaterEqual(detail["available_anchors"], 8)
            self.assertGreaterEqual(detail["available_pairs"], 2)

    def test_phase_c_summary_keeps_conditions_separate_from_phase_b(self):
        suites = [synthetic_suite(language, "C") for language in LANGUAGES]
        result = aggregate_reports(suites, "C")
        self.assertEqual(result["status"], "completed")
        self.assertEqual({row["condition"] for row in result["rows"]}, {"C0", "C1"})
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

    def test_actual_trial_interrupt_survives_independent_checkpoint_and_snapshot_failures(self):
        from mutual_lab.mastery_lab import safe_json
        from mutual_lab.neural import torch
        real_save = torch.save
        plan = replace(self.plan, budgets=(1,), benchmark_steps=0)
        cases = ((True, False, False), (False, True, False),
                 (True, True, False), (True, True, True))
        for fail_checkpoint, fail_snapshot, fail_suite in cases:
            with self.subTest(checkpoint=fail_checkpoint, snapshot=fail_snapshot, suite=fail_suite), \
                 tempfile.TemporaryDirectory() as temporary:
                original = KeyboardInterrupt("stop the experiment")
                cancellation_writes = []

                def persist(path, value):
                    name = Path(path).name
                    if isinstance(value, dict) and value.get("status") == "cancelled":
                        cancellation_writes.append((name, copy.deepcopy(value)))
                        if ((fail_snapshot and name == "report.json") or
                                (fail_suite and name == "mastery-suite.json")):
                            raise OSError("cancelled JSON storage unavailable")
                    return safe_json(path, value)

                def save(payload, path, *args, **kwargs):
                    if fail_checkpoint:
                        raise OSError("checkpoint storage unavailable")
                    return real_save(payload, path, *args, **kwargs)

                with patch("mutual_lab.mastery_lab.run_trial", wraps=run_trial) as trial, \
                     patch("mutual_lab.mastery_lab.evaluate_diagnostic", side_effect=original), \
                     patch("mutual_lab.mastery_lab.safe_json", side_effect=persist), \
                     patch("mutual_lab.neural.torch.save", side_effect=save) as checkpoint, \
                     self.assertRaises(ArmCancelled) as caught:
                    run_phase(experiment(), "ko", "A", plan, output=temporary, progress=None)
                self.assertIs(caught.exception.__cause__, original)
                self.assertEqual(trial.call_count, 1, "Cancellation must not start the next arm")
                self.assertEqual(checkpoint.call_count, 1, "Checkpoint persistence must still be attempted")
                report = caught.exception.report
                self.assertEqual(report["status"], "cancelled")
                self.assertEqual((report["condition"], report["replica"]), ("A0", 0))
                self.assertIn("stop the experiment", report["error"])
                json.dumps(report)  # The interrupt payload is metadata, never tensors or optimizer state.
                self.assertNotIn("model", report)
                self.assertNotIn("optimizer", report)
                self.assertEqual({item["operation"] for item in report["cancellation_persistence_errors"]},
                                 ({"checkpoint"} if fail_checkpoint else set()) |
                                 ({"snapshot"} if fail_snapshot else set()))
                self.assertTrue(any(name == "report.json" for name, _ in cancellation_writes),
                                "Snapshot persistence must be attempted even after checkpoint failure")
                suites = [value for name, value in cancellation_writes if name == "mastery-suite.json"]
                self.assertEqual(len(suites), 1)
                self.assertEqual(suites[0]["status"], "cancelled")
                self.assertEqual(suites[0]["reports"], [report],
                                 "The suite must prefer the in-memory cancelled report to a stale disk report")
                if not fail_suite:
                    path = next(Path(temporary).rglob("mastery-suite.json"))
                    self.assertEqual(json.loads(path.read_text()), json.loads(json.dumps(suites[0])))


if __name__ == "__main__":
    unittest.main()
