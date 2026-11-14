"""Decoder-axis provenance, coverage and a real two-step paired L1 fixture."""

import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from mutual_lab.config import ExperimentConfig
from mutual_lab.decoder_study import (RecordingFactory, aggregate_decoder_studies,
                                      decoder_pair_audit, main, run_decoder_study)
from mutual_lab.mastery_lab import MasteryPlan
from mutual_lab.native_preparation import LANGUAGES_ORDER
from mutual_lab.profiles import ModelProfile

ROOT = Path(__file__).resolve().parents[1]


class DecoderValidationTests(unittest.TestCase):
    def test_cancellation_prefers_memory_and_never_starts_another_arm_when_persistence_fails(self):
        from mutual_lab.composition_suite import ArmCancelled
        from mutual_lab.diagnostics import safe_json
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        report = {"status": "cancelled", "sentinel": "captured-before-disk-failure"}
        for source in ("memory", "disk", "unreadable"):
            for failure in (None, OSError, KeyboardInterrupt):
                with self.subTest(source=source, failure=failure), tempfile.TemporaryDirectory() as temporary:
                    interrupt = ArmCancelled(Path(temporary) / "unavailable-report.json")
                    if source == "memory":
                        interrupt.report = MappingProxyType(report)
                    snapshots = []

                    def cancel(*args, model_factory, **kwargs):
                        model_factory.initial_state = {"captured_initialization": True}
                        raise interrupt

                    def persist(path, payload):
                        snapshots.append(copy.deepcopy(payload))
                        if len(snapshots) > 1 and failure is not None:
                            raise failure("cancellation persistence failed")
                        safe_json(path, payload)

                    read_behavior = ({"return_value": json.dumps(report)} if source == "disk" else
                                     {"side_effect": OSError("report unreadable")})
                    with patch("mutual_lab.decoder_study.run_trial", side_effect=cancel) as trial, patch(
                            "mutual_lab.decoder_study.safe_json", side_effect=persist), patch.object(
                            Path, "read_text", **read_behavior) as read:
                        with self.assertRaises(KeyboardInterrupt) as caught:
                            run_decoder_study(config, "en", output=temporary, prior_bundle={})
                    self.assertIs(caught.exception, interrupt)
                    trial.assert_called_once()
                    self.assertEqual(read.call_count, int(source != "memory"))
                    self.assertEqual(len(snapshots), 2)
                    cancelled = snapshots[-1]
                    self.assertEqual(cancelled["status"], "cancelled")
                    self.assertEqual(len(cancelled["reports"]), 1)
                    record = cancelled["reports"][0]
                    self.assertEqual(record["status"], "cancelled")
                    self.assertEqual(record["initial_state"], {"captured_initialization": True})
                    if source != "unreadable":
                        self.assertEqual(record["engine_report"], report)
                    if failure is None:
                        saved = json.loads(next(Path(temporary).rglob("decoder-study.json")).read_text())
                        self.assertEqual(saved["status"], "cancelled")
                        self.assertEqual(len(saved["reports"]), 1)

    def test_invalid_inputs_fail_without_creating_an_output(self):
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            for language, prior, bundle in (("xx", "B1", {}), ("en", "C1", {}), ("en", "B1", None)):
                with self.subTest(language=language, prior=prior), self.assertRaises(ValueError):
                    run_decoder_study(config, language, output=output, prior_condition=prior, prior_bundle=bundle)
                self.assertFalse(output.exists())
        for condition, prior in (("C0", "B1"), ("D1", "C1")):
            with self.assertRaises(ValueError):
                RecordingFactory(condition, prior)

    def test_raw_c_artifacts_are_never_accepted_as_decoder_manifests(self):
        with self.assertRaisesRegex(ValueError, "raw C engine"):
            aggregate_decoder_studies([{"kind": "mastery-lab-v1-suite", "phase": "C"}])
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "mastery-suite.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "No decoder-study"):
                main(["--aggregate-dir", temporary])
        with self.assertRaises(ValueError):
            decoder_pair_audit([{"decoder_condition": "D0"}, {"decoder_condition": "D0"}], "B1")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class DecoderNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import development_world as world
        from mutual_lab.neural import configure_cpu
        from mutual_lab.spatial_prior import pretrain_prior
        configure_cpu()
        cls.temporary = tempfile.TemporaryDirectory()
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        cls.config = replace(config, native_training_profile=replace(config.native_training_profile, batch_size=4))
        cls.plan = MasteryPlan(budgets=(1, 2), data_seed=59003, b_seed_pair=(59007, 59009), benchmark_steps=1,
            model_profile=ModelProfile("decoder-test", embedding_dim=8, hidden_dim=16, max_sequence_length=128))
        full = world.development_splits(cls.config, cls.plan.data_seed)
        # Real C render/caption/evaluation/training paths on two protected-safe
        # four-arrangement pairs; small coverage deliberately remains indeterminate.
        fixture = {key: scenes[:4] for key, scenes in full.items()}
        cls.bundle = pretrain_prior([world.render_scene(scene) for scene in fixture["train"]],
                                   dim=16, steps=1, batch_size=4, eval_every=1)
        with patch.object(world, "development_splits", return_value=fixture), patch.object(
                world, "development_manifest", return_value={"fixture": "two-development-pairs"}):
            cls.path = run_decoder_study(cls.config, "en", cls.plan, cls.temporary.name,
                                        progress=None, prior_bundle=cls.bundle)
        cls.suite = json.loads((cls.path / "decoder-study.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def four_language_suites(self):
        suites = []
        for language in LANGUAGES_ORDER:
            suite = copy.deepcopy(self.suite)
            suite["language"] = language
            for record in suite["reports"]:
                record["engine_report"]["language"] = language
                record["engine_report"]["training_languages"] = [language]
                record["engine_report"]["data_sha256"] = "caption-data-" + language
            suites.append(suite)
        return suites

    def test_real_module_cli_factory_has_canonical_identity_and_unchanged_model_outputs(self):
        import argparse
        import inspect
        import runpy
        import warnings
        from mutual_lab import development_world as world
        from mutual_lab.neural import digest, torch

        class StopBeforeTraining(Exception):
            pass

        def intercept(_parser, *args, **kwargs):
            stopped = StopBeforeTraining()
            stopped.module_globals = inspect.currentframe().f_back.f_globals
            raise stopped

        with warnings.catch_warnings(), patch.object(argparse.ArgumentParser, "parse_args", new=intercept):
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(StopBeforeTraining) as stopped:
                runpy.run_module("mutual_lab.decoder_study", run_name="__main__")
        cli_factory_type = stopped.exception.module_globals["RecordingFactory"]
        self.assertEqual(cli_factory_type.__module__, "__main__")
        scene = world.development_splits(self.config, self.plan.data_seed)["train"][0]
        pngs, texts = [world.render_scene(scene)], [world.captions(scene, "en")[0]]
        for condition in ("D0", "D1"):
            imported, cli = RecordingFactory(condition, "B1"), cli_factory_type(condition, "B1")
            a = imported(self.plan.model_profile, 59007, self.bundle, condition="B1")
            b = cli(self.plan.model_profile, 59007, self.bundle, condition="B1")
            with self.subTest(condition=condition), torch.no_grad():
                self.assertEqual(cli.__module__ + "." + cli.__qualname__,
                                 "mutual_lab.decoder_study.RecordingFactory." + condition)
                self.assertEqual(imported.initial_state, cli.initial_state)
                self.assertEqual(digest(a.state_dict()), digest(b.state_dict()))
                ids, _ = a.tokenizer.batch(texts)
                def logits(model):
                    if hasattr(model, "teacher_forced_logits"):
                        return model.teacher_forced_logits(ids[:, :-1], pngs)
                    return model.producer(ids[:, :-1], model.visual(pngs))[0]
                self.assertTrue(torch.equal(logits(a), logits(b)))

    def test_actual_pair_matches_inputs_understanding_and_prior_but_changes_decoder(self):
        self.assertEqual(self.suite["status"], "completed", self.suite)
        audit = self.suite["pair_audit"]
        self.assertEqual(audit["status"], "matched", audit)
        self.assertTrue(all(audit["checks"].values()))
        zero, one = self.suite["reports"]
        self.assertEqual([r["decoder_condition"] for r in self.suite["reports"]], ["D0", "D1"])
        for key in ("understanding_sha256", "base_producer_sha256", "prior_sha256"):
            self.assertEqual(zero["initial_state"][key], one["initial_state"][key], key)
        self.assertEqual(one["initial_state"]["decoder_extra_parameter_count"], 5488)
        self.assertNotEqual(zero["engine_report"]["initial_producer_sha256"], one["engine_report"]["initial_producer_sha256"])
        self.assertNotEqual(zero["engine_report"]["final_producer_sha256"], one["engine_report"]["final_producer_sha256"])
        self.assertFalse(self.suite["capacity_matched"])
        for record in self.suite["reports"]:
            raw = json.loads((self.path / record["raw_report_path"]).read_text())
            self.assertEqual(raw, record["engine_report"])
            self.assertEqual((raw["phase"], raw["condition"]), ("C", "C1"))
            self.assertTrue(raw["benchmark"]["initial_weights_restored"])
            self.assertEqual(raw["training_steps_completed"], 2)
            self.assertTrue(all(point["candidate"]["status"] == "indeterminate_coverage" for point in raw["curves"]))

    def test_raw_development_checkpoint_retains_rng_optimizer_and_load_guard(self):
        from mutual_lab.neural import load_checkpoint, torch
        record = self.suite["reports"][1]
        path = self.path / Path(record["raw_report_path"]).parent / "step-000002.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(state["format"], "mutual-mastery-development-v1")
        self.assertEqual((state["phase"], state["condition"]), ("C", "C1"))
        self.assertIn("producer.attention_output.weight", state["model"])
        self.assertTrue(state["optimizer"]["state"])
        self.assertEqual(state["negative_exposure_offset"], 8)
        self.assertIn("torch_rng_state", state)
        with patch("mutual_lab.neural.torch.load") as deserialize:
            with self.assertRaises(ValueError):
                load_checkpoint(path, expected_kind="native")
            deserialize.assert_not_called()

    def test_actual_factory_captures_once_and_rejects_a_different_prior(self):
        factory = RecordingFactory("D0", "B0")
        with self.assertRaisesRegex(ValueError, "declared common prior"):
            factory(self.plan.model_profile, 59007, self.bundle, condition="B1")
        model = factory(self.plan.model_profile, 59007, self.bundle, condition="B0")
        self.assertEqual(factory.initial_state["prior_sha256"], self.bundle["report"]["random_prior_sha256"])
        self.assertEqual(factory.initial_state["decoder_extra_parameter_count"], 0)
        self.assertEqual(model.spatial_prior_metadata["condition"], "B0")
        with self.assertRaises(ValueError):
            factory(self.plan.model_profile, 59007, self.bundle, condition="B0")

    def test_pair_audit_detects_branch_leakage_and_engine_or_parameter_mislabeling(self):
        mutations = (
            ("final_understanding_sha256", "changed", "final_understanding_sha256"),
            ("condition", "C0", "engine_adapter"),
            ("source_sha256", {}, "source_sha256"),
            ("meaning_stream_sha256", "changed", "meaning_stream_sha256"),
            ("negative_exposures", {}, "negative_exposures"),
            ("status", "failed", "engine_completion"),
            ("protected_splits_used", ["final"], "input_boundary"),
        )
        for key, value, check in mutations:
            changed = copy.deepcopy(self.suite["reports"])
            changed[1]["engine_report"][key] = value
            with self.subTest(key=key):
                audit = decoder_pair_audit(changed, "B1")
                self.assertEqual(audit["status"], "mismatch")
                self.assertFalse(audit["checks"][check])
        changed = copy.deepcopy(self.suite["reports"])
        changed[1]["initial_state"]["decoder_extra_parameter_count"] = 1
        self.assertFalse(decoder_pair_audit(changed, "B1")["checks"]["parameter_accounting"])

    def test_aggregate_labels_decoder_axis_and_preserves_per_budget_counts_and_failures(self):
        suites = self.four_language_suites()
        result = aggregate_decoder_studies(suites)
        self.assertEqual(result["status"], "completed")
        self.assertEqual({row["condition"] for row in result["rows"]}, {"D0", "D1"})
        self.assertEqual(len(result["rows"]), 24)
        self.assertEqual(len(result["paired_deltas"]), 12)
        self.assertTrue(all(row["comprehension_delta"] == 0 for row in result["paired_deltas"]))
        self.assertTrue(all(row["completed_rows"] == 4 and row["comprehension_items"] == 64
                            and row["generation_scenes"] == 16 for row in result["budget_counts"]))
        suites[0]["reports"].pop()
        suites[1]["reports"][1]["status"] = "failed"
        result = aggregate_decoder_studies(suites[:-1])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_languages"], [LANGUAGES_ORDER[-1]])
        self.assertTrue(any(row.get("status") == "missing" for row in result["failed_trials"]))
        self.assertTrue(any(row.get("status") == "failed" for row in result["failed_trials"]))
        self.assertTrue(all(row["completed_rows"] == 1 for row in result["budget_counts"] if row["condition"] == "D1"))

    def test_aggregate_rejects_cross_language_source_drift_duplicate_arms_and_missing_budgets(self):
        suites = self.four_language_suites()
        suites[-1]["reports"][1]["engine_report"]["source_sha256"] = {"different": "source"}
        with self.assertRaisesRegex(ValueError, "provenance"):
            aggregate_decoder_studies(suites)
        suites = self.four_language_suites()
        suites[0]["reports"].append(copy.deepcopy(suites[0]["reports"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_decoder_studies(suites)
        suites = self.four_language_suites()
        suites[0]["reports"][1]["engine_report"]["curves"].pop()
        result = aggregate_decoder_studies(suites)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any(row["status"] == "missing_budget" for row in result["failed_trials"]))
        suites = self.four_language_suites()
        suites[0]["reports"][0]["engine_report"]["language"] = "xx"
        with self.assertRaisesRegex(ValueError, "manifest"):
            aggregate_decoder_studies(suites)

    def test_setup_failure_is_recorded_for_both_arms_instead_of_disappearing(self):
        with patch("mutual_lab.decoder_study.run_trial", side_effect=ValueError("fixture setup failure")):
            path = run_decoder_study(self.config, "en", self.plan, self.temporary.name,
                                     progress=None, prior_bundle=self.bundle)
        suite = json.loads((path / "decoder-study.json").read_text())
        self.assertEqual(suite["status"], "incomplete")
        self.assertEqual([r["status"] for r in suite["reports"]], ["failed", "failed"])
        self.assertTrue(all("fixture setup failure" in r["error"] for r in suite["reports"]))


if __name__ == "__main__":
    unittest.main()
