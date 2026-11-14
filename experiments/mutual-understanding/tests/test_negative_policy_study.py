"""E-axis provenance plus real standard/attentive 32-exposure Torch pairs."""

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
from mutual_lab.mastery_lab import MasteryPlan
from mutual_lab.native_preparation import LANGUAGES_ORDER
from mutual_lab.negative_policy_study import (FixedLanguageFactory, aggregate_negative_policy_studies,
    main, negative_policy_pair_audit, run_negative_policy_study)
from mutual_lab.profiles import ModelProfile

ROOT = Path(__file__).resolve().parents[1]


class NegativePolicyValidationTests(unittest.TestCase):
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
                    with patch("mutual_lab.negative_policy_study.run_trial", side_effect=cancel) as trial, patch(
                            "mutual_lab.negative_policy_study.safe_json", side_effect=persist), patch.object(
                            Path, "read_text", **read_behavior) as read:
                        with self.assertRaises(KeyboardInterrupt) as caught:
                            run_negative_policy_study(config, "en", output=temporary, prior_bundle={})
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
                        saved = json.loads(next(Path(temporary).rglob("negative-policy-study.json")).read_text())
                        self.assertEqual(saved["status"], "cancelled")
                        self.assertEqual(len(saved["reports"]), 1)

    def test_invalid_inputs_and_non_e_reports_fail_before_creating_an_output(self):
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            for language, model, prior, bundle in (("xx", "standard", "B1", {}),
                    ("en", "unknown", "B1", {}), ("en", "standard", "C1", {}),
                    ("en", "standard", "B1", None)):
                with self.subTest(language=language, model=model, prior=prior), self.assertRaises(ValueError):
                    run_negative_policy_study(config, language, output=output, model=model,
                                              prior_condition=prior, prior_bundle=bundle)
                self.assertFalse(output.exists())
            (Path(temporary) / "decoder-study.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "No negative-policy-study"):
                main(["--aggregate-dir", temporary])
        for report in ({"kind": "mastery-lab-v1-suite", "phase": "C"},
                       {"kind": "decoder-study-v1", "phase": "D", "axis": "decoder"}):
            with self.assertRaisesRegex(ValueError, "not E studies"):
                aggregate_negative_policy_studies([report])


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class NegativePolicyNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import development_world as world
        from mutual_lab.neural import configure_cpu
        from mutual_lab.spatial_prior import pretrain_prior
        configure_cpu()
        cls.temporary = tempfile.TemporaryDirectory()
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        cls.config = replace(config, native_training_profile=replace(config.native_training_profile, batch_size=4))
        cls.plan = MasteryPlan(budgets=(4, 8), data_seed=59003, b_seed_pair=(59007, 59009), benchmark_steps=1,
            model_profile=ModelProfile("negative-policy-test", embedding_dim=8, hidden_dim=16, max_sequence_length=128))
        full = world.development_splits(cls.config, cls.plan.data_seed)
        # Use genuine protected-safe C scenes with diverse fit-only negatives.
        fixture = {"train": full["train"][:4] + full["train"][100:104] + full["train"][-4:],
                   "dev": full["dev"][:4]}
        cls.bundle = pretrain_prior([world.render_scene(s) for s in fixture["train"]],
                                   dim=16, steps=1, batch_size=4, eval_every=1)
        cls.paths, cls.suites = {}, {}
        with patch.object(world, "development_splits", return_value=fixture), patch.object(
                world, "development_manifest", return_value={"fixture": "three-fit-one-dev-protected-safe-pairs"}):
            for model in ("standard", "attentive"):
                path = run_negative_policy_study(cls.config, "en", cls.plan, cls.temporary.name,
                                                 progress=None, prior_bundle=cls.bundle, model=model)
                cls.paths[model] = path
                cls.suites[model] = json.loads((path / "negative-policy-study.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def four_language_suites(self):
        suites = []
        for language in LANGUAGES_ORDER:
            suite = copy.deepcopy(self.suites["standard"])
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
                runpy.run_module("mutual_lab.negative_policy_study", run_name="__main__")
        cli_factory_type = stopped.exception.module_globals["FixedLanguageFactory"]
        self.assertEqual(cli_factory_type.__module__, "__main__")
        scene = world.development_splits(self.config, self.plan.data_seed)["train"][0]
        pngs, texts = [world.render_scene(scene)], [world.captions(scene, "en")[0]]
        for model_name in ("standard", "attentive"):
            imported, cli = FixedLanguageFactory(model_name, "B1"), cli_factory_type(model_name, "B1")
            a = imported(self.plan.model_profile, 59007, self.bundle, condition="B1")
            b = cli(self.plan.model_profile, 59007, self.bundle, condition="B1")
            with self.subTest(model=model_name), torch.no_grad():
                self.assertEqual(cli.__module__ + "." + cli.__qualname__,
                                 "mutual_lab.negative_policy_study.FixedLanguageFactory." + model_name)
                self.assertEqual(imported.initial_state, cli.initial_state)
                self.assertEqual(digest(a.state_dict()), digest(b.state_dict()))
                ids, _ = a.tokenizer.batch(texts)
                def logits(model):
                    if hasattr(model, "teacher_forced_logits"):
                        return model.teacher_forced_logits(ids[:, :-1], pngs)
                    return model.producer(ids[:, :-1], model.visual(pngs))[0]
                self.assertTrue(torch.equal(logits(a), logits(b)))

    def test_real_policy_pairs_preserve_full_initial_state_final_producer_and_prior(self):
        for model, suite in self.suites.items():
            with self.subTest(model=model):
                self.assertEqual(suite["status"], "completed", suite)
                self.assertEqual(suite["pair_audit"]["status"], "matched", suite["pair_audit"])
                self.assertTrue(all(suite["pair_audit"]["checks"].values()))
                zero, one = suite["reports"]
                self.assertEqual(zero["initial_state"], one["initial_state"])
                a, b = zero["engine_report"], one["engine_report"]
                self.assertEqual(a["final_producer_sha256"], b["final_producer_sha256"])
                self.assertNotEqual(a["final_understanding_sha256"], b["final_understanding_sha256"])
                self.assertEqual(a["initial_prior_sha256"], b["final_prior_sha256"])
                self.assertEqual(a["frozen_feature_policy"], "per_png_singleton_v1")
                self.assertEqual(a["frozen_feature_tables"], b["frozen_feature_tables"])
                self.assertEqual(a["frozen_feature_tables"]["step_0"], a["frozen_feature_tables"]["final"])
                self.assertEqual((a["policy"], b["policy"]), ("same_pair", "mixed_blocks"))
                self.assertEqual(b["negative_exposures"]["branch_same_pair_examples"], 16)
                self.assertEqual(b["negative_exposures"]["branch_attributes_examples"], 16)
                self.assertEqual(a["negative_exposures"]["examples"], 32)
                for record in (zero, one):
                    raw = json.loads((self.paths[model] / record["raw_report_path"]).read_text())
                    self.assertEqual(raw, record["engine_report"])
                    self.assertEqual((raw["phase"], raw["condition"]), ("C", "C1"))
                    self.assertTrue(raw["benchmark"]["initial_weights_restored"])
                self.assertTrue(suite["capacity_matched"])

    def test_first_sixteen_exposures_are_identical_then_only_understanding_diverges(self):
        from mutual_lab.neural import digest, torch
        for model, suite in self.suites.items():
            states = []
            for record in suite["reports"]:
                directory = self.paths[model] / Path(record["raw_report_path"]).parent
                states.append(torch.load(directory / "step-000004.pt", map_location="cpu", weights_only=True))
            with self.subTest(model=model):
                self.assertEqual(digest(states[0]["model"]), digest(states[1]["model"]))
                self.assertEqual(digest(states[0]["negative_rng_state"]), digest(states[1]["negative_rng_state"]))
                self.assertEqual(suite["reports"][0]["engine_report"]["curves"][1]["dev"],
                                 suite["reports"][1]["engine_report"]["curves"][1]["dev"])

    def test_factory_fixed_b0_choice_is_exact_and_cannot_be_reused_or_retargeted(self):
        factory = FixedLanguageFactory("attentive", "B0")
        with self.assertRaises(ValueError):
            factory(self.plan.model_profile, 59007, self.bundle, condition="B1")
        model = factory(self.plan.model_profile, 59007, self.bundle, condition="B0")
        self.assertEqual(factory.initial_state["prior_sha256"], self.bundle["report"]["random_prior_sha256"])
        self.assertEqual(model.spatial_prior_metadata["condition"], "B0")
        with self.assertRaises(ValueError):
            factory(self.plan.model_profile, 59007, self.bundle, condition="B0")

    def test_audit_detects_producer_leakage_schedule_capacity_and_policy_label_errors(self):
        mutations = (("final_producer_sha256", "changed", "final_producer_sha256"),
                     ("policy", "attributes", "negative_policy_axis"),
                     ("parameter_count", 1, "parameter_count"),
                     ("choice_count", 5, "four_choices"),
                     ("meaning_stream_sha256", "changed", "meaning_stream_sha256"),
                     ("condition", "C0", "engine_adapter"),
                     ("frozen_feature_policy", "old_batched_cache", "deterministic_frozen_features"),
                     ("frozen_feature_tables", {}, "deterministic_frozen_features"))
        for key, value, check in mutations:
            reports = copy.deepcopy(self.suites["standard"]["reports"])
            reports[1]["engine_report"][key] = value
            with self.subTest(key=key):
                audit = negative_policy_pair_audit(reports, "B1", "standard")
                self.assertEqual(audit["status"], "mismatch")
                self.assertFalse(audit["checks"][check])
        reports = copy.deepcopy(self.suites["standard"]["reports"])
        reports[1]["engine_report"]["negative_exposures"]["branch_attributes_examples"] = 15
        self.assertFalse(negative_policy_pair_audit(reports, "B1", "standard")["checks"]["mixed_branch_schedule"])

    def test_aggregate_uses_e_labels_and_retains_missing_failed_and_per_budget_counts(self):
        suites = self.four_language_suites()
        result = aggregate_negative_policy_studies(suites)
        self.assertEqual(result["status"], "completed")
        self.assertEqual((result["model"], result["fixed_prior_condition"]), ("standard", "B1"))
        self.assertEqual(result["negative_policies"]["E1"]["block_exposures"], 16)
        self.assertEqual({row["condition"] for row in result["rows"]}, {"E0", "E1"})
        self.assertEqual(len(result["rows"]), 24)
        self.assertEqual(len(result["paired_deltas"]), 12)
        self.assertTrue(all(row["generation_delta"] == 0 for row in result["paired_deltas"]))
        self.assertTrue(all(row["completed_rows"] == 4 and row["comprehension_items"] == 64
            and row["generation_scenes"] == 16 for row in result["budget_counts"]))
        suites[0]["reports"].pop()
        suites[1]["reports"][1]["status"] = "failed"
        result = aggregate_negative_policy_studies(suites[:-1])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_languages"], [LANGUAGES_ORDER[-1]])
        self.assertTrue(any(r.get("status") == "missing" for r in result["failed_trials"]))
        self.assertTrue(any(r.get("status") == "failed" for r in result["failed_trials"]))
        self.assertTrue(all(row["completed_rows"] == 1 for row in result["budget_counts"] if row["condition"] == "E1"))

    def test_aggregate_rejects_changed_model_provenance_duplicates_and_missing_budgets(self):
        suites = self.four_language_suites()
        suites[-1]["reports"][0]["engine_report"]["source_sha256"] = {"changed": "source"}
        with self.assertRaisesRegex(ValueError, "provenance"):
            aggregate_negative_policy_studies(suites)
        suites = self.four_language_suites()
        suites[0]["reports"].append(copy.deepcopy(suites[0]["reports"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_negative_policy_studies(suites)
        suites = self.four_language_suites()
        suites[0]["reports"][1]["engine_report"]["curves"].pop()
        result = aggregate_negative_policy_studies(suites)
        self.assertTrue(any(r["status"] == "missing_budget" for r in result["failed_trials"]))
        self.assertEqual(result["status"], "incomplete")

    def test_attention_checkpoint_remains_development_only_and_research_loader_rejects_it(self):
        from mutual_lab.neural import load_checkpoint, torch
        record = self.suites["attentive"]["reports"][1]
        path = self.paths["attentive"] / Path(record["raw_report_path"]).parent / "step-000008.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(state["format"], "mutual-mastery-development-v1")
        self.assertEqual(state["negative_exposure_offset"], 32)
        self.assertIn("producer.attention_output.weight", state["model"])
        self.assertTrue(state["optimizer"]["state"])
        with patch("mutual_lab.neural.torch.load") as deserialize:
            with self.assertRaises(ValueError):
                load_checkpoint(path, expected_kind="native")
            deserialize.assert_not_called()

    def test_failures_and_cancellation_are_saved_without_claiming_a_completed_pair(self):
        with patch("mutual_lab.negative_policy_study.run_trial", side_effect=ValueError("fixture failure")):
            path = run_negative_policy_study(self.config, "en", self.plan, self.temporary.name,
                                            progress=None, prior_bundle=self.bundle)
        suite = json.loads((path / "negative-policy-study.json").read_text())
        self.assertEqual([r["status"] for r in suite["reports"]], ["failed", "failed"])
        self.assertEqual(suite["status"], "incomplete")
        with tempfile.TemporaryDirectory() as temporary, patch(
                "mutual_lab.negative_policy_study.run_trial", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_negative_policy_study(self.config, "en", self.plan, temporary,
                                         progress=None, prior_bundle=self.bundle)
            path = next(Path(temporary).rglob("negative-policy-study.json"))
            suite = json.loads(path.read_text())
            self.assertEqual(suite["status"], "cancelled")
            self.assertEqual(suite["reports"][0]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
