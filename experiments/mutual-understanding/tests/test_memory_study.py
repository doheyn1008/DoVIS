"""Outside-package F preparation checks; tiny trials never qualify a model."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
from mutual_lab import memory_study as memory

from mutual_lab.config import ExperimentConfig
from mutual_lab.mastery_lab import MasteryPlan
from mutual_lab.native_preparation import LANGUAGES_ORDER
from mutual_lab.profiles import ModelProfile


class MemoryValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        self.plan = memory._reference_plan()

    def test_invalid_protocols_and_raw_cde_manifests_are_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            for changed in (replace(self.plan, budgets=(2400,)), replace(self.plan, production_min=.8),
                    replace(self.plan, data_seed=29003), replace(self.plan, b_seed_pair=(69007, 69009))):
                with self.subTest(plan=changed), self.assertRaises(ValueError):
                    memory.run_memory_study(self.config, "en", changed, output, prior_bundle={})
                self.assertFalse(output.exists())
            for language, policy, bundle in (("xx", "mixed_blocks", {}), ("en", "unknown", {}), ("en", "same_pair", None)):
                with self.assertRaises(ValueError):
                    memory.run_memory_study(self.config, language, self.plan, output, prior_bundle=bundle, policy=policy)
                self.assertFalse(output.exists())
        for phase in ("C", "D", "E"):
            with self.assertRaisesRegex(ValueError, "not F"):
                memory.aggregate_memory_studies([{"kind": phase, "phase": phase}])
        for conditions in ([{"memory_condition": "F0"}] * 2, [{"memory_condition": "F2"}]):
            with self.assertRaisesRegex(ValueError, "Duplicate or unknown"):
                memory.memory_pair_audit(conditions, "mixed_blocks")

    def test_cancelled_memory_report_survives_unreadable_disk_and_snapshot_failure_without_starting_f1(self):
        from mutual_lab import development_world as world
        from mutual_lab.composition_suite import ArmCancelled
        from mutual_lab.diagnostics import safe_json
        raw = {"status": "cancelled", "sentinel": "captured-before-fs-failure"}
        for source in ("memory", "disk", "unreadable"):
            for failure in (None, OSError, KeyboardInterrupt):
                with self.subTest(source=source, failure=failure), tempfile.TemporaryDirectory() as temporary:
                    interrupt = ArmCancelled(Path(temporary) / "missing.json")
                    if source == "memory":
                        interrupt.report = MappingProxyType(raw)
                    snapshots = []

                    def cancel(*args, model_factory, **kwargs):
                        model_factory.initial_state = {"captured": True}
                        raise interrupt

                    def persist(path, payload):
                        snapshots.append(copy.deepcopy(payload))
                        if len(snapshots) > 1 and failure is not None:
                            raise failure("snapshot failed")
                        safe_json(path, payload)

                    reader = {"return_value": json.dumps(raw)} if source == "disk" else {"side_effect": OSError("unreadable")}
                    with patch.object(memory, "_reference_plan", return_value=self.plan), patch.object(
                            world, "development_splits", return_value={"train": (), "dev": ()}), patch.object(
                            memory, "run_trial", side_effect=cancel) as execute, patch.object(
                            memory, "safe_json", side_effect=persist), patch.object(Path, "read_text", **reader) as read:
                        with self.assertRaises(KeyboardInterrupt) as caught:
                            memory.run_memory_study(self.config, "en", self.plan, temporary, prior_bundle={})
                    self.assertIs(caught.exception, interrupt)
                    execute.assert_called_once()
                    self.assertEqual(read.call_count, int(source != "memory"))
                    saved = snapshots[-1]
                    self.assertEqual(saved["status"], "cancelled")
                    self.assertEqual(len(saved["reports"]), 1)
                    self.assertEqual(saved["reports"][0]["memory_condition"], "F0")
                    self.assertEqual(saved["reports"][0]["initial_state"], {"captured": True})
                    if source != "unreadable":
                        self.assertEqual(saved["reports"][0]["engine_report"], raw)

    def test_request_requires_explicit_fixed_model_policy_and_matching_yx_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            content = {"schema_version": 1, "model": "attentive", "fixed_prior_condition": "B1", "policy": "mixed_blocks",
                "fixed_final_step": 9600, "data_version": "five-colors-three-shapes-v1",
                "memory_normalization": "l2_first32_raw_yx_v1", "rationale": "conditional preparation"}
            path.write_text(json.dumps(content))
            self.assertEqual(memory._request(path, self.plan, "mixed_blocks")["content"], content)
            with self.assertRaisesRegex(ValueError, "request differs"):
                memory._request(path, self.plan, "same_pair")
            content["memory_normalization"] = "l2_first32_raw_xy_v1"
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "request differs"):
                memory._request(path, self.plan, "mixed_blocks")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU Torch")
class MemoryNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import development_world as world
        from mutual_lab.neural import configure_cpu
        from mutual_lab.spatial_prior import pretrain_prior, save_prior_bundle, load_prior_bundle
        configure_cpu()
        cls.temporary = tempfile.TemporaryDirectory()
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        cls.config = replace(config, native_training_profile=replace(config.native_training_profile, batch_size=4))
        cls.plan = replace(memory._reference_plan(), budgets=(4, 8), benchmark_steps=1,
            model_profile=ModelProfile("memory-study-test", embedding_dim=8, hidden_dim=16, max_sequence_length=128))
        full = world.development_splits(cls.config, 59003)
        cls.fixture = {"train": full["train"][:4] + full["train"][100:104] + full["train"][-4:], "dev": full["dev"][:4]}
        cls.pngs = tuple(world.render_scene(scene) for scenes in cls.fixture.values() for scene in scenes)
        cls.prior_path = save_prior_bundle(pretrain_prior([world.render_scene(scene) for scene in cls.fixture["train"]],
            dim=16, steps=1, batch_size=4, eval_every=1), Path(cls.temporary.name) / "prior.pt")
        cls.bundle = load_prior_bundle(cls.prior_path)
        with patch.object(memory, "_reference_plan", return_value=cls.plan), patch.object(
                world, "development_splits", return_value=cls.fixture), patch.object(
                world, "development_manifest", return_value={"fixture": "four protected-safe C pairs"}):
            cls.path = memory.run_memory_study(cls.config, "en", cls.plan, cls.temporary.name,
                progress=None, prior_bundle=cls.bundle, policy="mixed_blocks")
        cls.suite = json.loads((cls.path / "memory-study.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def aggregate(self, suites):
        with patch.object(memory, "_reference_plan", return_value=self.plan):
            return memory.aggregate_memory_studies(suites, self.plan)

    def four_languages(self):
        suites = []
        for language in LANGUAGES_ORDER:
            suite = copy.deepcopy(self.suite)
            suite["language"] = language
            for record in suite["reports"]:
                record["engine_report"]["language"] = language
                record["engine_report"]["training_languages"] = [language]
                record["engine_report"]["data_sha256"] = "synthetic-native-captions-" + language
            suites.append(suite)
        return suites

    def test_real_f_pair_has_equal_full_initialization_final_understanding_and_frozen_prior(self):
        self.assertEqual(self.suite["status"], "completed", self.suite)
        self.assertEqual(self.suite["pair_audit"]["status"], "matched", self.suite["pair_audit"])
        self.assertTrue(all(self.suite["pair_audit"]["checks"].values()))
        zero, one = self.suite["reports"]
        for key in ("full_state_sha256", "understanding_sha256", "producer_sha256", "prior_sha256",
                    "parameter_count", "trainable_parameter_count"):
            self.assertEqual(zero["initial_state"][key], one["initial_state"][key], key)
        a, b = zero["engine_report"], one["engine_report"]
        self.assertEqual(a["final_understanding_sha256"], b["final_understanding_sha256"])
        self.assertNotEqual(a["final_producer_sha256"], b["final_producer_sha256"])
        self.assertEqual(a["frozen_feature_tables"]["step_0"]["caches"]["visual_cache"],
                         b["frozen_feature_tables"]["step_0"]["caches"]["visual_cache"])
        self.assertNotEqual(a["frozen_feature_tables"]["step_0"], b["frozen_feature_tables"]["step_0"])
        for record in (zero, one):
            audit = record["initial_state"]["memory_preflight"]
            self.assertEqual(audit["input_pngs"], len(self.pngs))
            self.assertEqual(audit["successful_pngs"], len(self.pngs))
            self.assertEqual(audit["fallback_count"], 0)
            self.assertTrue(audit["model_state_unchanged"])
            self.assertTrue(audit["rng_unchanged"])
            raw = record["engine_report"]
            self.assertEqual(raw["negative_exposures"]["branch_attributes_examples"], 16)
            self.assertEqual(raw["negative_exposures"]["branch_same_pair_examples"], 16)
            self.assertTrue(raw["benchmark"]["initial_weights_restored"])
            self.assertEqual(raw["frozen_feature_policy"], "per_png_singleton_v1")
            self.assertEqual(raw["frozen_feature_tables"]["step_0"], raw["frozen_feature_tables"]["final"])
            self.assertEqual([point["step"] for point in raw["curves"]], [0, 4, 8])
            self.assertTrue(all(point["candidate"]["status"] == "indeterminate_coverage" for point in raw["curves"]))
            self.assertEqual(raw["model_factory"], "mutual_lab.memory_study.RecordingFactory." + record["memory_condition"])
        self.assertTrue(self.suite["capacity_matched"])
        self.assertIn("NEW normalized CNN", self.suite["interpretation"])
        self.assertFalse(self.suite["eligible_for_fresh_experiment"])

    def test_final_f_checkpoints_restore_exact_saved_generation_with_canonical_cache_policy(self):
        from mutual_lab import development_world as world
        from mutual_lab.mastery_infer import load_development_model, infer_development
        for record in self.suite["reports"]:
            directory = self.path / Path(record["raw_report_path"]).parent
            model = load_development_model(directory / "step-000008.pt", self.prior_path)
            self.assertEqual(model.mastery_inference_metadata["frozen_feature_policy"], "per_png_singleton_v1")
            self.assertEqual(model.object_memory_source, record["memory_source"])
            generated = infer_development(model, world.render_scene(self.fixture["dev"][0]))["generation"]
            rows = json.loads((directory / "generations.json").read_text())
            expected = next(row for row in rows if row["step"] == 8 and row["split"] == "dev")
            self.assertEqual(generated, {key: expected[key] for key in generated})

    def test_actual_factories_preserve_initial_teacher_logits_and_validate_pngs_without_fallback(self):
        from mutual_lab import development_world as world
        from mutual_lab.neural import digest, torch
        factories = [memory.RecordingFactory(condition, self.pngs) for condition in ("F0", "F1")]
        a, b = [factory(self.plan.model_profile, 59007, self.bundle) for factory in factories]
        self.assertEqual(digest(a.state_dict()), digest(b.state_dict()))
        ids, _ = a.tokenizer.batch([world.captions(self.fixture["train"][0], "en")[0]])
        with torch.no_grad():
            self.assertTrue(torch.equal(a.teacher_forced_logits(ids[:, :-1], [self.pngs[0]]),
                                        b.teacher_forced_logits(ids[:, :-1], [self.pngs[0]])))
            self.assertFalse(torch.equal(a.visual_grid([self.pngs[0]]), b.visual_grid([self.pngs[0]])))
        self.assertEqual(factories[0].__module__, "mutual_lab.memory_study")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            factories[0](self.plan.model_profile, 59007, self.bundle)
        failed = memory.RecordingFactory("F1", [b"not a PNG"])
        with self.assertRaisesRegex(ValueError, "PNG"):
            failed(self.plan.model_profile, 59007, self.bundle)
        self.assertEqual(failed.initial_state["memory_preflight"]["status"], "failed")
        self.assertEqual(failed.initial_state["memory_preflight"]["fallback_count"], 0)
        self.assertTrue(failed.initial_state["memory_preflight"]["model_state_unchanged"])

    def test_pair_audit_detects_understanding_stream_capacity_memory_and_adapter_leakage(self):
        for key, value, check in (("final_understanding_sha256", "changed", "final_understanding_sha256"),
                ("meaning_stream_sha256", "changed", "meaning_stream_sha256"),
                ("condition", "C0", "engine_adapter"), ("parameter_count", 1, "parameter_count"),
                ("policy", "same_pair", "fixed_policy")):
            records = copy.deepcopy(self.suite["reports"])
            records[1]["engine_report"][key] = value
            with self.subTest(key=key):
                self.assertFalse(memory.memory_pair_audit(records, "mixed_blocks")["checks"][check])
        for key, value, check in (("full_state_sha256", "changed", "full_initial_state"),
                                  ("residual_vocabulary_projection_zero", False, "captured_initialization")):
            records = copy.deepcopy(self.suite["reports"])
            records[1]["initial_state"][key] = value
            self.assertFalse(memory.memory_pair_audit(records, "mixed_blocks")["checks"][check])
        records = copy.deepcopy(self.suite["reports"])
        records[1]["initial_state"]["memory_preflight"]["fallback_count"] = 1
        self.assertFalse(memory.memory_pair_audit(records, "mixed_blocks")["checks"]["png_preflight"])

    def test_aggregation_preserves_all_budget_counts_missing_failed_and_no_selection(self):
        suites = self.four_languages()
        result = self.aggregate(suites)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["rows"]), 24)
        self.assertEqual(len(result["paired_deltas"]), 12)
        self.assertTrue(all(row["comprehension_delta"] == 0 for row in result["paired_deltas"]))
        self.assertTrue(all(row["completed_rows"] == 4 and row["generation_scenes"] == 16 for row in result["budget_counts"]))
        self.assertTrue(result["selection"].startswith("undetermined"))
        suites[0]["reports"].pop()
        suites[1]["reports"][1]["status"] = "failed"
        result = self.aggregate(suites[:-1])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_languages"], [LANGUAGES_ORDER[-1]])
        self.assertTrue(any(item["status"] == "missing" for item in result["failed_trials"]))
        self.assertTrue(any(item["status"] == "failed" for item in result["failed_trials"]))
        suites = self.four_languages()
        suites[-1]["reports"][0]["engine_report"]["source_sha256"] = {"changed": "source"}
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.aggregate(suites)


if __name__ == "__main__":
    unittest.main()
