"""Outside-package G protocol/cancellation audits and genuine tiny L1 pairs."""

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
from mutual_lab import grounding_study as grounding

from mutual_lab.config import ExperimentConfig
from mutual_lab.native_preparation import LANGUAGES_ORDER
from mutual_lab.profiles import ModelProfile


class GroundingValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        self.plan = grounding._reference_plan()

    def test_protocol_and_producer_are_fixed_before_outputs_and_non_g_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            for plan in (replace(self.plan, budgets=(2400,)), replace(self.plan, data_seed=29003),
                         replace(self.plan, b_seed_pair=(69007, 69009)), replace(self.plan, contrast_min=.8)):
                with self.subTest(plan=plan), self.assertRaises(ValueError):
                    grounding.run_grounding_study(self.config, "en", plan, output, producer_memory="cnn", prior_bundle={})
                self.assertFalse(output.exists())
            for language, memory, prior in (("xx", "cnn", {}), ("en", None, {}), ("en", "objects", None)):
                with self.assertRaises(ValueError):
                    grounding.run_grounding_study(self.config, language, self.plan, output, producer_memory=memory, prior_bundle=prior)
                self.assertFalse(output.exists())
        for phase in ("C", "D", "E", "F"):
            with self.assertRaisesRegex(ValueError, "not G"):
                grounding.aggregate_grounding_studies([{"kind": phase, "phase": phase}])
        for rows in ([{"grounding_condition": "G0"}] * 2, [{"grounding_condition": "G2"}]):
            with self.assertRaisesRegex(ValueError, "Duplicate or unknown"):
                grounding.grounding_pair_audit(rows, "cnn")

    def test_all_four_canonical_factory_ids_encode_understanding_and_producer_memory(self):
        identities = set()
        for condition in ("G0", "G1"):
            for producer in ("cnn", "objects"):
                factory = grounding.RecordingFactory(condition, producer)
                identity = factory.__module__ + "." + factory.__qualname__
                self.assertEqual(identity, "mutual_lab.grounding_study.RecordingFactory." + condition + "_" + producer)
                identities.add(identity)
        self.assertEqual(len(identities), 4)

    def test_cancellation_keeps_in_memory_report_and_original_interrupt_when_disk_or_snapshot_fails(self):
        from mutual_lab import development_world as world
        from mutual_lab.composition_suite import ArmCancelled
        from mutual_lab.diagnostics import safe_json
        raw = {"status": "cancelled", "sentinel": "before-fs-failure"}
        for source in ("memory", "disk", "unreadable"):
            for failure in (None, OSError, KeyboardInterrupt):
                with self.subTest(source=source, failure=failure), tempfile.TemporaryDirectory() as temporary:
                    interrupt = ArmCancelled(Path(temporary) / "unavailable.json")
                    if source == "memory":
                        interrupt.report = MappingProxyType(raw)
                    snapshots = []

                    def cancel(*args, model_factory, **kwargs):
                        model_factory.initial_state = {"captured": True}
                        raise interrupt

                    def persist(path, value):
                        snapshots.append(copy.deepcopy(value))
                        if len(snapshots) > 1 and failure is not None:
                            raise failure("fixture snapshot failure")
                        safe_json(path, value)

                    reader = {"return_value": json.dumps(raw)} if source == "disk" else {"side_effect": OSError("unreadable")}
                    with patch.object(grounding, "_reference_plan", return_value=self.plan), patch.object(
                            world, "development_splits", return_value={"train": (), "dev": ()}), patch.object(
                            grounding, "run_trial", side_effect=cancel) as execute, patch.object(
                            grounding, "safe_json", side_effect=persist), patch.object(Path, "read_text", **reader) as read:
                        with self.assertRaises(KeyboardInterrupt) as caught:
                            grounding.run_grounding_study(self.config, "en", self.plan, temporary,
                                producer_memory="objects", prior_bundle={})
                    self.assertIs(caught.exception, interrupt)
                    execute.assert_called_once()
                    self.assertEqual(read.call_count, int(source != "memory"))
                    suite = snapshots[-1]
                    self.assertEqual(suite["status"], "cancelled")
                    self.assertEqual(len(suite["reports"]), 1)
                    self.assertEqual(suite["reports"][0]["grounding_condition"], "G0")
                    self.assertEqual(suite["reports"][0]["initial_state"], {"captured": True})
                    if source != "unreadable":
                        self.assertEqual(suite["reports"][0]["engine_report"], raw)

    def test_empty_aggregate_and_pinned_request_do_not_select_or_activate_a_producer(self):
        result = grounding.aggregate_grounding_studies([], self.plan)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_languages"], list(LANGUAGES_ORDER))
        self.assertEqual(result["selection"]["status"], "undetermined")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            request = {"schema_version": 1, "phase": "G", "fixed_prior_condition": "B1", "policy": "mixed_blocks",
                "fixed_final_step": 9600, "data_version": "five-colors-three-shapes-v1", "producer_memory": "cnn",
                "frozen_feature_policy": "per_png_singleton_v1"}
            path.write_text(json.dumps(request))
            self.assertEqual(grounding._request(path, self.plan, "cnn")["content"], request)
            with self.assertRaisesRegex(ValueError, "request differs"):
                grounding._request(path, self.plan, "objects")

    def test_predeclared_selection_requires_all_four_final_candidates_ranks_worst_six_and_ties_to_g0(self):
        def rows():
            result = []
            for condition in ("G0", "G1"):
                for language in LANGUAGES_ORDER:
                    categories, declared = {}, {}
                    for category, (anchors, pairs) in grounding.CONTRAST_COVERAGE.items():
                        categories[category] = {"hits": anchors * 4, "items": anchors * 4, "accuracy": 1.}
                        declared[category] = {**categories[category], "available_anchors": anchors,
                            "available_pairs": pairs, "coverage_sufficient": True, "score_target_met": True}
                    result.append({"condition": condition, "language": language, "step": 9600, "status": "completed",
                        "dev": {"comprehension_hits": 448, "comprehension_items": 448, "comprehension_accuracy": 1.,
                            "valid_generation_hits": 112, "generation_scenes": 112, "valid_generation_accuracy": 1.},
                        "contrasts": categories, "candidate": {"categories": declared, "status": "development_candidate",
                            "score_targets_met": True, "coverage_sufficient": True, "excluded_from_research": True,
                            "eligible_for_fresh_experiment": False}})
            return result

        perfect = rows()
        selected = grounding.select_grounding_candidate(perfect, self.plan, complete=True)
        self.assertEqual(selected["condition"], "G0")
        self.assertEqual(selected["eligible_conditions"], ["G0", "G1"])
        ranked = rows()
        ranked[0]["dev"].update(valid_generation_hits=101, valid_generation_accuracy=101 / 112)
        ranked[4]["dev"].update(valid_generation_hits=102, valid_generation_accuracy=102 / 112)
        selected = grounding.select_grounding_candidate(ranked, self.plan, complete=True)
        self.assertEqual(selected["condition"], "G1")
        self.assertEqual(selected["worst_scores"], {"G0": 101 / 112, "G1": 102 / 112})
        # All-language qualification precedes the minimum-score comparison.
        ranked[4]["dev"].update(valid_generation_hits=100, valid_generation_accuracy=100 / 112)
        ranked[4]["candidate"].update(status="targets_not_met", score_targets_met=False)
        self.assertEqual(grounding.select_grounding_candidate(ranked, self.plan, complete=True)["condition"], "G0")
        ranked[0]["candidate"]["categories"]["binding_swap"]["available_anchors"] = 4
        self.assertEqual(grounding.select_grounding_candidate(ranked, self.plan, complete=True)["reason"], "no_all_language_candidate")
        self.assertEqual(grounding.select_grounding_candidate(perfect, self.plan, complete=False)["status"], "undetermined")
        invalid = rows()
        for row in invalid:
            row["dev"]["valid_generation_hits"] = 111  # Accuracy left at1: inconsistent counts must fail.
        self.assertEqual(grounding.select_grounding_candidate(invalid, self.plan, complete=True)["reason"], "no_all_language_candidate")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU Torch")
class GroundingNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import development_world as world
        from mutual_lab.neural import configure_cpu
        from mutual_lab.spatial_prior import pretrain_prior, save_prior_bundle, load_prior_bundle
        configure_cpu()
        cls.temporary = tempfile.TemporaryDirectory()
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        cls.config = replace(config, native_training_profile=replace(config.native_training_profile, batch_size=4))
        cls.plan = replace(grounding._reference_plan(), budgets=(4, 8), benchmark_steps=1,
            model_profile=ModelProfile("grounding-test", embedding_dim=8, hidden_dim=16, max_sequence_length=128))
        full = world.development_splits(cls.config, 59003)
        cls.fixture = {"train": full["train"][:4] + full["train"][100:104] + full["train"][-4:], "dev": full["dev"][:4]}
        cls.pngs = tuple(world.render_scene(scene) for scenes in cls.fixture.values() for scene in scenes)
        cls.prior_path = save_prior_bundle(pretrain_prior([world.render_scene(scene) for scene in cls.fixture["train"]],
            dim=16, steps=1, batch_size=4, eval_every=1), Path(cls.temporary.name) / "prior.pt")
        cls.bundle = load_prior_bundle(cls.prior_path)
        cls.suites, cls.paths = {}, {}
        # Tests reduce the PNG fixture and budgets only through a local patched
        # reference; the public production guard rejects all these changes.
        with patch.object(grounding, "_reference_plan", return_value=cls.plan), patch.object(
                world, "development_splits", return_value=cls.fixture), patch.object(
                world, "development_manifest", return_value={"fixture": "protected-safe C pairs"}):
            for producer in ("cnn", "objects"):
                path = grounding.run_grounding_study(cls.config, "en", cls.plan, cls.temporary.name,
                    progress=None, producer_memory=producer, prior_bundle=cls.bundle, diagnostics=True)
                cls.paths[producer] = path
                cls.suites[producer] = json.loads((path / "grounding-study.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def four_languages(self, producer="objects"):
        suites = []
        for language in LANGUAGES_ORDER:
            suite = copy.deepcopy(self.suites[producer])
            suite["language"] = language
            for record in suite["reports"]:
                record["engine_report"]["language"] = language
                record["engine_report"]["training_languages"] = [language]
                record["engine_report"]["data_sha256"] = "synthetic-native-captions-" + language
            suites.append(suite)
        return suites

    def aggregate(self, suites):
        with patch.object(grounding, "_reference_plan", return_value=self.plan):
            return grounding.aggregate_grounding_studies(suites, self.plan)

    def test_real_g_pairs_hold_base_initialization_producer_and_prior_but_declare_extra_u_capacity(self):
        for producer, suite in self.suites.items():
            with self.subTest(producer=producer):
                self.assertEqual(suite["status"], "completed", suite)
                self.assertEqual(suite["pair_audit"]["status"], "matched", suite["pair_audit"])
                self.assertTrue(all(suite["pair_audit"]["checks"].values()))
                zero, one = suite["reports"]
                ia, ib = zero["initial_state"], one["initial_state"]
                for key in ("base_understanding_sha256", "producer_sha256", "prior_sha256", "base_understanding_parameters"):
                    self.assertEqual(ia[key], ib[key])
                self.assertEqual(ia["extra_understanding_parameters"], 0)
                self.assertEqual(ib["extra_understanding_parameters"], 1360)
                self.assertEqual(ib["parameter_count"] - ia["parameter_count"], 1360)
                a, b = zero["engine_report"], one["engine_report"]
                self.assertEqual(a["final_producer_sha256"], b["final_producer_sha256"])
                self.assertNotEqual(a["final_understanding_sha256"], b["final_understanding_sha256"])
                self.assertEqual(grounding._producer_tables(a), grounding._producer_tables(b))
                self.assertNotEqual(a["frozen_feature_tables"]["step_0"], b["frozen_feature_tables"]["step_0"])
                for record in (zero, one):
                    raw = record["engine_report"]
                    self.assertEqual(raw["initial_prior_sha256"], raw["final_prior_sha256"])
                    self.assertEqual(raw["frozen_feature_tables"]["step_0"], raw["frozen_feature_tables"]["final"])
                    self.assertEqual(raw["negative_exposures"]["branch_attributes_examples"], 16)
                    self.assertEqual(raw["negative_exposures"]["branch_same_pair_examples"], 16)
                    self.assertEqual([point["step"] for point in raw["curves"]], [0, 4, 8])
                    self.assertTrue(all(point["candidate"]["status"] == "indeterminate_coverage" for point in raw["curves"]))
                    self.assertEqual(raw["model_factory"], "mutual_lab.grounding_study.RecordingFactory." + record["grounding_condition"] + "_" + producer)
                    self.assertEqual(record["initial_state"]["memory_preflight"]["fallback_count"], 0)
                self.assertFalse(suite["capacity_matched"])
                self.assertFalse(suite["eligible_for_fresh_experiment"])
                self.assertIn("information/capacity", suite["interpretation"])

    def test_canonical_factories_have_exact_initial_logits_and_preflight_only_raw_two_object_u_tokens(self):
        from mutual_lab import development_world as world
        from mutual_lab.neural import digest, torch
        for producer in ("cnn", "objects"):
            factories = [grounding.RecordingFactory(condition, producer, self.pngs) for condition in ("G0", "G1")]
            a, b = [factory(self.plan.model_profile, 59007, self.bundle) for factory in factories]
            text = world.captions(self.fixture["train"][0], "en")[0]
            self.assertEqual(digest(a.understanding.base.state_dict()), digest(b.understanding.base.state_dict()))
            self.assertEqual(digest(a.producer.state_dict()), digest(b.producer.state_dict()))
            self.assertEqual(len(a.understanding_object_cache), 0)
            self.assertEqual(len(b.understanding_object_cache), len(self.pngs))
            with torch.no_grad():
                self.assertTrue(torch.equal(a.comprehension_logits([text], [list(self.pngs[:4])]),
                                            b.comprehension_logits([text], [list(self.pngs[:4])])))
                ids, _ = a.tokenizer.batch([text])
                self.assertTrue(torch.equal(a.teacher_forced_logits(ids[:, :-1], [self.pngs[0]]),
                                            b.teacher_forced_logits(ids[:, :-1], [self.pngs[0]])))
                self.assertEqual(b.understanding_objects([self.pngs[0]]).shape, (1, 2, 34))
            self.assertEqual(factories[1].__module__, "mutual_lab.grounding_study")
            self.assertTrue(bool((b.understanding.residual_output.weight == 0).all()))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                factories[0](self.plan.model_profile, 59007, self.bundle)

    def test_auxiliary_magnitudes_and_counter_caption_consistency_are_read_only_and_not_qualification_gates(self):
        for suite in self.suites.values():
            for record in suite["reports"]:
                diagnostic = record["auxiliary_diagnostics"]
                self.assertEqual(diagnostic["status"], "completed", diagnostic)
                self.assertFalse(diagnostic["qualification_gate"])
                self.assertTrue(diagnostic["read_only_verified"])
                for category, row in diagnostic["categories"].items():
                    self.assertGreaterEqual(row["counter_caption_pairs"], row["both_correct"])
                    if row["anchor_items"]:
                        self.assertGreaterEqual(row["mean_base_absolute"], 0)
                        self.assertGreaterEqual(row["mean_residual_absolute"], 0)
                    if record["grounding_condition"] == "G0":
                        self.assertEqual(row["residual_absolute_sum"], 0.)
                # Neither optional summary is added to the engine candidate.
                for point in record["engine_report"]["curves"]:
                    self.assertNotIn("counter_caption_consistency", point["candidate"])

    def test_all_four_final_g_checkpoint_ids_restore_saved_generation_and_understanding_predictions(self):
        from mutual_lab import development_world as world
        from mutual_lab.mastery_infer import infer_development, load_development_model
        for producer, suite in self.suites.items():
            for record in suite["reports"]:
                directory = self.paths[producer] / Path(record["raw_report_path"]).parent
                model = load_development_model(directory / "step-000008.pt", self.prior_path)
                example = world.make_examples(model, self.fixture["dev"][:1], "en")[0]
                actual = infer_development(model, example["image"], text=example["text"], choices_png=example["choices"])
                generated = next(row for row in json.loads((directory / "generations.json").read_text())
                    if row["step"] == 8 and row["split"] == "dev")
                predicted = next(row for row in json.loads((directory / "comprehension_predictions.json").read_text())
                    if row["step"] == 8 and row["split"] == "dev")
                self.assertEqual(actual["generation"], {key: generated[key] for key in actual["generation"]})
                self.assertEqual(actual["comprehension"]["prediction"], predicted["prediction"])
                self.assertEqual(model.object_memory_source, producer)
                self.assertEqual(model.understanding.object_context, record["grounding_condition"] == "G1")

    def test_audit_detects_producer_leakage_extra_capacity_and_raw_adapter_mislabeling(self):
        for field, value, check in (("final_producer_sha256", "changed", "final_producer_sha256"),
                ("meaning_stream_sha256", "changed", "meaning_stream_sha256"), ("condition", "C0", "engine_adapter"),
                ("frozen_feature_policy", "legacy", "canonical_features")):
            records = copy.deepcopy(self.suites["objects"]["reports"])
            records[1]["engine_report"][field] = value
            with self.subTest(field=field):
                self.assertFalse(grounding.grounding_pair_audit(records, "objects")["checks"][check])
        for field, value, check in (("extra_understanding_parameters", 0, "explicit_extra_capacity"),
                                  ("base_understanding_sha256", "changed", "base_understanding_initialization")):
            records = copy.deepcopy(self.suites["objects"]["reports"])
            records[1]["initial_state"][field] = value
            self.assertFalse(grounding.grounding_pair_audit(records, "objects")["checks"][check])
        records = copy.deepcopy(self.suites["objects"]["reports"])
        records[1]["engine_report"]["frozen_feature_tables"]["step_0"]["caches"]["visual_cache"]["sha256"] = "changed"
        self.assertFalse(grounding.grounding_pair_audit(records, "objects")["checks"]["producer_feature_tables_equal"])

    def test_aggregate_keeps_every_budget_failure_missing_language_and_explicit_nonselection(self):
        suites = self.four_languages()
        result = self.aggregate(suites)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["rows"]), 24)
        self.assertEqual(len(result["paired_deltas"]), 12)
        self.assertTrue(all(row["generation_delta"] == 0 for row in result["paired_deltas"]))
        self.assertTrue(all(row["completed_rows"] == 4 and row["generation_scenes"] == 16 for row in result["budget_counts"]))
        self.assertFalse(result["capacity_matched"])
        self.assertEqual(result["selection"]["reason"], "no_all_language_candidate")
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
