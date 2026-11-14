"""Fixed-seed validation audits and small genuine attentive/mixed Torch trials."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from mutual_lab import candidate_validation as validation
from mutual_lab.config import ExperimentConfig
from mutual_lab.mastery_lab import MasteryPlan
from mutual_lab.native_preparation import LANGUAGES_ORDER
from mutual_lab.profiles import ModelProfile

ROOT = Path(__file__).resolve().parents[1]


def declared_plan():
    return replace(MasteryPlan.load(ROOT / "configs/mastery-C-plan.json"),
        seed_pairs=validation.SEED_PAIRS, b_seed_pair=validation.SEED_PAIRS[0], metadata={"fixture": "audit-only"})


def audit_fixture(language, replica, plan):
    """Synthetic perfect counts exercise audit logic, never model performance."""
    pair = validation.SEED_PAIRS[replica]
    points = []
    for step in (0, *plan.budgets):
        categories, candidate_categories = {}, {}
        for category, (anchors, pairs) in validation.AVAILABILITY.items():
            categories[category] = {"hits": anchors * 4, "items": anchors * 4, "accuracy": 1.0}
            candidate_categories[category] = {**categories[category], "available_anchors": anchors,
                "available_pairs": pairs, "coverage_sufficient": True, "score_target_met": True}
        points.append({"step": step, "dev": {"comprehension_hits": 448, "comprehension_items": 448,
            "comprehension_accuracy": 1.0, "valid_generation_hits": 112, "generation_scenes": 112,
            "valid_generation_accuracy": 1.0}, "contrasts": {"categories": categories},
            "candidate": {"status": "development_candidate", "score_targets_met": True,
                "coverage_sufficient": True, "categories": candidate_categories,
                "excluded_from_research": True, "eligible_for_fresh_experiment": False}})
    initial = {"model": "attentive", "understanding_sha256": f"U-{replica}", "producer_sha256": f"P-{replica}",
        "prior_sha256": "shared-prior", "parameter_count": 100, "trainable_parameter_count": 80}
    total = plan.budgets[-1] * 16
    raw = {"status": "completed", "phase": "C", "condition": "C1", "language": language, "replica": replica,
        "seed_pair": list(pair), "plan": asdict(replace(plan, b_seed_pair=pair)),
        "model_factory": validation.FACTORY, "policy": "mixed_blocks", "curves": points,
        "training_steps_completed": plan.budgets[-1], "data_version": validation.DATA_VERSION,
        "split": {"version": validation.DATA_VERSION, "pair_sha256": validation.PAIR_SHA256,
            "scene_counts": {"train": 220, "dev": 112}, "png_pool_sha256": validation.PNG_SHA256},
        "initial_understanding_sha256": initial["understanding_sha256"],
        "initial_producer_sha256": initial["producer_sha256"], "initial_prior_sha256": "shared-prior",
        "final_prior_sha256": "shared-prior", "parameter_count": 100, "trainable_parameter_count": 80,
        "prior_metadata": {"condition": "B1", "prior_sha256": "shared-prior", "fit_pool_sha256": "one-fit-pool"},
        "training_languages": [language], "foreign_training_languages": [], "protected_splits_used": [],
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "optimization": "independent-branch-clipping-adam-v1", "clip_norm_per_branch": 1,
        "choice_count": 4, "training_chance_accuracy": .25, "comprehension_forms": 4,
        "production_forms": 1, "teacher_forced_evaluation_forms": 4,
        "negative_sampling_seed": pair[1] + plan.negative_seed_offset, "model_profile": asdict(plan.model_profile),
        "batch_size": 16, "learning_rate": .003, "config_sha256": "same-config",
        "source_sha256": {"synthetic_audit_fixture": "one-source"}, "meaning_stream_sha256": f"meaning-{replica}",
        "comprehension_caption_stream_sha256": f"native-{language}-{replica}",
        "benchmark": {"steps": plan.benchmark_steps, "initial_weights_restored": True},
        "negative_exposures": {"examples": total, "distractor_slots": total * 3,
            "semantic_duplicate_choices": 0, "png_duplicate_choices": 0,
            **{f"branch_{branch}_examples": count for branch, count in validation._mixed_branch_counts(total).items()}},
        "checkpoints": [{"step": step, "format": "mutual-mastery-development-v1", "kind": "development_only",
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "path": f"step-{step:06d}.pt", "sha256": "a" * 64} for step in (0, *plan.budgets)]}
    return {"kind": validation.VERSION, "language": language, "replica": replica, "seed_pair": list(pair),
        "plan": asdict(plan), "status": "completed", "model": "attentive", "fixed_prior_condition": "B1",
        "policy": "mixed_blocks", "negative_policy_specification": validation.negative_policy_manifest("mixed_blocks"),
        "candidate_request": None, "fresh_initialization": True, "resumed": False,
        "scratch_language_training": True, "shared_visual_prior": True, "independent_visual_replications": False,
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "initial_state": initial, "engine_report": raw}


class CandidateValidationTests(unittest.TestCase):
    def setUp(self):
        self.plan = declared_plan()
        self.config = ExperimentConfig.load(ROOT / "configs/neural.example.json")

    def all_suites(self):
        return [audit_fixture(language, replica, self.plan) for replica in range(3) for language in LANGUAGES_ORDER]

    def test_all_twelve_fixed_final_outcomes_and_fresh_seed_blocks_are_required(self):
        suites = self.all_suites()
        result = validation.aggregate_candidate_validations(suites, self.plan)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["robust_development_candidate"])
        self.assertEqual(result["final_candidate_count"], 12)
        self.assertEqual(len(result["rows"]), 48)
        self.assertTrue(all(block["all_four_final_candidates"] for block in result["seed_blocks"]))
        self.assertFalse(result["eligible_for_fresh_experiment"])
        self.assertFalse(result["independent_visual_replications"])
        # A perfect early checkpoint cannot rescue a failed fixed final budget.
        final = suites[-1]["engine_report"]["curves"][-1]
        final["dev"].update(valid_generation_hits=100, valid_generation_accuracy=100 / 112)
        final["candidate"].update(status="targets_not_met", score_targets_met=False)
        result = validation.aggregate_candidate_validations(suites, self.plan)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["robust_development_candidate"])
        self.assertEqual(result["final_candidate_count"], 11)
        self.assertEqual(result["noncandidate_final_trials"], [{"language": LANGUAGES_ORDER[-1], "replica": 2}])

    def test_cross_language_initialization_meanings_and_distinct_replicas_are_audited(self):
        for key, global_check in (("initial_understanding_sha256", "all_seed_blocks_matched"),
                                  ("meaning_stream_sha256", "all_seed_blocks_matched"),
                                  ("source_sha256", "same_protocol_provenance"),
                                  ("initial_prior_sha256", "one_shared_frozen_visual_prior")):
            suites = self.all_suites()
            suites[0]["engine_report"][key] = "different"
            result = validation.aggregate_candidate_validations(suites, self.plan)
            with self.subTest(key=key):
                self.assertFalse(result["checks"][global_check])
                self.assertFalse(result["robust_development_candidate"])
        suites = self.all_suites()
        for suite in suites:
            suite["initial_state"]["understanding_sha256"] = "reused-U"
            suite["engine_report"]["initial_understanding_sha256"] = "reused-U"
        result = validation.aggregate_candidate_validations(suites, self.plan)
        self.assertFalse(result["checks"]["distinct_language_initializations_across_seed_blocks"])

    def test_false_candidate_flags_bad_counts_coverage_and_missing_checkpoints_cannot_pass(self):
        for mutation in ("score", "coverage", "count", "checkpoint", "budget", "resume", "seed", "pool"):
            suites = self.all_suites()
            suite, raw = suites[0], suites[0]["engine_report"]
            final = raw["curves"][-1]
            if mutation == "score":
                final["candidate"]["score_targets_met"] = False
            elif mutation == "coverage":
                final["candidate"]["categories"]["binding_swap"]["available_anchors"] = 4
            elif mutation == "count":
                final["dev"]["valid_generation_hits"] = 111
            elif mutation == "checkpoint":
                raw["checkpoints"].pop()
            elif mutation == "budget":
                raw["curves"].pop(1)
            elif mutation == "resume":
                suite["resumed"] = True
            elif mutation == "seed":
                raw["seed_pair"] = [59007, 59009]
            else:
                raw["split"] = {**raw["split"], "pair_sha256": "different"}
            result = validation.aggregate_candidate_validations(suites, self.plan)
            with self.subTest(mutation=mutation):
                self.assertEqual(result["status"], "incomplete")
                self.assertFalse(result["robust_development_candidate"])
                self.assertTrue(result["failed_trials"])

    def test_failed_cancelled_and_missing_trials_remain_visible_without_retry_selection(self):
        suites = self.all_suites()
        suites[0].update(status="failed", error="fixture failure")
        suites[0]["engine_report"]["status"] = "failed"
        suites[1]["status"] = suites[1]["engine_report"]["status"] = "cancelled"
        suites[1]["engine_report"]["curves"] = suites[1]["engine_report"]["curves"][:1]
        result = validation.aggregate_candidate_validations(suites[:-1], self.plan)
        self.assertEqual(result["missing_trials"], [{"language": LANGUAGES_ORDER[-1], "replica": 2}])
        self.assertEqual({item["status"] for item in result["failed_trials"]}, {"failed", "cancelled"})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validation.aggregate_candidate_validations(suites + suites[:1], self.plan)
        with self.assertRaisesRegex(ValueError, "not discovery"):
            validation.aggregate_candidate_validations([{"kind": "negative-policy-study-v1"}], self.plan)

    def test_protocol_changes_are_rejected_before_creating_training_outputs(self):
        variants = (replace(self.plan, budgets=(2400,)), replace(self.plan, production_min=.8),
            replace(self.plan, data_seed=29003), replace(self.plan, seed_pairs=((59007, 59009),)),
            replace(self.plan, b_seed_pair=self.plan.seed_pairs[1]),
            replace(self.plan, model_profile=replace(self.plan.model_profile, hidden_dim=256)))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            for plan in variants:
                with self.subTest(plan=plan), patch.object(validation, "run_trial") as execute:
                    with self.assertRaises(ValueError):
                        validation.run_candidate_validation(self.config, "en", 0, plan, output, prior_bundle={})
                    execute.assert_not_called()
                    self.assertFalse(output.exists())
            for language, replica, prior in (("xx", 0, {}), ("en", True, {}), ("en", 3, {}), ("en", 0, None)):
                with self.assertRaises(ValueError):
                    validation.run_candidate_validation(self.config, language, replica, self.plan, output, prior_bundle=prior)
                self.assertFalse(output.exists())

    def test_cancellation_prefers_in_memory_report_and_rethrows_when_persistence_fails(self):
        from mutual_lab.composition_suite import ArmCancelled
        from mutual_lab.diagnostics import safe_json
        report = {"status": "cancelled", "sentinel": "before-disk-failure"}
        reference = validation._reference_plan()
        for source in ("memory", "disk", "unreadable"):
            for failure in (None, OSError, KeyboardInterrupt):
                with self.subTest(source=source, failure=failure), tempfile.TemporaryDirectory() as temporary:
                    interrupt = ArmCancelled(Path(temporary) / "unavailable.json")
                    if source == "memory":
                        interrupt.report = MappingProxyType(report)
                    snapshots = []

                    def cancel(*args, model_factory, **kwargs):
                        model_factory.initial_state = {"captured": True}
                        raise interrupt

                    def persist(path, payload):
                        snapshots.append(copy.deepcopy(payload))
                        if len(snapshots) > 1 and failure is not None:
                            raise failure("fixture persistence failure")
                        safe_json(path, payload)

                    read = {"return_value": json.dumps(report)} if source == "disk" else {"side_effect": OSError("unreadable")}
                    with patch.object(validation, "_reference_plan", return_value=reference), patch.object(
                            validation, "run_trial", side_effect=cancel) as execute, patch.object(
                            validation, "safe_json", side_effect=persist), patch.object(Path, "read_text", **read) as reader:
                        with self.assertRaises(KeyboardInterrupt) as caught:
                            validation.run_candidate_validation(self.config, "en", 2, self.plan, temporary, prior_bundle={})
                    self.assertIs(caught.exception, interrupt)
                    execute.assert_called_once()
                    self.assertEqual(reader.call_count, int(source != "memory"))
                    self.assertEqual(snapshots[-1]["status"], "cancelled")
                    self.assertEqual(snapshots[-1]["initial_state"], {"captured": True})
                    if source != "unreadable":
                        self.assertEqual(snapshots[-1]["engine_report"], report)

    def test_failed_execution_is_saved_once_and_empty_aggregate_cli_reports_all_twelve_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(validation, "run_trial", side_effect=ValueError("fixture failure")) as execute:
                path = validation.run_candidate_validation(self.config, "en", 0, self.plan, temporary, prior_bundle={})
            self.assertEqual(execute.call_count, 1)
            saved = json.loads((path / "candidate-validation.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertIn("fixture failure", saved["error"])
            empty, report, plan_path = Path(temporary) / "empty", Path(temporary) / "summary.json", Path(temporary) / "plan.json"
            empty.mkdir(); plan_path.write_text(json.dumps(asdict(self.plan)))
            with patch("builtins.print"):
                status = validation.main(["--aggregate-dir", str(empty), "--plan", str(plan_path), "--output", str(report)])
            result = json.loads(report.read_text())
            self.assertEqual(status, 1)
            self.assertEqual(len(result["missing_trials"]), 12)
            self.assertFalse(result["robust_development_candidate"])

    def test_candidate_request_bytes_and_plan_prior_pins_are_preserved_and_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path, prior, request_path = root / "plan.json", root / "prior.pt", root / "request.json"
            plan_path.write_text(json.dumps(asdict(self.plan)))
            prior.with_suffix(".json").write_text("prior-manifest")
            content = {"schema_version": 1, "model": "attentive", "fixed_prior_condition": "B1", "policy": "mixed_blocks",
                "fixed_final_step": 9600, "data_version": validation.DATA_VERSION, "source_e_revision": "pinned-source",
                "validation_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "prior_manifest_sha256": hashlib.sha256(prior.with_suffix(".json").read_bytes()).hexdigest()}
            request_path.write_text(json.dumps(content))
            record = validation._load_request(request_path, self.plan, plan_path=plan_path, prior_path=prior)
            self.assertEqual(record["content"], content)
            self.assertEqual(record["sha256"], hashlib.sha256(request_path.read_bytes()).hexdigest())
            suites = self.all_suites()
            for suite in suites:
                suite["candidate_request"] = copy.deepcopy(record)
            self.assertTrue(validation.aggregate_candidate_validations(suites, self.plan, candidate_request=record)["robust_development_candidate"])
            prior.with_suffix(".json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "prior_manifest_sha256"):
                validation._load_request(request_path, self.plan, plan_path=plan_path, prior_path=prior)
            suites[-1]["candidate_request"]["sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                validation.aggregate_candidate_validations(suites, self.plan, candidate_request=record)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class CandidateValidationNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import development_world as world
        from mutual_lab.neural import configure_cpu
        from mutual_lab.spatial_prior import load_prior_bundle, pretrain_prior, save_prior_bundle
        configure_cpu()
        cls.temporary = tempfile.TemporaryDirectory()
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        cls.config = replace(config, native_training_profile=replace(config.native_training_profile, batch_size=4))
        tiny = ModelProfile("validation-test", embedding_dim=8, hidden_dim=16, max_sequence_length=128)
        cls.plan = replace(declared_plan(), budgets=(1, 2), model_profile=tiny, benchmark_steps=0)
        reference = replace(validation._reference_plan(), budgets=(1, 2), model_profile=tiny, benchmark_steps=0)
        full = world.development_splits(cls.config, 59003)
        fixture = {"train": full["train"][:4] + full["train"][100:104] + full["train"][-4:], "dev": full["dev"][:4]}
        cls.prior_path = save_prior_bundle(pretrain_prior([world.render_scene(s) for s in fixture["train"]],
            dim=16, steps=1, batch_size=4, eval_every=1), Path(cls.temporary.name) / "prior.pt")
        bundle = load_prior_bundle(cls.prior_path)
        cls.suites, cls.paths = {}, {}
        # Only tests shrink the finite PNG fixture/budgets. Production rejects
        # such plan changes before output creation, covered by the guard test.
        with patch.object(validation, "_reference_plan", return_value=reference), patch.object(
                world, "development_splits", return_value=fixture), patch.object(
                world, "development_manifest", return_value={"fixture": "protected-safe test pairs"}):
            for replica in range(3):
                for language in ("en", "ko"):
                    path = validation.run_candidate_validation(cls.config, language, replica, cls.plan,
                        cls.temporary.name, progress=None, prior_bundle=bundle)
                    cls.paths[(replica, language)] = path
                    cls.suites[(replica, language)] = json.loads((path / "candidate-validation.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_real_fresh_attentive_trials_pair_initialization_and_meanings_across_languages(self):
        from mutual_lab.neural import torch
        initials = []
        for replica, pair in enumerate(validation.SEED_PAIRS):
            en, ko = (self.suites[(replica, language)] for language in ("en", "ko"))
            self.assertEqual(en["initial_state"], ko["initial_state"])
            initials.append(en["initial_state"])
            a, b = en["engine_report"], ko["engine_report"]
            self.assertEqual(a["meaning_stream_sha256"], b["meaning_stream_sha256"])
            self.assertNotEqual(a["comprehension_caption_stream_sha256"], b["comprehension_caption_stream_sha256"])
            for language in ("en", "ko"):
                suite = self.suites[(replica, language)]
                raw = suite["engine_report"]
                self.assertEqual(suite["status"], "completed", suite)
                self.assertEqual(raw["seed_pair"], list(pair))
                self.assertEqual(raw["plan"]["b_seed_pair"], list(pair))
                self.assertEqual(raw["model_factory"], validation.FACTORY)
                self.assertEqual(raw["policy"], "mixed_blocks")
                self.assertEqual(raw["training_steps_completed"], 2)
                self.assertEqual(raw["initial_prior_sha256"], raw["final_prior_sha256"])
                self.assertNotEqual(raw["initial_producer_sha256"], raw["final_producer_sha256"])
                checkpoint = self.paths[(replica, language)] / Path(suite["raw_report_path"]).parent / "step-000002.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertIn("producer.attention_output.weight", payload["model"])
                self.assertTrue(payload["optimizer"]["state"])
                self.assertEqual(payload["negative_exposure_offset"], 8)
                self.assertFalse(suite["resumed"])
        self.assertEqual(len({initial["understanding_sha256"] for initial in initials}), 3)
        self.assertEqual(len({initial["producer_sha256"] for initial in initials}), 3)
        self.assertEqual(len({initial["prior_sha256"] for initial in initials}), 1)

    def test_replica_two_checkpoint_is_readable_by_the_existing_development_inference_seed_guard(self):
        from mutual_lab.mastery_infer import load_development_model
        suite = self.suites[(2, "ko")]
        checkpoint = self.paths[(2, "ko")] / Path(suite["raw_report_path"]).parent / "step-000002.pt"
        model = load_development_model(checkpoint, self.prior_path)
        self.assertEqual(model.mastery_inference_metadata["model_factory"], validation.FACTORY)
        self.assertTrue(model.development_only)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
