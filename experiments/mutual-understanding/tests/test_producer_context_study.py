"""H discovery identity/selection/CLI controls and one genuine tiny H128 pair."""

import contextlib
import copy
import importlib.util
import io
import json
import shlex
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
from mutual_lab import producer_context_study as study
from mutual_lab.config import ExperimentConfig
from mutual_lab.mastery_lab import MasteryPlan


def plan():
    base = study._base_plan()
    return replace(base, seed_pairs=study.DISCOVERY_SEEDS, b_seed_pair=study.DISCOVERY_SEEDS[0])


def request_values(p):
    return {"schema_version": 1, "phase": "H", "model": "attentive", "fixed_prior_condition": "B1",
        "policy": "mixed_blocks", "producer_memory": "objects", "understanding": "objects", "hidden_dim": 128,
        "data_version": "five-colors-three-shapes-v1", "fixed_final_step": p.budgets[-1],
        "frozen_feature_policy": "per_png_singleton_v1", "memory_normalization": "l2_first32_raw_yx_v1", "source_rule": study.SOURCE_RULE,
        "context_implementation_sha256": "a" * 64, "study_implementation_sha256": "b" * 64,
        "inference_compatibility_sha256": "c" * 64, "source_cohort_run_id": 43647997707,
        "source_cohort_revision": "c65ab2752c117ddf66c98c4e871f553cf9c8f95a", "source_cohort_run_attempt": 1,
        "source_cohort_summary_sha256": "d" * 64, "source_plan_sha256": "bfc2900fb2153c4f267a87270d91b8ef2f05313304071b03480ea79ec526fd47",
        "plan_sha256": "a2622ee9bada8ce5f79365fa790d90bf2902a806c8af4d4a389b5081009a962c",
        "prior_manifest_sha256": "3ef0e0ce056c4c553468a0572f42098b09d3b3d7f9d68fe9021a5832a7e272a6",
        "prior_checkpoint_sha256": "862eed3ec93a4567019075adcce59faf75295987a5cf6b816cace607da926e6d"}


def workflow_arguments(kind):
    path = PACKAGE.parent.parent / ".github/workflows/l1-producer-context-H.yml"
    commands = {}
    for line in path.read_text().splitlines():
        if "python -m mutual_lab.producer_context_study " not in line: continue
        line = line.replace("${{ matrix.language }}", "ja").replace("${{ matrix.replica }}", "1")
        arguments = shlex.split(line.strip().removeprefix("run: "))
        arguments = arguments[arguments.index("mutual_lab.producer_context_study") + 1:]
        name = "aggregate" if "--aggregate-dir" in arguments else "train"
        if name in commands: raise AssertionError("Duplicate workflow CLI command")
        commands[name] = arguments
    if set(commands) != {"train", "aggregate"}: raise AssertionError("Missing H workflow commands")
    return commands[kind]


def option(arguments, name, value):
    arguments[arguments.index(name) + 1] = str(value)


def qualifying_rows(p):
    rows = []
    for condition in study.CONDITIONS:
        for replica in range(3):
            for language in study.LANGUAGES_ORDER:
                categories, gates = {}, {}
                for category, (anchors, pairs) in study.CONTRAST_COVERAGE.items():
                    categories[category] = {"hits": anchors * 4, "items": anchors * 4, "accuracy": 1.}
                    gates[category] = {**categories[category], "available_anchors": anchors, "available_pairs": pairs,
                        "coverage_sufficient": True, "score_target_met": True}
                p_hits = 105 if condition == "H0" else 110
                rows.append({"language": language, "replica": replica, "seed_pair": list(study.DISCOVERY_SEEDS[replica]),
                    "condition": condition, "step": p.budgets[-1], "status": "completed",
                    "dev": {"comprehension_hits": 404, "comprehension_items": 448, "comprehension_accuracy": 404 / 448,
                        "valid_generation_hits": p_hits, "generation_scenes": 112, "valid_generation_accuracy": p_hits / 112},
                    "contrasts": categories, "candidate": {"status": "development_candidate", "score_targets_met": True,
                        "coverage_sufficient": True, "categories": gates, "excluded_from_research": True, "eligible_for_fresh_experiment": False}})
    return rows


class ContextProtocolTests(unittest.TestCase):
    def test_three_discovery_seeds_only_and_all_other_numeric_controls_remain_fixed(self):
        p = plan(); study._validate_plan(p)
        for changed in (replace(p, seed_pairs=study.CONFIRMATION_SEEDS), replace(p, b_seed_pair=study.DISCOVERY_SEEDS[1]),
                        replace(p, budgets=(9600,)), replace(p, data_seed=29003)):
            with self.assertRaises(ValueError): study._validate_plan(changed)
        self.assertTrue(set(sum(study.DISCOVERY_SEEDS, ())).isdisjoint(sum(study.CONFIRMATION_SEEDS, ())))
        self.assertEqual(study.RecordingFactory("H0").__module__ + "." + study.RecordingFactory("H0").__qualname__,
                         "mutual_lab.producer_context_study.RecordingFactory.H0")

    def test_selection_requires_all_twelve_cells_and_ranks_minimum_P_not_minimum_six_scores(self):
        p, rows = plan(), qualifying_rows(plan())
        self.assertEqual(study.select_context_candidate(rows, p, complete=True)["condition"], "H1")
        # U is identical and below either P; a min-six rule would choose the wrong tie.
        for row in rows:
            row["dev"].update(valid_generation_hits=110, valid_generation_accuracy=110 / 112)
        self.assertEqual(study.select_context_candidate(rows, p, complete=True)["condition"], "H0")
        rows[12]["dev"].update(valid_generation_hits=99, valid_generation_accuracy=99 / 112)
        self.assertEqual(study.select_context_candidate(rows, p, complete=True)["condition"], "H0")
        rows[0]["dev"].update(valid_generation_hits=99, valid_generation_accuracy=99 / 112)
        self.assertIsNone(study.select_context_candidate(rows, p, complete=True)["condition"])
        self.assertIsNone(study.select_context_candidate(qualifying_rows(p), p, complete=False)["condition"])
        rows = qualifying_rows(p); rows[0]["seed_pair"] = [69007, 69009]
        self.assertEqual(study.select_context_candidate(rows, p, complete=True)["eligible_conditions"], ["H1"])

    def test_real_cli_empty_aggregate_and_request_schema_without_unsupported_flags(self):
        p = plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); plan_path, request, result = root / "plan.json", root / "request.json", root / "summary.json"
            plan_path.write_text(json.dumps(asdict(p))); request.write_text(json.dumps(request_values(p)))
            empty = root / "empty"; empty.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                status = study.main(["--aggregate-dir", str(empty), "--plan", str(plan_path), "--study-request", str(request), "--output", str(result)])
            self.assertEqual(status, 1)
            summary = json.loads(result.read_text())
            self.assertEqual(len(summary["missing_cells"]), 12)
            self.assertEqual(summary["selection"]["status"], "undetermined")
            content = request_values(p); content.pop("phase"); request.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "Pinned H request"):
                study.main(["--aggregate-dir", str(empty), "--plan", str(plan_path), "--study-request", str(request)])

    def test_actual_workflow_empty_aggregate_cli_and_full_phase_H_request(self):
        p = plan(); arguments = workflow_arguments("aggregate")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); plan_path, request, result = root / "plan.json", root / "request.json", root / "summary.json"
            plan_path.write_text(json.dumps(asdict(p))); request.write_text(json.dumps(request_values(p)))
            empty = root / "empty"; empty.mkdir()
            for key, value in (("--plan", plan_path), ("--study-request", request), ("--aggregate-dir", empty), ("--output", result)):
                option(arguments, key, value)
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(study.main(arguments), 1)
            self.assertEqual(len(json.loads(result.read_text())["missing_cells"]), 12)

    def test_cancellation_preserves_original_interrupt_and_never_starts_second_arm_on_disk_failure(self):
        from mutual_lab.composition_suite import ArmCancelled
        from mutual_lab import development_world as world
        p = plan(); config = ExperimentConfig.load(PACKAGE / "configs/neural.example.json")
        for persistence_failure in (False, True):
            with self.subTest(failure=persistence_failure), tempfile.TemporaryDirectory() as temporary:
                interrupt = ArmCancelled(Path(temporary) / "missing.json"); interrupt.report = {"status": "cancelled"}
                calls = []
                def cancel(*args, model_factory, **kwargs):
                    model_factory.initial_state = {"actual_initial": True}; raise interrupt
                def save(path, value):
                    calls.append(copy.deepcopy(value))
                    if len(calls) > 1 and persistence_failure: raise OSError("disk unavailable")
                with patch.object(world, "development_splits", return_value={"train": (), "dev": ()}), patch.object(
                        study, "run_trial", side_effect=cancel) as train, patch.object(study, "safe_json", side_effect=save):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        study.run_context_study(config, "ja", 1, p, temporary, prior_bundle={})
                self.assertIs(caught.exception, interrupt); train.assert_called_once()
                self.assertEqual(calls[-1]["status"], "cancelled")
                self.assertEqual(len(calls[-1]["reports"]), 1)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class ContextNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import development_world as world
        from mutual_lab.spatial_prior import pretrain_prior
        base = replace(study._base_plan(), budgets=(1, 2), benchmark_steps=1,
                       model_profile=replace(study._base_plan().model_profile, max_sequence_length=128))
        cls.base, cls.plan = base, replace(base, seed_pairs=study.DISCOVERY_SEEDS, b_seed_pair=study.DISCOVERY_SEEDS[0])
        config = ExperimentConfig.load(PACKAGE / "configs/neural.example.json")
        cls.config = replace(config, native_training_profile=replace(config.native_training_profile, batch_size=4))
        full = world.development_splits(config, base.data_seed)
        cls.fixture = {name: scenes[:4] for name, scenes in full.items()}
        cls.bundle = pretrain_prior([world.render_scene(scene) for scene in cls.fixture["train"]], dim=128, steps=1, batch_size=4, eval_every=1)
        cls.temporary = tempfile.TemporaryDirectory()
        with patch.object(study, "_base_plan", return_value=base), patch.object(world, "development_splits", return_value=cls.fixture), patch.object(
                world, "development_manifest", return_value={"fixture": "protected-safe-two-pairs"}):
            cls.path = study.run_context_study(cls.config, "ja", 1, cls.plan, cls.temporary.name, progress=None, prior_bundle=cls.bundle)
        cls.suite = json.loads((cls.path / "producer-context-study.json").read_text())

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_actual_H128_pair_preserves_U_weights_at_every_actual_saved_budget_and_input_tables(self):
        self.assertEqual(self.suite["status"], "completed", self.suite["pair_audit"])
        self.assertTrue(all(self.suite["pair_audit"]["checks"].values()))
        a, b = self.suite["reports"]
        self.assertEqual(a["understanding_checkpoints"], b["understanding_checkpoints"])
        self.assertEqual(set(a["understanding_checkpoints"]), {"0", "1", "2"})
        self.assertEqual(a["initial_state"]["full_state_sha256"], b["initial_state"]["full_state_sha256"])
        self.assertEqual(a["engine_report"]["frozen_feature_tables"], b["engine_report"]["frozen_feature_tables"])
        self.assertNotEqual(a["engine_report"]["final_producer_sha256"], b["engine_report"]["final_producer_sha256"])
        for record in (a, b):
            raw = record["engine_report"]
            self.assertEqual(raw["seed_pair"], list(study.DISCOVERY_SEEDS[1]))
            self.assertEqual(raw["plan"]["b_seed_pair"], list(study.DISCOVERY_SEEDS[1]))
            self.assertEqual(raw, json.loads((self.path / record["raw_report_path"]).read_text()))
            self.assertTrue(raw["benchmark"]["initial_weights_restored"])

    def test_pair_audit_detects_intermediate_U_mutation_and_same_capacity_or_cache_control_failure(self):
        reports = copy.deepcopy(self.suite["reports"])
        reports[1]["understanding_checkpoints"]["1"] = "changed"
        self.assertFalse(study.context_pair_audit(reports)["checks"]["understanding_exact_at_all_budgets"])
        reports = copy.deepcopy(self.suite["reports"]); reports[1]["engine_report"]["frozen_feature_tables"] = {}
        self.assertEqual(study.context_pair_audit(reports)["status"], "mismatch")
        reports = copy.deepcopy(self.suite["reports"]); reports[1]["initial_state"]["parameter_count"] += 1
        self.assertFalse(study.context_pair_audit(reports)["checks"]["all_initial_nn_exact"])

    def test_checkpoint_corruption_and_research_loading_are_rejected(self):
        from mutual_lab.neural import load_checkpoint
        record = self.suite["reports"][0]; directory = (self.path / record["raw_report_path"]).parent
        checkpoint = directory / "step-000002.pt"
        with self.assertRaises(ValueError): load_checkpoint(checkpoint, expected_kind="native")
        raw = copy.deepcopy(record["engine_report"]); raw["checkpoints"][-1]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            study._checkpoint_understanding(directory, raw)

    def test_actual_workflow_train_cli_validates_real_H_plan_request_and_prior_before_mock_training(self):
        from mutual_lab.spatial_prior import save_prior_bundle
        p = plan(); arguments = workflow_arguments("train")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); plan_path, request = root / "plan.json", root / "request.json"
            plan_path.write_text(json.dumps(asdict(p))); request.write_text(json.dumps(request_values(p)))
            prior = save_prior_bundle(self.bundle, root / "prior.pt")
            for key, value in (("--plan", plan_path), ("--study-request", request), ("--prior-bundle", prior), ("--output", root / "output")):
                option(arguments, key, value)
            def boundary(config, language, replica, actual_plan, output, **kwargs):
                config.validate(); actual_plan.validate(config); study._validate_plan(actual_plan)
                self.assertEqual((language, replica), ("ja", 1))
                self.assertEqual(actual_plan.seed_pairs, study.DISCOVERY_SEEDS)
                self.assertEqual(kwargs["study_request"]["content"]["phase"], "H")
                self.assertEqual(kwargs["study_request"]["content"]["hidden_dim"], 128)
                self.assertEqual(kwargs["prior_bundle"]["report"]["status"], "completed")
                path = Path(output); path.mkdir(parents=True)
                (path / "producer-context-study.json").write_text(json.dumps({"status": "completed"}))
                return path
            with patch.object(study, "run_context_study", side_effect=boundary) as train, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(study.main(arguments), 0)
            train.assert_called_once()


if __name__ == "__main__": unittest.main()
