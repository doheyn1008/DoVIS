"""Execute the workflow's real CLI contracts, replacing only L1 computation.

The request and qualified source rows are explicit synthetic evidence fixtures.
Configuration and image-prior serialization/loading are real. Set
DOVIS_COHORT_TEST_PRIOR to exercise an existing canonical prior; portable CI
creates a genuine one-step image-only prior without running discovery training.
"""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import unittest
from unittest.mock import patch

from mutual_lab import cohort_validation as cohort
from mutual_lab.config import ExperimentConfig
from mutual_lab.mastery_lab import MasteryPlan
from mutual_lab.native_preparation import LANGUAGES_ORDER
from test_cohort_validation import request_record, source_summary, suite_fixture

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT.parents[1] / ".github/workflows/l1-cohort-validation.yml"


def workflow_arguments(mode, *, language="en", replica=0):
    """Read the executable command itself; do not duplicate its option list."""
    candidates = []
    for line in WORKFLOW.read_text().splitlines():
        line = line.strip()
        if "python -m mutual_lab.cohort_validation " not in line:
            continue
        if ("--aggregate-dir " in line) != (mode == "aggregate"):
            continue
        if line.startswith("run: "):
            line = line[len("run: "):]
        line = line.replace("${{ matrix.language }}", language)
        line = line.replace("${{ matrix.replica }}", str(replica))
        if "${{" in line:
            raise AssertionError("An unresolved workflow expression reached the CLI fixture")
        command = shlex.split(line)
        position = command.index("mutual_lab.cohort_validation")
        if command[position - 2:position] != ["python", "-m"]:
            raise AssertionError("The workflow no longer executes the expected module")
        candidates.append(command[position + 1:])
    if len(candidates) != 1:
        raise AssertionError(f"Expected one actual {mode} command, found {len(candidates)}")
    return candidates[0]


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class CohortWorkflowCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import development_world as world
        from mutual_lab.neural import configure_cpu
        from mutual_lab.spatial_prior import pretrain_prior, save_prior_bundle

        configure_cpu()
        cls.temporary = tempfile.TemporaryDirectory(prefix="dovis-cohort-cli-")
        cls.directory = Path(cls.temporary.name)
        (cls.directory / "configs").mkdir()
        for name in ("neural.example.json", "mastery-C-plan.json", "mastery-cohort-plan.json"):
            shutil.copyfile(ROOT / "configs" / name, cls.directory / "configs" / name)
        cls.config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        cls.plan = MasteryPlan.load(ROOT / "configs/mastery-cohort-plan.json")
        cls.prior = cls.directory / "runs/cohort-input/prior/prior.pt"
        cls.prior.parent.mkdir(parents=True)
        original = os.environ.get("DOVIS_COHORT_TEST_PRIOR")
        if original:
            original = Path(original)
            shutil.copyfile(original, cls.prior)
            shutil.copyfile(original.with_suffix(".json"), cls.prior.with_suffix(".json"))
        else:
            pngs = [world.render_scene(scene) for scene in world.development_splits(
                cls.config, cls.plan.data_seed)["train"]]
            bundle = pretrain_prior(pngs, seed=59001, dim=cls.plan.model_profile.hidden_dim,
                steps=1, batch_size=16, eval_every=1)
            if bundle["report"]["status"] != "completed":
                raise AssertionError("The genuine CLI fixture image prior did not complete")
            save_prior_bundle(bundle, cls.prior)
        cls.prior_manifest = json.loads(cls.prior.with_suffix(".json").read_text())
        cls.prior_sha256 = hashlib.sha256(cls.prior.read_bytes()).hexdigest()
        cls.request_path = cls.directory / "configs/mastery-cohort-run.request.json"
        cls.summary_path = cls.directory / "runs/cohort-input/source-summary.json"
        qualified = source_summary(cls.plan, "F1")
        rows = []
        for original_row in qualified["rows"]:
            for condition in ("F0", "F1"):
                for step in (0, *cls.plan.budgets):
                    row = copy.deepcopy(original_row)
                    row.update(condition=condition, step=step,
                        memory_source="cnn" if condition == "F0" else "objects")
                    if condition == "F0":
                        row["dev"].update(valid_generation_hits=88,
                            valid_generation_accuracy=88 / 112)
                        row["candidate"].update(status="targets_not_met", score_targets_met=False)
                    rows.append(row)
        qualified.update(phase="F", axis="producer_perceptual_memory", rows=rows)
        cls.summary_path.write_text(json.dumps(qualified))
        request = request_record(cls.plan, "F1", actual_sources=True,
            prior_sha=cls.prior_sha256,
            prior_state=cls.prior_manifest["report"]["trained_prior_sha256"])["content"]
        files = {"validation_plan_sha256": cls.directory / "configs/mastery-cohort-plan.json",
            "source_plan_sha256": cls.directory / "configs/mastery-C-plan.json",
            "source_summary_sha256": cls.summary_path,
            "prior_manifest_sha256": cls.prior.with_suffix(".json"),
            "prior_checkpoint_sha256": cls.prior}
        request.update({key: hashlib.sha256(path.read_bytes()).hexdigest()
                        for key, path in files.items()})
        request["cohort_implementation_sha256"] = hashlib.sha256(Path(cohort.__file__).read_bytes()).hexdigest()
        cls.request_path.write_text(json.dumps(request))
        cls.request_bytes = cls.request_path.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_actual_workflow_training_and_aggregate_commands_accept_complete_real_assets(self):
        from mutual_lab.neural import digest

        called = []

        def fake_training(config, language, replica, plan, output, *, prior_bundle,
                          candidate_request, **kwargs):
            self.assertEqual(config.to_dict(), self.config.to_dict())
            self.assertEqual(plan, self.plan)
            self.assertEqual(prior_bundle["_bundle_sha256"], self.prior_sha256)
            self.assertEqual(digest(prior_bundle["trained_state"]),
                self.prior_manifest["report"]["trained_prior_sha256"])
            self.assertEqual(candidate_request["sha256"], hashlib.sha256(self.request_bytes).hexdigest())
            self.assertEqual(candidate_request["content"]["candidate_spec"], {
                "candidate_kind": "memory", "producer_memory": "objects", "understanding": "base"})
            called.append((language, replica))
            path = Path(output) / f"cohort-{language}-r{replica}-training-fixture"
            path.mkdir(parents=True)
            suite = suite_fixture(language, replica, plan, candidate_request)
            (path / "cohort-validation.json").write_text(json.dumps(suite))
            return path

        with contextlib.chdir(self.directory), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(cohort, "run_cohort_validation", side_effect=fake_training):
                for replica in range(3):
                    for language in LANGUAGES_ORDER:
                        self.assertEqual(cohort.main(workflow_arguments("training",
                            language=language, replica=replica)), 0)
            self.assertEqual(set(called), {(language, replica) for replica in range(3)
                for language in LANGUAGES_ORDER})
            self.assertEqual(len(called), 12)
            shutil.copytree("runs/cohort", "runs/cohort-downloads")
            self.assertEqual(cohort.main(workflow_arguments("aggregate")), 0)
            summary = json.loads(Path("runs/cohort-summary.json").read_text())
        self.assertTrue(summary["execution_completed"])
        self.assertTrue(summary["provenance_valid"])
        self.assertTrue(summary["robust_development_candidate"])
        self.assertEqual(summary["final_candidate_count"], 12)
        self.assertEqual(summary["missing_trials"], [])
        self.assertEqual(summary["failed_trials"], [])

    def test_required_request_fields_are_checked_before_any_training(self):
        incomplete = json.loads(self.request_bytes)
        incomplete.pop("validation_run_attempt")
        self.request_path.write_text(json.dumps(incomplete))
        try:
            with contextlib.chdir(self.directory), patch.object(cohort, "run_cohort_validation") as train:
                with self.assertRaises(ValueError):
                    cohort.main(workflow_arguments("training"))
                train.assert_not_called()
        finally:
            self.request_path.write_bytes(self.request_bytes)

    def test_unknown_workflow_option_cannot_silently_reach_training(self):
        with contextlib.chdir(self.directory), contextlib.redirect_stderr(io.StringIO()), \
                patch.object(cohort, "run_cohort_validation") as train:
            with self.assertRaises(SystemExit) as error:
                cohort.main(workflow_arguments("training") + ["--policy", "mixed_blocks"])
            self.assertEqual(error.exception.code, 2)
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
