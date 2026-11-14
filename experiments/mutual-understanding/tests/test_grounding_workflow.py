"""Execute the checked-in G workflow's literal commands through its real CLI."""

import contextlib
import importlib.util
import io
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mutual_lab import grounding_study as grounding
from mutual_lab.config import ExperimentConfig
from mutual_lab.mastery_lab import MasteryPlan

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT.parents[1] / ".github/workflows/l1-grounding-G.yml"
REQUEST = ROOT / "configs/mastery-G-run.request.json"


def workflow_arguments():
    """Read the actual train/aggregate shell lines without a YAML dependency."""
    commands = {}
    replacements = {"${{ matrix.language }}": "en",
                    "${{ needs.verify.outputs.producer_memory }}": "objects"}
    for line in WORKFLOW.read_text().splitlines():
        if "python -m mutual_lab.grounding_study " not in line:
            continue
        for source, target in replacements.items():
            line = line.replace(source, target)
        arguments = shlex.split(line.strip().removeprefix("run: "))
        arguments = arguments[arguments.index("mutual_lab.grounding_study") + 1:]
        kind = "aggregate" if "--aggregate-dir" in arguments else "train"
        if kind in commands:
            raise AssertionError("Duplicate workflow CLI command: " + kind)
        commands[kind] = arguments
    if set(commands) != {"train", "aggregate"}:
        raise AssertionError("Expected one actual train command and one aggregate command")
    return commands


def replace_option(arguments, option, value):
    arguments[arguments.index(option) + 1] = str(value)


def real_paths(arguments):
    arguments = list(arguments)
    replace_option(arguments, "--plan", ROOT / "configs/mastery-C-plan.json")
    replace_option(arguments, "--study-request", REQUEST)
    return arguments


class GroundingWorkflowContractTests(unittest.TestCase):
    def test_checked_in_request_passes_the_actual_cli_schema(self):
        plan = MasteryPlan.load(ROOT / "configs/mastery-C-plan.json")
        request = grounding._request(REQUEST, plan)
        self.assertEqual(request["content"]["phase"], "G")
        self.assertEqual(request["content"]["model"], "attentive")
        self.assertEqual(request["content"]["policy"], "mixed_blocks")
        self.assertEqual(request["content"]["producer_memory"], "objects")

    def test_actual_empty_aggregate_command_executes_and_preserves_all_missing_languages(self):
        arguments = real_paths(workflow_arguments()["aggregate"])
        self.assertNotIn("--producer-memory", arguments)
        with tempfile.TemporaryDirectory() as temporary:
            empty, output = Path(temporary) / "empty", Path(temporary) / "summary.json"
            empty.mkdir()
            replace_option(arguments, "--aggregate-dir", empty)
            replace_option(arguments, "--output", output)
            with patch.object(grounding, "run_grounding_study") as train, contextlib.redirect_stdout(io.StringIO()):
                status = grounding.main(arguments)
            train.assert_not_called()
            self.assertEqual(status, 1)
            summary = json.loads(output.read_text())
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(set(summary["missing_languages"]), {"en", "es", "ja", "ko"})
            self.assertEqual(summary["selection"]["status"], "undetermined")

    def test_missing_phase_in_actual_request_is_rejected_before_training(self):
        arguments = real_paths(workflow_arguments()["train"])
        content = json.loads(REQUEST.read_text())
        content.pop("phase")
        with tempfile.TemporaryDirectory() as temporary:
            malformed = Path(temporary) / "missing-phase.json"
            malformed.write_text(json.dumps(content))
            replace_option(arguments, "--study-request", malformed)
            replace_option(arguments, "--prior-bundle", Path(temporary) / "must-not-be-loaded.pt")
            replace_option(arguments, "--output", Path(temporary) / "unused")
            with patch.object(grounding, "run_grounding_study") as train:
                with self.assertRaisesRegex(ValueError, "Pinned G request"):
                    grounding.main(arguments)
            train.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class GroundingWorkflowPriorTests(unittest.TestCase):
    def test_actual_train_command_loads_real_prior_config_plan_and_checked_in_request(self):
        from PIL import Image, ImageDraw
        from mutual_lab.spatial_prior import pretrain_prior, save_prior_bundle
        arguments = real_paths(workflow_arguments()["train"])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            specified = os.environ.get("DOVIS_G_WORKFLOW_TEST_PRIOR")
            staged = ROOT / "runs/grounding-G-input/source-F/source-E/prior/prior.pt"
            prior = Path(specified) if specified else staged
            if specified:
                self.assertTrue(prior.is_file(), "The explicitly requested real prior must exist")
            if not prior.is_file():
                # Standalone CI has no downloaded upstream asset. This creates a
                # real one-step image-only bundle and still uses the real loader.
                images = []
                for color in ((21, 89, 177), (191, 37, 97)):
                    image = Image.new("RGB", (48, 32), (246, 246, 239))
                    ImageDraw.Draw(image).rectangle((8, 7, 20, 24), fill=color)
                    encoded = io.BytesIO(); image.save(encoded, format="PNG"); images.append(encoded.getvalue())
                bundle = pretrain_prior(images, dim=128, steps=1, batch_size=2, eval_every=1)
                prior = save_prior_bundle(bundle, directory / "prior.pt")
            replace_option(arguments, "--prior-bundle", prior)
            replace_option(arguments, "--output", directory / "output")
            expected_request = grounding._request(REQUEST, MasteryPlan.load(ROOT / "configs/mastery-C-plan.json"), "objects")

            def train_boundary(config, language, plan, output, **kwargs):
                self.assertIsInstance(config, ExperimentConfig)
                config.validate(); plan.validate(config); grounding._validate_plan(plan)
                self.assertEqual(language, "en")
                self.assertEqual(plan.b_seed_pair, (59007, 59009))
                self.assertEqual(plan.budgets, (2400, 4800, 9600))
                self.assertEqual(kwargs["producer_memory"], "objects")
                self.assertEqual(kwargs["study_request"], expected_request)
                self.assertEqual(kwargs["study_request"]["content"]["phase"], "G")
                self.assertEqual(kwargs["study_request"]["content"]["model"], "attentive")
                self.assertEqual(kwargs["prior_bundle"]["report"]["status"], "completed")
                self.assertEqual(kwargs["prior_bundle"]["report"]["dimension"], 128)
                path = Path(output) / "mocked-training-boundary"
                path.mkdir(parents=True)
                (path / "grounding-study.json").write_text(json.dumps({"status": "completed"}))
                return path

            with patch.object(grounding, "run_grounding_study", side_effect=train_boundary) as train, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(grounding.main(arguments), 0)
            train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
