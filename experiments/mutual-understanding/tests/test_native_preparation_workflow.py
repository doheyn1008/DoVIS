"""Run the real aggregate step without optional shell search utilities."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import asdict
from pathlib import Path

from mutual_lab.native_preparation import PreparationPlan

EXPERIMENT = Path(__file__).resolve().parents[1]
WORKFLOW = EXPERIMENT.parents[1] / ".github/workflows/native-preparation.yml"
BASH = shutil.which("bash")


def aggregate_step():
    blocks = re.findall(r"        run: \|\n((?:          [^\n]*\n)+)", WORKFLOW.read_text())
    return next(textwrap.dedent(block) for block in blocks if "classify_all_languages" in block)


@unittest.skipUnless(BASH, "workflow integration requires bash")
class AggregationWorkflowTests(unittest.TestCase):
    def run_step(self, root):
        # The Actions runner lacked rg. Make only Python available on PATH.
        binary = root / "bin"
        binary.mkdir()
        (binary / "python").symlink_to(sys.executable)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(EXPERIMENT)
        environment["PATH"] = str(binary)
        self.assertIsNone(shutil.which("rg", path=environment["PATH"]))
        return subprocess.run([BASH, "-e", "-o", "pipefail", "-c", aggregate_step()],
                              cwd=root, env=environment, capture_output=True, text=True)

    def prepare(self, root):
        (root / ".gitignore").write_text("runs/\n")
        directory = root / "runs/native-preparation-downloads"
        directory.mkdir(parents=True)
        return directory

    def test_all_language_reports_are_aggregated_without_optional_search_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.prepare(root)
            for language in ("ko", "ja", "en", "es"):
                output = directory / language
                output.mkdir()
                report = {"language": language, "status": "completed",
                          "plan": asdict(PreparationPlan(budgets=(1,))),
                          "model_profile": {"id": "test"}, "config_sha256": "config",
                          "semantic_data_sha256": "semantics", "source_sha256": {"module": "source"},
                          "curves": [{"step": 1, "dev": {"comprehension_accuracy": 1.0,
                                      "valid_generation_accuracy": 1.0}}]}
                (output / "report.json").write_text(json.dumps(report))
            completed = self.run_step(root)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((root / "runs/native-preparation-summary.json").read_text())
            self.assertEqual(summary["missing_languages"], [])
            self.assertEqual(summary["common_budget"], 1)

    def test_empty_artifacts_fail_instead_of_publishing_a_successful_empty_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            self.assertNotEqual(self.run_step(root).returncode, 0)
            self.assertFalse((root / "runs/native-preparation-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
