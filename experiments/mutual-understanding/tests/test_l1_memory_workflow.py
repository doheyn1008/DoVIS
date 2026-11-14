"""Execute F's real aggregate wrapper without turning execution failure into data."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

EXPERIMENT = Path(__file__).resolve().parents[1]
WORKFLOW = EXPERIMENT.parents[1] / '.github/workflows/l1-memory-F.yml'
REQUEST = Path('configs/mastery-F-run.request.json')
SUMMARY = Path('runs/memory-F-summary.json')


def aggregate_wrapper():
    blocks = re.findall(r'        run: \|\n((?:          [^\n]*\n)+)', WORKFLOW.read_text())
    block = next(textwrap.dedent(block) for block in blocks if 'aggregate_exit_code = subprocess.call' in block)
    return block.split("python - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]


class MemoryWorkflowFallbackTests(unittest.TestCase):
    def execute(self, root, *, code, verification='failure', request=False, existing=None):
        if request:
            (root / REQUEST).parent.mkdir(parents=True)
            (root / REQUEST).write_text('UNVALIDATED_PRIVATE_REQUEST_SENTINEL')

        def process(command):
            self.assertEqual(command[:3], [sys.executable, '-m', 'mutual_lab.memory_study'])
            self.assertNotIn('--policy', command)
            if existing is not None:
                (root / SUMMARY).parent.mkdir(parents=True, exist_ok=True)
                (root / SUMMARY).write_bytes(existing)
            return code

        previous = Path.cwd()
        try:
            os.chdir(root)
            with patch.dict(os.environ, {'F_VERIFY_RESULT': verification}), patch.object(
                    subprocess, 'call', side_effect=process) as called:
                with self.assertRaises(SystemExit) as stopped:
                    exec(compile(aggregate_wrapper(), str(WORKFLOW), 'exec'), {'__name__': '__main__'})
                self.assertEqual(stopped.exception.code, code)
                called.assert_called_once()
        finally:
            os.chdir(previous)
        return root / SUMMARY

    def test_failed_verification_without_a_request_preserves_the_module_error_and_missing_languages(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.execute(Path(temporary), code=1)
            summary = json.loads(path.read_text())
            self.assertEqual(summary['status'], 'invalid_or_unavailable_input')
            self.assertEqual(summary['verification_job_status'], 'failure')
            self.assertEqual(summary['aggregation_module_exit_code'], 1)
            self.assertFalse(summary['request_present'])
            self.assertEqual(summary['missing_languages'], ['ko', 'ja', 'en', 'es'])
            self.assertEqual(summary['rows'], [])
            self.assertTrue(summary['excluded_from_research'])
            self.assertFalse(summary['eligible_for_fresh_experiment'])
            self.assertFalse(summary['robust_development_candidate'])

    def test_present_unvalidated_request_is_not_copied_into_failure_context(self):
        for verification, status in (('failure', 'invalid_or_unavailable_input'), ('success', 'execution_failed')):
            with self.subTest(verification=verification), tempfile.TemporaryDirectory() as temporary:
                path = self.execute(Path(temporary), code=7, verification=verification, request=True)
                summary = json.loads(path.read_text())
                self.assertEqual(summary['status'], status)
                self.assertEqual(summary['aggregation_module_exit_code'], 7)
                self.assertTrue(summary['request_present'])
                self.assertNotIn('UNVALIDATED_PRIVATE_REQUEST_SENTINEL', path.read_text())
                self.assertFalse(summary['native_aggregation_summary_available'])

    def test_existing_completed_or_incomplete_aggregation_is_never_rewritten(self):
        for code, status in ((0, 'completed'), (1, 'incomplete')):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                original = (json.dumps({'kind': 'memory-study-v1-comparison', 'status': status,
                                       'rows': [{'preserved': True}]}, indent=1) + '\n').encode()
                path = self.execute(Path(temporary), code=code, existing=original)
                self.assertEqual(path.read_bytes(), original)

    def test_failure_to_write_fallback_does_not_replace_the_original_module_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                Path, 'write_text', side_effect=OSError('fallback persistence unavailable')):
            path = self.execute(Path(temporary), code=7)
            self.assertFalse(path.exists())

    def test_real_module_missing_and_invalid_requests_still_leave_a_failure_summary(self):
        for request in (False, True):
            with self.subTest(request=request), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'configs').mkdir()
                shutil.copyfile(EXPERIMENT / 'configs/mastery-C-plan.json', root / 'configs/mastery-C-plan.json')
                if request:
                    (root / REQUEST).write_text('UNVALIDATED_PRIVATE_REQUEST_SENTINEL')
                environment = os.environ.copy()
                environment['PYTHONPATH'] = str(EXPERIMENT)
                environment['F_VERIFY_RESULT'] = 'failure'
                process = subprocess.run([sys.executable, '-c', aggregate_wrapper()], cwd=root,
                    env=environment, capture_output=True, text=True)
                self.assertNotEqual(process.returncode, 0)
                summary = json.loads((root / SUMMARY).read_text())
                self.assertEqual(summary['status'], 'invalid_or_unavailable_input')
                self.assertEqual(summary['aggregation_module_exit_code'], process.returncode)
                self.assertEqual(summary['request_present'], request)
                self.assertEqual(summary['rows'], [])
                self.assertNotIn('UNVALIDATED_PRIVATE_REQUEST_SENTINEL', process.stdout + process.stderr)


if __name__ == '__main__':
    unittest.main()
