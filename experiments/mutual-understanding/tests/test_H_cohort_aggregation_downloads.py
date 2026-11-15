"""Exercise the workflow's literal aggregate download-failure branch with stdlib."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import textwrap
import unittest


def workflow_path():
    here = Path(__file__).resolve()
    sibling = here.with_name('l1-H-cohort-validation.yml')
    if sibling.is_file():
        return sibling
    for root in here.parents:
        candidate = root / '.github/workflows/l1-H-cohort-validation.yml'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('Expected prototype sibling or repository H cohort workflow')


WORKFLOW = workflow_path()
STEP = 'Retain all twelve fixed-final outcomes and separate execution, provenance and targets'
NEXT = 'Record blocked provenance and every unstarted cell when verification failed'
EXPECTED_COMMAND = ['-m', 'mutual_lab.cohort_context_validation', '--config', 'configs/neural.example.json',
    '--plan', 'configs/mastery-H-cohort-plan.json', '--candidate-request', 'configs/mastery-H-cohort-run.request.json',
    '--source-summary', 'runs/H-cohort-input/source-summary.json', '--source-plan', 'configs/mastery-H-plan.json',
    '--prior-bundle', 'runs/H-cohort-input/prior/prior.pt', '--aggregate-dir', 'runs/H-cohort-downloads',
    '--output', 'runs/H-cohort-summary.json']


def aggregate_step():
    return WORKFLOW.read_text().split('      - name: ' + STEP + '\n', 1)[1].split('      - name: ' + NEXT + '\n', 1)[0]


def run_block():
    return textwrap.dedent(aggregate_step().split('        run: |\n', 1)[1])


def suite(language='ko', replica=0, status='completed'):
    return {'kind': 'cohort-context-validation-v2', 'language': language, 'replica': replica, 'status': status,
        'engine_report': {'kind': 'mastery-lab-v1-trial', 'status': status, 'training_steps_completed': 9600,
            'curves': [{'step': x} for x in (0, 2400, 4800, 9600)],
            'checkpoints': [{'step': x} for x in (0, 2400, 4800, 9600)],
            'candidate': {'score_targets_met': True}}}


class AggregationDownloadTests(unittest.TestCase):
    def execute(self, *, input_outcome='failure', trials_outcome='success', records=(), canonical_exit=0):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            shim = directory / 'shim'
            shim.mkdir()
            executable = shim / 'python'
            executable.write_text(textwrap.dedent('''\
                #!/usr/bin/env python3
                import json, os, pathlib, sys
                if sys.argv[1:3] == ['-m', 'mutual_lab.cohort_context_validation']:
                    pathlib.Path(os.environ['CANONICAL_CALL_LOG']).write_text(json.dumps(sys.argv[1:]))
                    target = pathlib.Path('runs/H-cohort-summary.json')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps({'canonical_cli': True}))
                    raise SystemExit(int(os.environ['CANONICAL_EXIT']))
                if sys.argv[1:] == ['-']:
                    os.execv(sys.executable, [sys.executable, '-'])
                raise SystemExit('Unexpected workflow Python command')
                '''))
            executable.chmod(0o700)
            for index, value in enumerate(records):
                path = directory / 'runs/H-cohort-downloads' / str(index) / 'cohort-context-validation.json'
                path.parent.mkdir(parents=True)
                path.write_text(value if isinstance(value, str) else json.dumps(value))
            log = directory / 'canonical-call.json'
            environment = dict(os.environ, PATH=str(shim) + os.pathsep + os.environ.get('PATH', ''),
                SOURCE_VERIFICATION='success', INPUT_DOWNLOAD_RESULT=input_outcome,
                TRIAL_DOWNLOAD_RESULT=trials_outcome, CANONICAL_CALL_LOG=str(log), CANONICAL_EXIT=str(canonical_exit))
            completed = subprocess.run(['bash', '-e', '-c', run_block()], cwd=directory,
                env=environment, text=True, capture_output=True, timeout=15)
            summary_path = directory / 'runs/H-cohort-summary.json'
            self.assertTrue(summary_path.is_file(), completed.stderr)
            result = json.loads(summary_path.read_text())
            command = json.loads(log.read_text()) if log.exists() else None
        return completed, result, command

    def assert_incomplete(self, completed, result, command):
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIsNone(command)
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['provenance_valid'])
        self.assertFalse(result['robust_development_candidate'])
        self.assertFalse(result['qualification_assessed'])
        self.assertIsNone(result['final_candidate_count'])
        self.assertFalse(result['native_training_rerun'])
        self.assertTrue(result['excluded_from_research'])
        self.assertFalse(result['eligible_for_fresh_experiment'])

    def test_input_download_failure_retains_one_completed_available_model(self):
        completed, result, command = self.execute(records=[suite()])
        self.assert_incomplete(completed, result, command)
        self.assertEqual(result['reported_jobs'], 1)
        self.assertEqual(len(result['missing_trials']), 11)
        self.assertFalse(result['execution_completed'])
        record = result['available_trials'][0]
        self.assertEqual((record['language'], record['replica'], record['status'], record['engine_status']),
            ('ko', 0, 'completed', 'completed'))
        self.assertEqual(record['reported_budget_steps'], [0, 2400, 4800, 9600])
        self.assertEqual(record['checkpoint_manifests'], 4)
        self.assertEqual(len(record['source_sha256']), 64)
        self.assertNotIn({'language': 'ko', 'replica': 0}, result['missing_trials'])
        self.assertEqual(result['artifact_downloads'], {'input': 'failure', 'trials': 'success'})

    def test_missing_all_trials_is_unavailable_rather_than_claimed_unstarted(self):
        completed, result, command = self.execute(input_outcome='failure', trials_outcome='failure')
        self.assert_incomplete(completed, result, command)
        self.assertEqual(result['reported_jobs'], 0)
        self.assertEqual(result['available_trials'], [])
        self.assertEqual(len(result['missing_trials']), 12)
        self.assertTrue(all(set(row) == {'language', 'replica'} for row in result['missing_trials']))
        self.assertFalse(result['execution_completed'])

    def test_trial_download_failure_retains_partial_completed_and_cancelled_cells(self):
        completed, result, command = self.execute(input_outcome='success', trials_outcome='failure',
            records=[suite(), suite('ja', 1, 'cancelled')])
        self.assert_incomplete(completed, result, command)
        self.assertEqual(result['reported_jobs'], 2)
        self.assertEqual(len(result['missing_trials']), 10)
        self.assertEqual(len(result['failed_trials']), 1)
        self.assertEqual(result['failed_trials'][0]['status'], 'cancelled')
        self.assertEqual(result['artifact_downloads']['trials'], 'failure')

    def test_unreadable_unknown_and_duplicate_metadata_do_not_suppress_summary(self):
        completed, result, command = self.execute(records=[suite(), suite(), '{broken', suite('foreign', 0), suite('ko', False)])
        self.assert_incomplete(completed, result, command)
        self.assertEqual(result['available_trial_files'], 5)
        self.assertEqual(len(result['available_trials']), 4)
        self.assertEqual(result['reported_jobs'], 1)
        self.assertEqual(len(result['unreadable_trial_files']), 1)
        self.assertEqual(len(result['unidentified_trial_files']), 2)
        self.assertEqual(len(result['duplicate_trials']), 1)
        self.assertEqual(len(result['missing_trials']), 11)

    def test_all_completed_metadata_does_not_bypass_missing_input_provenance(self):
        records = [suite(language, replica) for replica in range(3) for language in ('ko', 'ja', 'en', 'es')]
        completed, result, command = self.execute(records=records)
        self.assert_incomplete(completed, result, command)
        self.assertTrue(result['execution_completed'])
        self.assertEqual(result['reported_jobs'], 12)
        self.assertEqual(result['missing_trials'], [])
        self.assertEqual(result['failed_trials'], [])
        self.assertIn('Unverified', result['execution_evidence'])

    def test_successful_downloads_invoke_exact_canonical_cli(self):
        completed, result, command = self.execute(input_outcome='success', trials_outcome='success')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(command, EXPECTED_COMMAND)
        self.assertEqual(result, {'canonical_cli': True})

    def test_canonical_failure_is_propagated_without_qualification_fallback(self):
        completed, result, command = self.execute(input_outcome='success', trials_outcome='success', canonical_exit=17)
        self.assertEqual(completed.returncode, 17)
        self.assertEqual(command, EXPECTED_COMMAND)
        self.assertEqual(result, {'canonical_cli': True})

    def test_outcomes_use_distinct_download_ids_and_existing_verify_failed_path(self):
        source = WORKFLOW.read_text().split('  aggregate_cohort:\n', 1)[1]
        self.assertEqual(source.count('id: aggregate_input'), 1)
        self.assertEqual(source.count('id: aggregate_trials'), 1)
        step = aggregate_step()
        self.assertIn("if: always() && needs.verify.result == 'success'", step)
        self.assertIn('${{ steps.aggregate_input.outcome }}', step)
        self.assertIn('${{ steps.aggregate_trials.outcome }}', step)
        self.assertNotIn('.conclusion', step)
        blocked = source.split('      - name: ' + NEXT + '\n', 1)[1]
        self.assertIn("if: needs.verify.result != 'success'", blocked)
        self.assertIn("'status': 'blocked'", blocked)
        self.assertIn('if: always()', blocked)


if __name__ == '__main__':
    unittest.main()
