"""Exercise the actual cross-branch workflow admission and wrapper guards."""

import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import textwrap
import unittest
from unittest.mock import patch
import urllib.error
import urllib.parse

def workflow_and_configs():
    test_path = Path(__file__).resolve()
    sibling_workflow = test_path.with_name("l1-H-cohort-validation.yml")
    if sibling_workflow.is_file():
        return sibling_workflow, test_path.parent
    for root in test_path.parents:
        workflow = root / ".github/workflows/l1-H-cohort-validation.yml"
        configs = root / "experiments/mutual-understanding/configs"
        if workflow.is_file() and configs.is_dir():
            return workflow, configs
    raise FileNotFoundError("Expected sibling prototype files or the promoted repository workflow/configs")


WORKFLOW, CONFIGS = workflow_and_configs()
WORKFLOW_PATH = ".github/workflows/l1-H-cohort-validation.yml"
REPOSITORY = "fixture/DoVIS"
REQUEST = b'{"pinned_candidate":"same-cohort-request"}\n'
OLD_WRAPPER = "830c3a81d3e6aac87d2ab8bd38583a3b8ae1a7b2fc330bcc1006cc1e11989d53"
NEW_WRAPPER = "3e5bc9318a98bc8775988059d5654103e0136330987a7dad1854b8d618596c19"


def python_block(step_name):
    source = WORKFLOW.read_text().split("      - name: " + step_name + "\n", 1)[1]
    source = source.split("          python - <<'PY'\n", 1)[1].split("          PY", 1)[0]
    return textwrap.dedent(source)


def run_record(run_id, *, branch="codex/l1-mastery", revision=None):
    return {"id": run_id, "path": WORKFLOW_PATH, "head_branch": branch,
        "head_sha": revision or f"{run_id:040x}", "repository": {"full_name": REPOSITORY},
        "created_at": "2026-11-14T01:00:00Z", "status": "completed",
        "conclusion": "failure", "run_attempt": 1,
        "html_url": f"https://github.com/{REPOSITORY}/actions/runs/{run_id}"}


def verify_job(run_id, conclusion="success", status="completed"):
    return {"id": run_id * 10, "name": "verify", "status": status, "conclusion": conclusion}


class CohortAdmissionTests(unittest.TestCase):
    request_path = "configs/mastery-H-cohort-run.request.json"
    evidence_path = "runs/H-cohort-input/provenance/admission.json"
    concurrency_group = "l1-H-cohort-validation"

    def execute(self, *, history=None, contents=None, jobs=None, attempt="1", failure=None):
        current = run_record(900)
        current.update(created_at="2026-11-14T02:00:00Z", status="in_progress", conclusion=None)
        history = history if history is not None else {1: [current], 2: []}
        contents, jobs = contents or {}, jobs or {}
        calls = []

        class API:
            def open(self, request, timeout=30):
                calls.append(request.full_url)
                parsed = urllib.parse.urlparse(request.full_url)
                self_outer.assertEqual(parsed.scheme, "https")
                self_outer.assertEqual(parsed.netloc, "api.github.com")
                self_outer.assertEqual(timeout, 30)
                path = parsed.path.removeprefix("/repos/" + REPOSITORY)
                query = urllib.parse.parse_qs(parsed.query)
                page = int(query.get("page", ["1"])[0])
                if path == "/actions/runs/900":
                    result = current
                elif path == "/actions/runs":
                    if failure == "history":
                        raise urllib.error.HTTPError(request.full_url, 503, "fixture unavailable", {}, None)
                    result = {"workflow_runs": history.get(page, [])}
                elif path.startswith("/commits/"):
                    if failure == "revision_missing":
                        raise urllib.error.HTTPError(request.full_url, 404, "historical commit uncertain", {}, None)
                    result = {"sha": path.rsplit("/", 1)[-1]}
                elif path.startswith("/contents/"):
                    revision = query["ref"][0]
                    value = contents.get(revision)
                    if value is None:
                        raise urllib.error.HTTPError(request.full_url, 404, "no pre-activation request", {}, None)
                    if failure == "request_forbidden":
                        raise urllib.error.HTTPError(request.full_url, 403, "fixture denied", {}, None)
                    result = {"type": "file", "encoding": "base64",
                        "content": base64.b64encode(value).decode()}
                else:
                    matched = re.fullmatch(r"/actions/runs/([0-9]+)/attempts/1/jobs", path)
                    if not matched:
                        raise AssertionError("Unexpected API path or a latest-attempt jobs query: " + path)
                    if failure == "jobs_missing":
                        raise urllib.error.HTTPError(request.full_url, 404, "jobs uncertain", {}, None)
                    result = {"jobs": jobs.get((int(matched.group(1)), page), [])}
                return io.StringIO(json.dumps(result))

        self_outer = self
        source = python_block("Reject a previously admitted identical cohort request across both branches")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "configs").mkdir()
            (directory / self.request_path).write_bytes(REQUEST)
            environment = {"GITHUB_RUN_ATTEMPT": attempt, "GITHUB_RUN_ID": "900",
                "GITHUB_REPOSITORY": REPOSITORY, "SOURCE_GITHUB_TOKEN": "offline-fixture"}
            error = None
            with contextlib.chdir(directory), patch.dict(os.environ, environment), \
                    patch("urllib.request.build_opener", return_value=API()), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    exec(compile(source, str(WORKFLOW) + ":admission", "exec"), {"__name__": "__fixture__"})
                except Exception as raised:
                    error = raised
            evidence = json.loads((directory / self.evidence_path).read_text())
        return evidence, error, calls

    def test_first_admission_records_canonical_identity_and_raw_provenance(self):
        content = {"candidate": "H1", "rationale": "모국어 cohort", "seed_pair": [99007, 99009]}
        first = json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8")
        reordered = json.dumps(dict(reversed(tuple(content.items()))), ensure_ascii=True).encode("utf-8")
        fingerprints = []
        for current in (first, reordered):
            with self.subTest(current=current), patch.dict(globals(), {"REQUEST": current}):
                evidence, error, _ = self.execute()
                self.assertIsNone(error)
                self.assertEqual(evidence["status"], "no_prior_admission")
                self.assertEqual(evidence["request_sha256"], hashlib.sha256(current).hexdigest())
                self.assertEqual(evidence["request_identity_policy"], "canonical_parsed_json_v1")
                fingerprints.append(evidence["request_identity_sha256"])
        self.assertNotEqual(hashlib.sha256(first).hexdigest(), hashlib.sha256(reordered).hexdigest())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_whitespace_and_key_order_cannot_replace_an_admitted_request(self):
        content = {"candidate": "H1", "rationale": "모국어 cohort", "seed_pair": [99007, 99009]}
        current = json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8")
        variants = (json.dumps(content, ensure_ascii=True, separators=(",", ":")).encode("utf-8"),
                    json.dumps(dict(reversed(tuple(content.items()))), ensure_ascii=False, indent=4).encode("utf-8"))
        prior = run_record(100, branch="codex/l1-memory")
        for historical in variants:
            with self.subTest(historical=historical), patch.dict(globals(), {"REQUEST": current}):
                self.assertNotEqual(hashlib.sha256(current).hexdigest(), hashlib.sha256(historical).hexdigest())
                evidence, error, _ = self.execute(history={1: [prior]},
                    contents={prior["head_sha"]: historical}, jobs={(100, 1): [verify_job(100)]})
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(evidence["status"], "previously_admitted")
                self.assertEqual(evidence["blocking_run"]["id"], 100)
                self.assertEqual(evidence["request_sha256"], hashlib.sha256(current).hexdigest())
                self.assertEqual(evidence["blocking_run"]["request_sha256"], hashlib.sha256(historical).hexdigest())

    def test_malformed_historical_request_fails_closed_before_admission(self):
        prior = run_record(100)
        for historical in (b"not JSON", b"\xff", b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'):
            with self.subTest(historical=historical):
                evidence, error, calls = self.execute(history={1: [prior]},
                    contents={prior["head_sha"]: historical}, jobs={(100, 1): [verify_job(100)]})
                self.assertIsNotNone(error)
                self.assertEqual(evidence["status"], "blocked_uncertain")
                self.assertFalse(any("/attempts/" in call for call in calls))

    def test_distinct_parsed_candidate_or_rationale_remains_independent(self):
        content = {"candidate": "H1", "rationale": "same native cohort"}
        current = json.dumps(content).encode("utf-8")
        prior = run_record(100)
        for historical_content in (dict(content, candidate="H0"),
                                   dict(content, rationale="a different declared experiment")):
            with self.subTest(historical_content=historical_content), patch.dict(globals(), {"REQUEST": current}):
                historical = json.dumps(historical_content).encode("utf-8")
                evidence, error, calls = self.execute(history={1: [prior]},
                    contents={prior["head_sha"]: historical}, jobs={(100, 1): [verify_job(100)]})
                self.assertIsNone(error)
                self.assertEqual(evidence["status"], "no_prior_admission")
                self.assertEqual(evidence["matching_prior_requests"], [])
                self.assertFalse(any("/attempts/" in call for call in calls))

    def test_global_concurrency_serializes_both_allowed_branches(self):
        source = WORKFLOW.read_text()
        found = re.search(r"(?m)^concurrency:\n  group: ([^\n]+)\n  cancel-in-progress: false$", source)
        self.assertIsNotNone(found)
        self.assertEqual(found.group(1), self.concurrency_group)
        self.assertNotIn("$" + "{{", found.group(1))
        self.assertIn("branches: [codex/l1-mastery, codex/l1-memory]", source)

    def test_unseen_request_is_clear_to_verify(self):
        evidence, error, _ = self.execute()
        self.assertIsNone(error)
        self.assertEqual(evidence["status"], "no_prior_admission")

    def test_cross_branch_redispatch_preserves_previous_admission_and_failure(self):
        prior = run_record(100, branch="codex/l1-memory")
        evidence, error, _ = self.execute(history={1: [prior]}, contents={prior["head_sha"]: REQUEST},
            jobs={(100, 1): [verify_job(100)]})
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(evidence["status"], "previously_admitted")
        self.assertEqual(evidence["blocking_run"]["id"], 100)
        self.assertEqual(evidence["blocking_run"]["head_branch"], "codex/l1-memory")
        self.assertEqual(evidence["blocking_run"]["conclusion"], "failure")
        self.assertIn(prior["html_url"], str(error))

    def test_newer_created_run_with_successful_original_verify_still_blocks(self):
        prior = run_record(1000)
        prior["created_at"] = "2026-11-14T03:00:00Z"
        evidence, error, _ = self.execute(history={1: [prior]}, contents={prior["head_sha"]: REQUEST},
            jobs={(1000, 1): [verify_job(1000)]})
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(evidence["blocking_run"]["id"], 1000)

    def test_unstarted_queued_first_attempt_waits_behind_the_global_lock(self):
        prior = run_record(1000, branch="codex/l1-memory")
        prior.update(created_at="2026-11-14T03:00:00Z", status="queued", conclusion=None)
        evidence, error, _ = self.execute(history={1: [prior]}, contents={prior["head_sha"]: REQUEST})
        self.assertIsNone(error)
        self.assertEqual(evidence["matching_prior_requests"][0]["admission"], "first_attempt_not_started")

    def test_queued_later_attempt_cannot_hide_original_verify_success(self):
        prior = run_record(1000)
        prior.update(created_at="2026-11-14T03:00:00Z", status="queued", conclusion=None, run_attempt=2)
        evidence, error, _ = self.execute(history={1: [prior]}, contents={prior["head_sha"]: REQUEST},
            jobs={(1000, 1): [verify_job(1000)]})
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(evidence["status"], "previously_admitted")

    def test_different_request_does_not_replace_or_block_an_independent_candidate(self):
        prior = run_record(100)
        evidence, error, calls = self.execute(history={1: [prior]},
            contents={prior["head_sha"]: b'{"different":"request"}'})
        self.assertIsNone(error)
        self.assertEqual(evidence["status"], "no_prior_admission")
        self.assertFalse(any("/attempts/" in call for call in calls))

    def test_request_absent_before_activation_is_skipped(self):
        evidence, error, _ = self.execute(history={1: [run_record(100)]})
        self.assertIsNone(error)
        self.assertEqual(evidence["requests_missing_before_activation"], 1)

    def test_request_404_does_not_hide_an_unavailable_historical_revision(self):
        prior = run_record(100)
        evidence, error, calls = self.execute(history={1: [prior]}, failure="revision_missing")
        self.assertIsInstance(error, urllib.error.HTTPError)
        self.assertEqual(evidence["status"], "blocked_uncertain")
        self.assertEqual(evidence["requests_missing_before_activation"], 0)
        self.assertTrue(any("/commits/" in call for call in calls))

    def test_failed_verification_before_admission_may_be_corrected(self):
        prior = run_record(100)
        evidence, error, _ = self.execute(history={1: [prior]}, contents={prior["head_sha"]: REQUEST},
            jobs={(100, 1): [verify_job(100, conclusion="failure")]})
        self.assertIsNone(error)
        self.assertEqual(evidence["matching_prior_requests"][0]["admission"],
            "not_admitted_verify_did_not_succeed")

    def test_later_failed_attempt_cannot_hide_original_verify_success(self):
        prior = run_record(100)
        prior.update(run_attempt=3, conclusion="failure")
        evidence, error, calls = self.execute(history={1: [prior]}, contents={prior["head_sha"]: REQUEST},
            jobs={(100, 1): [verify_job(100)]})
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(evidence["status"], "previously_admitted")
        self.assertTrue(any("/attempts/1/jobs" in call for call in calls))
        self.assertFalse(any("/attempts/3/" in call for call in calls))

    def test_history_and_first_attempt_jobs_are_fully_paginated(self):
        a, b = run_record(100), run_record(200, branch="codex/l1-memory")
        evidence, error, calls = self.execute(history={1: [a], 2: [b], 3: []},
            contents={a["head_sha"]: b'{"different":"request-a"}', b["head_sha"]: REQUEST},
            jobs={(200, 1): [{"id": 2001, "name": "validate_candidate (ko, 0)",
                              "status": "completed", "conclusion": "failure"}],
                  (200, 2): [verify_job(200)]})
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(evidence["blocking_run"]["id"], 200)
        self.assertTrue(any("/actions/runs?per_page=100&page=2" in call for call in calls))
        self.assertTrue(any("/attempts/1/jobs?per_page=100&page=2" in call for call in calls))

    def test_current_second_attempt_is_rejected_before_history(self):
        evidence, error, calls = self.execute(attempt="2")
        self.assertIsInstance(error, ValueError)
        self.assertEqual(evidence["status"], "blocked_uncertain")
        self.assertEqual(calls, [])

    def test_api_uncertainty_never_means_no_prior_admission(self):
        prior = run_record(100)
        for failure in ("history", "request_forbidden", "jobs_missing"):
            with self.subTest(failure=failure):
                evidence, error, _ = self.execute(history={1: [prior]},
                    contents={prior["head_sha"]: REQUEST}, jobs={(100, 1): [verify_job(100)]},
                    failure=failure)
                self.assertIsNotNone(error)
                self.assertEqual(evidence["status"], "blocked_uncertain")

    def test_missing_or_unresolved_first_verify_fails_closed(self):
        prior = run_record(100)
        for job in ([], [verify_job(100, conclusion=None, status="in_progress")]):
            with self.subTest(job=job):
                evidence, error, _ = self.execute(history={1: [prior]},
                    contents={prior["head_sha"]: REQUEST}, jobs={(100, 1): job})
                self.assertIsInstance(error, ValueError)
                self.assertEqual(evidence["status"], "blocked_uncertain")

    def test_repeated_history_page_fails_instead_of_silently_capping_search(self):
        prior = run_record(100)
        evidence, error, _ = self.execute(history={1: [prior], 2: [prior]},
            contents={prior["head_sha"]: b'{"different":"request-b"}'})
        self.assertIsInstance(error, ValueError)
        self.assertEqual(evidence["status"], "blocked_uncertain")


class HDiscoveryAdmissionTests(CohortAdmissionTests):
    request_path = "configs/mastery-H-run.request.json"
    evidence_path = "runs/producer-context-H-input/provenance/admission.json"
    concurrency_group = "l1-producer-context-H"

    def setUp(self):
        discovery = WORKFLOW.with_name("l1-producer-context-H.yml")
        workflow_patch = patch.dict(globals(), {"WORKFLOW": discovery,
            "WORKFLOW_PATH": ".github/workflows/l1-producer-context-H.yml"})
        workflow_patch.start()
        self.addCleanup(workflow_patch.stop)


class HConfirmationWorkflowTests(unittest.TestCase):
    def test_all_embedded_verifier_python_compiles(self):
        source = WORKFLOW.read_text()
        blocks = re.findall(r"          python - <<'PY'\n(.*?)          PY", source, re.S)
        self.assertGreaterEqual(len(blocks), 6)
        for number, block in enumerate(blocks):
            compile(textwrap.dedent(block), f"{WORKFLOW}:block{number}", "exec")

    def test_h_confirmation_routes_with_only_reviewed_aggregation_compatibility(self):
        source = WORKFLOW.read_text()
        self.assertIn("--source-plan configs/mastery-H-plan.json", source)
        for name in ("mastery-H-plan.json", "mastery-H-cohort-plan.json"):
            self.assertIn(hashlib.sha256((CONFIGS / name).read_bytes()).hexdigest(), source)
        self.assertIn("--plan configs/mastery-H-cohort-plan.json", source)
        self.assertIn("--candidate-request configs/mastery-H-cohort-run.request.json", source)
        self.assertIn("python -m mutual_lab.cohort_context_validation", source)
        self.assertIn("[[99007,99009],[99017,99019],[99027,99029]]", source)
        self.assertNotIn("830c3a81", source)
        self.assertNotIn("mastery-cohort-run.request.json", source)
        self.assertNotIn("mutual_lab.cohort_validation", source)
        self.assertIn("recorded_hashes = context.get('source_sha256')", source)
        self.assertIn("changed != {name}", source)
        self.assertIn("7ede9cd0a40a585d11fc94d08738d8a74abc2607b252c3ccc6e4b4614ef6eba4", source)
        self.assertIn("nonaggregation(old_tree) != nonaggregation(new_tree)", source)
        self.assertIn("source_run.get('path') != '.github/workflows/l1-producer-context-H.yml'", source)
        self.assertIn("name: l1-H-cohort-trial-${{ matrix.language }}-r${{ matrix.replica }}-${{ github.run_id }}", source)
        self.assertIn("'kind': 'cohort-context-validation-v2-comparison'", source)


if __name__ == "__main__":
    unittest.main()
