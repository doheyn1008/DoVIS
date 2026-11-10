import json
import os
import random
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from mutual_lab.agents import EndpointAgent, SharedImage
from mutual_lab.backend import BackendError, ChatTransport, parse_choice
from mutual_lab.config import AgentConfig, ExperimentConfig
from mutual_lab.images import scene_png
from mutual_lab.runner import run_experiment
from mutual_lab.world import make_dataset


class FixtureServer:
    """An HTTP protocol fixture, deliberately not a language model."""
    def __init__(self):
        self.requests = []
        self.headers = []
        self.status = 200
        self.finish_reason = "stop"
        self.malformed = False
        self.answer = "1"
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.append(body)
                fixture.headers.append(dict(self.headers))
                content = body["messages"][-1]["content"][0]["text"]
                selecting = "one digit" in content or "숫자" in content or "数字" in content
                message = fixture.answer if selecting else "A red circle is to the left of a blue square."
                data = {"model": "fixture-model", "system_fingerprint": "fixture-v1", "choices": [
                    {"message": {"role": "assistant", "content": message}, "finish_reason": fixture.finish_reason}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 2}}
                raw = b"invalid-json" if fixture.malformed else json.dumps(data).encode()
                self.send_response(fixture.status)
                if fixture.status == 302:
                    self.send_header("Location", "/unexpected-target")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = FixtureServer()
        self.addCleanup(self.fixture.close)
        self.agent = AgentConfig("A", "en", "endpoint", "fixture-model", self.fixture.base_url)
        self.config = replace(ExperimentConfig(), agents=(self.agent, replace(self.agent, name="B", language="ko")),
                              max_exchanges=2, eval_every=1, eval_items=4, stop_on_threshold=False)
        self.records = []
        self.transport = ChatTransport(self.config, self.records.append)
        self.scene = make_dataset(self.config).train[0]

    def test_pixels_only_stateless_interface_and_private_memory(self):
        agent = EndpointAgent(self.agent, self.config, self.transport)
        image = SharedImage(scene_png(self.scene))
        agent.observe(image, "partner observation")
        before = agent.state_digest()
        selected, _ = agent.choose("partner test utterance", image, random.Random(1))
        self.assertEqual(selected, 1)
        self.assertEqual(before, agent.state_digest())
        body = self.fixture.requests[-1]
        self.assertNotIn("tools", body)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(len(body["messages"]), 3)
        encoded = json.dumps(body)
        self.assertNotIn("correct_choice", encoded)
        self.assertNotIn("subject_color", encoded)
        self.assertNotIn(".png", encoded)
        self.assertIn("data:image/png;base64,", encoded)
        agent.speak(image, frozen=True)
        self.assertEqual(len(self.fixture.requests[-1]["messages"]), 2)
        self.assertNotIn("partner observation", json.dumps(self.fixture.requests[-1]))

    def test_key_is_used_only_in_header_and_never_saved(self):
        agent = replace(self.agent, api_key_env="MUTUAL_LAB_TEST_KEY")
        config = replace(self.config, agents=(agent, replace(agent, name="B", language="ko")))
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MUTUAL_LAB_TEST_KEY": "secret-test-marker"}):
            path = run_experiment(config, directory)
            serialized = b"".join(file.read_bytes() for file in path.iterdir() if file.is_file())
            self.assertNotIn(b"secret-test-marker", serialized)
            self.assertTrue(all(header.get("Authorization") == "Bearer secret-test-marker" for header in self.fixture.headers))
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual(summary["transport"]["requests"], config.request_upper_bound())
            self.assertEqual(summary["transport"]["reported_prompt_tokens"], len(self.fixture.requests) * 8)
            self.assertTrue(all(d["baseline_accuracy"] == .25 for d in summary["directions"]))
            records = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
            purposes = [e["purpose"] for e in records if e["type"] == "model_request"]
            self.assertEqual(purposes[:8], ["frozen_caption"] * 8)
            self.assertEqual(purposes[8:16], ["evaluation"] * 8)

    def test_missing_key_sends_no_request(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(BackendError):
            self.transport.complete(replace(self.agent, api_key_env="MISSING_KEY"), "dialogue", [])
        self.assertFalse(self.fixture.requests)
        self.assertEqual(self.transport.budget.used, 0)

    def test_http_errors_are_not_retried(self):
        self.fixture.status = 503
        with self.assertRaisesRegex(BackendError, "HTTP 503"):
            EndpointAgent(self.agent, self.config, self.transport).speak(SharedImage(scene_png(self.scene)))
        self.assertEqual(len(self.fixture.requests), 1)
        self.assertEqual(self.records[0]["status"], "http_503")

    def test_redirects_are_not_followed(self):
        self.fixture.status = 302
        with self.assertRaisesRegex(BackendError, "HTTP 302"):
            EndpointAgent(self.agent, self.config, self.transport).speak(SharedImage(scene_png(self.scene)))
        self.assertEqual(len(self.fixture.requests), 1)

    def test_truncated_and_malformed_generations_fail(self):
        agent = EndpointAgent(self.agent, self.config, self.transport)
        image = SharedImage(scene_png(self.scene))
        self.fixture.finish_reason = "length"
        with self.assertRaisesRegex(BackendError, "max_tokens"):
            agent.speak(image)
        self.fixture.finish_reason = "stop"
        self.fixture.malformed = True
        with self.assertRaisesRegex(BackendError, "invalid Chat Completions"):
            agent.speak(image)

    def test_request_budget_stops_before_extra_http_call(self):
        transport = ChatTransport(replace(self.config, max_requests=1), self.records.append)
        agent = EndpointAgent(self.agent, self.config, transport)
        image = SharedImage(scene_png(self.scene))
        agent.speak(image)
        with self.assertRaisesRegex(BackendError, "budget exhausted"):
            agent.speak(image)
        self.assertEqual(len(self.fixture.requests), 1)

    def test_invalid_answer_is_counted_wrong_without_guessing(self):
        self.fixture.answer = "I think panel 1 is correct."
        with tempfile.TemporaryDirectory() as directory:
            path = run_experiment(self.config, directory)
            points = json.loads((path / "evaluations.json").read_text())
            self.assertTrue(all(p["accuracy"] == 0 and p["invalid_answers"] == 4 for p in points))
            self.assertEqual(len(self.fixture.requests), self.config.request_upper_bound())

    def test_underbudget_run_makes_no_http_calls_or_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Worst-case"):
                run_experiment(replace(self.config, max_requests=1), directory)
            self.assertFalse(list(Path(directory).iterdir()))
        self.assertFalse(self.fixture.requests)


class AnswerParsingTests(unittest.TestCase):
    def test_strict_answer_contract(self):
        self.assertEqual(parse_choice("1"), 1)
        self.assertEqual(parse_choice('{"choice": 4}'), 4)
        for invalid in ("0", "5", "answer: 2", '{"choice": true}', '{"choice": 2, "explanation": "x"}', ""):
            with self.subTest(answer=invalid):
                self.assertIsNone(parse_choice(invalid))


if __name__ == "__main__":
    unittest.main()
