import io
import json
import tempfile
import threading
import unittest
import zipfile
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mutual_lab.config import ExperimentConfig
from mutual_lab.runner import run_experiment
from mutual_lab.server import make_server


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.run_path = run_experiment(replace(ExperimentConfig(), max_exchanges=5, eval_items=4), cls.temp.name)
        cls.server, cls.app = make_server(cls.temp.name, port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.app.pool.shutdown(wait=True)
        cls.temp.cleanup()

    def get(self, path):
        with urlopen(self.base + path) as response:
            return response.status, response.headers, response.read()

    def test_static_assets_and_result_api(self):
        for path, mime in (("/", "text/html"), ("/style.css", "text/css"), ("/app.js", "text/javascript")):
            status, headers, body = self.get(path)
            self.assertEqual(status, 200)
            self.assertIn(mime, headers["Content-Type"])
            self.assertTrue(body)
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        result = json.loads(self.get("/api/runs")[2])
        self.assertTrue(any(run["run_id"] == self.run_path.name for run in result["runs"]))
        detail = json.loads(self.get("/api/run/" + self.run_path.name)[2])
        self.assertEqual(detail["summary"]["status"], "completed")
        self.assertEqual(len(detail["dialogue"]), 5)
        self.assertEqual(len(detail["points"]), 4)

    def test_export_contains_reusable_original_artifacts(self):
        status, headers, body = self.get(f"/api/run/{self.run_path.name}/export")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertIn("metrics.csv", archive.namelist())
            self.assertIn("frozen_captions.json", archive.namelist())
            self.assertTrue(any(name.startswith("assets/") for name in archive.namelist()))
            self.assertIsNone(archive.testzip())

    def test_unknown_routes_and_traversal_are_not_served(self):
        for path in ("/config.json", "/api/run/..", f"/assets/{self.run_path.name}/%2e%2e/summary.json",
                     f"/api/run/{self.run_path.name}/unexpected"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as caught:
                self.get(path)
            self.assertEqual(caught.exception.code, 404)

    def test_bad_host_and_cross_origin_launch_are_rejected(self):
        request = Request(self.base + "/api/runs", headers={"Host": "unexpected.example"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code, 403)
        request = Request(self.base + "/api/demo", b"{}", {
            "Content-Type": "application/json", "X-Mutual-Lab": "1", "Origin": "https://unexpected.example"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code, 403)

    def test_ui_can_only_launch_synthetic_configuration(self):
        request = Request(self.base + "/api/demo", json.dumps({"backend": "endpoint"}).encode(), {
            "Content-Type": "application/json", "X-Mutual-Lab": "1"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code, 400)
        request = Request(self.base + "/api/demo", json.dumps({"seed": 23, "max_exchanges": 5,
                          "lang_a": "ko", "lang_b": "ja"}).encode(), {
            "Content-Type": "application/json", "X-Mutual-Lab": "1"})
        with urlopen(request) as response:
            result = json.loads(response.read())
            self.assertEqual(response.status, 202)
            self.assertEqual(result["mode"], "demo")


if __name__ == "__main__":
    unittest.main()
