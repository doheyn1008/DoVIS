"""Loopback-only dashboard. No cloud, model, or API-key actions are exposed by its UI."""

from __future__ import annotations

import io
import json
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .config import ExperimentConfig, integer
from .runner import ExperimentFailed, run_experiment


def load_json(path: Path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


class Dashboard:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mutual-lab-demo")
        self.lock = threading.Lock()
        self.running = False
        self.job_error = None

    def run_path(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid run ID")
        path = self.root / run_id
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(run_id)
        return path

    def launch(self, data: dict) -> None:
        allowed = {"seed", "max_exchanges", "lang_a", "lang_b"}
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError("Unsupported demo option")
        maximum = data.get("max_exchanges", 80)
        integer(maximum, "max_exchanges", 5, 1000)
        config = ExperimentConfig.from_dict({"seed": data.get("seed", 7), "max_exchanges": maximum,
                                            "eval_every": min(5, maximum), "agents": [
            {"name": "A", "language": data.get("lang_a", "en"), "backend": "symbolic"},
            {"name": "B", "language": data.get("lang_b", "ko"), "backend": "symbolic"},
        ]})
        with self.lock:
            if self.running:
                raise RuntimeError("A demo is already running")
            self.running = True
            self.job_error = None

        def task():
            try:
                run_experiment(config, self.root)
            except (ExperimentFailed, ValueError, OSError) as exc:
                self.job_error = str(exc)
            finally:
                with self.lock:
                    self.running = False

        self.pool.submit(task)


def make_server(root: str | Path, port: int = 8765) -> tuple[ThreadingHTTPServer, Dashboard]:
    integer(port, "port", 0, 65535)
    app = Dashboard(root)
    static = files("mutual_lab").joinpath("static")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status: int, body: bytes, content_type="application/json; charset=utf-8", **headers):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def json(self, status: int, data: object):
            self.respond(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))

        def trusted_host(self) -> bool:
            try:
                host = urlsplit("http://" + self.headers.get("Host", "")).hostname
                return host in {"localhost", "127.0.0.1", "::1"}
            except ValueError:
                return False

        def do_GET(self):
            if not self.trusted_host():
                self.json(403, {"error": "Loopback host required"})
                return
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/":
                    self.respond(200, static.joinpath("index.html").read_bytes(), "text/html; charset=utf-8")
                elif path in {"/app.js", "/style.css"}:
                    content_type = "text/javascript; charset=utf-8" if path.endswith("js") else "text/css; charset=utf-8"
                    self.respond(200, static.joinpath(path[1:]).read_bytes(), content_type)
                elif path == "/api/runs":
                    summaries = []
                    for directory in sorted(app.root.iterdir(), reverse=True):
                        if directory.is_dir() and not directory.is_symlink():
                            summary = load_json(directory / "summary.json")
                            if summary:
                                summaries.append(summary)
                    summaries.sort(key=lambda item: item.get("started_at", ""), reverse=True)
                    self.json(200, {"runs": summaries, "demo_running": app.running, "job_error": app.job_error})
                elif path.startswith("/api/run/"):
                    parts = path.split("/")
                    if len(parts) not in {4, 5}:
                        raise FileNotFoundError(path)
                    run = app.run_path(parts[3])
                    if len(parts) == 4:
                        events = []
                        event_path = run / "events.jsonl"
                        if event_path.exists():
                            with event_path.open(encoding="utf-8") as stream:
                                for line in stream:
                                    try:
                                        event = json.loads(line)
                                        if event.get("type") == "dialogue":
                                            events.append(event)
                                    except ValueError:
                                        continue  # A last line may be in the middle of an append.
                        self.json(200, {"summary": load_json(run / "summary.json"),
                                        "points": load_json(run / "evaluations.json", []), "dialogue": events})
                    elif parts[4] == "export":
                        buffer = io.BytesIO()
                        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                            for file in sorted(run.rglob("*")):
                                if file.is_file() and not file.is_symlink() and file.suffix in {".json", ".jsonl", ".csv", ".png"}:
                                    archive.write(file, arcname=str(file.relative_to(run)))
                        self.respond(200, buffer.getvalue(), "application/zip",
                                     Content_Disposition=f'attachment; filename="{run.name}.zip"')
                    else:
                        raise FileNotFoundError(path)
                elif path.startswith("/assets/"):
                    parts = path.split("/")
                    if len(parts) != 4 or not re.fullmatch(r"[A-Za-z0-9_-]+\.png", parts[3]):
                        raise FileNotFoundError(path)
                    asset = app.run_path(parts[2]) / "assets" / parts[3]
                    if asset.is_symlink() or not asset.resolve().is_relative_to(app.run_path(parts[2])):
                        raise FileNotFoundError(path)
                    self.respond(200, asset.read_bytes(), "image/png")
                else:
                    raise FileNotFoundError(path)
            except (FileNotFoundError, ValueError):
                self.json(404, {"error": "Not found"})
            except OSError:
                self.json(500, {"error": "Could not read run artifacts"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed_origin = "http://" + self.headers.get("Host", "")
            if (not self.trusted_host() or (origin is not None and origin != allowed_origin)
                    or self.headers.get("X-Mutual-Lab") != "1"
                    or self.headers.get("Content-Type", "").split(";")[0] != "application/json"):
                self.json(403, {"error": "Same-origin JSON request required"})
                return
            if urlsplit(self.path).path != "/api/demo":
                self.json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                app.launch(json.loads(self.rfile.read(length)))
                self.json(202, {"status": "started", "mode": "demo"})
            except (ValueError, TypeError) as exc:
                self.json(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.json(409, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server, app


def serve(root: str | Path, port: int = 8765) -> None:
    server, app = make_server(root, port)
    print(f"Dashboard: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        app.pool.shutdown(wait=False, cancel_futures=True)
