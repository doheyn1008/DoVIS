"""Optional real-browser smoke check; not imported by standard unittest discovery."""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import zipfile
from dataclasses import replace
from pathlib import Path

from mutual_lab.config import ExperimentConfig
from mutual_lab.runner import run_experiment
from mutual_lab.server import make_server


def main() -> None:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="browser-artifacts")
    args = parser.parse_args()
    artifacts = Path(args.artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        for seed in (7, 11):
            run_experiment(replace(ExperimentConfig(), seed=seed), directory)
        server, app = make_server(directory, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        errors: list[str] = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 1120})
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
                    page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="networkidle")
                    page.wait_for_function("document.querySelector('#exchanges').textContent !== '—'")
                    assert page.locator("#shared-image").is_visible()
                    page.locator("#intervals").check()
                    other = page.locator("#compare option").nth(1).get_attribute("value")
                    assert other
                    page.locator("#compare").select_option(other)
                    page.wait_for_function("document.querySelector('#compare-label').textContent.length > 0")
                    page.locator("#next").click()
                    assert page.locator("#replay-count").inner_text().startswith("2 /")
                    page.locator("#play").click()
                    page.wait_for_function("Number(document.querySelector('#scrub').value) >= 2")
                    page.locator("#play").click()
                    with page.expect_download() as download:
                        page.locator("#export").click()
                    export_path = artifacts / "result.zip"
                    download.value.save_as(export_path)
                    with zipfile.ZipFile(export_path) as archive:
                        assert "metrics.csv" in archive.namelist()
                        assert archive.testzip() is None
                    page.screenshot(path=str(artifacts / "desktop.png"), full_page=True)
                    page.locator("#seed").fill("29")
                    page.locator("#maximum").fill("10")
                    page.locator("#lang-a").select_option("ko")
                    page.locator("#lang-b").select_option("ja")
                    page.locator("#launch").click()
                    page.wait_for_function("document.querySelector('#run-list .active small')?.textContent.includes('Seed 29')")
                    page.wait_for_function("document.querySelector('#status').textContent === 'COMPLETED'")
                    assert page.locator("#exchanges").inner_text() == "10회"
                    page.set_viewport_size({"width": 390, "height": 844})
                    page.locator("#shared-image").scroll_into_view_if_needed()
                    assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth"), "Mobile overflow"
                    page.screenshot(path=str(artifacts / "mobile.png"), full_page=True)
                    assert not errors, errors
                    print(json.dumps({"desktop": "1440x1120", "mobile": "390x844", "errors": errors,
                                      "launch": "ko/ja seed 29 completed", "export": "valid ZIP"}))
                finally:
                    browser.close()
        finally:
            server.shutdown()
            server.server_close()
            app.pool.shutdown(wait=True)
            thread.join()


if __name__ == "__main__":
    main()
