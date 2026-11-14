"""Opaque state preservation and provenance checks; no tensor runtime required."""

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_l1_states.py"
spec = importlib.util.spec_from_file_location("package_l1_states", SCRIPT)
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def trial(root, name="trial", *, steps=(0, 4), status="completed", prior=None, exact_curve=True):
    path = root / name
    path.mkdir(parents=True)
    checkpoints = []
    for step in steps:
        data = b"opaque-not-pickle-state\x00optimizer-and-rng\xff" + str(step).encode()
        state = path / f"step-{step:06d}.pt"
        state.write_bytes(data)
        manifest = {"format": packaging.CHECKPOINT_FORMAT, "kind": "development_only",
                    "excluded_from_research": True, "eligible_for_fresh_experiment": False,
                    "step": step, "path": state.name, "sha256": sha(data)}
        write_json(state.with_suffix(".json"), manifest)
        checkpoints.append(manifest)
    latest = max(steps) if steps else 0
    report = {"kind": packaging.TRIAL_KIND, "excluded_from_research": True,
              "eligible_for_fresh_experiment": False, "protected_splits_used": [], "phase": "A",
              "condition": "A0", "language": "ko", "replica": 0, "seed_pair": [17, 19],
              "status": status, "training_steps_completed": latest, "policy": "same_pair",
              "model_factory": "default", "model_profile": {"hidden_dim": 16},
              "revision": {"commit": "a" * 40, "working_tree_dirty": False},
              "prior_metadata": prior or {"kind": "fixed_pixel_random_projection"},
              "checkpoints": checkpoints, "curves": [{"step": 0, "candidate": {"status": "best_at_zero"}}]}
    if steps and latest and exact_curve:
        report["curves"].append({"step": latest, "candidate": {"status": "targets_not_met"}})
    write_json(path / "report.json", report)
    for filename in packaging.CONTEXT_FILES:
        write_json(path / filename, {"fixture": filename})
    write_json(path / "generations.json", [{"step": 0, "text": "earlier"}, {"step": latest, "text": "latest"}])
    return path, report


def prior(root, name="prior", data=b"opaque-prior", pool="fit-pool"):
    path = root / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    report = {"format": packaging.PRIOR_FORMAT, "kind": "image_only_prior_development",
              "excluded_from_research": True, "eligible_for_fresh_experiment": False,
              "fit_pool_sha256": pool, "fit_png_sha256": ["png1", "png2"]}
    write_json(path.with_suffix(".json"), {"format": packaging.PRIOR_FORMAT, "kind": "development_only",
                                          "excluded_from_research": True, "sha256": sha(data), "report": report})
    return path


def read_index(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("index.json"))


class L1StatePackagingTests(unittest.TestCase):
    def test_concurrent_publication_preserves_the_winning_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial(root / "A")
            output = root / "all.zip"
            winner = b"another verified archive won publication"
            real_link = os.link

            def publish_after_competitor(source, destination):
                Path(destination).write_bytes(winner)
                return real_link(source, destination)

            with patch("os.link", side_effect=publish_after_competitor):
                with self.assertRaises(FileExistsError):
                    packaging.package_states({"A": root / "A"}, {}, output)
            self.assertEqual(output.read_bytes(), winner)
            self.assertFalse(list(root.glob("all.zip.*.tmp")))

    def test_latest_opaque_bytes_all_statuses_and_exact_step_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one, raw = trial(root / "A")
            trial(root / "A", "failed", steps=(0, 6), status="failed", exact_curve=False)
            trial(root / "A", "abrupt", steps=(), status="running")
            write_json(root / "A" / "ignored" / "report.json", {"kind": "context-report"})
            (root / "A" / "source.py").write_text("must not be bundled")
            output = root / "all.zip"
            result = packaging.package_states({"A": root / "A"}, {}, output, loader_revision="abc123")
            self.assertEqual(result["complete"]["trial_count"], 3)
            self.assertEqual(result["complete"]["checkpoint_count"], 2)
            index = read_index(output)
            completed = next(t for t in index["trials"] if t["status"] == "completed")
            failed = next(t for t in index["trials"] if t["status"] == "failed")
            abrupt = next(t for t in index["trials"] if t["status"] == "running")
            self.assertEqual(completed["checkpoint"]["step"], 4)
            self.assertEqual(completed["final_candidate_at_latest_step"]["status"], "targets_not_met")
            self.assertEqual(failed["checkpoint"]["step"], 6)
            self.assertIsNone(failed["final_candidate_at_latest_step"])
            self.assertIsNone(abrupt["checkpoint"])
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.read(completed["checkpoint"]["member"]), (one / "step-000004.pt").read_bytes())
                self.assertEqual(archive.read(completed["report_member"]), (one / "report.json").read_bytes())
                self.assertFalse(any(name.endswith(".py") or name.endswith("step-000000.pt") for name in archive.namelist()))
                for name, item in index["members"].items():
                    data = archive.read(name)
                    self.assertEqual((sha(data), len(data)), (item["sha256"], item["bytes"]))
                rows = json.loads(archive.read(completed["context"]["generations.json"]["member"]))
                self.assertEqual(rows, [{"step": 4, "text": "latest"}])
                self.assertIn(b"git checkout abc123", archive.read("README.txt"))

    def test_invalid_comparison_and_selected_stage_do_not_filter_failed_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial(root / "old", status="completed")
            trial(root / "selected", "good")
            trial(root / "selected", "failed", status="cancelled")
            write_json(root / "old" / "negative-policy-study.json", {"status": "completed"})
            packaging.package_states({"E_original": root / "old", "cohort": root / "selected"}, {}, root / "all.zip",
                                     invalid_stages=["E_original"], selected_stage="cohort", selected_output=root / "light.zip")
            full, light = read_index(root / "all.zip"), read_index(root / "light.zip")
            old = next(t for t in full["trials"] if t["stage"] == "E_original")
            self.assertFalse(old["control_valid"])
            self.assertEqual(old["status"], "completed")
            self.assertEqual(len(light["trials"]), 2)
            self.assertEqual({t["status"] for t in light["trials"]}, {"completed", "cancelled"})
            self.assertEqual(light["archive_scope"], "selected_stage")
            self.assertEqual(light["priors"], {})

    def test_corrupt_latest_state_never_falls_back_to_earlier_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, report = trial(root / "A")
            (path / "step-000004.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                packaging.package_states({"A": root / "A"}, {}, root / "out.zip")
            self.assertFalse((root / "out.zip").exists())

    def test_manifest_flags_and_report_sidecar_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, report = trial(root / "A")
            sidecar = path / "step-000004.json"
            manifest = json.loads(sidecar.read_text())
            manifest["eligible_for_fresh_experiment"] = True
            write_json(sidecar, manifest)
            with self.assertRaisesRegex(ValueError, "development-only"):
                packaging.package_states({"A": root / "A"}, {}, root / "flags.zip")
            manifest["eligible_for_fresh_experiment"] = False
            manifest["step"] = 3
            write_json(sidecar, manifest)
            with self.assertRaisesRegex(ValueError, "disagree"):
                packaging.package_states({"A": root / "A"}, {}, root / "mismatch.zip")

    def test_priors_deduplicate_bytes_and_match_recorded_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = prior(root, "first")
            alias = prior(root, "alias")
            other = prior(root, "other", b"different-prior")
            metadata = {"fit_pool_sha256": "fit-pool", "fit_png_sha256": ["png2", "png1"],
                        "source_bundle_sha256": sha(first.read_bytes())}
            trial(root / "F", prior=metadata)
            packaging.package_states({"F": root / "F"}, {"first": first, "alias": alias, "other": other}, root / "all.zip",
                                     selected_stage="F", selected_output=root / "light.zip")
            full, light = read_index(root / "all.zip"), read_index(root / "light.zip")
            self.assertEqual(len(full["priors"]), 2)
            self.assertEqual(len(light["priors"]), 1)
            matched = light["trials"][0]["prior_sha256"]
            self.assertEqual(len(light["priors"][matched]["aliases"]), 2)
            with zipfile.ZipFile(root / "light.zip") as archive:
                self.assertEqual(archive.read(light["priors"][matched]["member"]), first.read_bytes())

    def test_ambiguous_missing_and_corrupt_priors_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = prior(root, "first")
            second = prior(root, "second", b"second")
            trial(root / "C", prior={"fit_pool_sha256": "fit-pool"})
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                packaging.package_states({"C": root / "C"}, {"a": first, "b": second}, root / "ambiguous.zip")
            with self.assertRaisesRegex(ValueError, "Missing"):
                packaging.package_states({"C": root / "C"}, {}, root / "missing.zip")
            first.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                packaging.package_states({"C": root / "C"}, {"a": first}, root / "corrupt.zip")

    def test_duplicate_stage_paths_names_and_checkpoint_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, report = trial(root / "A")
            for stages in ([('A', root / 'A'), ('A', root / 'A')],
                           [('A', root / 'A'), ('B', root / 'A')], [('../A', root / 'A')]):
                with self.subTest(stages=stages), self.assertRaises(ValueError):
                    packaging.package_states(stages, {}, root / "out.zip")
            report["checkpoints"][-1]["path"] = "../escaped.pt"
            write_json(path / "report.json", report)
            with self.assertRaisesRegex(ValueError, "basename"):
                packaging.package_states({"A": root / "A"}, {}, root / "out.zip")

    def test_identical_reports_in_distinct_trials_are_both_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial(root / "A", "one")
            trial(root / "A", "two")
            packaging.package_states({"A": root / "A"}, {}, root / "out.zip")
            index = read_index(root / "out.zip")
            self.assertEqual(len(index["trials"]), 2)
            self.assertEqual(len({t["checkpoint"]["member"] for t in index["trials"]}), 2)

    def test_grounding_and_cohort_suite_provenance_preserved_in_both_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial(root / "cohort")
            originals = {}
            for name in ("grounding-study.json", "cohort-validation.json"):
                path = root / "cohort" / "wrapper" / name
                path.parent.mkdir(exist_ok=True)
                data = b'{ "request": "retain whitespace and original bytes" }\n\n'
                path.write_bytes(data)
                originals[str(path)] = data
            packaging.package_states({"cohort": root / "cohort"}, {}, root / "all.zip",
                                     selected_stage="cohort", selected_output=root / "light.zip")
            for output in (root / "all.zip", root / "light.zip"):
                with self.subTest(output=output), zipfile.ZipFile(output) as archive:
                    index = json.loads(archive.read("index.json"))
                    found = {item["source_path"]: archive.read(name) for name, item in index["members"].items()
                             if item["purpose"] == "original_runtime_suite_provenance"}
                    self.assertEqual(found, originals)

    def test_runtime_provenance_json_txt_retained_without_code_or_png_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial(root / "G")
            trial(root / "other")
            expected = {}
            for name in ("request.json", "run-context.json", "plan.json", "data-manifest.json",
                         "nested/selection.json", "dependencies.txt"):
                path = root / "G" / "download" / "provenance-ko" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = ("original context for " + name + "\n").encode()
                path.write_bytes(data)
                expected[str(path)] = data
            for name in ("module.py", "image.png"):
                (root / "G" / "download" / "provenance-ko" / name).write_bytes(b"not included")
            write_json(root / "other" / "provenance" / "request.json", {"stage": "other"})
            packaging.package_states({"G": root / "G", "other": root / "other"}, {}, root / "all.zip",
                                     selected_stage="G", selected_output=root / "light.zip")
            for output in (root / "all.zip", root / "light.zip"):
                with self.subTest(output=output), zipfile.ZipFile(output) as archive:
                    index = json.loads(archive.read("index.json"))
                    found = {item["source_path"]: archive.read(name) for name, item in index["members"].items()
                             if item["purpose"] == "original_runtime_context_provenance"
                             and item["source_stage"] == "G"}
                    self.assertEqual(found, expected)
                    self.assertFalse(any(name.endswith((".py", ".png")) for name in archive.namelist()))
                    if output.name == "light.zip":
                        self.assertFalse(any(name.startswith("provenance/other/") for name in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
