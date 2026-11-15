"""Full-history byte preservation, publication races and hostile source paths."""

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_l1_history.py"
SPEC = importlib.util.spec_from_file_location("archive_l1_history", SCRIPT)
history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history)
REVISION = "a" * 40


class FullHistoryArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stage = self.root / "raw"; self.stage.mkdir()
        self.output = self.root / "history.zip"

    def write(self, name, data):
        path = self.stage / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_every_budget_prior_predictions_failure_context_and_image_is_byte_exact(self):
        originals = {
            "trial/step-000000.pt": b"opaque initial tensor bytes\x00\xff",
            "trial/step-002400.pt": b"opaque middle optimizer RNG bytes\x00",
            "trial/step-009600.pt": b"opaque final tensor bytes\xfe",
            "trial/step-000000.json": b'{"step":0,"excluded_from_research":true}\n',
            "trial/step-002400.json": b'{"step":2400}\n',
            "trial/step-009600.json": b'{"step":9600}\n',
            "trial/report.json": b'{"status":"failed","unmodified_historical_claim":true}\n',
            "trial/generations.json": b'[{"step":0},{"step":2400},{"step":9600}]\n',
            "trial/comprehension_predictions.json": b'[{"all":"raw rows, not a summary"}]\n',
            "inputs/prior.pt": b"original common prior, never unpickled",
            "inputs/provenance/request.json": b'{"source_run":123}\n',
            "failed/provenance/run-context.json": b'{"failure":"retained"}\n',
            "failure.txt": b"exact traceback\r\n",
            "figures/curve.svg": b"<svg>unchanged</svg>\n",
            "images/scene.PNG": b"\x89PNG\r\n\x1a\nopaque bytes",
        }
        for name, data in originals.items(): self.write(name, data)
        self.write("source.py", b"raise RuntimeError('must not be archived')")
        self.write(".git/nested/config.json", b"git internals excluded")
        self.write(".github/workflows/context.json", b"git code excluded")
        self.write("notes.md", b"unsupported source/document extension")
        other = self.root / "other"; other.mkdir(); (other / "report.json").write_bytes(b"different stage")
        receipt = history.archive_history({"A_B_C": self.stage, "D_E": other}, self.output, REVISION)
        index = history.verify_archive(self.output)
        self.assertEqual(receipt["files"], len(originals) + 1)
        self.assertEqual(receipt["checkpoints"], 4)
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(receipt["compressed_bytes"], self.output.stat().st_size)
        with zipfile.ZipFile(self.output) as archive:
            self.assertIsNone(archive.testzip())
            for name, data in originals.items():
                self.assertEqual(archive.read("stages/A_B_C/" + name), data)
            self.assertEqual(archive.read("stages/D_E/report.json"), b"different stage")
            self.assertFalse(any(name.endswith(".py") or "/.git/" in name or "/.github/" in name for name in archive.namelist()))
            self.assertTrue(all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist()))
            self.assertIn(REVISION.encode(), archive.read("RESTORE.txt"))
            self.assertIn(b"not a model-training revision", archive.read("RESTORE.txt"))
        self.assertEqual(index["loader_revision"], REVISION)
        self.assertFalse(index["eligible_for_fresh_experiment"])
        self.assertEqual({row["original_relative_path"] for row in index["files"] if row["stage"] == "A_B_C"}, set(originals))
        self.assertFalse(list(self.root.glob(".history.zip.*.tmp")))

    def test_existing_output_and_racing_winner_are_never_overwritten(self):
        self.write("report.json", b"{}")
        self.output.write_bytes(b"winner")
        with self.assertRaises(FileExistsError):
            history.archive_history({"A": self.stage}, self.output, REVISION)
        self.assertEqual(self.output.read_bytes(), b"winner")
        self.output.unlink()
        real_link = os.link

        def race(source, destination, **kwargs):
            self.output.write_bytes(b"racing winner")
            return real_link(source, destination, **kwargs)

        with patch.object(history.os, "link", side_effect=race):
            with self.assertRaises(FileExistsError):
                history.archive_history({"A": self.stage}, self.output, REVISION)
        self.assertEqual(self.output.read_bytes(), b"racing winner")
        self.assertFalse(list(self.root.glob(".history.zip.*.tmp")))

    def test_changed_or_new_source_artifacts_abort_without_publication(self):
        source = self.write("step-000001.pt", b"original")
        original = history._copy_member

        def mutate(archive, root, entry):
            source.write_bytes(b"replaced")
            return original(archive, root, entry)

        with patch.object(history, "_copy_member", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                history.archive_history({"A": self.stage}, self.output, REVISION)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".history.zip.*.tmp")))

        def add(archive, root, entry):
            result = original(archive, root, entry)
            self.write("late-step-009600.pt", b"new artifact")
            return result

        with patch.object(history, "_copy_member", side_effect=add):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                history.archive_history({"A": self.stage}, self.output, REVISION)
        self.assertFalse(self.output.exists())

    def test_zip_corruption_is_detected_before_publication_and_on_later_verification(self):
        self.write("report.json", b"complete raw predictions" * 100)
        original = history._verify_zip

        def corrupt(stream, expected_index=None):
            stream.seek(0); stream.write(b"BAD!"); stream.flush()
            return original(stream, expected_index)

        with patch.object(history, "_verify_zip", side_effect=corrupt):
            with self.assertRaises(zipfile.BadZipFile):
                history.archive_history({"A": self.stage}, self.output, REVISION)
        self.assertFalse(self.output.exists())
        history.archive_history({"A": self.stage}, self.output, REVISION)
        with self.output.open("r+b") as stream:
            stream.write(b"BAD!")
        with self.assertRaises(zipfile.BadZipFile): history.verify_archive(self.output)

    def test_symlinked_roots_members_ancestors_outputs_and_unsafe_labels_are_rejected(self):
        self.write("report.json", b"{}")
        root_link = self.root / "root-link"; root_link.symlink_to(self.stage, target_is_directory=True)
        for source in (root_link, root_link / "child"):
            with self.assertRaises((OSError, ValueError)):
                history.archive_history({"A": source}, self.output, REVISION)
        (self.stage / "linked.json").symlink_to(self.stage / "report.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            history.archive_history({"A": self.stage}, self.output, REVISION)
        (self.stage / "linked.json").unlink()
        self.output.symlink_to(self.root / "nonexistent")
        with self.assertRaises(FileExistsError): history.archive_history({"A": self.stage}, self.output, REVISION)
        self.assertTrue(self.output.is_symlink()); self.output.unlink()
        for stages in ({"../A": self.stage}, {"A/B": self.stage}, {"A": self.stage, "a": self.stage}):
            with self.assertRaises(ValueError): history.archive_history(stages, self.output, REVISION)
        with self.assertRaises(ValueError):
            history.archive_history({"A": self.stage / ".." / "raw"}, self.output, REVISION)
        self.write("..\\escape.json", b"unsafe extraction name")
        with self.assertRaises(ValueError): history.archive_history({"A": self.stage}, self.output, REVISION)

    def test_size_limit_and_bad_loader_revision_leave_no_archive(self):
        self.write("report.json", b"{}")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            history.archive_history({"A": self.stage}, self.output, REVISION, max_bytes=1)
        self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError): history.archive_history({"A": self.stage}, self.output, "main")

    def test_output_parent_creation_and_symlink_parent_rejection(self):
        self.write("report.json", b"{}")
        output = self.root / "new" / "nested" / "history.zip"
        history.archive_history({"A": self.stage}, output, REVISION)
        self.assertEqual(len(history.verify_archive(output)["files"]), 1)
        linked = self.root / "linked-parent"
        linked.symlink_to(output.parent, target_is_directory=True)
        with self.assertRaises(OSError):
            history.archive_history({"A": self.stage}, linked / "other.zip", REVISION)
        self.assertFalse((output.parent / "other.zip").exists())

    def test_source_replaced_by_fifo_cannot_block_or_publish(self):
        source = self.write("state.pt", b"original")
        original = history._copy_member

        def replace_with_fifo(archive, root, entry):
            source.unlink(); os.mkfifo(source)
            return original(archive, root, entry)

        with patch.object(history, "_copy_member", side_effect=replace_with_fifo):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                history.archive_history({"A": self.stage}, self.output, REVISION)
        self.assertFalse(self.output.exists())


if __name__ == "__main__": unittest.main()
