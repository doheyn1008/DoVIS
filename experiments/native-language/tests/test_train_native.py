"""Authored tiny native fixtures: train/dev boundary, updates and strict restore."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from corpus_loader import load_corpus
from native_lm import ByteTokenizer, ModelConfig
from train_native import _digest, load_checkpoint, run_training
from test_corpus_loader import fixture_documents, write_corpus

ROOT = Path(__file__).resolve().parents[1]
SMALL = ModelConfig(id="trainer-fixture", hidden_dim=24, layers=1, heads=3, feedforward_dim=48, sequence_length=32)


class NativeTrainingTests(unittest.TestCase):
    def test_actual_updates_all_four_languages_keep_test_unencoded_and_save_complete_dev_only_state(self):
        for language in ("ko", "ja", "en", "es"):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = write_corpus(root, fixture_documents(language), language)
                corpus = load_corpus(manifest, language)
                original = ByteTokenizer.encode
                with patch("train_native.ByteTokenizer.encode", wraps=original) as encode:
                    path = run_training(manifest, language, output=root / "runs", profile=SMALL, steps=2,
                        batch_size=2, eval_every=1, max_seconds=30)
                observed = {call.args[0] for call in encode.call_args_list}
                self.assertEqual(observed, {doc.text for name in ("train", "dev") for doc in corpus["splits"][name]})
                self.assertTrue(observed.isdisjoint({doc.text for doc in corpus["splits"]["test"]}))
                report = json.loads((path / "report.json").read_text())
                self.assertEqual(report["status"], "completed", report)
                self.assertEqual(report["step"], 2)
                self.assertFalse(report["test_encoded"])
                self.assertFalse(report["trained_general_l1"])
                self.assertFalse(report["eligible_for_l2"])
                train_ids = {doc.document_id for doc in corpus["splits"]["train"]}
                self.assertTrue(all(item["document_id"] in train_ids for step in report["sampling_trace"] for item in step["windows"]))
                self.assertEqual(report["training_target_units"], report["training_utf8_bytes"] + report["training_eos_targets"])
                self.assertEqual([point["step"] for point in report["dev_history"]], [0, 1, 2])
                for point in report["dev_history"]:
                    self.assertTrue(point["read_only_verified"])
                    self.assertEqual(point["target_units"], point["utf8_bytes"] + point["eos_targets"])
                    self.assertAlmostEqual(point["bits_per_byte"], point["byte_negative_log_likelihood_sum"] / (point["utf8_bytes"] * math.log(2)))
                restored = load_checkpoint(path / "step-000002.pt", corpus)
                self.assertEqual(_digest(restored["model"].state_dict()), report["final_model_sha256"])
                self.assertTrue(restored["optimizer"].state)
                self.assertEqual(restored["payload"]["step"], 2)
                initial = load_checkpoint(path / "step-000000.pt", corpus)
                self.assertNotEqual(_digest(initial["model"].state_dict()), report["final_model_sha256"])

    def test_cancelled_complete_boundary_resume_matches_continuous_four_steps_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_corpus(root, fixture_documents())
            options = dict(profile=SMALL, steps=4, batch_size=2, initialization_seed=31, data_seed=33,
                           eval_every=2, max_seconds=30)
            baseline = run_training(manifest, "en", output=root / "continuous", **options)
            interrupt = KeyboardInterrupt("authored fixture boundary")

            def stop(step):
                if step == 2:
                    raise interrupt

            with self.assertRaises(KeyboardInterrupt) as caught:
                run_training(manifest, "en", output=root / "partial", progress=stop, **options)
            self.assertIs(caught.exception, interrupt)
            partial = next((root / "partial").glob("native-*"))
            self.assertEqual(json.loads((partial / "report.json").read_text())["status"], "cancelled")
            resumed = run_training(manifest, "en", output=root / "resumed", resume=partial / "step-000002.pt", **options)
            a, b = (json.loads((path / "report.json").read_text()) for path in (baseline, resumed))
            self.assertEqual(a["final_model_sha256"], b["final_model_sha256"])
            self.assertEqual(a["sampling_stream_sha256"], b["sampling_stream_sha256"])
            self.assertEqual(a["dev_history"], b["dev_history"])
            corpus = load_corpus(manifest, "en")
            left, right = (load_checkpoint(path / "step-000004.pt", corpus) for path in (baseline, resumed))
            self.assertEqual(_digest(left["optimizer"].state_dict()), _digest(right["optimizer"].state_dict()))
            self.assertEqual(left["rng"].getstate(), right["rng"].getstate())

    def test_changing_only_dev_and_test_text_never_changes_training_weights_or_sampler_stream(self):
        from dataclasses import replace
        from native_lm import document_split
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = fixture_documents()
            first, second = root / "first", root / "second"
            first.mkdir(); second.mkdir()
            changed = []
            for document in documents:
                if document_split(document, 79003) != "train":
                    text = document.text + " This deliberately changes held-out content and the number of windows." * 3
                    document = replace(document, text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())
                changed.append(document)
            a = run_training(write_corpus(first, documents), "en", output=first / "runs", profile=SMALL,
                steps=2, batch_size=2, eval_every=1, max_seconds=30)
            b = run_training(write_corpus(second, changed), "en", output=second / "runs", profile=SMALL,
                steps=2, batch_size=2, eval_every=1, max_seconds=30)
            left, right = (json.loads((path / "report.json").read_text()) for path in (a, b))
            self.assertNotEqual(left["corpus_documents_sha256"], right["corpus_documents_sha256"])
            self.assertEqual(left["final_model_sha256"], right["final_model_sha256"])
            self.assertEqual(left["sampling_stream_sha256"], right["sampling_stream_sha256"])
            restored = [load_checkpoint(path / "step-000002.pt", load_corpus(manifest, "en")) for path, manifest in
                ((a, first / "manifest.json"), (b, second / "manifest.json"))]
            self.assertEqual(_digest(restored[0]["optimizer"].state_dict()), _digest(restored[1]["optimizer"].state_dict()))

    def test_checkpoint_write_failure_still_records_failed_outcome_and_preserves_last_safe_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_corpus(root, fixture_documents())
            original_save, calls = torch.save, []

            def fail_later(*args, **kwargs):
                calls.append(args[1])
                if len(calls) > 1:
                    raise OSError("injected checkpoint storage failure")
                return original_save(*args, **kwargs)

            with patch("train_native.torch.save", side_effect=fail_later):
                path = run_training(manifest, "en", output=root / "runs", profile=SMALL,
                    steps=2, eval_every=1, max_seconds=30)
            report = json.loads((path / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("checkpoint storage failure", report["error"])
            self.assertTrue(any(row["operation"] == "checkpoint" for row in report["persistence_errors"]))
            self.assertEqual(load_checkpoint(path / "step-000000.pt", load_corpus(manifest, "en"))["payload"]["step"], 0)

    def test_mid_update_interrupt_preserves_safe_checkpoint_and_marks_live_state_nonresumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_corpus(root, fixture_documents())
            with patch("train_native.next_token_loss", side_effect=KeyboardInterrupt("after sampler draw")):
                with self.assertRaises(KeyboardInterrupt):
                    run_training(manifest, "en", output=root / "runs", profile=SMALL, steps=2, max_seconds=30)
            path = next((root / "runs").glob("native-*"))
            safe = path / "step-000000.pt"
            unsafe = path / "incomplete-after-000000.pt"
            self.assertTrue(safe.is_file())
            self.assertFalse(json.loads(unsafe.with_suffix(".json").read_text())["resume_safe"])
            corpus = load_corpus(manifest, "en")
            with patch("train_native.torch.load") as deserialize, self.assertRaises(ValueError):
                load_checkpoint(unsafe, corpus)
            deserialize.assert_not_called()
            self.assertEqual(load_checkpoint(safe, corpus)["payload"]["step"], 0)

    def test_setup_time_limit_is_preserved_without_training_and_corrupt_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_corpus(root, fixture_documents())
            clock = iter(range(1000))
            with patch("train_native.time.monotonic", side_effect=lambda: next(clock)):
                path = run_training(manifest, "en", output=root / "runs", profile=SMALL, steps=2, max_seconds=1)
            report = json.loads((path / "report.json").read_text())
            self.assertEqual(report["status"], "time_limit")
            self.assertEqual(report["step"], 0)
            self.assertFalse(report["sampling_trace"])
            checkpoint = path / "step-000000.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
            with patch("train_native.torch.load") as deserialize, self.assertRaises(ValueError):
                load_checkpoint(checkpoint, load_corpus(manifest, "en"))
            deserialize.assert_not_called()

    def test_late_foreign_record_fails_before_model_or_encoding_and_cli_runs_tiny_fixture(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = fixture_documents()
            documents[-1] = replace(documents[-1], language="ja")
            manifest = write_corpus(root, documents)
            with patch("train_native.build_model") as construct, patch("train_native.ByteTokenizer.encode") as encode:
                with self.assertRaises(ValueError):
                    run_training(manifest, "en", output=root / "absent", profile=SMALL, steps=2)
                construct.assert_not_called(); encode.assert_not_called()
            self.assertFalse((root / "absent").exists())
            manifest = write_corpus(root, fixture_documents())
            profile = root / "profile.json"
            profile.write_text(json.dumps(asdict(SMALL)))
            process = subprocess.run([sys.executable, "train_native.py", "--manifest", str(manifest), "--language", "en",
                "--profile", str(profile), "--steps", "2", "--batch-size", "1", "--eval-every", "1",
                "--max-seconds", "30", "--output", str(root / "cli")], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((Path(process.stdout.strip()) / "report.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["parameter_count"], 11904)


if __name__ == "__main__":
    unittest.main()
