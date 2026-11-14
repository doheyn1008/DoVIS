"""Complete-file validation and reviewed document-group split boundaries."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus_loader import load_corpus
from native_lm import ByteTokenizer, NativeDocument, document_split


def document(identifier, text, language="en", group=None, **changes):
    value = NativeDocument(document_id=identifier, language=language, text=text,
        source_url="https://example.invalid/authored-fixture", revision="fixture-v1", license_id="CC0-1.0",
        attribution="Unit-test author; not a downloaded corpus", text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        near_duplicate_group=group or identifier, rights_review="approved", foreign_passage_audit="passed")
    return replace(value, **changes)


def fixture_documents(language="en", include_test=True):
    text = {"ko": "한국어 검사용 문장", "ja": "日本語の検査文", "en": "An authored fixture", "es": "Una oración de prueba"}[language]
    result = {}
    for index in range(10000):
        value = document(f"document-{index}", f"{text} {index}.", language)
        name = document_split(value, 79003)
        result.setdefault(name, value)
        if all(name in result for name in ("train", "dev", "test")):
            break
    return [result[name] for name in ("train", "dev", "test") if include_test or name != "test"]


def write_corpus(root, documents, language="en", manifest_changes=None, raw=None):
    source = root / "documents.jsonl"
    source.write_bytes(raw if raw is not None else b"".join(
        (json.dumps(asdict(value), ensure_ascii=False) + "\n").encode() for value in documents))
    manifest = {"schema_version": 1, "kind": "native-monolingual-corpus-v1", "language": language,
                "document_file": source.name, "document_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "split_seed": 79003}
    manifest.update(manifest_changes or {})
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class CorpusLoaderTests(unittest.TestCase):
    def test_four_scripts_exact_source_hashes_deterministic_group_splits_without_encoding(self):
        for language in ("ko", "ja", "en", "es"):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = write_corpus(root, fixture_documents(language), language)
                with patch.object(ByteTokenizer, "encode", side_effect=AssertionError("loader must not encode")):
                    result, repeat = load_corpus(path, language), load_corpus(path, language)
                self.assertEqual(result, repeat)
                self.assertEqual(result["split_counts"], {"train": 1, "dev": 1, "test": 1})
                self.assertEqual(result["manifest_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(result["documents_sha256"], hashlib.sha256((root / "documents.jsonl").read_bytes()).hexdigest())
                for name, documents in result["splits"].items():
                    self.assertIsInstance(documents, tuple)
                    self.assertTrue(all(document_split(doc, 79003) == name and doc.language == language for doc in documents))
                self.assertTrue(result["audit"]["attestations_are_supplied_metadata"])
                self.assertFalse(result["audit"]["independent_rights_or_language_verification"])
                self.assertTrue(all(not any(item.values()) for item in result["audit"]["split_intersections"].values()))

    def test_late_foreign_row_fails_before_any_encoding_and_no_silent_drop(self):
        with tempfile.TemporaryDirectory() as temporary:
            docs = fixture_documents() + [document("foreign-last", "日本語。", "ja")]
            path = write_corpus(Path(temporary), docs)
            with patch.object(ByteTokenizer, "encode") as encoder, self.assertRaisesRegex(ValueError, "line 4.*native language"):
                load_corpus(path, "en")
            encoder.assert_not_called()

    def test_exact_duplicate_text_cannot_claim_different_group_even_in_same_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            docs = fixture_documents()
            docs.append(replace(docs[0], document_id="copy", near_duplicate_group="other-claimed-group"))
            path = write_corpus(Path(temporary), docs)
            with self.assertRaisesRegex(ValueError, "identical text.*near_duplicate_group"):
                load_corpus(path, "en")

    def test_duplicate_ids_rejected_and_shared_reviewed_group_stays_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = fixture_documents()
            path = write_corpus(root, docs + [docs[0]])
            with self.assertRaisesRegex(ValueError, "duplicate document_id"):
                load_corpus(path, "en")
            path = write_corpus(root, docs + [replace(docs[0], document_id="same-group-copy")])
            result = load_corpus(path, "en")
            self.assertEqual(result["split_counts"], {"train": 2, "dev": 1, "test": 1})
            self.assertEqual(result["audit"]["exact_text_count"], 3)
            self.assertEqual(result["audit"]["near_duplicate_group_count"], 3)

    def test_unsafe_paths_and_file_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in ("../outside.jsonl", "/absolute.jsonl", "folder/file.jsonl", "..\\outside.jsonl", ".."):
                path = write_corpus(root, fixture_documents(), manifest_changes={"document_file": filename})
                with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "basename"):
                    load_corpus(path, "en")
            path = write_corpus(root, fixture_documents(), manifest_changes={"document_file_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_corpus(path, "en")

    def test_source_review_text_hash_and_required_fields_rejected(self):
        for changes in ({"rights_review": "pending"}, {"foreign_passage_audit": "pending"},
                        {"license_id": "unknown"}, {"source_url": " "}, {"revision": ""},
                        {"attribution": ""}, {"document_id": " "}, {"near_duplicate_group": ""},
                        {"text_sha256": "0" * 64}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temporary:
                docs = fixture_documents()
                docs[-1] = replace(docs[-1], **changes)
                path = write_corpus(Path(temporary), docs)
                with self.assertRaises(ValueError):
                    load_corpus(path, "en")

    def test_strict_manifest_record_schema_utf8_and_blank_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for changes in ({"extra": 1}, {"schema_version": True}, {"split_seed": True}, {"language": "ja"}):
                path = write_corpus(root, fixture_documents(), manifest_changes=changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    load_corpus(path, "en")
            valid = (json.dumps(asdict(fixture_documents()[0])) + "\n").encode()
            extra = asdict(fixture_documents()[1]); extra["unexpected"] = "field"
            bad_records = (valid + b"\xff\n", valid + b"\n", valid + json.dumps(extra).encode(),
                           valid + b'{"document_id":"one","document_id":"two"}\n')
            for raw in bad_records:
                path = write_corpus(root, [], raw=raw)
                with self.subTest(raw=raw[-30:]), self.assertRaises(ValueError):
                    load_corpus(path, "en")

    def test_nonempty_train_dev_required_without_moving_groups_test_optional(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = fixture_documents()
            for values in ([], [docs[0]], [docs[1]], [docs[2]]):
                path = write_corpus(root, values)
                with self.subTest(values=values), self.assertRaises(ValueError):
                    load_corpus(path, "en")
            path = write_corpus(root, fixture_documents(include_test=False))
            result = load_corpus(path, "en")
            self.assertEqual(result["splits"]["test"], ())
            self.assertEqual(result["split_counts"], {"train": 1, "dev": 1, "test": 0})


if __name__ == "__main__":
    unittest.main()
