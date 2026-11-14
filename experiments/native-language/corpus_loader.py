"""Validate one explicitly reviewed monolingual JSONL corpus before exposure.

Rights, language, and near-duplicate groups are supplied attestations. This
loader checks their declared values and consistency; it does not independently
establish rights, detect languages, or discover near duplicates. No text is
tokenized here. The caller must restrict training to the returned train split.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re

from native_lm import LANGUAGES, NativeDocument, document_split


MANIFEST_FIELDS = {"schema_version", "kind", "language", "document_file", "document_file_sha256", "split_seed"}
DOCUMENT_FIELDS = {field.name for field in fields(NativeDocument)}
SPLITS = ("train", "dev", "test")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON field: {key}")
        value[key] = item
    return value


def _json(raw, label):
    try:
        return json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid UTF-8 JSON in {label}: {error}") from error


def _identity_hash(values):
    raw = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8", errors="strict")
    return hashlib.sha256(raw).hexdigest()


def load_corpus(manifest_path, native_language):
    """Return validated NativeDocument tuples and split hashes; never encode text.

    The manifest has exactly MANIFEST_FIELDS and references a plain local JSONL
    basename. Every row has exactly the NativeDocument fields. Invalid records
    abort the complete load, including invalid rows at the end of the file.
    ``documents_sha256`` is the hash of the original JSONL bytes, not a fitted or
    tokenized representation. Empty train/dev splits are rejected; test may be
    empty and is returned separately for a caller to keep outside its trainer.
    """
    if native_language not in LANGUAGES:
        raise ValueError("Choose exactly one supported native language")
    manifest_path = Path(manifest_path).resolve()
    raw_manifest = manifest_path.read_bytes()
    manifest = _json(raw_manifest, str(manifest_path))
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise ValueError("Corpus manifest must contain exactly the schema-v1 fields")
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["kind"] != "native-monolingual-corpus-v1"
            or manifest["language"] != native_language):
        raise ValueError("Corpus schema, kind, or declared native language does not match")
    seed = manifest["split_seed"]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Corpus split_seed must be an unsigned 32-bit integer")
    filename = manifest["document_file"]
    if (not isinstance(filename, str) or not filename or Path(filename).name != filename
            or filename in (".", "..") or "/" in filename or "\\" in filename):
        raise ValueError("document_file must be a plain local basename")
    expected = manifest["document_file_sha256"]
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("document_file_sha256 must be a lowercase SHA256 hex digest")
    source = manifest_path.parent / filename
    if source.is_symlink() or source.resolve().parent != manifest_path.parent or not source.is_file():
        raise ValueError("document_file must be an existing nonsymlink local file")

    documents, identifiers, text_groups = [], set(), {}
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for number, raw_line in enumerate(stream, 1):
            digest.update(raw_line)
            label = f"{source.name} line {number}"
            record = _json(raw_line, label)
            if not isinstance(record, dict) or set(record) != DOCUMENT_FIELDS:
                raise ValueError(f"{label}: expected exactly the NativeDocument fields")
            if any(not isinstance(value, str) for value in record.values()):
                raise ValueError(f"{label}: all NativeDocument fields must be strings")
            try:
                for value in record.values():
                    value.encode("utf-8", errors="strict")
                document = NativeDocument(**record).validate(native_language)
            except (UnicodeError, ValueError) as error:
                raise ValueError(f"{label}: {error}") from error
            if document.document_id in identifiers:
                raise ValueError(f"{label}: duplicate document_id {document.document_id!r}")
            identifiers.add(document.document_id)
            previous = text_groups.setdefault(document.text_sha256, document.near_duplicate_group)
            if previous != document.near_duplicate_group:
                raise ValueError(f"{label}: identical text has contradictory near_duplicate_group attestations")
            documents.append(document)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError("Corpus document file SHA256 mismatch")
    if not documents:
        raise ValueError("Corpus document file must contain records")

    # No split is exposed until every input record and the whole-file hash pass.
    splits = {name: [] for name in SPLITS}
    for document in documents:
        name = document_split(document, seed)
        if name not in splits:
            raise ValueError("Document splitter returned an unknown split")
        splits[name].append(document)
    groups = {name: {doc.near_duplicate_group for doc in values} for name, values in splits.items()}
    texts = {name: {doc.text_sha256 for doc in values} for name, values in splits.items()}
    ids = {name: {doc.document_id for doc in values} for name, values in splits.items()}
    intersections = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            key = left + "_" + right
            intersections[key] = {"document_ids": len(ids[left] & ids[right]),
                                  "near_duplicate_groups": len(groups[left] & groups[right]),
                                  "exact_text_hashes": len(texts[left] & texts[right])}
            if any(intersections[key].values()):
                raise ValueError("Corpus split leakage: document IDs, groups, or exact texts overlap")
    counts = {name: len(values) for name, values in splits.items()}
    if not counts["train"] or not counts["dev"]:
        raise ValueError("Corpus must provide nonempty train and dev splits without moving groups")
    audit = {"document_count": len(documents), "exact_text_count": len(text_groups),
             "near_duplicate_group_count": len(set.union(*groups.values())),
             "split_group_counts": {name: len(values) for name, values in groups.items()},
             "split_document_ids_sha256": {name: _identity_hash(values) for name, values in ids.items()},
             "split_groups_sha256": {name: _identity_hash(values) for name, values in groups.items()},
             "split_text_hashes_sha256": {name: _identity_hash(values) for name, values in texts.items()},
             "split_intersections": intersections,
             "split_policy": "sha256-seed-reviewed-near-duplicate-group-90-5-5-v1",
             "split_hash_policy": "sha256-utf8-json-sorted-unique-values-compact-v1",
             "document_file": filename, "all_records_validated_before_exposure": True,
             "tokenization_performed": False, "training_split": "train", "test_for_training": False,
             "attestations_are_supplied_metadata": True,
             "independent_rights_or_language_verification": False,
             "near_duplicate_detection_performed": False,
             "rights_review_values": sorted({doc.rights_review for doc in documents}),
             "foreign_passage_audit_values": sorted({doc.foreign_passage_audit for doc in documents})}
    return {"language": native_language, "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "documents_sha256": actual, "split_seed": seed, "split_counts": counts,
            "splits": {name: tuple(values) for name, values in splits.items()}, "audit": audit}
