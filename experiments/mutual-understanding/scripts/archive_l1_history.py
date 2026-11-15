"""Preserve complete raw L1 development histories as verified, opaque bytes.

Unlike package_l1_states.py, this keeps EVERY checkpoint and all allowed raw
artifacts, including failed runs and original predictions. It never imports
tensor libraries, parses a model, selects a winner or promotes research status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import zipfile
import zlib
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "dovis-l1-full-history-v1"
EXTENSIONS = {".json", ".pt", ".txt", ".png", ".svg"}
CHUNK = 1024 * 1024
REPOSITORY = "https://github.com/doheyn1008/DoVIS"


def _json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()


def _name(name):
    if (not name or name in (".", "..") or any(character in name for character in ("/", "\\", ":", "\x00", "\n", "\r"))):
        raise ValueError("Unsafe archive path component: " + repr(name))
    return name


def _absolute(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Parent traversal is not accepted in archive paths")
    return Path(os.path.abspath(path))


def _directory(path, *, create=False):
    """Open each ancestor without following symlinks; callers retain the fd."""
    path = _absolute(path)
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor); descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _scan(descriptor, stage, prefix=()):
    result = []
    with os.scandir(descriptor) as entries:
        entries = sorted(entries, key=lambda entry: entry.name)
    for entry in entries:
        name = _name(entry.name)
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("Symlinked source artifacts are not accepted: " + name)
        relative = (*prefix, name)
        if stat.S_ISDIR(info.st_mode):
            if name in (".git", ".github"):
                continue
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                if _signature(os.fstat(child)) != _signature(info):
                    raise RuntimeError("Source directory changed while scanning")
                result.extend(_scan(child, stage, relative))
            finally:
                os.close(child)
        elif not stat.S_ISREG(info.st_mode):
            raise ValueError("Only regular source artifacts are accepted: " + name)
        elif Path(name).suffix.lower() in EXTENSIONS:
            original = "/".join(relative)
            result.append({"stage": stage, "original_relative_path": original,
                "archive_path": "stages/" + stage + "/" + original,
                "bytes": info.st_size, "_parts": relative, "_signature": _signature(info)})
    return result


def _source(root, entry):
    descriptor = os.dup(root)
    try:
        for part in entry["_parts"][:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor); descriptor = child
        source = os.open(entry["_parts"][-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    stream = os.fdopen(source, "rb")
    if _signature(os.fstat(stream.fileno())) != entry["_signature"]:
        stream.close()
        raise RuntimeError("Source artifact changed before reading: " + entry["archive_path"])
    return stream


def _read_source(root, entry, destination=None):
    digest, size = hashlib.sha256(), 0
    with _source(root, entry) as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            digest.update(chunk); size += len(chunk)
            if destination is not None:
                destination.write(chunk)
        if _signature(os.fstat(source.fileno())) != entry["_signature"]:
            raise RuntimeError("Source artifact changed during reading: " + entry["archive_path"])
    if size != entry["bytes"]:
        raise RuntimeError("Source artifact size changed")
    return digest.hexdigest()


def _copy_member(archive, root, entry):
    with archive.open(entry["archive_path"], "w", force_zip64=True) as destination:
        if _read_source(root, entry, destination) != entry["sha256"]:
            raise RuntimeError("Source artifact SHA256 changed: " + entry["archive_path"])


def _verify_zip(stream, expected_index=None):
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        if len(infos) != len({info.filename for info in infos}):
            raise ValueError("Duplicate ZIP members")
        index_bytes = archive.read("INDEX.json")
        if expected_index is not None and index_bytes != expected_index:
            raise ValueError("The archive index changed")
        index = json.loads(index_bytes)
        if index.get("schema") != SCHEMA:
            raise ValueError("Not a full L1 history archive")
        expected = {row["archive_path"]: row for row in index["files"]}
        if len(expected) != len(index["files"]):
            raise ValueError("Duplicate indexed artifacts")
        expected["RESTORE.txt"] = index["restore"]
        expected["INDEX.json"] = {"sha256": hashlib.sha256(index_bytes).hexdigest(), "bytes": len(index_bytes)}
        if {info.filename for info in infos} != set(expected):
            raise ValueError("ZIP members differ from the complete index")
        for info in infos:
            if info.is_dir() or info.filename.startswith("/") or "\\" in info.filename:
                raise ValueError("Unsafe ZIP member")
            for part in info.filename.split("/"):
                _name(part)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Symlinked ZIP member")
            digest, crc, size = hashlib.sha256(), 0, 0
            with archive.open(info) as source:
                for chunk in iter(lambda: source.read(CHUNK), b""):
                    digest.update(chunk); crc = zlib.crc32(chunk, crc); size += len(chunk)
            row = expected[info.filename]
            if size != row["bytes"] or digest.hexdigest() != row["sha256"] or crc & 0xffffffff != info.CRC:
                raise ValueError("ZIP SHA256/size/CRC verification failed: " + info.filename)
    return index


def verify_archive(path):
    """Check every member's CRC, indexed SHA256, byte count and safe name."""
    path = _absolute(path)
    parent = _directory(path.parent)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Archive verification requires a regular ZIP file")
            return _verify_zip(stream)
    finally:
        os.close(parent)


def archive_history(stages, output, loader_revision, *, max_bytes=None):
    """Archive all eligible original files; atomically publish without replacing."""
    if not isinstance(stages, Mapping) or not stages:
        raise ValueError("Supply a nonempty stage-name/path mapping")
    if not isinstance(loader_revision, str) or not re.fullmatch("[0-9a-f]{40}", loader_revision):
        raise ValueError("Pin a full lowercase GitHub loader source revision")
    names = list(stages)
    if (any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names)
            or len({name.casefold() for name in names}) != len(names)):
        raise ValueError("Supply unique safe stage prefixes")
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("The optional compressed-byte limit must be positive")
    output = _absolute(output); _name(output.name)
    parent = _directory(output.parent, create=True)
    roots, temporary = {}, None
    try:
        try:
            os.stat(output.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(output)
        for name, path in stages.items():
            roots[name] = _directory(path)
        entries = [entry for name, root in roots.items() for entry in _scan(root, name)]
        entries.sort(key=lambda row: row["archive_path"])
        if not entries:
            raise ValueError("The stages contain no supported raw artifacts")
        for entry in entries:
            entry["sha256"] = _read_source(roots[entry["stage"]], entry)
        restore = ("FULL ORIGINAL L1 DEVELOPMENT HISTORY -- NOT RESEARCH QUALIFICATION\n\n"
            "Every included raw checkpoint, prediction, prior, failure context and supported image is preserved as opaque original bytes.\n"
            "No PT file was deserialized; historical JSON claims remain unchanged, including invalid and failed studies.\n"
            "1. Keep the external archive SHA256 receipt and verify INDEX.json plus every member CRC/SHA256 before extraction.\n"
            "2. Extract into a new empty directory; stages/<name>/ keeps every original relative path and all checkpoint budgets.\n"
            "3. Obtain the inspection/restore helper source from " + REPOSITORY + "/tree/" + loader_revision + "/experiments/mutual-understanding\n"
            "4. Inspect the original trial reports, sidecars, source revisions, factory IDs and prior provenance. Use the pinned helper's --help to restore the declared development architecture with its matching prior.\n"
            "The loader revision pins helper source only; it is not a model-training revision claim or a guarantee that every legacy state is supported.\n"
            "No source code is bundled, no winner is selected and no state is promoted to a native/research learner.\n").encode()
        index = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "scope": "Read-only complete original development artifacts; all budgets and failures, no selection or deserialization",
            "loader_revision": loader_revision, "loader_source": REPOSITORY + "/tree/" + loader_revision,
            "loader_revision_scope": "Inspection helper source only, not model training provenance",
            "stages": names, "files": [{key: value for key, value in entry.items() if not key.startswith("_")} for entry in entries],
            "restore": {"bytes": len(restore), "sha256": hashlib.sha256(restore).hexdigest()}}
        index_bytes = _json(index)
        temporary = "." + output.name + "." + secrets.token_hex(12) + ".tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
                for entry in entries:
                    _copy_member(archive, roots[entry["stage"]], entry)
                archive.writestr("RESTORE.txt", restore)
                archive.writestr("INDEX.json", index_bytes)
            stream.flush(); os.fsync(stream.fileno())
            _verify_zip(stream, index_bytes)
            final = {entry["archive_path"]: entry["_signature"] for name, root in roots.items() for entry in _scan(root, name)}
            if final != {entry["archive_path"]: entry["_signature"] for entry in entries}:
                raise RuntimeError("Source history changed while archiving; nothing was published")
            compressed = os.fstat(stream.fileno()).st_size
            if max_bytes is not None and compressed > max_bytes:
                raise ValueError(f"Compressed archive exceeds its limit: {compressed} > {max_bytes}")
            stream.seek(0); digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(CHUNK), b""):
                digest.update(chunk)
            receipt = {"output": str(output), "sha256": digest.hexdigest(), "compressed_bytes": compressed,
                "original_bytes": sum(entry["bytes"] for entry in entries), "files": len(entries),
                "checkpoints": sum(Path(entry["original_relative_path"]).suffix.lower() == ".pt" for entry in entries),
                "verified_members": len(entries) + 2, "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "loader_revision": loader_revision, "excluded_from_research": True}
            # The only publication step: a racing winner is never overwritten.
            os.link(temporary, output.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            os.fsync(parent)
        return receipt
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
        for root in roots.values():
            os.close(root)
        os.close(parent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loader-revision", required=True)
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args(argv)
    stages = {}
    for item in args.stage:
        name, separator, path = item.partition("=")
        if not separator or not path or name in stages:
            raise ValueError("Supply unique stage NAME=PATH arguments")
        stages[name] = path
    print(json.dumps(archive_history(stages, args.output, args.loader_revision, max_bytes=args.max_bytes), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
