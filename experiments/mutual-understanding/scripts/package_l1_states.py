"""Archive original, excluded L1 development states without importing tensor code.

Run from the experiment directory, for example::

    python scripts/package_l1_states.py --stage A=runs/A --stage E_original=runs/E \
        --invalid-stage E_original --prior common=runs/prior.pt --output l1-all.zip

Only the latest checkpoint declared by each raw trial is retained. Selection is
by step, never by a score. Model files are copied as opaque bytes, including any
optimizer and random-generator state already present in the original file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


TRIAL_KIND = "mastery-lab-v1-trial"
CHECKPOINT_FORMAT = "mutual-mastery-development-v1"
PRIOR_FORMAT = "mutual-spatial-prior-development-v1"
SCHEMA = "dovis-l1-state-archive-v1"
CONTEXT_FILES = ("dataset.json", "config.json", "contrast_probes.json", "training_catalog.json")
SUITE_FILES = ("mastery-suite.json", "decoder-study.json", "negative-policy-study.json",
               "memory-study.json", "grounding-study.json", "cohort-validation.json")
REPOSITORY = "https://github.com/doheyn1008/DoVIS"


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token(value):
    original = str(value)
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", original)[:72]
    if clean in ("", ".", ".."):
        clean = "value"
    return clean if clean == original else clean + "-" + hashlib.sha256(original.encode()).hexdigest()[:10]


def _labels(items, description):
    """Accept mappings or pairs so duplicate CLI names cannot disappear in dict()."""
    result, paths = {}, set()
    for name, value in (items.items() if hasattr(items, "items") else items):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Unsafe {description} name: {name!r}")
        path = Path(value).resolve()
        if name in result or path in paths:
            raise ValueError(f"Duplicate {description} name or path: {name}")
        result[name] = path
        paths.add(path)
    return result


def _development(value, expected_format=None, *, eligibility=True):
    if (not isinstance(value, dict) or value.get("kind") != "development_only"
            or value.get("excluded_from_research") is not True
            or (expected_format and value.get("format") != expected_format)
            or (eligibility and value.get("eligible_for_fresh_experiment") is not False)):
        raise ValueError("Expected an explicitly excluded development-only manifest")


def _local_file(directory, name):
    if not isinstance(name, str) or Path(name).name != name or "\\" in name or name in (".", ".."):
        raise ValueError(f"Checkpoint must use a plain local basename: {name!r}")
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise ValueError(f"Missing or unsafe artifact: {path}")
    return path


def _add(payload, name, *, path=None, data=None, purpose):
    if name.startswith("/") or "\\" in name or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError(f"Unsafe archive member: {name}")
    if name in payload:
        raise ValueError(f"Archive member collision: {name}")
    if path is not None:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked artifact: {path}")
        item = {"path": path, "sha256": _sha(path), "bytes": path.stat().st_size,
                "source_path": str(path), "purpose": purpose}
    else:
        item = {"data": data, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                "purpose": purpose}
    payload[name] = item
    return name


def _load_priors(priors):
    records = {}
    for label, path in priors.items():
        if path.suffix != ".pt":
            raise ValueError(f"Prior must be a .pt file: {path}")
        sidecar = path.with_suffix(".json")
        if path.is_symlink() or sidecar.is_symlink():
            raise ValueError("Symlinked prior files are not accepted")
        manifest = _read_json(sidecar)
        _development(manifest, PRIOR_FORMAT, eligibility=False)
        report = manifest.get("report", {})
        if (report.get("format") != PRIOR_FORMAT or report.get("excluded_from_research") is not True
                or report.get("eligible_for_fresh_experiment") is not False or not report.get("fit_pool_sha256")):
            raise ValueError("Prior report must declare its fit pool and development exclusion")
        digest = _sha(path)
        if digest != manifest.get("sha256"):
            raise ValueError(f"Prior SHA256 mismatch: {path}")
        entry = records.setdefault(digest, {"path": path, "sidecar": sidecar, "report": report, "aliases": []})
        if entry["report"] != report:
            raise ValueError("Identical prior bytes have contradictory sidecar reports")
        entry["aliases"].append({"label": label, "path": str(path), "sidecar": str(sidecar),
                                 "sidecar_sha256": _sha(sidecar)})
    return records


def _match_prior(report, priors):
    metadata = report.get("prior_metadata", {})
    pool = metadata.get("fit_pool_sha256")
    if not pool:
        if metadata.get("kind") != "fixed_pixel_random_projection":
            raise ValueError("Trial does not declare a fit-pool prior or the original random projection")
        return None
    matches = [sha for sha, item in priors.items() if item["report"].get("fit_pool_sha256") == pool]
    recorded = metadata.get("source_bundle_sha256")
    if recorded:
        matches = [sha for sha in matches if sha == recorded]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous prior bundle for fit pool {pool}")
    digest = matches[0]
    if ("fit_png_sha256" in metadata and sorted(metadata["fit_png_sha256"])
            != sorted(priors[digest]["report"].get("fit_png_sha256", ()))):
        raise ValueError("Trial and supplied prior PNG hashes disagree")
    return digest


def _trial(report_path, stage, stage_root, invalid, priors, payload):
    report = _read_json(report_path)
    if report.get("excluded_from_research") is not True or report.get("eligible_for_fresh_experiment") is not False:
        raise ValueError(f"Trial is not explicitly research-excluded: {report_path}")
    if report.get("protected_splits_used") != []:
        raise ValueError(f"Trial must explicitly declare no protected-split use: {report_path}")
    directory = report_path.parent
    report_sha = _sha(report_path)
    seed_pair = report.get("seed_pair")
    if not isinstance(seed_pair, list) or len(seed_pair) != 2 or any(type(seed) is not int for seed in seed_pair):
        raise ValueError(f"Invalid trial seed pair: {report_path}")
    trial_identity = report_sha + "\0" + report_path.relative_to(stage_root).as_posix()
    trial_digest = hashlib.sha256(trial_identity.encode()).hexdigest()[:16]
    prefix = (f"stages/{stage}/{_token(report.get('language', 'unknown'))}/"
              f"r{_token(report.get('replica', 'unknown'))}-{'-'.join(map(str, seed_pair))}/"
              f"{_token(report.get('condition', 'unknown'))}-{trial_digest}")
    report_member = _add(payload, prefix + "/report.json", path=report_path, purpose="original_raw_trial_report")
    context = {}
    for name in CONTEXT_FILES:
        path = directory / name
        if path.exists():
            context[name] = _add(payload, prefix + "/" + name, path=path, purpose="original_trial_context")
        elif name == "dataset.json":
            raise ValueError(f"Missing original trial dataset: {path}")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError(f"Trial checkpoints must be a list: {report_path}")
    steps = [item.get("step") for item in checkpoints if isinstance(item, dict)]
    if len(steps) != len(checkpoints) or any(type(step) is not int or step < 0 for step in steps):
        raise ValueError(f"Invalid declared checkpoint steps: {report_path}")
    checkpoint, candidate, evaluation = None, None, None
    if checkpoints:
        step = max(steps)
        latest = [item for item in checkpoints if item["step"] == step]
        if len(latest) != 1:
            raise ValueError(f"Ambiguous latest checkpoint at step {step}: {report_path}")
        declared = latest[0]
        _development(declared, CHECKPOINT_FORMAT)
        path = _local_file(directory, declared.get("path"))
        if path.suffix != ".pt":
            raise ValueError("Declared model state must have the .pt suffix")
        sidecar = _local_file(directory, path.with_suffix(".json").name)
        manifest = _read_json(sidecar)
        _development(manifest, CHECKPOINT_FORMAT)
        if manifest != declared:
            raise ValueError(f"Report and checkpoint sidecar disagree: {sidecar}")
        if _sha(path) != declared.get("sha256"):
            raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
        completed = report.get("training_steps_completed")
        if type(completed) is not int or completed < step:
            raise ValueError("Checkpoint step exceeds reported completed training")
        checkpoint = {**declared, "member": _add(payload, prefix + "/" + path.name, path=path,
                                                  purpose="original_full_training_state"),
                      "sidecar_member": _add(payload, prefix + "/" + sidecar.name, path=sidecar,
                                             purpose="original_checkpoint_sidecar")}
        matching = [row for row in report.get("curves", []) if row.get("step") == step]
        if len(matching) > 1:
            raise ValueError("Ambiguous evaluation at the latest checkpoint step")
        if matching:
            evaluation = matching[0]
            candidate = evaluation.get("candidate")
        for name in ("generations.json", "comprehension_predictions.json"):
            path = directory / name
            if path.exists():
                rows = _read_json(path)
                if not isinstance(rows, list):
                    raise ValueError(f"Expected evaluation rows: {path}")
                selected = [row for row in rows if row.get("step") == step]
                member = _add(payload, prefix + "/latest-" + name, data=_json_bytes(selected),
                              purpose="latest_checkpoint_evaluation_rows")
                context[name] = {"member": member, "selected_step": step, "selected_rows": len(selected),
                                 "original_source_path": str(path), "original_sha256": _sha(path)}
    revision = report.get("revision")
    commit = revision.get("commit") if isinstance(revision, dict) else None
    return {"stage": stage, "language": report.get("language"), "phase": report.get("phase"),
            "condition": report.get("condition"), "replica": report.get("replica"), "seed_pair": seed_pair,
            "policy": report.get("policy"), "model_factory": report.get("model_factory"),
            "model_profile": report.get("model_profile"), "parameter_count": report.get("parameter_count"),
            "source_revision": revision, "source_url": f"{REPOSITORY}/tree/{commit}" if commit else None,
            "source_sha256": report.get("source_sha256"), "source_report": str(report_path.relative_to(stage_root)),
            "report_member": report_member, "report_sha256": report_sha, "context": context,
            "status": report.get("status"), "training_steps_completed": report.get("training_steps_completed"),
            "declared_checkpoint_steps": steps, "checkpoint": checkpoint,
            "checkpoint_status": "latest_declared" if checkpoint else "no_declared_checkpoint",
            "latest_step_evaluation": evaluation, "final_candidate_at_latest_step": candidate,
            "control_valid": stage not in invalid,
            "comparison_validity_reason": "explicitly_invalid_stage" if stage in invalid else "not_marked_invalid_by_packager",
            "prior_sha256": _match_prior(report, priors), "development_only": True,
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "protected_splits_used": []}


def _readme(index):
    revision = index.get("loader_revision")
    lines = ["DoVIS original L1 development states", "", "All states are development-only and excluded from research.",
             "A development candidate is not a human-fluent or research-qualified learner.",
             "Every discovered raw trial is indexed, including failed and partial trials.",
             "Checkpoint selection uses maximum declared step, never maximum score.",
             "control_valid=false identifies a stage whose comparison was explicitly invalidated.",
             "control_valid=true only means no invalid-stage flag was supplied; inspect raw audits.",
             "No model was loaded, unpickled, retrained, stripped, or reserialized by this packer.",
             "PT files retain their original optimizer and RNG contents, where originally saved.",
             "index.json lists SHA256 and size for every other archive member.", "",
             "Code is kept in GitHub, not bundled in this archive:", REPOSITORY,
             "Each trial records its training revision, factory, profile and source hashes.",
             "Dirty or unavailable source revisions require the recorded source hashes to be resolved.", "",
             "Restore into an empty directory using a ZIP tool, then obtain the experiment source:",
             f"git clone {REPOSITORY}.git", "cd DoVIS"]
    if revision:
        lines.append("git checkout " + revision)
    else:
        lines.append("# Check out a loader revision supporting the recorded trial factories.")
    lines += ["cd experiments/mutual-understanding", "python -m mutual_lab.mastery_infer --help",
              "# Install the selected source revision's declared runtime dependencies.",
              "# For each trial, use index.json checkpoint.member and prior_sha256 mapping:",
              "python -m mutual_lab.mastery_infer --checkpoint /restored/<checkpoint.member> \\",
              "    --prior-bundle /restored/<matching-prior.member> --image /path/to/input.png",
              "# Omit --prior-bundle for original fixed-pixel-random-projection A models.",
              "# The loader verifies development flags, hashes, architecture and source compatibility.",
              "# It may reject a historical state whose exact implementation is unavailable.", ""]
    return "\n".join(lines).encode()


def _write_archive(output, index, payload):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Refusing to replace an existing archive: {output}")
    payload = dict(payload)
    _add(payload, "README.txt", data=_readme(index), purpose="generated_restore_instructions")
    index = dict(index, members={name: {key: value for key, value in item.items() if key not in ("path", "data")}
                               for name, item in sorted(payload.items())})
    index_bytes = _json_bytes(index)
    handle, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    import os
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for name, item in sorted(payload.items()):
                if "path" in item:
                    with item["path"].open("rb") as source, archive.open(name, "w", force_zip64=True) as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                else:
                    archive.writestr(name, item["data"])
            archive.writestr("index.json", index_bytes)
        with zipfile.ZipFile(temporary) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"ZIP CRC verification failed: {corrupt}")
            if set(archive.namelist()) != set(payload) | {"index.json"} or len(archive.namelist()) != len(payload) + 1:
                raise ValueError("ZIP member set mismatch")
            for name, item in payload.items():
                digest, size = hashlib.sha256(), 0
                with archive.open(name) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                if digest.hexdigest() != item["sha256"] or size != item["bytes"]:
                    raise ValueError(f"ZIP member integrity mismatch or changing source: {name}")
            if archive.read("index.json") != index_bytes:
                raise ValueError("ZIP index integrity mismatch")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(output), "sha256": _sha(output), "bytes": output.stat().st_size,
            "trial_count": len(index["trials"]), "checkpoint_count": sum(bool(t["checkpoint"]) for t in index["trials"]),
            "prior_count": len(index["priors"]), "verified": True}


def package_states(stages, priors, output, *, invalid_stages=(), selected_stage=None,
                   selected_output=None, loader_revision=None):
    """Validate and create full/optional selected-stage ZIPs using only stdlib.

    ``stages`` and ``priors`` accept mappings or NAME/path pairs. A selected stage
    retains all of its trials, including failures; it is never a score filter.
    Existing destination files are never overwritten. Returns small file metadata.
    """
    stages, priors = _labels(stages, "stage"), _labels(priors, "prior")
    invalid = set(invalid_stages)
    if not stages or invalid - stages.keys():
        raise ValueError("Supply stages and declare invalidity only for named stages")
    if bool(selected_stage) != bool(selected_output) or (selected_stage and selected_stage not in stages):
        raise ValueError("Selected stage and output must be supplied together and name a supplied stage")
    outputs = [Path(output).resolve()] + ([Path(selected_output).resolve()] if selected_output else [])
    if len(set(outputs)) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Archive output paths must be distinct and must not already exist")
    if loader_revision and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*", loader_revision):
        raise ValueError("Unsafe loader revision")
    prior_records = _load_priors(priors)
    payload, trials, seen = {}, [], set()
    for stage, root in stages.items():
        if not root.is_dir():
            raise ValueError(f"Stage directory does not exist: {root}")
        count = 0
        for path in sorted(root.rglob("report.json")):
            raw = _read_json(path)
            if not isinstance(raw, dict) or raw.get("kind") != TRIAL_KIND:
                continue
            if path.is_symlink() or path.resolve() in seen:
                raise ValueError(f"Duplicate or symlinked raw trial report: {path}")
            seen.add(path.resolve())
            trials.append(_trial(path, stage, root, invalid, prior_records, payload))
            count += 1
        if not count:
            raise ValueError(f"No raw {TRIAL_KIND} reports in stage {stage}")
        suite_paths = {path for basename in SUITE_FILES for path in root.rglob(basename)}
        runtime_paths = {path for path in root.rglob("*")
                         if path.is_file() and path.suffix.lower() in (".json", ".txt")
                         and any(part.startswith("provenance") for part in path.relative_to(root).parts[:-1])}
        for path in sorted(suite_paths | runtime_paths):
            relative = path.relative_to(root).as_posix()
            purpose = ("original_runtime_suite_provenance" if path in suite_paths
                       else "original_runtime_context_provenance")
            member = _add(payload, f"provenance/{stage}/{hashlib.sha256(relative.encode()).hexdigest()[:12]}/{_token(path.name)}",
                          path=path, purpose=purpose)
            payload[member].update(source_stage=stage, source_relative_path=relative)
    prior_index = {}
    for digest, item in sorted(prior_records.items()):
        prefix = "priors/" + digest
        prior_index[digest] = {"member": _add(payload, prefix + "/" + item["path"].name, path=item["path"],
                                              purpose="original_full_prior_state"),
                               "sidecar_member": _add(payload, prefix + "/" + item["sidecar"].name, path=item["sidecar"],
                                                      purpose="original_prior_sidecar"),
                               "fit_pool_sha256": item["report"]["fit_pool_sha256"], "aliases": item["aliases"]}
        for alias in item["aliases"][1:]:
            _add(payload, prefix + "/aliases/" + alias["label"] + "/" + Path(alias["sidecar"]).name,
                 path=Path(alias["sidecar"]), purpose="original_alias_prior_sidecar")
    base = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "development_only": True,
            "excluded_from_research": True, "eligible_for_fresh_experiment": False, "protected_splits_used": [],
            "checkpoint_selection": "maximum_declared_step_per_raw_trial_including_failed_and_partial",
            "original_model_bytes_preserved": True, "model_deserialization_performed": False,
            "source_repository": REPOSITORY, "loader_revision": loader_revision,
            "invalid_stages": sorted(invalid), "stages": {name: str(root) for name, root in stages.items()},
            "trials": trials, "priors": prior_index, "archive_scope": "complete"}
    selected = None
    if selected_stage:
        subset = [trial for trial in trials if trial["stage"] == selected_stage]
        used = {trial["prior_sha256"] for trial in subset if trial["prior_sha256"]}
        selected_payload = {name: item for name, item in payload.items()
                            if name.startswith((f"stages/{selected_stage}/", f"provenance/{selected_stage}/"))
                            or any(name.startswith(f"priors/{digest}/") for digest in used)}
        selected = (dict(base, archive_scope="selected_stage", stages={selected_stage: str(stages[selected_stage])},
                         trials=subset, priors={digest: prior_index[digest] for digest in sorted(used)}), selected_payload)
    result = {"complete": _write_archive(output, base, payload)}
    if selected:
        result["selected"] = _write_archive(selected_output, *selected)
    return result


def _assignment(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected nonempty NAME=PATH")
    return name, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", type=_assignment, required=True)
    parser.add_argument("--prior", action="append", type=_assignment, default=[])
    parser.add_argument("--invalid-stage", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-stage")
    parser.add_argument("--selected-output", type=Path)
    parser.add_argument("--loader-revision")
    args = parser.parse_args(argv)
    try:
        result = package_states(args.stage, args.prior, args.output, invalid_stages=args.invalid_stage,
                                selected_stage=args.selected_stage, selected_output=args.selected_output,
                                loader_revision=args.loader_revision)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Archive validation failed: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
