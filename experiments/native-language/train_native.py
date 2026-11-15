"""Bounded scratch monolingual training; every checkpoint remains development-only."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import torch
from torch.nn import functional as F

from native_lm import ByteTokenizer, LANGUAGES, ModelConfig, PAD, build_model, next_token_loss

FORMAT = "native-scratch-development-v1"
OPTIMIZER = {"name": "AdamW", "learning_rate": .0003, "betas": [.9, .95], "eps": 1e-8,
             "weight_decay": .1, "gradient_clip_norm": 1.}


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(value):
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(str((str(item.dtype), tuple(item.shape))).encode())
            result.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            result.update(json.dumps(item, sort_keys=True, ensure_ascii=True).encode())
        result.update(b"\x00")

    visit(value)
    return result.hexdigest()


def _sources():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("native_lm.py", "corpus_loader.py", "train_native.py")}


def _device(value):
    device = torch.device(value)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Select CPU or an explicitly available CUDA device")
    return device


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=OPTIMIZER["learning_rate"],
        betas=tuple(OPTIMIZER["betas"]), eps=OPTIMIZER["eps"], weight_decay=OPTIMIZER["weight_decay"])


def _windows(documents, language, length, *, check_deadline=None):
    """Encode only the explicitly supplied train/dev tuple, never test."""
    check = check_deadline or (lambda: None)
    check()
    result = []
    for document in documents:
        check()
        document.validate(language)
        check()
        tokens = ByteTokenizer.encode(document.text)
        check()
        for index, start in enumerate(range(0, len(tokens) - 1, length)):
            check()
            values = tokens[start:start + length + 1]
            targets = values[1:]
            result.append({"document_id": document.document_id, "window": index,
                "target_units": len(targets), "target_bytes": sum(value >= 3 for value in targets),
                "sha256": _digest(values), "tokens": values + [PAD] * (length + 1 - len(values))})
            check()
    if not result:
        raise ValueError("Each train/dev split needs at least one next-token window")
    return result


def evaluate_dev(model, optimizer, rng, windows, batch_size, device, *, check_deadline=None):
    check = check_deadline or (lambda: None)
    check()
    before = _digest({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler": rng.getstate()})
    check()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    training = model.training
    total_nll, byte_nll, units, byte_count, eos_count, pad_count = 0., 0., 0, 0, 0, 0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                check()
                rows = windows[start:start + batch_size]
                tokens = torch.tensor([row["tokens"] for row in rows], dtype=torch.long, device=device)
                logits, targets = model(tokens[:, :-1]), tokens[:, 1:]
                losses = F.cross_entropy(logits.reshape(-1, 259), targets.reshape(-1), ignore_index=PAD, reduction="none").reshape_as(targets)
                total_nll += float(losses.sum())
                byte_nll += float(losses[targets >= 3].sum())
                units += sum(row["target_units"] for row in rows)
                byte_count += sum(row["target_bytes"] for row in rows)
                eos_count += int((targets == 2).sum()); pad_count += int((targets == PAD).sum())
                check()
    finally:
        model.train(training)
    after = _digest({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampler": rng.getstate()})
    unchanged = before == after and bool(torch.equal(cpu_rng, torch.random.get_rng_state()))
    if cuda_rng is not None:
        unchanged &= bool(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
    if not unchanged:
        raise RuntimeError("Dev evaluation changed model/optimizer or RNG")
    check()
    return {"loss_per_target_unit": total_nll / units, "bits_per_prediction_unit": total_nll / (units * math.log(2)),
        "bits_per_byte": byte_nll / (byte_count * math.log(2)), "byte_negative_log_likelihood_sum": byte_nll,
        "negative_log_likelihood_sum": total_nll, "target_units": units, "utf8_bytes": byte_count,
        "eos_targets": eos_count, "pad_targets": pad_count,
        "windows": len(windows), "read_only_verified": True,
        "bpb_definition": "Only byte target IDs>=3 contribute to numerator and denominator; EOS/PAD/BOS excluded",
        "scope": "Held-out dev only; not an L1 qualification or language difficulty score"}


def load_checkpoint(path, corpus, *, device="cpu"):
    """Verify development header/SHA/source/corpus before strict tensor restore."""
    path, device = Path(path), _device(device)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if (manifest.get("format") != FORMAT or manifest.get("kind") != "development_only"
            or manifest.get("excluded_from_research") is not True or manifest.get("eligible_for_l2") is not False
            or manifest.get("eligible_for_fresh_experiment") is not False
            or manifest.get("resume_safe") is not True
            or manifest.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError("Expected an intact development-only scratch checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != FORMAT or payload.get("kind") != "development_only"
            or payload.get("excluded_from_research") is not True or payload.get("eligible_for_l2") is not False
            or payload.get("eligible_for_fresh_experiment") is not False or payload.get("source_sha256") != _sources()
            or payload.get("step") != manifest.get("step") or payload.get("language") != corpus["language"]
            or payload.get("corpus_manifest_sha256") != corpus["manifest_sha256"]
            or payload.get("corpus_documents_sha256") != corpus["documents_sha256"]
            or payload.get("split_seed") != corpus["split_seed"] or payload.get("optimizer_specification") != OPTIMIZER):
        raise ValueError("Checkpoint source/corpus/protocol metadata differs")
    if (payload.get("resume_safe") is not True or type(payload.get("step")) is not int
            or not 0 <= payload["step"] <= payload["plan"]["steps"] or len(payload.get("sampling_trace", [])) != payload["step"]
            or any(manifest.get(key) != payload.get(key) for key in ("language", "model_profile", "source_sha256",
                "corpus_manifest_sha256", "corpus_documents_sha256", "split_seed"))):
        raise ValueError("Checkpoint is not a complete resumable training boundary")
    model = build_model(ModelConfig(**payload["model_profile"]), payload["plan"]["initialization_seed"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = _optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    rng = random.Random(); rng.setstate(payload["sampler_rng_state"])
    return {"model": model, "optimizer": optimizer, "rng": rng, "payload": payload, "checkpoint_sha256": manifest["sha256"]}


class _Deadline(Exception):
    pass


def run_training(manifest_path, language, *, output="runs/native-scratch", profile=ModelConfig(), steps=100,
                 batch_size=1, initialization_seed=79007, data_seed=79009, eval_every=25,
                 max_seconds=600, threads=1, device="cpu", resume=None, progress=None):
    """One explicit bounded job. No fetches, pretrained weights or test encoding."""
    began = time.monotonic()
    from corpus_loader import load_corpus
    profile.validate()
    bounds = ((steps, 1, 10000), (batch_size, 1, 16), (eval_every, 1, 10000), (max_seconds, 1, 3600), (threads, 1, 8),
              (initialization_seed, 0, 2**32 - 1), (data_seed, 0, 2**32 - 1))
    if language not in LANGUAGES or any(type(value) is not int or not low <= value <= high for value, low, high in bounds):
        raise ValueError("Invalid language or bounded training controls")
    deadline = began + max_seconds

    def check_deadline():
        if time.monotonic() >= deadline:
            raise _Deadline()

    plan = {"steps": steps, "batch_size": batch_size, "initialization_seed": initialization_seed,
        "data_seed": data_seed, "eval_every": eval_every, "threads": threads, "device": str(device)}
    corpus, restored = None, None

    def setup_outcome(status, error=None):
        path = Path(output).resolve() / f"native-{language}-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True)
        value = {"schema_version": 1, "kind": "native-scratch-development-run-v1", "status": status,
            "stage": "setup", "language": language, "model_profile": asdict(profile), "plan": plan,
            "max_seconds": max_seconds, "elapsed_seconds": time.monotonic() - began,
            "step": restored["payload"]["step"] if restored else 0, "sampling_trace": [], "dev_history": [],
            "checkpoints": [], "checkpoint_created": False, "test_encoded": False,
            "corpus_validation_completed": corpus is not None, "source_sha256": _sources(),
            "excluded_from_research": True, "eligible_for_l2": False, "eligible_for_fresh_experiment": False,
            "trained_general_l1": False, "new_training_steps": 0}
        if error is not None:
            value["error"] = f"{type(error).__name__}: {error}"
        _json(path / "report.json", value)
        return path

    try:
        # Every row is validated before any model/window exposure. Cooperative
        # checks surround I/O, encoding and batches; individual operations are
        # not preemptible and persistence is attempted outside the deadline.
        check_deadline()
        corpus = load_corpus(manifest_path, language, check_deadline=check_deadline)
        check_deadline()
        device = _device(device); plan["device"] = str(device)
        torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)
        windows = {name: _windows(corpus["splits"][name], language, profile.sequence_length,
            check_deadline=check_deadline) for name in ("train", "dev")}
        check_deadline()
        restored = load_checkpoint(resume, corpus, device=device) if resume else None
        if restored and (restored["payload"]["plan"] != plan or restored["payload"]["model_profile"] != asdict(profile)):
            raise ValueError("Resume must retain the original fixed training plan and profile")
        model = restored["model"] if restored else build_model(profile, initialization_seed).to(device)
        check_deadline()
        optimizer = restored["optimizer"] if restored else _optimizer(model)
        rng = restored["rng"] if restored else random.Random(data_seed)
        check_deadline()
    except _Deadline as error:
        return setup_outcome("time_limit", error)
    except KeyboardInterrupt as error:
        try:
            setup_outcome("cancelled", error)
        finally:
            raise error
    directory = Path(output).resolve() / f"native-{language}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    step = restored["payload"]["step"] if restored else 0
    trace = list(restored["payload"]["sampling_trace"]) if restored else []
    history = list(restored["payload"]["dev_history"]) if restored else []
    sources = _sources()
    report = {"schema_version": 1, "kind": "native-scratch-development-run-v1", "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(), "language": language, "model_profile": asdict(profile),
        "parameter_count": sum(p.numel() for p in model.parameters()), "plan": plan, "max_seconds": max_seconds,
        "source_sha256": sources, "corpus_manifest_sha256": corpus["manifest_sha256"],
        "corpus_documents_sha256": corpus["documents_sha256"], "split_seed": corpus["split_seed"],
        "split_counts": corpus["split_counts"], "corpus_audit": corpus["audit"],
        "window_counts": {key: len(value) for key, value in windows.items()}, "test_encoded": False,
        "training_languages": [language], "foreign_training_languages": [], "optimizer_specification": OPTIMIZER,
        "resumed_from_sha256": restored["checkpoint_sha256"] if restored else None,
        "excluded_from_research": True, "eligible_for_fresh_experiment": False, "eligible_for_l2": False,
        "trained_general_l1": False, "dev_history": history, "checkpoints": [],
        "selection": "Fixed final budget only; no best-dev checkpoint selection or automatic qualification"}
    in_update = False
    cuda_indices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []

    def snapshot():
        counts = Counter((item["document_id"], item["window"]) for row in trace for item in row["windows"])
        report.update(step=step, elapsed_seconds=time.monotonic() - began, sampling_trace=trace,
            sampling_stream_sha256=_digest(trace), training_target_units=sum(row["target_units"] for row in trace),
            training_utf8_bytes=sum(row["target_bytes"] for row in trace),
            training_eos_targets=sum(row["target_units"] - row["target_bytes"] for row in trace),
            training_pad_targets=step * batch_size * profile.sequence_length - sum(row["target_units"] for row in trace),
            sampled_windows=[{"document_id": key[0], "window": key[1], "exposures": count} for key, count in sorted(counts.items())],
            final_model_sha256=_digest(model.state_dict()))
        _json(directory / "report.json", report)

    def checkpoint():
        prefix = "incomplete-after" if in_update else "step"
        path = directory / f"{prefix}-{step:06d}-{report['status']}-{uuid.uuid4().hex}.pt"
        payload = {"format": FORMAT, "kind": "development_only", "excluded_from_research": True,
            "eligible_for_fresh_experiment": False, "eligible_for_l2": False, "language": language,
            "step": step, "model_profile": asdict(profile), "plan": plan, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "optimizer_specification": OPTIMIZER,
            "sampler_rng_state": rng.getstate(), "torch_cpu_rng_state": torch.random.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "source_sha256": sources, "corpus_manifest_sha256": corpus["manifest_sha256"],
            "corpus_documents_sha256": corpus["documents_sha256"], "split_seed": corpus["split_seed"],
            "sampling_trace": trace, "dev_history": history, "resume_safe": not in_update,
            "status_at_save": report["status"]}
        temporary = path.with_suffix(".pt.tmp")
        try:
            with temporary.open("xb") as stream:
                torch.save(payload, stream)
            # Publish a new inode/name, never replace a previous checkpoint.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        sidecar = {"format": FORMAT, "kind": "development_only", "excluded_from_research": True,
            "eligible_for_fresh_experiment": False, "eligible_for_l2": False, "language": language,
            "step": step, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "model_profile": asdict(profile),
            "resume_safe": not in_update, "path": path.name,
            "status_at_save": report["status"],
            "source_sha256": sources, "corpus_manifest_sha256": corpus["manifest_sha256"],
            "corpus_documents_sha256": corpus["documents_sha256"], "split_seed": corpus["split_seed"]}
        _json(path.with_suffix(".json"), sidecar)
        report["checkpoints"] = [row for row in report["checkpoints"] if row["path"] != path.name] + [sidecar]

    def persist_outcome():
        # A failed checkpoint must not prevent the plain JSON outcome from
        # recording failure/timeout/cancellation and the last safe checkpoint.
        for operation, persist in (("checkpoint", checkpoint), ("report", snapshot)):
            try:
                persist()
            except (Exception, KeyboardInterrupt) as error:
                report.setdefault("persistence_errors", []).append(
                    {"operation": operation, "error": f"{type(error).__name__}: {error}"})

    def evaluate_current_if_due():
        scheduled = step == 0 or step % eval_every == 0 or step == steps
        if scheduled and not any(point["step"] == step for point in history):
            check_deadline()
            metrics = evaluate_dev(model, optimizer, rng, windows["dev"], batch_size, device,
                check_deadline=check_deadline)
            history.append({"step": step, **metrics})
            checkpoint(); snapshot(); check_deadline()

    with torch.random.fork_rng(devices=cuda_indices):
        torch.random.set_rng_state(restored["payload"]["torch_cpu_rng_state"] if restored else
            torch.Generator(device="cpu").manual_seed(data_seed).get_state())
        if device.type == "cuda":
            state = restored["payload"]["torch_cuda_rng_state"] if restored else torch.Generator(device=device).manual_seed(data_seed).get_state()
            torch.cuda.set_rng_state(state, device)
        try:
            snapshot(); check_deadline()
            # A safe resumed checkpoint can precede its scheduled evaluation,
            # including the final update. Fill that point exactly once first.
            evaluate_current_if_due()
            for next_step in range(step + 1, steps + 1):
                check_deadline()
                in_update = True
                chosen = rng.choices(windows["train"], k=batch_size)
                tokens = torch.tensor([row["tokens"] for row in chosen], dtype=torch.long, device=device)
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = next_token_loss(model, tokens)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Non-finite scratch language loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
                step = next_step
                trace.append({"step": step, "windows": [{key: row[key] for key in
                    ("document_id", "window", "target_units", "target_bytes", "sha256")} for row in chosen],
                    "target_units": sum(row["target_units"] for row in chosen), "target_bytes": sum(row["target_bytes"] for row in chosen)})
                in_update = False
                evaluate_current_if_due()
                if progress:
                    progress(step)
            check_deadline(); report["status"] = "completed"; checkpoint(); snapshot(); check_deadline()
        except _Deadline:
            report["status"] = "time_limit"; persist_outcome()
        except KeyboardInterrupt as exc:
            report.update(status="cancelled", error=f"{type(exc).__name__}: {exc}")
            try:
                persist_outcome()
            finally:
                raise exc
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            persist_outcome()
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--language", required=True, choices=LANGUAGES)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--initialization-seed", type=int, default=79007)
    parser.add_argument("--data-seed", type=int, default=79009)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/native-scratch"))
    args = parser.parse_args(argv)
    profile = ModelConfig(**json.loads(args.profile.read_text())) if args.profile else ModelConfig()
    path = run_training(args.manifest, args.language, output=args.output, profile=profile, steps=args.steps,
        batch_size=args.batch_size, initialization_seed=args.initialization_seed, data_seed=args.data_seed,
        eval_every=args.eval_every, max_seconds=args.max_seconds, threads=args.threads, device=args.device, resume=args.resume)
    print(path, flush=True)
    return 0 if json.loads((path / "report.json").read_text())["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
