"""Bounded synthetic CPU resource smoke; never a language-learning result."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import torch

from native_lm import BOS, EOS, ModelConfig, build_model, next_token_loss


def benchmark(*, steps=10, batch_size=1, threads=1):
    if type(steps) is not int or not 10 <= steps <= 20:
        raise ValueError("Synthetic smoke is bounded to10–20 measured steps")
    if type(batch_size) is not int or not 1 <= batch_size <= 4 or type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("Supply bounded batch size1–4 and CPU threads1–8")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    config = ModelConfig()
    began = time.perf_counter()
    model = build_model(config, seed=79007).cpu().train()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer_settings = {"name": "AdamW", "learning_rate": .0003, "betas": [.9, .95], "eps": 1e-8,
                          "weight_decay": .1, "gradient_clip_norm": 1.}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, betas=(.9, .95), eps=1e-8, weight_decay=.1)
    generator = torch.Generator(device="cpu").manual_seed(79009)
    batches = [torch.randint(3, 259, (batch_size, config.sequence_length + 1), generator=generator) for _ in range(steps + 2)]
    for tokens in batches:
        tokens[:, 0] = BOS
        tokens[:, -1] = EOS

    def update(tokens):
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, tokens)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite synthetic smoke loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())

    # Two disposable steps allocate optimizer state and warm CPU kernels. Their
    # temporary random state is never saved or used as a native checkpoint.
    warmup_started = time.perf_counter()
    warmup_losses = [update(tokens) for tokens in batches[:2]]
    warmup_seconds = time.perf_counter() - warmup_started
    measured_started = time.perf_counter()
    losses = [update(tokens) for tokens in batches[2:]]
    measured_seconds = time.perf_counter() - measured_started
    predicted_tokens = steps * batch_size * config.sequence_length
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    optimizer_bytes = sum(value.numel() * value.element_size() for state in optimizer.state.values()
                          for value in state.values() if isinstance(value, torch.Tensor))
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ("native_lm.py", "benchmark_cpu.py")}
    cpu = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    return {"schema_version": 1, "scope": "SMOKE_NOT_LANGUAGE_LEARNING", "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "native-language-cpu-resource-smoke-v1", "language": None, "corpus": None,
        "input": "Uniform synthetic byte IDs plus BOS/EOS; no natural-language documents", "pretrained_weights": False,
        "trained_tokenizer": False, "model": asdict(config), "parameter_count": parameters,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dtype": "float32", "device": "cpu", "batch_size": batch_size, "sequence_length": config.sequence_length,
        "initialization_seed": 79007, "synthetic_data_seed": 79009, "optimizer": optimizer_settings,
        "warmup_steps": 2, "warmup_seconds": warmup_seconds, "warmup_losses": warmup_losses,
        "measured_steps": steps, "measured_seconds": measured_seconds, "predicted_tokens": predicted_tokens,
        "tokens_per_second": predicted_tokens / measured_seconds, "seconds_per_step": measured_seconds / steps,
        "measured_losses": losses, "total_benchmark_seconds": time.perf_counter() - began,
        "total_time_scope": "Model/optimizer construction, synthetic batches, warmup and measured steps; excludes imports and thread setup",
        "parameter_tensor_bytes": parameter_bytes, "optimizer_state_tensor_bytes": optimizer_bytes,
        "max_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "Linux process high-water RSS includes interpreter, torch, activations, gradients and temporary allocations; not GPU VRAM",
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "cpu": cpu},
        "source_sha256": sources, "checkpoint_saved": False, "gpu_vram_measured": False,
        "interpretation": "Throughput/memory feasibility only. Losses on random byte samples measure no native-language competence. No long training or corpus ingestion occurred."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/cpu-smoke.json"))
    args = parser.parse_args(argv)
    result = benchmark(steps=args.steps, batch_size=args.batch_size, threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("scope", "parameter_count", "tokens_per_second", "measured_seconds", "max_process_rss_kib")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
