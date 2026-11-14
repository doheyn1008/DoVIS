"""CI-only tiny synthetic update, not the14M resource measurement or L1 training."""

import argparse
import json

import torch
from native_lm import BOS, EOS, ModelConfig, build_model, next_token_loss


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = ModelConfig(id="ci-small-fixture", hidden_dim=24, layers=2, heads=3,
                         feedforward_dim=48, sequence_length=32)
    model = build_model(config, 17)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator(device="cpu").manual_seed(19)
    losses = []
    for _ in range(args.steps):
        tokens = torch.randint(3, 259, (2, 33), generator=generator)
        tokens[:, 0], tokens[:, -1] = BOS, EOS
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, tokens)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite tiny CI smoke")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    print(json.dumps({"scope": "SMOKE_NOT_LANGUAGE_LEARNING", "profile": config.id,
        "parameter_count": sum(p.numel() for p in model.parameters()), "steps": args.steps,
        "synthetic_losses": losses, "checkpoint_saved": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
