"""Image-only spatial priors for a development-only B0/B1 comparison.

The pretrainer accepts PNG bytes, never captions, Scene records or filenames.
Its caller must supply fit images only and verify the recorded image hashes.
Tensor dependencies stay optional until an actual image/model operation runs.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path

FORMAT = "mutual-spatial-prior-development-v1"
WIDTH, HEIGHT = 48, 32
MASK_THRESHOLD = .02


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _pngs(pngs):
    if not isinstance(pngs, (tuple, list)) or not pngs:
        raise ValueError("Supply a nonempty explicit sequence of fit PNG bytes")
    if any(type(png) is not bytes or not png.startswith(b"\x89PNG\r\n\x1a\n") for png in pngs):
        raise ValueError("The image-only pretrainer accepts PNG bytes, not labels or filenames")
    hashes = [hashlib.sha256(png).hexdigest() for png in pngs]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate fit PNGs would change the image sampling distribution")
    return hashes


def fit_pool_hash(pngs):
    """Hash membership independently of image order; the full order is also recorded."""
    return hashlib.sha256(json.dumps(sorted(_pngs(pngs)), separators=(",", ":")).encode()).hexdigest()


def png_tensor(pngs):
    """Fixed RGB/255 preprocessing; no fitted statistics, augmentation or labels."""
    from .neural import np, torch, Image
    rows = []
    for png in pngs:
        if type(png) is not bytes or not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Spatial inputs must be PNG bytes")
        with Image.open(io.BytesIO(png)) as image:
            pixels = np.asarray(image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR),
                                dtype=np.float32).copy() / 255
        rows.append(torch.from_numpy(pixels).permute(2, 0, 1))
    return torch.stack(rows)


def create_spatial_prior(seed=49001, dim=128):
    _integer(seed, "prior seed", 0, 2**32 - 1)
    _integer(dim, "prior output dimension", 8, 2048)
    from .neural import torch, nn, F

    class SpatialEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU())
            self.readout = nn.Linear(32 * 4 * 6, dim)
            self.output_dim = dim

        def forward(self, pixels):
            # Flattening preserves which feature came from each of 4x6 positions.
            spatial = self.features(pixels).flatten(1)
            return F.normalize(self.readout(spatial), dim=-1)

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return SpatialEncoder()


def _decoder(seed, dim):
    from .neural import torch, nn
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(dim, 32 * 4 * 6), nn.Unflatten(1, (32, 4, 6)),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid())


def reconstruction_losses(reconstructed, target):
    """Equal foreground/background MSE, using a pixel-derived corner background."""
    # Background is the top-left RGB pixel of EACH supplied image, not a label.
    background = target[:, :, :1, :1]
    foreground = ((target - background).abs().amax(dim=1, keepdim=True) > MASK_THRESHOLD).to(target.dtype)
    inverse = 1 - foreground
    error = (reconstructed - target).square()
    fg = (error * foreground).sum() / (3 * foreground.sum()).clamp_min(1)
    bg = (error * inverse).sum() / (3 * inverse.sum()).clamp_min(1)
    return .5 * fg + .5 * bg, fg, bg


def pretrain_prior(fit_pngs, seed=49001, dim=128, steps=1000, batch_size=16,
                   learning_rate=.001, max_seconds=300, eval_every=200):
    """Train only on the explicitly supplied fit PNGs at a fixed image budget.

    There is no development input, early-stopping selector, augmentation, Scene
    record or language objective. A time-limited B1 must not be treated as a
    completed prior by the language model factory.
    """
    begun = time.monotonic()
    hashes = _pngs(fit_pngs)
    _integer(seed, "prior seed", 0, 2**32 - 1)
    _integer(dim, "prior output dimension", 8, 2048)
    _integer(steps, "pretraining steps", 1, 10000)
    _integer(batch_size, "image batch size", 1, 256)
    _integer(eval_every, "image evaluation interval", 1, 10000)
    if type(learning_rate) not in {int, float} or not math.isfinite(learning_rate) or not 0 < learning_rate <= .1:
        raise ValueError("Image learning rate must be finite and in (0, .1]")
    if type(max_seconds) not in {int, float} or not math.isfinite(max_seconds) or not 0 < max_seconds <= 3600:
        raise ValueError("Image runtime must be finite and in (0, 3600]")
    deadline = begun + max_seconds
    from .neural import configure_cpu, digest, torch, nn
    configure_cpu()
    encoder = create_spatial_prior(seed, dim)
    decoder = _decoder((seed + 1) % 2**32, dim)
    initial = copy.deepcopy(encoder.state_dict())
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=learning_rate)
    rng = random.Random((seed + 2) % 2**32)
    images = png_tensor(fit_pngs)
    mask = ((images - images[:, :, :1, :1]).abs().amax(dim=1) > MASK_THRESHOLD)
    if any(int(row.sum()) == 0 or int(row.sum()) == row.numel() for row in mask):
        raise ValueError("Each fit image must contain pixel-derived foreground and background")
    source = Path(__file__).resolve()
    report = {"format": FORMAT, "kind": "image_only_prior_development", "excluded_from_research": True,
        "eligible_for_fresh_experiment": False, "status": "running", "seed": seed,
        "decoder_seed": (seed + 1) % 2**32, "training_seed": (seed + 2) % 2**32,
        "dimension": dim, "steps": steps, "steps_completed": 0, "batch_size": batch_size,
        "learning_rate": learning_rate, "runtime_allowance_seconds": max_seconds,
        "fit_png_sha256": sorted(hashes), "fit_png_order_sha256": hashes,
        "fit_pool_sha256": fit_pool_hash(fit_pngs),
        "image_count": len(hashes), "input_scope": "Only the explicit fit PNG byte sequence supplied by caller",
        "input_policy": "RGB/255 at 48x32; no fitted normalization, labels, captions, filenames or augmentation",
        "mask_policy": "RGB distance from each image top-left pixel > 0.02; no scene labels",
        "loss_weights": {"foreground": .5, "background": .5},
        "spatial_readout": "32 channels x 4 rows x 6 columns, flattened then linear, L2 normalized",
        "encoder_parameter_count": sum(p.numel() for p in encoder.parameters()),
        "encoder_buffer_count": sum(b.numel() for b in encoder.buffers()),
        "decoder_parameter_count": sum(p.numel() for p in decoder.parameters()),
        "random_prior_sha256": digest(initial), "torch_version": str(torch.__version__),
        "source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest(),
                          "neural.py": hashlib.sha256(source.with_name("neural.py").read_bytes()).hexdigest()},
        "curves": []}
    sampled = hashlib.sha256()

    def evaluate(step):
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            features = encoder(images)
            total, fg, bg = reconstruction_losses(decoder(features), images)
            distances = torch.pdist(features) if len(features) > 1 else None
        row = {"step": step, "loss": float(total), "foreground_mse": float(fg), "background_mse": float(bg),
               "foreground_pixels": int(mask.sum()), "background_pixels": int((~mask).sum()),
               "latent_coordinate_variance": float(features.var(dim=0, unbiased=False).mean()),
               "minimum_pairwise_feature_distance": float(distances.min()) if distances is not None else None}
        if not all(math.isfinite(row[key]) for key in ("loss", "foreground_mse", "background_mse")):
            raise ValueError("Non-finite image reconstruction evaluation")
        report["curves"].append(row)
        encoder.train(); decoder.train()

    try:
        evaluate(0)
        if time.monotonic() >= deadline:
            report["status"] = "time_limit"
        else:
            for step in range(1, steps + 1):
                if time.monotonic() >= deadline:
                    report["status"] = "time_limit"
                    break
                indices = rng.choices(range(len(images)), k=batch_size)
                sampled.update(json.dumps(indices).encode() + b"\n")
                batch = images[indices]
                optimizer.zero_grad(set_to_none=True)
                total, _, _ = reconstruction_losses(decoder(encoder(batch)), batch)
                if not bool(torch.isfinite(total)):
                    raise ValueError("Non-finite image reconstruction training loss")
                total.backward()
                nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters()], 1, error_if_nonfinite=True)
                optimizer.step()
                report["steps_completed"] = step
                if step % eval_every == 0 or step == steps:
                    evaluate(step)
                if time.monotonic() >= deadline:
                    report["status"] = "time_limit"
                    break
            else:
                report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    encoder.eval(); decoder.eval()
    trained = copy.deepcopy(encoder.state_dict())
    report.update(elapsed_seconds=time.monotonic() - begun,
                  trained_prior_sha256=digest(trained), sampling_stream_sha256=sampled.hexdigest())
    return {"format": FORMAT, "kind": "development_only", "excluded_from_research": True,
            "random_state": initial, "trained_state": trained, "report": report,
            "decoder_state": copy.deepcopy(decoder.state_dict()), "optimizer_state": optimizer.state_dict(),
            "training_rng_state": rng.getstate(), "torch_rng_state": torch.random.get_rng_state()}


def save_prior_bundle(bundle, path):
    from .neural import torch
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(bundle, temporary)
    temporary.replace(path)
    manifest = {"format": FORMAT, "kind": "development_only", "excluded_from_research": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "report": bundle["report"]}
    from .diagnostics import safe_json
    safe_json(path.with_suffix(".json"), manifest)
    return path


def load_prior_bundle(path, *, expected_fit_pool_sha256=None):
    from .neural import torch, digest
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("kind") != "development_only":
        raise ValueError("Not a development spatial-prior bundle")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("Spatial prior SHA256 mismatch")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format") != FORMAT or bundle.get("report") != manifest["report"]:
        raise ValueError("Spatial prior manifest differs from saved state")
    report = bundle["report"]
    if digest(bundle["random_state"]) != report["random_prior_sha256"] or digest(bundle["trained_state"]) != report["trained_prior_sha256"]:
        raise ValueError("Spatial prior tensor hash mismatch")
    if expected_fit_pool_sha256 is not None and report["fit_pool_sha256"] != expected_fit_pool_sha256:
        raise ValueError("Spatial prior used a different fit image pool")
    bundle["_bundle_sha256"] = manifest["sha256"]
    return bundle


def spatial_prior_hash(model):
    from .neural import digest
    return digest(model.visual_encoder.state_dict())


prior_state_digest = spatial_prior_hash


def build_spatial_model(profile, language_initialization_seed, bundle, condition="B1"):
    """Return an L1 learner whose common spatial prior stays entirely frozen."""
    profile.validate()
    _integer(language_initialization_seed, "language initialization seed", 0, 2**32 - 1)
    if condition not in {"B0", "B1"} or bundle.get("format") != FORMAT:
        raise ValueError("Choose B0 or B1 from a spatial development bundle")
    report = bundle["report"]
    if report["dimension"] != profile.hidden_dim:
        raise ValueError("Spatial output dimension must equal the language model visual dimension")
    if condition == "B1" and report["status"] != "completed":
        raise ValueError("B1 requires the fixed image-pretraining budget to complete")
    from .neural import NativeModel, digest, torch
    from .development_models import development_model_spec

    class SpatialNativeModel(NativeModel):
        def train(self, mode=True):
            super().train(mode)
            if hasattr(self, "visual_encoder"):
                self.visual_encoder.eval()
            return self

        @torch.no_grad()
        def visual(self, pngs):
            if any(parameter.requires_grad for parameter in self.visual_encoder.parameters()):
                raise RuntimeError("Spatial priors must stay frozen during language learning")
            self.visual_encoder.eval()
            missing = {hashlib.sha256(png).digest(): png for png in pngs
                       if hashlib.sha256(png).digest() not in self.visual_cache}
            if missing:
                keys = list(missing)
                vectors = self.visual_encoder(png_tensor([missing[key] for key in keys]))
                self.visual_cache.update((key, vector.detach().clone()) for key, vector in zip(keys, vectors))
            return torch.stack([self.visual_cache[hashlib.sha256(png).digest()] for png in pngs])

    model = SpatialNativeModel(profile, language_initialization_seed)
    model.visual_encoder = create_spatial_prior(report["seed"], profile.hidden_dim)
    state = bundle["random_state" if condition == "B0" else "trained_state"]
    expected = report["random_prior_sha256" if condition == "B0" else "trained_prior_sha256"]
    if digest(state) != expected:
        raise ValueError("Spatial prior tensor state does not match its declared hash")
    model.visual_encoder.load_state_dict(state, strict=True)
    model.visual_encoder.requires_grad_(False)
    model.visual_encoder.eval()
    model.visual_cache.clear()
    specification = development_model_spec(profile, "last")
    specification.update(kind="development_spatial_prior", visual_encoder_parameters=report["encoder_parameter_count"],
        trainable_language_parameters=specification["parameter_count"],
        parameter_count=specification["parameter_count"] + report["encoder_parameter_count"])
    model.development_spec = specification
    model.development_only = True
    model.spatial_prior_metadata = {"format": FORMAT, "condition": condition, "prior_sha256": expected,
        "prior_hash": expected, "source_bundle_sha256": bundle.get("_bundle_sha256"),
        "fit_pool_sha256": report["fit_pool_sha256"], "fit_png_sha256": report["fit_png_sha256"],
        "pretraining_steps": report["steps_completed"] if condition == "B1" else 0,
        "prior_status": report["status"], "spatial_readout": report["spatial_readout"],
        "historical_visual_projection": "Unused buffer retained only for NativeModel state compatibility",
        "shared_scope": "This image-only prior is common across language runs; not an independent visual replica"}
    return model
