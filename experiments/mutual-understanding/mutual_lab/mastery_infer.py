"""Read-only inference for explicitly development-only mastery checkpoints.

This loader never returns a research-qualified native/acquisition learner.
Only known local model factories are used; saved Python objects are not loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .profiles import ModelProfile

FORMAT = "mutual-mastery-development-v1"
PRIOR_FORMAT = "mutual-spatial-prior-development-v1"
LANGUAGES = ("ko", "ja", "en", "es")
FACTORIES = {
    "default": "standard",
    "mutual_lab.attentive_spatial.build_attentive_spatial_model": "attentive",
    "mutual_lab.decoder_study.RecordingFactory.D0": "standard",
    "mutual_lab.decoder_study.RecordingFactory.D1": "attentive",
    "mutual_lab.negative_policy_study.FixedLanguageFactory.standard": "standard",
    "mutual_lab.negative_policy_study.FixedLanguageFactory.attentive": "attentive",
}
# Running the original D/E study entrypoints with ``python -m`` recorded their
# wrapper class under __main__. These are exact historical CLI spellings, not
# import targets; retain the recorded name and verify the usual tensor/core
# implementation provenance before using the corresponding known factory.
LEGACY_CLI_FACTORIES = {
    "__main__.RecordingFactory.D0": "mutual_lab.decoder_study.RecordingFactory.D0",
    "__main__.RecordingFactory.D1": "mutual_lab.decoder_study.RecordingFactory.D1",
    "__main__.FixedLanguageFactory.standard": "mutual_lab.negative_policy_study.FixedLanguageFactory.standard",
    "__main__.FixedLanguageFactory.attentive": "mutual_lab.negative_policy_study.FixedLanguageFactory.attentive",
}
ATTENTION_KEYS = {"producer.query.weight", "producer.key.weight", "producer.value.weight",
                  "producer.attention_output.weight", "spatial_coordinates"}


def _manifest(path, expected_format, *, require_eligibility):
    path = Path(path).resolve()
    manifest = json.loads(path.with_suffix(".json").read_text())
    if (manifest.get("format") != expected_format or manifest.get("kind") != "development_only"
            or manifest.get("excluded_from_research") is not True
            or (require_eligibility and manifest.get("eligible_for_fresh_experiment") is not False)):
        raise ValueError("Expected an excluded development-only checkpoint format")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError("Development checkpoint SHA256 mismatch")
    return path, manifest


def _verify_sources(recorded, names):
    if not isinstance(recorded, dict):
        raise ValueError("Development checkpoint has no implementation source hashes")
    root = Path(__file__).resolve().parent
    verified = {}
    for name in names:
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if recorded.get(name) != actual:
            raise ValueError(f"Implementation source mismatch: {name}; use the checkpoint's code revision")
        verified[name] = actual
    return verified


def load_development_model(checkpoint_path, prior_bundle_path=None):
    """Rebuild a known local model and load tensor-only state for read-only use.

    B/C checkpoints require their explicit common-prior bundle. Relevant model
    implementation hashes must match; unrelated study/evaluation modules may
    have changed. The returned model has ``mastery_inference_metadata``.
    """
    path, manifest = _manifest(checkpoint_path, FORMAT, require_eligibility=True)
    from .neural import digest, torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != FORMAT or payload.get("kind") != "development_only"
            or payload.get("excluded_from_research") is not True
            or payload.get("eligible_for_fresh_experiment") is not False
            or type(payload.get("step")) is not int or payload["step"] < 0
            or payload["step"] != manifest.get("step")):
        raise ValueError("Development manifest and checkpoint metadata differ")
    phase, condition = payload.get("phase"), payload.get("condition")
    if phase not in ("A", "B", "C") or condition not in (phase + "0", phase + "1"):
        raise ValueError("Unsupported development phase/condition")
    language = payload.get("language")
    if language not in LANGUAGES:
        raise ValueError("Unsupported checkpoint language")
    # The first A/B v1 writer had no configurable factory. Its known tensor
    # layout plus exact implementation hashes identifies the standard model.
    # Later C/custom checkpoints must declare their factory explicitly.
    legacy_factory = "model_factory" not in payload and phase in ("A", "B")
    recorded_factory = payload.get("model_factory")
    legacy_cli = recorded_factory in LEGACY_CLI_FACTORIES
    if legacy_cli and phase != "C":
        raise ValueError("Historical D/E CLI factories require the C engine phase")
    factory = "default" if legacy_factory else LEGACY_CLI_FACTORIES.get(recorded_factory, recorded_factory)
    if factory not in FACTORIES:
        raise ValueError("Unsupported or missing development model factory")
    plan = payload.get("plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("model_profile"), dict):
        raise ValueError("Development checkpoint lacks its model profile")
    profile = ModelProfile(**plan["model_profile"])
    profile.validate()
    seed_pair = payload.get("seed_pair")
    if (not isinstance(seed_pair, (tuple, list)) or len(seed_pair) != 2
            or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in seed_pair)):
        raise ValueError("Checkpoint initialization/training seed pair is invalid")
    planned = plan.get("seed_pairs", ()) if phase == "A" else (plan.get("b_seed_pair"),)
    if list(seed_pair) not in [list(pair) for pair in planned if isinstance(pair, (tuple, list))]:
        raise ValueError("Checkpoint seed pair differs from its plan")
    if "model_profile" in payload and payload["model_profile"] != plan["model_profile"]:
        raise ValueError("Conflicting checkpoint model profiles")
    weights = payload.get("model")
    if not isinstance(weights, dict):
        raise ValueError("Checkpoint has no tensor model state")
    attentive = FACTORIES[factory] == "attentive"
    present_attention = ATTENTION_KEYS & set(weights)
    if (attentive and present_attention != ATTENTION_KEYS) or (not attentive and present_attention):
        raise ValueError("Ambiguous model factory and attention state")
    if phase == "A" and (factory != "default" or prior_bundle_path is not None):
        raise ValueError("A checkpoints use the original random projection without a spatial bundle")
    relevant = ["neural.py", "development_models.py"]
    if phase != "A":
        relevant.append("spatial_prior.py")
    if attentive:
        relevant.append("attentive_spatial.py")
    sources = _verify_sources(payload.get("source_sha256"), relevant)
    prior_metadata = payload.get("prior_metadata", {})
    if not isinstance(prior_metadata, dict):
        raise ValueError("Checkpoint prior metadata is invalid")
    bundle = None
    if phase != "A":
        if prior_bundle_path is None:
            raise ValueError("An explicit spatial prior bundle is required for B/C inference")
        prior_path, prior_manifest = _manifest(prior_bundle_path, PRIOR_FORMAT, require_eligibility=False)
        if prior_metadata.get("condition") != "B" + condition[-1]:
            raise ValueError("Checkpoint and spatial prior condition differ")
        from .spatial_prior import load_prior_bundle
        bundle = load_prior_bundle(prior_path, expected_fit_pool_sha256=prior_metadata.get("fit_pool_sha256"))
        if (bundle.get("kind") != "development_only" or bundle.get("excluded_from_research") is not True
                or bundle["report"].get("excluded_from_research") is not True
                or bundle["report"].get("eligible_for_fresh_experiment") is not False):
            raise ValueError("Spatial bundle is not excluded from research")
        _verify_sources(bundle["report"].get("source_sha256"), ("neural.py", "spatial_prior.py"))
        if (not prior_metadata.get("fit_pool_sha256")
                or prior_metadata.get("fit_pool_sha256") != bundle["report"]["fit_pool_sha256"]
                or sorted(prior_metadata.get("fit_png_sha256", ())) != sorted(bundle["report"]["fit_png_sha256"])):
            raise ValueError("Spatial bundle has a different fit PNG pool")
        expected_bundle = prior_metadata.get("source_bundle_sha256")
        if expected_bundle is not None and expected_bundle != prior_manifest["sha256"]:
            raise ValueError("Spatial prior bundle differs from the training bundle")
    # Rebuilding should not consume the caller's inference/training RNG stream.
    with torch.random.fork_rng():
        if phase == "A":
            from .development_models import build_development_model
            model = build_development_model(profile, seed_pair[0])
        elif attentive:
            from .attentive_spatial import build_attentive_spatial_model
            model = build_attentive_spatial_model(profile, seed_pair[0], bundle, prior_metadata["condition"])
        else:
            from .spatial_prior import build_spatial_model
            model = build_spatial_model(profile, seed_pair[0], bundle, prior_metadata["condition"])
        if attentive and not torch.equal(model.spatial_coordinates, weights["spatial_coordinates"]):
            raise ValueError("Checkpoint changed the fixed spatial coordinate grid")
        if (not isinstance(weights.get("visual_projection"), torch.Tensor)
                or not torch.equal(model.visual_projection, weights["visual_projection"])):
            raise ValueError("Checkpoint changed its frozen original visual projection")
        try:
            model.load_state_dict(weights, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("Checkpoint tensor state does not match its declared local factory") from exc
    if phase != "A":
        from .spatial_prior import spatial_prior_hash
        expected_prior = prior_metadata.get("prior_sha256")
        if not expected_prior or spatial_prior_hash(model) != expected_prior:
            raise ValueError("Restored visual parameters/buffers differ from the declared prior")
        if prior_metadata.get("prior_hash", expected_prior) != expected_prior:
            raise ValueError("Conflicting spatial prior tensor hashes")
    model.eval()
    model.requires_grad_(False)
    model.development_only = True
    model.mastery_inference_metadata = {
        "scope": "Development-only finite-world model; not qualified L1 or language-distance evidence",
        "excluded_from_research": True, "eligible_for_fresh_experiment": False,
        "language": language, "phase": phase, "condition": condition, "step": payload["step"],
        "model_factory": factory, "checkpoint_sha256": manifest["sha256"],
        "recorded_model_factory": recorded_factory,
        "factory_provenance": ("legacy_cli_alias_verified" if legacy_cli else
                               "legacy_v1_standard_layout" if legacy_factory else "explicit_saved_factory"),
        "factory_compatibility_note": ("Exact historical D/E __main__ CLI alias; C phase, tensor layout and core source hashes verified"
                                       if legacy_cli else None),
        "prior_sha256": prior_metadata.get("prior_sha256"), "verified_source_sha256": sources,
        "restored_model_sha256": digest(model.state_dict()), "training_revision": payload.get("revision"),
        "training_torch_version": payload.get("environment", {}).get("torch"),
        "inference_torch_version": str(torch.__version__),
    }
    return model


def _png(value):
    if not isinstance(value, (bytes, bytearray)) or not value.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Inference input must be PNG bytes")
    return bytes(value)


def infer_development(model, image_png=None, *, text=None, choices_png=None):
    """Generate and/or select an image, verifying unchanged weights and RNG."""
    if not getattr(model, "development_only", False) or not hasattr(model, "mastery_inference_metadata"):
        raise ValueError("Use load_development_model for explicitly development-only inference")
    if image_png is None and text is None:
        raise ValueError("Provide an image or a text/choices comprehension query")
    if (text is None) != (choices_png is None):
        raise ValueError("Comprehension requires both text and PNG choices")
    if choices_png is not None and (not isinstance(text, str) or not text.strip() or len(choices_png) not in (2, 4)):
        raise ValueError("Provide nonempty text and two or four PNG choices")
    from .neural import digest, torch
    before, rng_before = digest(model.state_dict()), torch.random.get_rng_state().clone()
    if before != model.mastery_inference_metadata.get("restored_model_sha256"):
        raise ValueError("Model no longer matches its restored development checkpoint")
    result = dict(model.mastery_inference_metadata)
    with torch.no_grad(), torch.random.fork_rng():
        if image_png is not None:
            result["generation"] = model.generate_with_trace(_png(image_png))
        if text is not None:
            choices = [_png(png) for png in choices_png]
            result["comprehension"] = {"text": text, "choice_count": len(choices),
                                       "prediction": model.predict(text, choices), "index_base": 0}
        if not torch.equal(rng_before, torch.random.get_rng_state()):
            raise RuntimeError("Development inference consumed RNG state")
    if before != digest(model.state_dict()):
        raise RuntimeError("Development inference changed model weights or buffers")
    result["read_only_state_verified"] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prior-bundle", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--choices", nargs="+", type=Path)
    args = parser.parse_args(argv)
    model = load_development_model(args.checkpoint, args.prior_bundle)
    result = infer_development(model, args.image.read_bytes() if args.image else None,
                               text=args.text, choices_png=[p.read_bytes() for p in args.choices] if args.choices else None)
    # Keep invalid generated Unicode in the trace while making stdout valid
    # JSON even when a Unicode-BMP model emits an unpaired surrogate.
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
