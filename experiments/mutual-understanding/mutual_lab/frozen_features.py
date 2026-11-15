"""Versioned, per-PNG extraction for frozen development visual features.

Frozen CNN weights do not make batched floating-point features bit-identical.
Always use one PNG per extractor call so a negative policy cannot change a
producer's cached inputs by changing the first batch that contains an image.
No parameters, buffers, encoder operations, or saved legacy states are changed.
"""

from __future__ import annotations

from types import MethodType

FROZEN_FEATURE_POLICY = "per_png_singleton_v1"


def _feature_caches(model):
    return {name: value for name, value in vars(model).items()
            if name.endswith("_cache") and isinstance(value, dict)}


def clear_feature_caches(model):
    """Remove all per-instance image cache contents without changing weights."""
    counts = {}
    for name, cache in sorted(_feature_caches(model).items()):
        counts[name] = len(cache)
        cache.clear()
    return counts


def _singleton_method(original, name):
    def singleton(self, pngs):
        from .neural import torch
        pngs = list(pngs)
        if not pngs:
            raise ValueError("Frozen feature extraction requires at least one PNG")
        rows = [original(self, [png]) for png in pngs]
        if any(not isinstance(row, torch.Tensor) or row.shape[0] != 1 for row in rows):
            raise ValueError("A singleton visual extractor must return one tensor row")
        return torch.cat(rows, dim=0)

    singleton.__name__ = name
    singleton.__qualname__ = "singleton_" + name
    singleton.frozen_feature_policy = FROZEN_FEATURE_POLICY
    return singleton


def enforce_singleton_features(model):
    """Wrap this instance's vector/grid methods, including bound overrides.

    Repeated calls preserve existing canonical caches. First wrapping clears
    caches that may already contain features computed in a larger batch.
    Binding the original function, rather than its instance, also keeps copied
    models independent and supports an instance-provided visual_grid method.
    """
    policy = getattr(model, "frozen_feature_policy", None)
    if policy not in (None, FROZEN_FEATURE_POLICY):
        raise ValueError("Unknown frozen feature policy")
    methods = {}
    for name in ("visual", "visual_grid"):
        if not hasattr(model, name):
            continue
        method = getattr(model, name)
        original = getattr(method, "__func__", None)
        if getattr(original, "frozen_feature_policy", None) == FROZEN_FEATURE_POLICY:
            continue
        if original is None or getattr(method, "__self__", None) is not model:
            raise TypeError("Frozen visual extractors must be methods bound to this model")
        methods[name] = _singleton_method(original, name)
    if not hasattr(model, "visual"):
        raise TypeError("A model must provide visual PNG features")
    if methods:
        clear_feature_caches(model)
        for name, method in methods.items():
            setattr(model, name, MethodType(method, model))
    model.frozen_feature_policy = FROZEN_FEATURE_POLICY
    return model


def feature_cache_manifest(model):
    """Hash sorted tensor cache entries, including shape, dtype, keys and count."""
    from .neural import digest, torch
    caches = {}
    for name, cache in sorted(_feature_caches(model).items()):
        rows = []
        for key, value in cache.items():
            if isinstance(key, bytes):
                key = ("bytes", key.hex())
            elif isinstance(key, str):
                key = ("str", key)
            else:
                raise TypeError("Feature cache keys must be PNG hashes or strings")
            if not isinstance(value, torch.Tensor):
                raise TypeError("Feature cache values must be tensors")
            rows.append((key, value))
        rows.sort(key=lambda row: row[0])
        caches[name] = {"entries": len(rows), "sha256": digest(rows)}
    manifest = {"policy": getattr(model, "frozen_feature_policy", None), "caches": caches}
    return {**manifest, "sha256": digest(manifest)}
