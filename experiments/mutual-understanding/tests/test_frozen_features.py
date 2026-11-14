"""Frozen CNN features must not depend on a negative policy's cache history."""

import copy
import hashlib
import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path
from types import MethodType

from mutual_lab.config import ExperimentConfig
from mutual_lab import development_world as world
from mutual_lab.frozen_features import (FROZEN_FEATURE_POLICY, clear_feature_caches,
                                       enforce_singleton_features, feature_cache_manifest)
from mutual_lab.profiles import ModelProfile

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class FrozenFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab.neural import configure_cpu
        from mutual_lab.spatial_prior import pretrain_prior
        configure_cpu()
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        cls.scenes = world.development_splits(config, 59003)["train"][:4]
        cls.pngs = [world.render_scene(scene) for scene in cls.scenes]
        cls.profile = ModelProfile("singleton-feature-test", embedding_dim=8, hidden_dim=16,
                                   max_sequence_length=128)
        cls.bundle = pretrain_prior(cls.pngs, dim=16, steps=1, batch_size=4, eval_every=1)

    def model(self):
        from mutual_lab.attentive_spatial import build_attentive_spatial_model
        return build_attentive_spatial_model(self.profile, 59007, self.bundle, condition="B1")

    def test_actual_cnn_canonical_vectors_grids_and_producer_updates_match_cache_histories(self):
        from mutual_lab.neural import digest, torch
        a, b = self.model(), self.model()
        pngs = self.pngs[:3]
        a.visual([pngs[0]]); a.visual(pngs[1:])
        a.visual_grid([pngs[0]]); a.visual_grid(pngs[1:])
        # Real backends may or may not differ bitwise across batch sizes. The
        # post-correction invariant must hold on either kind of backend.
        a.visual(pngs); b.visual(pngs)
        b.visual_grid(pngs)
        enforce_singleton_features(a); enforce_singleton_features(b)
        a.visual([pngs[0]]); a.visual(pngs[1:])
        a.visual_grid([pngs[0]]); a.visual_grid(pngs[1:])
        self.assertTrue(torch.equal(a.visual(pngs), b.visual(pngs)))
        self.assertTrue(torch.equal(a.visual_grid(pngs), b.visual_grid(pngs)))
        self.assertEqual(feature_cache_manifest(a), feature_cache_manifest(b))
        examples = [{"image": png, "text": world.captions(scene, "en")[0]}
                    for png, scene in zip(pngs, self.scenes)]
        for model in (a, b):
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            optimizer.zero_grad(set_to_none=True)
            model.production_loss(examples).backward()
            torch.nn.utils.clip_grad_norm_(model.producer.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        self.assertEqual(digest(a.producer.state_dict()), digest(b.producer.state_dict()))

    def test_singleton_policy_removes_a_deterministic_batch_sensitive_cache_fixture(self):
        from mutual_lab.neural import torch

        class BatchSensitiveCache:
            def __init__(self):
                self.visual_cache, self.spatial_grid_cache = {}, {}

            def cached(self, pngs, cache, shape):
                for png in pngs:
                    key = hashlib.sha256(png).digest()
                    if key not in cache:
                        cache[key] = torch.full(shape, float(len(pngs)))
                return torch.stack([cache[hashlib.sha256(png).digest()] for png in pngs])

            def visual(self, pngs):
                return self.cached(pngs, self.visual_cache, (2,))

            def visual_grid(self, pngs):
                return self.cached(pngs, self.spatial_grid_cache, (24, 34))

        a, b = BatchSensitiveCache(), BatchSensitiveCache()
        pngs = self.pngs[:3]
        for name in ("visual", "visual_grid"):
            getattr(a, name)(pngs[:1]); getattr(a, name)(pngs[1:])
            self.assertFalse(torch.equal(getattr(a, name)(pngs), getattr(b, name)(pngs)))
        enforce_singleton_features(a); enforce_singleton_features(b)
        for name, expected in (("visual", torch.ones(3, 2)), ("visual_grid", torch.ones(3, 24, 34))):
            getattr(a, name)(pngs[:1]); getattr(a, name)(pngs[1:])
            self.assertTrue(torch.equal(getattr(a, name)(pngs), getattr(b, name)(pngs)))
            self.assertTrue(torch.equal(getattr(a, name)(pngs), expected))
        self.assertEqual(feature_cache_manifest(a), feature_cache_manifest(b))

    def test_wrapping_clears_preflight_caches_without_changing_state_flags_or_copy_independence(self):
        from mutual_lab.neural import digest, torch
        model = self.model(); model.train()
        model.visual(self.pngs); model.visual_grid(self.pngs)
        model.object_memory_cache = {b"preflight": torch.ones(2, 34)}
        state = digest(model.state_dict())
        flags = ({name: item.training for name, item in model.named_modules()},
                 {name: item.requires_grad for name, item in model.named_parameters()})
        self.assertIs(enforce_singleton_features(model), model)
        self.assertEqual(model.frozen_feature_policy, FROZEN_FEATURE_POLICY)
        self.assertEqual(digest(model.state_dict()), state)
        self.assertEqual(flags, ({name: item.training for name, item in model.named_modules()},
                                 {name: item.requires_grad for name, item in model.named_parameters()}))
        self.assertTrue(all(not cache for name, cache in vars(model).items()
                            if name.endswith("_cache") and isinstance(cache, dict)))
        model.visual(self.pngs); model.visual_grid(self.pngs)
        manifest = feature_cache_manifest(model); method = model.visual.__func__
        enforce_singleton_features(model)
        self.assertIs(model.visual.__func__, method)
        self.assertEqual(feature_cache_manifest(model), manifest)
        cloned = copy.deepcopy(model)
        clear_feature_caches(cloned)
        self.assertEqual(feature_cache_manifest(model), manifest)
        self.assertTrue(torch.equal(cloned.visual(self.pngs), model.visual(self.pngs)))

    def test_instance_bound_grid_override_and_every_image_cache_are_supported(self):
        from mutual_lab.neural import torch
        model = self.model(); original = model.visual_grid.__func__
        sizes = []

        def custom_grid(self, pngs):
            sizes.append(len(pngs))
            return original(self, pngs)

        model.visual_grid = MethodType(custom_grid, model)
        model.object_memory_cache = {b"old": torch.ones(2, 34)}
        enforce_singleton_features(model)
        self.assertEqual(model.object_memory_cache, {})
        model.visual_grid(self.pngs[:3])
        self.assertEqual(sizes, [1, 1, 1])
        counts = clear_feature_caches(model)
        self.assertEqual(counts["spatial_grid_cache"], 3)
        self.assertIn("object_memory_cache", counts)

    def test_manifest_ignores_insertion_order_and_detects_cached_tensor_mutation(self):
        model = enforce_singleton_features(self.model()); model.visual(self.pngs)
        before = feature_cache_manifest(model)
        values = list(model.visual_cache.items()); model.visual_cache.clear()
        model.visual_cache.update(reversed(values))
        self.assertEqual(feature_cache_manifest(model), before)
        next(iter(model.visual_cache.values()))[0] += 1
        after = feature_cache_manifest(model)
        self.assertNotEqual(after["sha256"], before["sha256"])
        self.assertEqual(after["caches"]["visual_cache"]["entries"], 4)


if __name__ == "__main__":
    unittest.main()
