"""Image-only provenance, matched language initialization and a frozen spatial prior."""

import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mutual_lab.images import scene_png
from mutual_lab.profiles import ModelProfile
from mutual_lab.spatial_prior import (
    build_spatial_model, create_spatial_prior, fit_pool_hash, load_prior_bundle,
    png_tensor, pretrain_prior, reconstruction_losses, save_prior_bundle, spatial_prior_hash,
)
from mutual_lab.world import Object, variants


def fit_images():
    return [scene_png(scene) for scene in variants((Object("blue", "square"), Object("red", "circle")))]


class PriorInputTests(unittest.TestCase):
    def test_membership_hash_ignores_order_but_rejects_duplicate_or_nonpixel_inputs(self):
        images = fit_images()
        self.assertEqual(fit_pool_hash(images), fit_pool_hash(images[::-1]))
        self.assertNotEqual(fit_pool_hash(images), fit_pool_hash(images[:-1]))
        for invalid in ([], ["red-circle.png"], [Object("red", "circle")],
                        [b"not a PNG"], images + images[:1]):
            with self.subTest(invalid=type(invalid)), self.assertRaises(ValueError):
                fit_pool_hash(invalid)

    def test_invalid_budgets_fail_before_optional_tensor_work(self):
        images = fit_images()
        for kwargs in ({"steps": 0}, {"steps": True}, {"batch_size": 0}, {"seed": True},
                       {"dim": 0}, {"max_seconds": 0}, {"learning_rate": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pretrain_prior(images, **kwargs)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class SpatialPriorNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab.neural import configure_cpu
        configure_cpu()
        cls.images = fit_images()
        cls.bundle = pretrain_prior(cls.images, seed=49001, dim=16, steps=3,
                                   batch_size=4, eval_every=1, max_seconds=120)
        cls.profile = ModelProfile("spatial-prior-test", embedding_dim=8, hidden_dim=16)

    def test_encoder_keeps_a_spatial_grid_and_restores_the_callers_rng(self):
        from mutual_lab.neural import torch
        torch.manual_seed(719)
        before = torch.random.get_rng_state().clone()
        encoder = create_spatial_prior(seed=49001, dim=16)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        pixels = png_tensor(self.images)
        self.assertEqual(tuple(encoder.features(pixels).shape), (4, 32, 4, 6))
        features = encoder(pixels)
        self.assertEqual(tuple(features.shape), (4, 16))
        self.assertTrue(torch.allclose(features.norm(dim=1), torch.ones(4), atol=1e-6))
        # Moving which object occupies each position is not globally averaged away.
        self.assertFalse(torch.allclose(features[0], features[2], atol=1e-7, rtol=1e-7))
        self.assertEqual(len(list(encoder.buffers())), 0)

    def test_pixel_masks_weight_foreground_and_background_separately(self):
        from mutual_lab.neural import torch
        target = torch.ones(1, 3, 4, 4)
        target[:, :, 1:3, 1:3] = 0
        total, foreground, background = reconstruction_losses(torch.ones_like(target), target)
        self.assertEqual(float(foreground), 1)
        self.assertEqual(float(background), 0)
        self.assertEqual(float(total), .5)

    def test_pretraining_changes_only_an_image_model_and_records_the_exact_fit_pool(self):
        report = self.bundle["report"]
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["steps_completed"], 3)
        self.assertEqual(report["image_count"], 4)
        self.assertEqual(report["fit_png_sha256"], sorted(hashlib.sha256(png).hexdigest() for png in self.images))
        self.assertEqual(report["fit_pool_sha256"], fit_pool_hash(self.images))
        self.assertNotEqual(report["random_prior_sha256"], report["trained_prior_sha256"])
        self.assertEqual([row["step"] for row in report["curves"]], [0, 1, 2, 3])
        self.assertTrue(report["excluded_from_research"])
        self.assertTrue(self.bundle["optimizer_state"]["state"])
        for row in report["curves"]:
            self.assertTrue(all(math.isfinite(row[key]) for key in ("loss", "foreground_mse", "background_mse")))
            self.assertGreater(row["foreground_pixels"], 0)
            self.assertGreater(row["background_pixels"], 0)

    def test_both_conditions_share_language_weights_and_keep_all_visual_state_frozen(self):
        from mutual_lab.neural import NativeModel, digest, native_examples, torch
        original = NativeModel(self.profile, 49007)
        models = [build_spatial_model(self.profile, 49007, self.bundle, condition) for condition in ("B0", "B1")]
        for model in models:
            self.assertEqual(digest(model.understanding.state_dict()), digest(original.understanding.state_dict()))
            self.assertEqual(digest(model.producer.state_dict()), digest(original.producer.state_dict()))
            self.assertTrue(torch.equal(model.visual_projection, original.visual_projection))
            self.assertTrue(all(not parameter.requires_grad for parameter in model.visual_encoder.parameters()))
            model.train()
            self.assertFalse(model.visual_encoder.training)
            before = spatial_prior_hash(model)
            examples = native_examples(model,
                variants((Object("blue", "square"), Object("red", "circle"))), "ko")
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            optimizer.zero_grad(set_to_none=True)
            (model.grounding_loss(examples) + model.production_loss(examples)).backward()
            optimizer.step()
            self.assertEqual(spatial_prior_hash(model), before)
            self.assertNotEqual(digest(model.understanding.state_dict()), digest(original.understanding.state_dict()))
            self.assertNotEqual(digest(model.producer.state_dict()), digest(original.producer.state_dict()))
            self.assertIn("visual_encoder.features.0.weight", model.state_dict())
        self.assertNotEqual(spatial_prior_hash(models[0]), spatial_prior_hash(models[1]))

    def test_manifest_checks_and_nonresearch_format_guard_the_saved_prior(self):
        from mutual_lab.neural import load_checkpoint, torch
        with tempfile.TemporaryDirectory() as temporary:
            path = save_prior_bundle(self.bundle, Path(temporary) / "prior.pt")
            loaded = load_prior_bundle(path, expected_fit_pool_sha256=fit_pool_hash(self.images))
            self.assertEqual(loaded["report"], self.bundle["report"])
            with self.assertRaisesRegex(ValueError, "different fit"):
                load_prior_bundle(path, expected_fit_pool_sha256="different")
            with patch("mutual_lab.neural.torch.load") as deserialize:
                for expected in ("native", "acquisition"):
                    with self.assertRaises(ValueError):
                        load_checkpoint(path, expected_kind=expected)
                deserialize.assert_not_called()
            manifest = json.loads(path.with_suffix(".json").read_text())
            manifest["sha256"] = "wrong"
            path.with_suffix(".json").write_text(json.dumps(manifest))
            with patch.object(torch, "load") as deserialize:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    load_prior_bundle(path)
                deserialize.assert_not_called()

    def test_incomplete_pretraining_is_not_accepted_as_a_completed_b1_condition(self):
        partial = copy.deepcopy(self.bundle)
        partial["report"]["status"] = "time_limit"
        with self.assertRaisesRegex(ValueError, "budget to complete"):
            build_spatial_model(self.profile, 49007, partial, "B1")
        build_spatial_model(self.profile, 49007, partial, "B0")


if __name__ == "__main__":
    unittest.main()
