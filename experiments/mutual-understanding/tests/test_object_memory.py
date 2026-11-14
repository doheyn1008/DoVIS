"""Small actual-Torch checks for the unactivated PNG memory-provider prototype."""

import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
from mutual_lab import object_memory as memory


def picture(color=(17, 89, 203), shape="rectangle", shift=(0, 0), same_color=False):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (96, 64), (246, 246, 239))
    draw = ImageDraw.Draw(canvas)
    dx, dy = shift
    first = (10 + dx, 13 + dy, 30 + dx, 39 + dy)
    second = (57 + dx, 15 + dy, 78 + dx, 40 + dy)
    if shape == "rectangle":
        draw.rectangle(first, fill=color)
    elif shape == "ellipse":
        draw.ellipse(first, fill=color)
    else:
        draw.polygon([(10 + dx, 14 + dy), (28 + dx, 13 + dy), (23 + dx, 21 + dy),
                      (30 + dx, 34 + dy), (18 + dx, 39 + dy), (12 + dx, 29 + dy)], fill=color)
    draw.ellipse(second, fill=color if same_color else (191, 47, 111))
    output = io.BytesIO(); canvas.save(output, format="PNG")
    return output.getvalue()


def ordered(descriptors):
    # Alignment is test-only; the actual provider never ranks components.
    return descriptors[descriptors[:, -1].argsort()]


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class ObjectMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab.neural import configure_cpu
        from mutual_lab.profiles import ModelProfile
        from mutual_lab.spatial_prior import pretrain_prior
        configure_cpu()
        cls.pngs = [picture(), picture(shape="ellipse"), picture(shape="irregular"), picture(color=(23, 177, 61))]
        cls.profile = ModelProfile("object-memory-test", embedding_dim=8, hidden_dim=16, max_sequence_length=128)
        cls.bundle = pretrain_prior(cls.pngs, dim=16, steps=2, batch_size=4, eval_every=1)

    def model(self, source="objects"):
        return memory.build_object_memory_model(self.profile, 79007, self.bundle, memory=source)

    def test_arbitrary_rgb_geometry_and_actual_centroids_remain_continuous(self):
        from mutual_lab.neural import torch
        original = ordered(memory.component_descriptors(self.pngs[0]))
        color_changed = ordered(memory.component_descriptors(self.pngs[3]))
        self.assertEqual(tuple(original.shape), (2, 34))
        self.assertTrue(torch.allclose(original[0, :3], torch.tensor([17, 89, 203]) / 255))
        self.assertTrue(torch.equal(original[0, 3:], color_changed[0, 3:]))
        self.assertFalse(torch.equal(original[0, :3], color_changed[0, :3]))
        for png in self.pngs[1:3]:
            changed = ordered(memory.component_descriptors(png))
            self.assertTrue(torch.equal(original[0, :3], changed[0, :3]))
            self.assertFalse(torch.equal(original[0, 3:32], changed[0, 3:32]))
        translated = ordered(memory.component_descriptors(picture(shift=(5, 4))))
        self.assertTrue(torch.allclose(original[:, :32], translated[:, :32], atol=1e-7))
        expected_delta = torch.tensor([8 / 64, 10 / 96]).expand(2, -1)
        self.assertTrue(torch.allclose(translated[:, -2:] - original[:, -2:], expected_delta, atol=1e-7))
        # An off-diagonal centroid catches swapped axes; diagonal points cannot.
        expected_yx = torch.tensor([2 * 26.5 / 64 - 1, 2 * 20.5 / 96 - 1])
        self.assertTrue(torch.allclose(original[0, -2:], expected_yx, atol=1e-7))
        self.assertNotEqual(float(expected_yx[0]), float(expected_yx[1]))

    def test_identical_color_objects_and_tiny_components_are_not_collapsed_or_nan(self):
        from mutual_lab.neural import Image, torch
        same = memory.component_descriptors(picture(same_color=True))
        self.assertEqual(tuple(same.shape), (2, 34))
        self.assertTrue(torch.equal(same[0, :3], same[1, :3]))
        canvas = Image.new("RGB", (20, 20), "white")
        canvas.putpixel((4, 4), (19, 31, 151)); canvas.putpixel((14, 14), (19, 31, 151))
        output = io.BytesIO(); canvas.save(output, format="PNG")
        tiny = ordered(memory.component_descriptors(output.getvalue()))
        self.assertTrue(bool(torch.isfinite(tiny).all()))
        self.assertTrue(torch.allclose(tiny[:, 6:30], torch.ones(2, 24)))
        self.assertTrue(torch.equal(tiny[:, 30:32], torch.zeros(2, 2)))
        self.assertTrue(torch.allclose(tiny[0, -2:], torch.tensor([-.55, -.55])))

    def test_component_assumption_violations_raise_without_silently_filtering_inputs(self):
        from PIL import Image, ImageDraw
        for count in (0, 1, 3):
            image = Image.new("RGB", (64, 32), "white")
            draw = ImageDraw.Draw(image)
            for offset in range(count):
                draw.rectangle((3 + offset * 18, 5, 12 + offset * 18, 20), fill=(51, 93, 141))
            output = io.BytesIO(); image.save(output, format="PNG")
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "exactly two"):
                memory.component_descriptors(output.getvalue())
        touching = Image.new("RGB", (64, 32), "white")
        draw = ImageDraw.Draw(touching)
        draw.rectangle((5, 5, 20, 20), fill=(20, 70, 150))
        draw.rectangle((21, 5, 36, 20), fill=(170, 40, 80))
        output = io.BytesIO(); touching.save(output, format="PNG")
        with self.assertRaisesRegex(ValueError, "exactly two"):
            memory.component_descriptors(output.getvalue())
        for value in ("scene.png", Path("scene.png"), {"shape": "circle"}, b"not a PNG"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "PNG bytes only"):
                memory.component_descriptors(value)
        transparent = Image.new("RGBA", (24, 24), (255, 255, 255, 0))
        output = io.BytesIO(); transparent.save(output, format="PNG")
        with self.assertRaisesRegex(ValueError, "opaque"):
            memory.component_descriptors(output.getvalue())

    def test_current_development_pngs_all_satisfy_the_declared_component_assumption(self):
        from mutual_lab.config import ExperimentConfig
        from mutual_lab.development_world import development_splits, render_scene
        from mutual_lab.neural import torch
        config = ExperimentConfig.load(PACKAGE_ROOT / "configs/neural.example.json")
        splits = development_splits(config, 59003)
        count = 0
        for scenes in splits.values():
            for scene in scenes:
                # The provider receives only the bytes; these Scene values are
                # test-fixture generation, never descriptor/model arguments.
                descriptors = memory.component_descriptors(render_scene(scene))
                self.assertEqual(tuple(descriptors.shape), (2, 34))
                self.assertTrue(bool(torch.isfinite(descriptors).all()))
                count += 1
        self.assertEqual(count, 332)

    def test_both_sources_use_common_feature_normalization_and_keep_centroids_raw(self):
        from mutual_lab.neural import F, torch
        for source in ("cnn", "objects"):
            model = self.model(source)
            grid = model.visual_grid(self.pngs[:2])
            self.assertEqual(tuple(grid.shape), (2, 24, 34))
            raw = (model.object_memory_cnn_grid(self.pngs[:2]) if source == "cnn" else
                   torch.stack([memory.component_descriptors(png).repeat_interleave(12, 0) for png in self.pngs[:2]]))
            self.assertTrue(torch.equal(grid[:, :, :32], F.normalize(raw[:, :, :32], dim=-1)))
            self.assertTrue(torch.equal(grid[:, :, 32:], raw[:, :, 32:]))
            self.assertEqual(model.object_memory_metadata["coordinate_order"], "y,x")
            self.assertEqual(model.object_memory_metadata["memory_normalization"], "l2_first32_raw_yx_v1")
            if source == "cnn":
                self.assertTrue(torch.allclose(grid[0, 1, -2:], torch.tensor([-1., -.6])))
            if source == "objects":
                self.assertTrue(torch.equal(grid[:, :12], grid[:, :1].expand(-1, 12, -1)))
                self.assertTrue(torch.equal(grid[:, 12:], grid[:, 12:13].expand(-1, 12, -1)))
            self.assertIn("NEW baseline", model.object_memory_metadata["baseline"])

    def test_registered_state_flags_global_vision_and_initial_logits_equal_old_attentive(self):
        from mutual_lab.attentive_spatial import build_attentive_spatial_model
        from mutual_lab.neural import digest, torch
        original = build_attentive_spatial_model(self.profile, 79007, self.bundle)
        ids, _ = original.tokenizer.batch(["two objects", "another geometry"])
        before = digest(original.state_dict())
        with torch.no_grad():
            expected = original.teacher_forced_logits(ids[:, :-1], self.pngs[:2])
        for source in ("cnn", "objects"):
            model = self.model(source)
            self.assertEqual(digest(model.state_dict()), before)
            self.assertEqual([(n, p.requires_grad) for n, p in original.named_parameters()],
                             [(n, p.requires_grad) for n, p in model.named_parameters()])
            self.assertEqual(sum(p.numel() for p in original.parameters()), sum(p.numel() for p in model.parameters()))
            with torch.no_grad():
                self.assertTrue(torch.equal(original.visual(self.pngs), model.visual(self.pngs)))
                actual = model.teacher_forced_logits(ids[:, :-1], self.pngs[:2])
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(digest(model.state_dict()), before)

    def test_object_memory_cache_is_png_keyed_read_only_and_calls_no_cnn_provider(self):
        from mutual_lab.neural import digest, torch
        model = self.model()
        before = digest(model.state_dict())
        with patch.object(memory, "component_descriptors", wraps=memory.component_descriptors) as extract, patch.object(
                model, "object_memory_cnn_grid", side_effect=AssertionError("objects must not call CNN grid")):
            forward = model.visual_grid(self.pngs[:2])
            reverse = model.visual_grid(self.pngs[:2][::-1])
            self.assertEqual(extract.call_count, 2)
        self.assertTrue(torch.equal(reverse, forward.flip(0)))
        self.assertFalse(forward.requires_grad)
        forward.zero_()
        self.assertGreater(float(model.visual_grid(self.pngs[:1]).abs().sum()), 0)
        self.assertEqual(digest(model.state_dict()), before)

    def test_object_order_and_all_token_permutations_are_irrelevant_after_nonzero_readout(self):
        from mutual_lab.neural import torch
        model = self.model()
        pngs = self.pngs[:1]
        ids, _ = model.tokenizer.batch(["describe the two objects"])
        with torch.no_grad():
            zero_head = model.teacher_forced_logits(ids[:, :-1], pngs)
            with torch.random.fork_rng():
                torch.manual_seed(79011)
                model.producer.attention_output.weight.normal_(std=.2)
            expected = model.teacher_forced_logits(ids[:, :-1], pngs)
            grid, trace = model.visual_grid(pngs), model.generate_with_trace(pngs[0])
            self.assertGreater(float((expected - zero_head).abs().max()), 1e-4)
            with torch.random.fork_rng():
                torch.manual_seed(79013)
                shuffled = torch.randperm(24)
            orders = (torch.cat((torch.arange(12, 24), torch.arange(12))), torch.arange(23, -1, -1), shuffled)
            for order in orders:
                with patch.object(model, "visual_grid", return_value=grid[:, order]):
                    actual = model.teacher_forced_logits(ids[:, :-1], pngs)
                    self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
                    self.assertEqual(model.generate_with_trace(pngs[0]), trace)

    def test_two_producer_steps_reach_attention_without_changing_understanding_or_visual_state(self):
        from mutual_lab.neural import digest, torch
        from mutual_lab.spatial_prior import spatial_prior_hash
        for source in ("cnn", "objects"):
            model = self.model(source)
            before_u, before_v = digest(model.understanding.state_dict()), spatial_prior_hash(model)
            examples = [{"image": png, "text": text} for png, text in zip(self.pngs[:2], ("first image", "second image"))]
            optimizer = torch.optim.Adam(model.producer.parameters(), lr=.003)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                model.production_loss(examples).backward()
                optimizer.step()
            for name in ("query.weight", "key.weight", "value.weight", "attention_output.weight"):
                self.assertGreater(float(dict(model.producer.named_parameters())[name].grad.norm()), 0, (source, name))
            self.assertEqual(digest(model.understanding.state_dict()), before_u)
            self.assertEqual(spatial_prior_hash(model), before_v)
            self.assertTrue(all(p.grad is None for p in model.understanding.parameters()))
            self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
