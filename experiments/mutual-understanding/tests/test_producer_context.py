"""Actual H128 few-step preparation; does not test or claim native qualification."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
from mutual_lab import producer_context as module


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "numpy", "PIL")),
                     "requires optional CPU torch, NumPy and Pillow")
class ProducerContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import development_world as world
        from mutual_lab.config import ExperimentConfig
        from mutual_lab.neural import configure_cpu
        from mutual_lab.profiles import ModelProfile
        from mutual_lab.spatial_prior import pretrain_prior
        configure_cpu()
        cls.world = world
        cls.profile = ModelProfile("h128-prototype", embedding_dim=64, hidden_dim=128, max_sequence_length=128)
        cls.splits = world.development_splits(ExperimentConfig.load(ROOT / "configs/neural.example.json"), 59003)
        cls.pngs = tuple(world.render_scene(scene) for values in cls.splits.values() for scene in values)
        cls.bundle = pretrain_prior([world.render_scene(scene) for scene in cls.splits["train"]],
            dim=128, steps=1, batch_size=4, eval_every=1)

    def models(self):
        return [module.build_producer_context_model(self.profile, 79007, self.bundle, context=source)
                for source in ("cnn", "object_moments")]

    def test_all332_pngs_cache_exactly_both_arms_and_moments_are_unit_permutation_invariant(self):
        from mutual_lab.neural import digest, torch
        from mutual_lab.frozen_features import clear_feature_caches
        a, b = self.models()
        self.assertEqual(digest(a.state_dict()), digest(b.state_dict()))
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        tables = [module.preflight_context(model, self.pngs) for model in (a, b)]
        self.assertEqual(tables[0], tables[1])
        self.assertEqual(tables[0]["caches"]["producer_context_cache"]["entries"], 332)
        values = b.understanding_objects([self.pngs[0]])
        self.assertTrue(torch.equal(module.position_moments(values), module.position_moments(values.flip(1))))
        vector = module.position_moments(values)
        self.assertTrue(torch.allclose(vector.norm(dim=-1), torch.ones(1)))
        self.assertTrue(torch.equal(vector[:, 102:], torch.zeros(1, 26)))
        # Both H0/H1 warm moments again after the actual benchmark cache reset.
        for model in (a, b):
            clear_feature_caches(model)
            model.producer_visual([self.pngs[0]])
            self.assertEqual(len(model.producer_context_cache), 1)

    def test_h0_exact_source_g1_logits_and_generation_h1_uses_producer_hook_without_mutating_visual(self):
        from mutual_lab.understanding_memory import build_understanding_memory_model
        from mutual_lab.neural import digest, torch
        source = build_understanding_memory_model(self.profile, 79007, self.bundle, understanding="objects", producer_memory="objects")
        a, b = self.models()
        text = self.world.captions(self.splits["train"][0], "ja")[0]
        ids, _ = a.tokenizer.batch([text])
        self.assertEqual(digest(a.state_dict()), digest(source.state_dict()))
        with torch.no_grad():
            self.assertTrue(torch.equal(source.teacher_forced_logits(ids[:, :-1], [self.pngs[0]]),
                                        a.teacher_forced_logits(ids[:, :-1], [self.pngs[0]])))
            self.assertFalse(torch.equal(a.teacher_forced_logits(ids[:, :-1], [self.pngs[0]]),
                                         b.teacher_forced_logits(ids[:, :-1], [self.pngs[0]])))
        self.assertEqual(a.generate_with_trace(self.pngs[0]), source.generate_with_trace(self.pngs[0]))
        before_visual = b.visual
        example = self.world.make_examples(b, self.splits["train"][:1], "ja")
        with patch.object(b, "producer_visual", wraps=b.producer_visual) as hook, patch.object(b, "visual", wraps=b.visual) as global_visual:
            b.teacher_forced_logits(ids[:, :-1], [self.pngs[0]])
            b.production_loss(example)
            b.generate_with_trace(self.pngs[0])
            self.assertEqual(hook.call_count, 3)
            self.assertEqual(global_visual.call_count, 0)
        self.assertEqual(b.visual, before_visual)
        with torch.no_grad():
            self.assertTrue(torch.equal(a.comprehension_logits([text], [list(self.pngs[:4])]),
                                        b.comprehension_logits([text], [list(self.pngs[:4])])))

    def test_actual_h128_three_updates_leave_final_understanding_bitexact_and_tables_immutable(self):
        from mutual_lab.neural import digest, nn, torch
        from mutual_lab.frozen_features import feature_cache_manifest
        a, b = self.models()
        tables = [module.preflight_context(model, self.pngs) for model in (a, b)]
        for model in (a, b):
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            examples = self.world.make_examples(model, self.splits["train"][:4], "ja")
            for offset in range(3):
                batch = examples[offset * 4:(offset + 1) * 4]
                production = [dict(row, text=row["_first_caption"]) for row in batch]
                optimizer.zero_grad(set_to_none=True)
                loss = model.grounding_loss(batch) + model.production_loss(production)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                nn.utils.clip_grad_norm_(model.understanding.parameters(), 1., error_if_nonfinite=True)
                nn.utils.clip_grad_norm_(model.producer.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
        self.assertEqual(digest(a.understanding.state_dict()), digest(b.understanding.state_dict()))
        self.assertNotEqual(digest(a.producer.state_dict()), digest(b.producer.state_dict()))
        self.assertEqual(feature_cache_manifest(a), tables[0])
        self.assertEqual(feature_cache_manifest(b), tables[1])

    def test_only_png_two_objects_and_h128_are_accepted(self):
        from dataclasses import replace
        from mutual_lab.neural import torch
        with self.assertRaises(ValueError):
            module.build_producer_context_model(replace(self.profile, hidden_dim=64), 79007, self.bundle)
        model = self.models()[1]
        with self.assertRaises(ValueError):
            model.producer_visual([self.splits["train"][0]])
        with self.assertRaises(ValueError):
            module.position_moments(torch.ones(1, 24, 34))

    def test_swapping_colors_at_fixed_same_shape_positions_changes_moments_while_mean_is_unchanged(self):
        from mutual_lab.neural import torch
        from mutual_lab.world import Object, Scene
        model = self.models()[1]
        first = Scene(Object("red", "circle"), "left_of", Object("blue", "circle"))
        swapped = Scene(Object("blue", "circle"), "left_of", Object("red", "circle"))
        tokens = model.understanding_objects([self.world.render_scene(first), self.world.render_scene(swapped)])
        self.assertTrue(torch.allclose(tokens[0].mean(0), tokens[1].mean(0), atol=1e-7))
        self.assertFalse(torch.allclose(module.position_moments(tokens[:1]), module.position_moments(tokens[1:])))


if __name__ == "__main__":
    unittest.main()
