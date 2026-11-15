"""Conditional spatial decoder identity, gradients and image-only boundaries."""

import importlib.util
import unittest
from unittest.mock import patch

from mutual_lab.attentive_spatial import build_attentive_spatial_model
from mutual_lab.images import scene_png
from mutual_lab.profiles import ModelProfile
from mutual_lab.spatial_prior import build_spatial_model, pretrain_prior, spatial_prior_hash
from mutual_lab.world import Object, native_captions, variants


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class AttentiveSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab.neural import configure_cpu
        configure_cpu()
        cls.scenes = variants((Object("blue", "square"), Object("red", "circle")))
        cls.pngs = [scene_png(scene) for scene in cls.scenes]
        cls.profile = ModelProfile("attentive-test", embedding_dim=8, hidden_dim=16, max_sequence_length=128)
        cls.bundle = pretrain_prior(cls.pngs, dim=16, steps=2, batch_size=4, eval_every=1)

    def models(self):
        return (build_spatial_model(self.profile, 59007, self.bundle),
                build_attentive_spatial_model(self.profile, 59007, self.bundle))

    def test_initial_language_tensors_and_logits_match_the_global_decoder_exactly(self):
        from mutual_lab.neural import digest, torch
        baseline, attentive = self.models()
        self.assertEqual(digest(baseline.understanding.state_dict()), digest(attentive.understanding.state_dict()))
        self.assertEqual(spatial_prior_hash(baseline), spatial_prior_hash(attentive))
        for name, parameter in baseline.producer.state_dict().items():
            self.assertTrue(torch.equal(parameter, attentive.producer.state_dict()[name]), name)
        texts = [native_captions(scene, "en")[0] for scene in self.scenes]
        ids, _ = baseline.tokenizer.batch(texts)
        expected, _ = baseline.producer(ids[:, :-1], baseline.visual(self.pngs))
        actual = attentive.teacher_forced_logits(ids[:, :-1], self.pngs)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(attentive.development_spec["decoder_extra_parameters"], 5488)
        self.assertFalse(attentive.development_spec["capacity_matched"])
        with self.assertRaisesRegex(ValueError, "requires its image grid"):
            attentive.producer(ids[:, :-1], attentive.visual(self.pngs))

    def test_grid_preserves_positions_and_all_visual_state_stays_frozen(self):
        from mutual_lab.neural import torch
        _, model = self.models()
        before = spatial_prior_hash(model)
        model.train()
        self.assertFalse(model.visual_encoder.training)
        grid = model.visual_grid(self.pngs)
        self.assertEqual(tuple(grid.shape), (4, 24, 34))
        self.assertFalse(grid.requires_grad)
        self.assertTrue(torch.equal(grid[0, :, -2:], model.spatial_coordinates))
        self.assertFalse(torch.equal(grid[0, :, :-2], grid[2, :, :-2]))
        self.assertTrue(torch.equal(model.visual_grid(self.pngs[::-1]), grid.flip(0)))
        self.assertEqual(spatial_prior_hash(model), before)

    def test_attention_learns_after_two_steps_without_changing_understanding_or_vision(self):
        from mutual_lab.neural import digest, torch
        _, model = self.models()
        before_u, before_v = digest(model.understanding.state_dict()), spatial_prior_hash(model)
        examples = [{"text": native_captions(scene, "en")[0], "image": png}
                    for scene, png in zip(self.scenes, self.pngs)]
        optimizer = torch.optim.Adam(model.producer.parameters(), lr=.003)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model.production_loss(examples)
            loss.backward()
            optimizer.step()
        for name in ("query.weight", "key.weight", "value.weight", "attention_output.weight"):
            parameter = dict(model.producer.named_parameters())[name]
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.norm()), 0, name)
        self.assertEqual(digest(model.understanding.state_dict()), before_u)
        self.assertEqual(spatial_prior_hash(model), before_v)

    def test_production_loss_uses_the_same_teacher_forced_grid_logits(self):
        from mutual_lab.neural import torch, F
        _, model = self.models()
        examples = [{"text": native_captions(scene, "ko")[0], "image": png}
                    for scene, png in zip(self.scenes, self.pngs)]
        ids, _ = model.tokenizer.batch([example["text"] for example in examples])
        logits = model.teacher_forced_logits(ids[:, :-1], self.pngs)
        expected = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=0)
        self.assertTrue(torch.equal(expected, model.production_loss(examples)))

    def test_generation_preserves_trace_flags_and_requires_the_current_image_grid(self):
        from mutual_lab.neural import torch
        _, model = self.models()
        # Invalid UTF-8 byte 0xff, then EOS. The trace must keep the raw byte.
        outputs = iter((258, 2))

        def forward(ids, visual, hidden=None, spatial=None):
            self.assertEqual(tuple(spatial.shape), (1, 24, 34))
            logits = torch.full((1, 1, model.tokenizer.vocab_size), -1000.)
            logits[0, 0, next(outputs)] = 1
            return logits, hidden

        with patch.object(model.producer, "forward", side_effect=forward):
            trace = model.generate_with_trace(self.pngs[0])
        self.assertTrue(trace["terminated"])
        self.assertTrue(trace["invalid_encoding"])
        self.assertEqual(trace["token_ids"], [258])
        self.assertEqual(trace["text"], "")
        self.assertEqual(trace["special_tokens_in_body"], 0)
        self.assertEqual(trace["output_units"], 1)


if __name__ == "__main__":
    unittest.main()
