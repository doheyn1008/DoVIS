"""Actual tiny Torch checks for the conditional G understanding prototype."""

import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
from mutual_lab import understanding_memory as prototype


def png(index):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (96, 64), (246, 246, 239))
    draw = ImageDraw.Draw(image)
    red, blue = (193, 41, 97), (13, 89, 207)
    colors = (red, blue) if index % 2 == 0 else (blue, red)
    boxes = ((8, 12, 28, 36), (58, 17, 81, 42)) if index < 2 else ((14, 5, 35, 23), (57, 37, 80, 58))
    draw.rectangle(boxes[0], fill=colors[0]); draw.ellipse(boxes[1], fill=colors[1])
    output = io.BytesIO(); image.save(output, format="PNG")
    return output.getvalue()


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class UnderstandingMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab.neural import configure_cpu
        from mutual_lab.profiles import ModelProfile
        from mutual_lab.spatial_prior import pretrain_prior
        configure_cpu()
        cls.pngs = [png(index) for index in range(4)]
        cls.profile = ModelProfile("understanding-memory-test", embedding_dim=8, hidden_dim=16, max_sequence_length=128)
        cls.bundle = pretrain_prior(cls.pngs, dim=16, steps=2, batch_size=4, eval_every=1)
        cls.texts = ["a red object left of a blue object", "blue below red"]

    def model(self, understanding="objects", producer_memory="objects"):
        return prototype.build_understanding_memory_model(self.profile, 89007, self.bundle,
            understanding=understanding, producer_memory=producer_memory)

    def active_model(self):
        from mutual_lab.neural import torch
        model = self.model()
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(89011)
            model.understanding.residual_output.weight.normal_(std=.5)
        return model

    def test_initial_cosine_logits_base_weights_producer_and_rng_match_canonical_baseline(self):
        from mutual_lab.frozen_features import enforce_singleton_features
        from mutual_lab.neural import digest, torch
        from mutual_lab.object_memory import build_object_memory_model
        for producer_memory in ("cnn", "objects"):
            reference = build_object_memory_model(self.profile, 89007, self.bundle, memory=producer_memory)
            enforce_singleton_features(reference)
            # Different cache population histories must not alter frozen inputs.
            reference.visual(self.pngs[:2]); reference.visual_grid(self.pngs[2:][::-1])
            ids, lengths = reference.tokenizer.batch(self.texts)
            images = reference.visual(self.pngs * 2).reshape(2, 4, -1)
            with torch.no_grad():
                expected = torch.einsum("bh,bkh->bk", reference.understanding(ids, lengths), images) / .1
            for mode in ("global", "objects"):
                rng = torch.random.get_rng_state().clone()
                model = self.model(mode, producer_memory)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
                model.visual(self.pngs[1:]); model.visual_grid(self.pngs[:1])
                self.assertEqual(digest(model.understanding.base.state_dict()), digest(reference.understanding.state_dict()))
                self.assertEqual(digest(model.producer.state_dict()), digest(reference.producer.state_dict()))
                with torch.no_grad():
                    parts = model.comprehension_logit_parts(self.texts, [self.pngs] * 2)
                self.assertTrue(torch.equal(parts["total"], expected), (producer_memory, mode))
                self.assertTrue(torch.equal(parts["base"], expected))
                self.assertEqual(float(parts["residual"].abs().sum()), 0)
                self.assertEqual(model.understanding_memory_metadata["understanding_extra_parameter_count"], 1360 if mode == "objects" else 0)
                self.assertFalse(model.understanding_memory_metadata["capacity_matched"])
                self.assertEqual(model.development_spec["parameter_count"], sum(p.numel() for p in model.parameters()))
                self.assertEqual(model.development_spec["trainable_parameter_count"], sum(p.numel() for p in model.parameters() if p.requires_grad))
                self.assertTrue(all(not p.requires_grad for p in model.visual_encoder.parameters()))

    def test_shared_logits_drive_binary_and_four_choice_loss_predict_and_one_gru_pass(self):
        from mutual_lab.neural import F, torch
        model = self.active_model()
        for count in (2, 4):
            choices = [self.pngs[:count]] * 2
            with patch.object(model.understanding.base.gru, "forward", wraps=model.understanding.base.gru.forward) as gru:
                logits = model.comprehension_logits(self.texts, choices)
                self.assertEqual(gru.call_count, 1)
            examples = [{"text": text, "choices": row, "answer": answer}
                        for text, row, answer in zip(self.texts, choices, (0, 1))]
            self.assertTrue(torch.equal(model.grounding_loss(examples), F.cross_entropy(logits, torch.tensor([0, 1]))))
            for text in self.texts:
                expected = model.comprehension_logits([text], [self.pngs[:count]])[0].argmax()
                with patch.object(model, "comprehension_logits", wraps=model.comprehension_logits) as shared:
                    self.assertEqual(model.predict(text, self.pngs[:count]), int(expected))
                    shared.assert_called_once()
        ids, lengths = model.tokenizer.batch(self.texts)
        with self.assertRaisesRegex(ValueError, "requires comprehension_logits"):
            model.understanding(ids, lengths)

    def test_nonzero_object_context_is_choice_equivariant_and_object_permutation_invariant(self):
        from mutual_lab.neural import torch
        model = self.active_model()
        choices = [self.pngs] * 2
        parts = model.comprehension_logit_parts(self.texts, choices)
        self.assertGreater(float(parts["residual"].detach().abs().max()), 1e-4)
        self.assertGreater(float((parts["residual"][0] - parts["residual"][1]).detach().abs().max()), 1e-5)
        order = [2, 0, 3, 1]
        changed = model.comprehension_logits(self.texts, [[row[index] for index in order] for row in choices])
        self.assertTrue(torch.allclose(changed, parts["total"][:, order], atol=1e-6, rtol=1e-5))
        original = model.understanding_objects
        with patch.object(model, "understanding_objects", side_effect=lambda images: original(images).flip(1)):
            reversed_objects = model.comprehension_logits(self.texts, choices)
        self.assertTrue(torch.allclose(reversed_objects, parts["total"], atol=1e-6, rtol=1e-5))

    def test_pad_and_batch_single_parity_after_nonzero_head(self):
        from mutual_lab.neural import torch
        model = self.active_model()
        ids, lengths = model.tokenizer.batch(self.texts)
        choices = [self.pngs] * 2
        visual = model.visual(self.pngs * 2).reshape(2, 4, -1)
        objects = model.understanding_objects(self.pngs * 2).reshape(2, 4, 2, 34)
        expected = model.understanding(ids, lengths, visual, objects)
        padded = torch.cat((ids, torch.zeros(len(ids), 11, dtype=torch.long)), -1)
        self.assertTrue(torch.allclose(model.understanding(padded, lengths, visual, objects), expected, atol=1e-6, rtol=1e-5))
        batch = model.comprehension_logits(self.texts, choices)
        singles = torch.cat([model.comprehension_logits([text], [self.pngs]) for text in self.texts])
        self.assertTrue(torch.allclose(batch, singles, atol=1e-6, rtol=1e-5))

    def test_two_understanding_updates_reach_attention_but_preserve_all_producer_outputs_and_caches(self):
        from mutual_lab.frozen_features import feature_cache_manifest
        from mutual_lab.neural import digest, torch
        from mutual_lab.spatial_prior import spatial_prior_hash
        for source in ("cnn", "objects"):
            model = self.model(producer_memory=source)
            choices = [self.pngs] * 2
            model.comprehension_logits(self.texts, choices)
            ids, _ = model.tokenizer.batch(["producer first", "producer second", "producer third", "producer fourth"])
            with torch.no_grad():
                before_logits = model.teacher_forced_logits(ids[:, :-1], self.pngs)
                before_trace = model.generate_with_trace(self.pngs[0])
            before_p, before_v = digest(model.producer.state_dict()), spatial_prior_hash(model)
            before_cache = feature_cache_manifest(model)
            examples = [{"text": text, "choices": self.pngs, "answer": target}
                        for text, target in zip(self.texts, (0, 3))]
            optimizer = torch.optim.Adam(model.understanding.parameters(), lr=.003)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                model.grounding_loss(examples).backward()
                optimizer.step()
            for name in ("query.weight", "key.weight", "value.weight", "residual_output.weight"):
                self.assertGreater(float(dict(model.understanding.named_parameters())[name].grad.norm()), 0, (source, name))
            self.assertEqual(digest(model.producer.state_dict()), before_p)
            self.assertEqual(spatial_prior_hash(model), before_v)
            self.assertEqual(feature_cache_manifest(model), before_cache)
            with torch.no_grad():
                self.assertTrue(torch.equal(model.teacher_forced_logits(ids[:, :-1], self.pngs), before_logits))
                self.assertEqual(model.generate_with_trace(self.pngs[0]), before_trace)
            self.assertTrue(all(parameter.grad is None for parameter in model.producer.parameters()))

    def test_raw_object_tokens_cache_read_only_and_invalid_cnn_grid_nan_inputs_fail_closed(self):
        from mutual_lab import object_memory
        from mutual_lab.neural import digest, torch
        model = self.model()
        before = digest(model.state_dict())
        with patch.object(object_memory, "component_descriptors", wraps=object_memory.component_descriptors) as extract:
            slots = model.understanding_objects(self.pngs[:2])
            self.assertEqual(tuple(slots.shape), (2, 2, 34))
            self.assertTrue(torch.equal(model.understanding_objects(self.pngs[:2][::-1]), slots.flip(0)))
            self.assertEqual(extract.call_count, 2)
        self.assertFalse(slots.requires_grad)
        slots.zero_()
        self.assertGreater(float(model.understanding_objects(self.pngs[:1]).abs().sum()), 0)
        self.assertEqual(digest(model.state_dict()), before)
        fresh = self.model()
        with patch.object(object_memory, "component_descriptors", return_value=torch.zeros(24, 34)):
            with self.assertRaisesRegex(ValueError, "two finite"):
                fresh.understanding_objects(self.pngs[:1])
        with patch.object(object_memory, "component_descriptors", return_value=torch.full((2, 34), float("nan"))):
            with self.assertRaisesRegex(ValueError, "two finite"):
                fresh.understanding_objects(self.pngs[:1])
        with self.assertRaisesRegex(ValueError, "PNG bytes"):
            fresh.comprehension_logits(["text"], [["one.png", "two.png"]])
        ids, lengths = fresh.tokenizer.batch(["text"])
        visual = fresh.visual(self.pngs[:2])[None]
        with self.assertRaisesRegex(ValueError, "never CNN grid"):
            fresh.understanding(ids, lengths, visual, torch.zeros(1, 2, 24, 34))

    def test_c_development_evaluation_and_binary_contrasts_use_image_aware_overrides(self):
        from mutual_lab.config import ExperimentConfig
        from mutual_lab import development_world as world
        model = self.model()
        config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        full = world.development_splits(config, 59003)
        splits = {key: scenes[:4] for key, scenes in full.items()}
        examples = world.make_examples(model, splits["dev"], "en")
        probes = world.build_development_contrasts(config, splits)
        with patch.object(model, "comprehension_logits", wraps=model.comprehension_logits) as shared:
            metrics, _ = world.evaluate_development(model, examples, splits["dev"], "en", 4)
            before = shared.call_count
            diagnostic = world.score_development_contrasts(model, probes, "en")
            self.assertGreater(before, 0)
            self.assertGreater(shared.call_count, before)
        self.assertEqual(metrics["comprehension_items"], 16)
        self.assertTrue(diagnostic["categories"])


if __name__ == "__main__":
    unittest.main()
