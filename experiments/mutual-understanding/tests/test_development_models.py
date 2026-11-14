"""Encoder isolation, capacity matching and actual directional gradient checks."""

import importlib.util
import unittest

from mutual_lab.development_models import build_development_model, development_model_spec
from mutual_lab.profiles import ModelProfile


class DevelopmentModelDescriptionTests(unittest.TestCase):
    def test_bidirectional_and_width_control_are_capacity_matched(self):
        for dimension, embedding, layers in ((16, 8, 1), (64, 32, 1), (128, 64, 1), (64, 32, 2)):
            profile = ModelProfile("comparison", embedding_dim=embedding, hidden_dim=dimension, layers=layers)
            bi = development_model_spec(profile, "bidirectional")
            wide = development_model_spec(profile, "wide_last")
            with self.subTest(profile=profile):
                self.assertEqual(bi["output_dim"], wide["output_dim"])
                self.assertEqual(bi["producer_parameters"], wide["producer_parameters"])
                self.assertLess(wide["nonembedding_count_relative_difference_from_bidirectional"], .02)
                self.assertEqual(wide, development_model_spec(profile, "wide_last"))
                self.assertTrue(bi["excluded_from_research"])

    def test_invalid_variants_and_vocabulary_fail_without_tensor_dependencies(self):
        profile = ModelProfile()
        with self.assertRaises(ValueError):
            development_model_spec(profile, "attention")
        with self.assertRaises(ValueError):
            development_model_spec(profile, vocab_size=65539)
        for seed in (True, -1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                build_development_model(profile, seed)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class DevelopmentEncoderTests(unittest.TestCase):
    def setUp(self):
        from mutual_lab.neural import configure_cpu
        configure_cpu()
        self.profile = ModelProfile("test-development", embedding_dim=8, hidden_dim=16)

    def test_baseline_is_exact_and_all_variants_preserve_unmodified_initial_tensors(self):
        from mutual_lab.neural import NativeModel, digest, torch
        torch.manual_seed(517)
        before = torch.random.get_rng_state().clone()
        original = NativeModel(self.profile, 173)
        models = [build_development_model(self.profile, 173, name)
                  for name in ("last", "bidirectional", "wide_last")]
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(digest(original.state_dict()), digest(models[0].state_dict()))
        for model in models:
            with self.subTest(encoder=model.development_spec["encoder_variant"]):
                self.assertEqual(digest(model.producer.state_dict()), digest(original.producer.state_dict()))
                self.assertTrue(torch.equal(model.visual_projection, original.visual_projection))
                self.assertTrue(torch.equal(model.understanding.embedding.weight,
                                            original.understanding.embedding.weight))
                self.assertEqual(sum(parameter.numel() for parameter in model.understanding.parameters()),
                                 model.development_spec["understanding_parameters"])
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()),
                                 model.development_spec["parameter_count"])

    def test_bidirectional_readout_ignores_padding_and_both_directions_receive_gradients(self):
        from mutual_lab.neural import torch
        for layers in (1, 2):
            profile = ModelProfile("test-bidirectional", embedding_dim=8, hidden_dim=16, layers=layers)
            model = build_development_model(profile, 173, "bidirectional")
            ids, lengths = model.tokenizer.batch(["원", "빨간 원의 왼쪽에 파란 사각형이 있다."])
            actual = model.understanding(ids, lengths)
            self.assertEqual(tuple(actual.shape), (2, profile.hidden_dim))
            for index, text in enumerate(("원", "빨간 원의 왼쪽에 파란 사각형이 있다.")):
                single, length = model.tokenizer.batch([text])
                self.assertTrue(torch.allclose(actual[index], model.understanding(single, length)[0],
                                               atol=1e-6, rtol=1e-5))
            actual[:, 0].sum().backward()
            for name in ("weight_ih_l0", "weight_ih_l0_reverse"):
                gradient = getattr(model.understanding.gru, name).grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.norm()), 0)
            self.assertGreater(float(model.understanding.embedding.weight.grad.norm()), 0)

    def test_all_variants_train_both_language_branches_on_the_same_image_examples(self):
        from mutual_lab.neural import digest, native_examples, torch
        from mutual_lab.world import Object, variants
        scenes = variants((Object("blue", "square"), Object("red", "circle")))
        for name in ("last", "bidirectional", "wide_last"):
            model = build_development_model(self.profile, 173, name)
            examples = native_examples(model, scenes, "ko")
            before = [digest(module.state_dict()) for module in (model.understanding, model.producer)]
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            optimizer.zero_grad(set_to_none=True)
            loss = model.grounding_loss(examples) + model.production_loss(examples)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            after = [digest(module.state_dict()) for module in (model.understanding, model.producer)]
            self.assertTrue(all(first != second for first, second in zip(before, after)))
            representation = model.understanding(*model.tokenizer.batch(["빨간 원"]))
            self.assertEqual(tuple(representation.shape), (1, 16))


if __name__ == "__main__":
    unittest.main()
