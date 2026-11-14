"""The expanded development population preserves all old protected boundaries."""

import importlib.util
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mutual_lab import development_world as world
from mutual_lab.composition_diagnostics import balanced_splits, is_contrast
from mutual_lab.config import ExperimentConfig
from mutual_lab.images import RGB, scene_png
from mutual_lab.world import COLORS, Object, Scene, _WORDS, make_research_dataset, native_captions, variants

ROOT = Path(__file__).resolve().parents[1]


class DevelopmentWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        cls.splits = world.development_splits(cls.config)
        cls.probes = world.build_development_contrasts(cls.config, cls.splits)

    def test_new_population_preserves_old_membership_and_excludes_every_protected_pair(self):
        base = balanced_splits(self.config)
        groups = {name: {scene.group for scene in scenes} for name, scenes in self.splits.items()}
        self.assertEqual({name: len(pool) for name, pool in self.splits.items()}, {"train": 220, "dev": 112})
        self.assertEqual({name: len(pool) for name, pool in groups.items()}, {"train": 55, "dev": 28})
        self.assertFalse(groups["train"] & groups["dev"])
        self.assertEqual(len({world.render_scene(scene) for pool in self.splits.values() for scene in pool}), 332)
        for name in ("train", "dev"):
            self.assertTrue({scene.group for scene in base[name]} <= groups[name])
        source = make_research_dataset(self.config)
        for name in ("native", "pretest", "curve", "final"):
            protected = {probe.scene.group for probe in source.view(name).probes}
            self.assertFalse((groups["train"] | groups["dev"]) & protected)
        self.assertEqual(self.splits, world.development_splits(self.config))
        self.assertEqual(world.development_manifest(self.config)["selection_attempt"], 7)
        self.assertEqual(world.development_manifest(self.config), world.development_manifest(self.config))

    def test_partner_and_diagnostic_coverage_is_large_enough_for_the_fixed_candidate_rule(self):
        for name, minimum in (("train", 4), ("dev", 2)):
            counts = Counter(obj for pair in {scene.group for scene in self.splits[name]} for obj in pair)
            self.assertEqual(len(counts), 15)
            self.assertGreaterEqual(min(counts.values()), minimum)
        self.assertEqual(self.probes["chance_accuracy"], .5)
        expected = {"role_reversal": 112, "color_only": 112, "shape_only": 108, "binding_swap": 60}
        for category, availability in self.probes["availability"].items():
            self.assertEqual(availability["available"], expected[category])
            self.assertGreaterEqual(availability["available_object_pairs"], 2)
            self.assertLessEqual(abs(availability["correct_position_counts"][0]
                                     - availability["correct_position_counts"][1]), 1)
        for probe in self.probes["probes"]:
            self.assertEqual(probe.choices[probe.correct_choice], probe.scene)
            alternate = probe.choices[1 - probe.correct_choice]
            self.assertTrue(is_contrast(probe.scene, alternate, probe.category))
            self.assertNotEqual(world.render_scene(probe.scene), world.render_scene(alternate))

    def test_original_unprotected_rendering_and_caption_bytes_are_unchanged(self):
        for scene in make_research_dataset(self.config).train:
            self.assertEqual(world.render_scene(scene), scene_png(scene))
            for language in ("ko", "ja", "en", "es"):
                self.assertEqual(world.captions(scene, language), native_captions(scene, language))
        self.assertEqual(COLORS, ("red", "blue", "green"))
        self.assertNotIn("yellow", RGB)
        self.assertNotIn("purple", _WORDS["en"])

    def test_extended_palette_and_exact_semantics_are_a_separate_version(self):
        scene = Scene(Object("yellow", "circle"), "above", Object("purple", "triangle"))
        self.assertTrue(world.render_scene(scene).startswith(b"\x89PNG\r\n\x1a\n"))
        first = {language: world.captions(scene, language)[0] for language in ("ko", "ja", "en", "es")}
        self.assertEqual(first["ko"], "노란 원은 보라색 삼각형의 위쪽에 있다.")
        self.assertEqual(first["ja"], "黄色い円は紫の三角の上にある。")
        self.assertEqual(first["en"], "The yellow circle is above the purple triangle.")
        self.assertEqual(first["es"], "El círculo amarillo está encima del triángulo morado.")
        trace = {"text": first["en"], "terminated": True, "invalid_encoding": False,
                 "special_tokens_in_body": 0}
        graded = world.grade_generation(trace, scene, "en")
        self.assertTrue(graded["exact_match"])
        self.assertTrue(graded["recognized_semantic_correct"])
        trace["text"] = "The yellow circle is above the yellow circle."
        self.assertIsNone(world.grade_generation(trace, scene, "en")["recognized_semantic_correct"])

    def test_example_builder_pairs_each_native_caption_with_its_actual_rendered_meaning(self):
        seen = []
        model = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text: seen.append(text)))
        scenes = self.splits["train"][:4]
        rows = world.make_examples(model, scenes, "ko")
        self.assertEqual(len(rows), 16)
        self.assertEqual(len(seen), 16)
        for index, row in enumerate(rows):
            scene = scenes[index // 4]
            self.assertEqual(row["image"], world.render_scene(scene))
            self.assertEqual(row["choices"][row["answer"]], row["image"])
            self.assertEqual(row["_scene_key"], scene.key)
            self.assertEqual(row["_first_caption"], world.captions(scene, "ko")[0])

    def test_protected_or_duplicate_scenes_are_rejected_by_the_new_diagnostic(self):
        source = make_research_dataset(self.config)
        contaminated = {**self.splits, "train": (*self.splits["train"], source.native_test[0].scene)}
        with self.assertRaises(ValueError):
            world.build_development_contrasts(self.config, contaminated)
        duplicate = {**self.splits, "dev": (*self.splits["dev"], self.splits["dev"][0])}
        with self.assertRaises(ValueError):
            world.build_development_contrasts(self.config, duplicate)
        with self.assertRaises(ValueError):
            world.development_splits(self.config, data_seed=True)
        with self.assertRaises(ValueError):
            world.render_scene(Scene(Object("orange", "circle"), "above", Object("blue", "square")))

    def test_new_diagnostic_scorer_accepts_only_text_and_pngs_and_keeps_raw_denominators(self):
        targets = {text: world.render_scene(scene) for scene in self.splits["dev"]
                   for text in world.captions(scene, "en")}

        class Oracle:
            def predict(self, text, pngs):
                return pngs.index(targets[text])

        scores = world.score_development_contrasts(Oracle(), self.probes, "en")
        for category, metrics in scores["categories"].items():
            self.assertEqual(metrics["accuracy"], 1)
            self.assertEqual(metrics["items"], self.probes["availability"][category]["available"] * 4)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class DevelopmentWorldNeuralTests(unittest.TestCase):
    def test_attentive_decoder_evaluation_uses_its_teacher_forced_image_grid_hook(self):
        from mutual_lab.attentive_spatial import build_attentive_spatial_model
        from mutual_lab.neural import configure_cpu, digest, torch
        from mutual_lab.profiles import ModelProfile
        from mutual_lab.spatial_prior import pretrain_prior, spatial_prior_hash
        configure_cpu()
        scenes = tuple(variants((Object("yellow", "circle"), Object("purple", "triangle")))[:2])
        pngs = [world.render_scene(scene) for scene in scenes]
        bundle = pretrain_prior(pngs, dim=16, steps=2, batch_size=2, eval_every=1)
        profile = ModelProfile("development-world-attention-test", embedding_dim=8, hidden_dim=16,
                               max_sequence_length=128)
        model = build_attentive_spatial_model(profile, 59101, bundle)
        examples = world.make_examples(model, scenes, "ko")
        before = digest({"weights": model.state_dict(), "rng": torch.random.get_rng_state()})
        prior_before = spatial_prior_hash(model)
        with patch.object(model, "teacher_forced_logits", wraps=model.teacher_forced_logits) as hook:
            metrics, generated = world.evaluate_development(model, examples, scenes, "ko", 4)
        self.assertEqual(hook.call_count, 2)
        self.assertEqual(metrics["comprehension_items"], 8)
        self.assertEqual(metrics["generation_scenes"], 2)
        self.assertEqual(len(generated), 2)
        self.assertEqual(prior_before, spatial_prior_hash(model))
        self.assertEqual(before, digest({"weights": model.state_dict(), "rng": torch.random.get_rng_state()}))

    def test_real_evaluation_uses_extended_captions_without_mutating_weights_or_rng(self):
        from mutual_lab.neural import NativeModel, configure_cpu, digest, torch
        from mutual_lab.profiles import ModelProfile
        configure_cpu()
        model = NativeModel(ModelProfile("development-world-test", embedding_dim=8, hidden_dim=16,
                                         max_sequence_length=128), 59101)
        scenes = tuple(variants((Object("yellow", "circle"), Object("purple", "triangle")))[:2])
        examples = world.make_examples(model, scenes, "ko")
        before = digest({"weights": model.state_dict(), "rng": torch.random.get_rng_state()})
        metrics, generated = world.evaluate_development(model, examples, scenes, "ko", 4)
        self.assertEqual(metrics["comprehension_items"], 8)
        self.assertEqual(metrics["generation_scenes"], 2)
        self.assertEqual(len(generated), 2)
        self.assertEqual(before, digest({"weights": model.state_dict(), "rng": torch.random.get_rng_state()}))


if __name__ == "__main__":
    unittest.main()
