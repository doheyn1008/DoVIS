"""Meaning-level coverage and discrimination for development composition work."""

import hashlib
import importlib.util
import json
import random
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from mutual_lab.composition_diagnostics import (
    CATEGORIES, balanced_splits, build_contrast_probes, contrast_manifest,
    is_contrast, score_contrasts, split_manifest,
)
from mutual_lab.config import ExperimentConfig
from mutual_lab.images import scene_png
from mutual_lab.native_preparation import PreparationPlan, build_preparation_splits
from mutual_lab.world import COLORS, SHAPES, Object, Scene, make_research_dataset, native_captions


ROOT = Path(__file__).resolve().parents[1]


def experiment():
    return ExperimentConfig.load(ROOT / "configs/neural.example.json")


class CompositionSplitTests(unittest.TestCase):
    def test_balanced_policy_preserves_protected_pools_and_gives_every_object_two_partners(self):
        config = experiment()
        original = build_preparation_splits(config)
        before = split_manifest(original)
        balanced = balanced_splits(config)
        self.assertEqual(balanced, balanced_splits(config))
        self.assertEqual(before, split_manifest(build_preparation_splits(config)))
        self.assertNotEqual(before["pair_sha256"], split_manifest(balanced)["pair_sha256"])
        self.assertEqual(before["minimum_train_distinct_partners"], 1)
        self.assertEqual(split_manifest(balanced)["minimum_train_distinct_partners"], 2)
        self.assertEqual(before["split_version"], "composition-split-manifest-v1")
        self.assertEqual({name: len(scenes) for name, scenes in balanced.items()}, {"train": 40, "dev": 16})
        research = make_research_dataset(config)
        used = set(balanced["train"]) | set(balanced["dev"])
        self.assertTrue(used <= set(research.train))
        self.assertFalse({s.group for s in balanced["train"]} & {s.group for s in balanced["dev"]})
        for name in ("native", "pretest", "curve", "final"):
            self.assertFalse({s.group for s in used} & {p.scene.group for p in research.view(name).probes})
        train_objects = {obj for s in balanced["train"] for obj in (s.subject, s.object)}
        self.assertEqual(train_objects, {Object(c, s) for c in COLORS for s in SHAPES})
        rows = split_manifest(balanced)["partner_counts"]["train"]
        self.assertTrue(all(row["distinct_partners"] >= 2 for row in rows))
        for attribute, expected in (("color", set(COLORS)), ("shape", set(SHAPES))):
            self.assertEqual({getattr(o, attribute) for s in balanced["dev"] for o in (s.subject, s.object)}, expected)

    def test_impossible_partner_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nine"):
            balanced_splits(experiment(), PreparationPlan(train_pairs=8))


class ContrastProbeTests(unittest.TestCase):
    def setUp(self):
        self.config = experiment()
        self.dev = balanced_splits(self.config)["dev"]
        self.probes = build_contrast_probes(self.config, self.dev)

    def test_known_counterfactuals_change_only_the_intended_category(self):
        anchor = Scene(Object("red", "circle"), "left_of", Object("blue", "square"))
        alternatives = {
            "role_reversal": Scene(Object("blue", "square"), "left_of", Object("red", "circle")),
            "color_only": Scene(Object("green", "circle"), "left_of", Object("blue", "square")),
            "shape_only": Scene(Object("red", "triangle"), "left_of", Object("blue", "square")),
            "binding_swap": Scene(Object("blue", "circle"), "left_of", Object("red", "square")),
        }
        for expected, alternative in alternatives.items():
            for category in CATEGORIES:
                with self.subTest(expected=expected, category=category):
                    self.assertEqual(is_contrast(anchor, alternative, category), category == expected)
                    axis_changed = Scene(alternative.subject, "above", alternative.object)
                    self.assertFalse(is_contrast(anchor, axis_changed, category))

    def test_rendering_collisions_are_reported_and_never_used_as_choices(self):
        with patch("mutual_lab.composition_diagnostics.scene_png", return_value=b"identical PNG"):
            collision = build_contrast_probes(self.config, self.dev)
        self.assertEqual(collision["probes"], ())
        for availability in collision["availability"].values():
            self.assertEqual(availability["available"], 0)
            self.assertEqual(availability["unavailable"], len(self.dev))
            self.assertGreater(availability["duplicate_images_rejected"], 0)

    def test_binary_contrasts_are_category_correct_unique_and_position_balanced(self):
        self.assertEqual(self.probes, build_contrast_probes(self.config, self.dev))
        allowed = set(make_research_dataset(self.config).train)
        self.assertEqual(self.probes["choice_count"], 2)
        self.assertEqual(self.probes["chance_accuracy"], .5)
        self.assertEqual(set(self.probes["anchor_scene_keys"]), {s.key for s in self.dev})
        self.assertEqual(set(self.probes["availability"]), set(CATEGORIES))
        for probe in self.probes["probes"]:
            self.assertIn(probe.scene, self.dev)
            self.assertEqual(probe.choices[probe.correct_choice], probe.scene)
            self.assertTrue(set(probe.choices) <= allowed)
            self.assertEqual(len(set(probe.choices)), 2)
            self.assertNotEqual(*(hashlib.sha256(scene_png(s)).hexdigest() for s in probe.choices))
            self.assertTrue(is_contrast(probe.scene, probe.choices[1 - probe.correct_choice], probe.category))
            if probe.category == "role_reversal":
                alternative = probe.choices[1 - probe.correct_choice]
                self.assertEqual((alternative.subject, alternative.object), (probe.scene.object, probe.scene.subject))
                self.assertEqual(alternative.relation, probe.scene.relation)
            with self.assertRaises(FrozenInstanceError):
                probe.correct_choice = 1
        for category, availability in self.probes["availability"].items():
            self.assertEqual(availability["available"] + availability["unavailable"], len(self.dev))
            self.assertEqual(availability["unavailable"], len(availability["unavailable_scene_keys"]))
            self.assertLessEqual(abs(availability["correct_position_counts"][0]
                                     - availability["correct_position_counts"][1]), 1)
            actual = [p for p in self.probes["probes"] if p.category == category]
            self.assertEqual(len(actual), availability["available"])
            self.assertTrue(all(p.scene.key not in availability["unavailable_scene_keys"] for p in actual))
        self.assertGreater(self.probes["availability"]["binding_swap"]["unavailable"], 0)
        json.dumps(contrast_manifest(self.probes), allow_nan=False)

    def test_protected_or_duplicate_anchors_and_invalid_seed_are_rejected(self):
        protected = make_research_dataset(self.config).native_test[0].scene
        with self.assertRaisesRegex(ValueError, "research TRAIN"):
            build_contrast_probes(self.config, (protected,))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_contrast_probes(self.config, (self.dev[0], self.dev[0]))
        with self.assertRaises(ValueError):
            build_contrast_probes(self.config, self.dev, seed=True)

    def test_read_only_scorer_uses_native_captions_and_reports_each_category_separately(self):
        class Oracle:
            def __init__(self, scenes, language):
                self.meanings = {text: scene_png(s) for s in scenes for text in native_captions(s, language)}

            def predict(self, text, choices):
                return choices.index(self.meanings[text])

        state = random.getstate()
        for language in ("ko", "ja", "en", "es"):
            model = Oracle(self.dev, language)
            before = dict(model.meanings)
            scored = score_contrasts(model, self.probes, language)
            self.assertEqual(model.meanings, before)
            self.assertEqual(scored["chance_accuracy"], .5)
            for category, values in scored["categories"].items():
                n = 4 * self.probes["availability"][category]["available"]
                self.assertEqual(values["items"], n)
                self.assertEqual(values["hits"], n)
                self.assertEqual(values["accuracy"], 1)
                self.assertEqual(values["corrected_accuracy"], 1)
        self.assertEqual(random.getstate(), state)
        with self.assertRaises(ValueError):
            score_contrasts(Oracle(self.dev, "ko"), self.probes, "not-a-language")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class ContrastNeuralTests(unittest.TestCase):
    def test_real_neural_inference_preserves_weights_and_torch_rng(self):
        from mutual_lab.neural import NativeModel, configure_cpu, digest, torch
        from mutual_lab.profiles import ModelProfile
        configure_cpu()
        model = NativeModel(ModelProfile("contrast-test", embedding_dim=8, hidden_dim=16), 39007)
        probes = build_contrast_probes(experiment(), balanced_splits(experiment())["dev"])
        before = digest({"weights": model.state_dict(), "rng": torch.random.get_rng_state()})
        result = score_contrasts(model, probes, "ko")
        self.assertEqual(before, digest({"weights": model.state_dict(), "rng": torch.random.get_rng_state()}))
        self.assertEqual(len(result["predictions"]), 4 * len(probes["probes"]))


if __name__ == "__main__":
    unittest.main()
