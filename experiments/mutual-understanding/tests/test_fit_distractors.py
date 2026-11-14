"""Fit-only sampling, matched positives, and availability-aware four-choice training."""

import hashlib
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from mutual_lab.composition_diagnostics import balanced_splits
from mutual_lab.config import ExperimentConfig
from mutual_lab.fit_distractors import build_fit_catalog, make_training_examples
from mutual_lab.images import scene_png
from mutual_lab.world import make_research_dataset, native_captions, variants

ROOT = Path(__file__).resolve().parents[1]


def examples(scenes):
    result = []
    for scene in scenes:
        alternatives = variants(scene.group)
        for caption in native_captions(scene, "ko"):
            result.append({"text": caption, "image": scene_png(scene),
                           "choices": [scene_png(s) for s in alternatives],
                           "answer": alternatives.index(scene), "_scene_key": scene.key,
                           "_first_caption": native_captions(scene, "ko")[0]})
    return result


class FitDistractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ExperimentConfig.load(ROOT / "configs/neural.example.json")
        cls.splits = balanced_splits(cls.config)
        cls.catalog = build_fit_catalog(cls.splits["train"])
        cls.batch = examples(cls.splits["train"])

    def test_catalog_is_explicit_deterministic_and_reports_real_fit_availability(self):
        report = self.catalog.manifest()
        reversed_catalog = build_fit_catalog(reversed(self.splits["train"]))
        self.assertEqual(report, reversed_catalog.manifest())
        self.assertEqual(report["scene_count"], 40)
        self.assertEqual(report["pair_count"], 10)
        self.assertEqual({key: value["available"] for key, value in report["availability"].items()},
                         {"role_reversal": 40, "color_only": 20, "shape_only": 20, "binding_swap": 8})
        self.assertEqual(report["color_and_shape_available"], 4)
        self.assertEqual(report["choice_count"], 4)
        self.assertEqual(report["chance_accuracy"], .25)
        self.assertEqual(len({row["png_sha256"] for row in report["pool"]}), 40)

    def test_no_choice_or_positive_is_borrowed_from_dev_or_protected_pools(self):
        fit = set(self.splits["train"])
        research = make_research_dataset(self.config)
        forbidden = set(self.splits["dev"])
        for name in ("native", "pretest", "curve", "final"):
            forbidden.update(probe.scene for probe in research.view(name).probes)
        forbidden_pngs = {scene_png(scene) for scene in forbidden}
        fit_pngs = {scene_png(scene) for scene in fit}
        for policy in ("same_pair", "attributes"):
            output, audit = make_training_examples(self.batch, self.catalog, random.Random(917), policy=policy)
            self.assertEqual(len(output), len(self.batch))
            for original, sampled in zip(self.batch, output):
                for key in ("text", "image", "_scene_key", "_first_caption"):
                    self.assertEqual(original[key], sampled[key])
                self.assertEqual(len(sampled["choices"]), 4)
                self.assertEqual(len(set(sampled["choices"])), 4)
                self.assertTrue(set(sampled["choices"]) <= fit_pngs)
                self.assertFalse(set(sampled["choices"]) & forbidden_pngs)
                self.assertEqual(sampled["choices"][sampled["answer"]], original["image"])
            self.assertEqual(audit["examples"], len(self.batch))
            self.assertEqual(audit["distractor_slots"], 3 * len(self.batch))

    def test_baseline_preserves_choices_answers_and_does_not_advance_negative_rng(self):
        rng = random.Random(918)
        before = rng.getstate()
        output, audit = make_training_examples(self.batch, self.catalog, rng, policy="same_pair")
        self.assertEqual(rng.getstate(), before)
        self.assertEqual(output, self.batch)
        self.assertEqual(audit["fallback_slots"], 0)
        for original, copied in zip(self.batch, output):
            self.assertIsNot(original, copied)
            self.assertIs(original["choices"], copied["choices"])
        output[0]["text"] = "changed copy"
        self.assertNotEqual(output[0]["text"], self.batch[0]["text"])

    def test_category_omission_and_answer_position_are_crossed_not_tied(self):
        # One anchor repeated exposes the omission/position schedule independently
        # of batch composition or accidental category availability.
        batch = [self.batch[0]] * 16
        output, audit = make_training_examples(batch, self.catalog, random.Random(919))
        self.assertEqual([row["answer"] for row in output], list(range(4)) * 4)
        for position in range(4):
            self.assertEqual(audit[f"answer_position_{position}"], 4)
        for category in ("role_reversal", "color_only", "shape_only", "binding_swap"):
            self.assertEqual(audit[f"omitted_{category}"], 4)
            self.assertEqual(audit[f"requested_{category}"], 12)
        again, again_audit = make_training_examples(batch, self.catalog, random.Random(919))
        self.assertEqual(output, again)
        self.assertEqual(audit, again_audit)
        # Batch boundaries do not change the global exposure schedule or RNG.
        rng = random.Random(919)
        first, _ = make_training_examples(batch[:7], self.catalog, rng)
        second, _ = make_training_examples(batch[7:], self.catalog, rng, exposure_offset=7)
        self.assertEqual(first + second, output)

    def test_sparse_catalog_uses_fallback_without_dropping_or_resampling_anchors(self):
        scenes = tuple(variants(self.splits["train"][0].group))
        catalog = build_fit_catalog(scenes)
        batch = [examples(scenes)[0]] * 16
        output, audit = make_training_examples(batch, catalog, random.Random(920))
        self.assertEqual(len(output), 16)
        self.assertTrue(all(row["_scene_key"] == batch[0]["_scene_key"] for row in output))
        self.assertEqual(audit["fallback_slots"], 36)
        self.assertEqual(audit["selected_by_policy_role_reversal"], 12)
        # The omitted-role fallback also contains a role reversal: intrinsic
        # contrast exposure must be counted separately from its sampling source.
        self.assertEqual(audit["selected_actual_role_reversal"], 16)
        self.assertEqual(audit["selected_actual_other"], 32)
        self.assertEqual(audit["selected_by_policy_color_only"], 0)
        self.assertEqual(audit["selected_by_policy_shape_only"], 0)
        self.assertEqual(audit["selected_by_policy_binding_swap"], 0)

    def test_wrong_anchors_images_and_baseline_leakage_are_rejected(self):
        dev = examples(self.splits["dev"])[0]
        with self.assertRaisesRegex(ValueError, "outside"):
            make_training_examples([dev], self.catalog, random.Random(921))
        wrong_image = dict(self.batch[0], image=dev["image"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            make_training_examples([wrong_image], self.catalog, random.Random(921))
        leaked = dict(self.batch[0], choices=list(self.batch[0]["choices"]))
        leaked["choices"][(leaked["answer"] + 1) % 4] = dev["image"]
        with self.assertRaisesRegex(ValueError, "outside"):
            make_training_examples([leaked], self.catalog, random.Random(921), policy="same_pair")
        duplicate = dict(self.batch[0], choices=[self.batch[0]["image"]] * 4, answer=0)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            make_training_examples([duplicate], self.catalog, random.Random(921), policy="same_pair")

    def test_semantic_and_rendering_duplicates_are_rejected_at_catalog_boundary(self):
        with self.assertRaisesRegex(ValueError, "semantic"):
            build_fit_catalog((self.splits["train"][0],) * 4)
        with patch("mutual_lab.fit_distractors.scene_png", return_value=b"same image"):
            with self.assertRaisesRegex(ValueError, "PNG"):
                build_fit_catalog(self.splits["train"])
        with self.assertRaisesRegex(ValueError, "Four unique"):
            build_fit_catalog(self.splits["train"][:3])

    def test_explicit_renderer_defines_the_exact_allowed_png_pool(self):
        renderer = lambda scene: b"custom-rendering:" + scene.key.encode()
        catalog = build_fit_catalog(self.splits["train"], render_fn=renderer)
        pool = {renderer(scene) for scene in self.splits["train"]}
        scene = self.splits["train"][0]
        item = dict(self.batch[0], image=renderer(scene))
        output, _ = make_training_examples([item], catalog, random.Random(922))
        self.assertTrue(set(output[0]["choices"]) <= pool)
        self.assertEqual(output[0]["choices"][output[0]["answer"]], renderer(scene))
        self.assertEqual({row["png_sha256"] for row in catalog.manifest()["pool"]},
                         {hashlib.sha256(png).hexdigest() for png in pool})


if __name__ == "__main__":
    unittest.main()
