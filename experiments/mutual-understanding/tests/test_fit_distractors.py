"""Fit-only sampling, matched positives, and availability-aware four-choice training."""

import hashlib
import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from mutual_lab.composition_diagnostics import balanced_splits
from mutual_lab.config import ExperimentConfig
from mutual_lab.fit_distractors import build_fit_catalog, make_training_examples, negative_policy_manifest
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
        for policy in ("same_pair", "attributes", "mixed_blocks"):
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

    def test_existing_policy_outputs_audits_and_rng_match_the_frozen_baseline(self):
        # These hashes were captured before adding mixed_blocks. Images are
        # canonicalized to their scene keys so the test does not depend on the
        # platform's zlib byte-stream choices; the exact selected PNGs are still
        # checked against the same catalog by the sampler and scope tests.
        expected = {"same_pair": "ec34bae73635e97f069920263b3f35849b613d5eae825f80ab0ed4f5a0a50cd9",
                    "attributes": "815ab0fe34383e801c983f512f907b7cd1090df8773d8a84b23c9a7847e7ad66"}

        def meaning(png):
            return self.catalog.scenes[self.catalog.by_png[hashlib.sha256(png).hexdigest()]].key

        for policy, checksum in expected.items():
            rng = random.Random(9271)
            output, audit = make_training_examples(self.batch[3:40], self.catalog, rng,
                                                  exposure_offset=7, policy=policy)
            canonical = [{**row, "image": meaning(row["image"]),
                          "choices": [meaning(png) for png in row["choices"]]} for row in output]
            data = {"output": canonical, "audit": audit, "rng": rng.getstate()}
            encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            with self.subTest(policy=policy):
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), checksum)

    def test_mixed_blocks_match_explicit_baseline_and_compact_attribute_delegation(self):
        batch = [self.batch[(i * 17) % len(self.batch)] for i in range(64)]
        mixed_rng, expected_rng = random.Random(9272), random.Random(9272)
        output, audit = make_training_examples(batch, self.catalog, mixed_rng, policy="mixed_blocks")
        expected, branch_audits = [], {"same_pair": {}, "attributes": {}}
        for block in range(4):
            policy = "same_pair" if block % 2 == 0 else "attributes"
            offset = block * 16 if policy == "same_pair" else (block // 2) * 16
            rows, counts = make_training_examples(batch[block * 16:(block + 1) * 16], self.catalog,
                                                  expected_rng, exposure_offset=offset, policy=policy)
            expected.extend(rows)
            for key, value in counts.items():
                branch_audits[policy][key] = branch_audits[policy].get(key, 0) + value
        self.assertEqual(output, expected)
        self.assertEqual(mixed_rng.getstate(), expected_rng.getstate())
        self.assertEqual(audit["branch_same_pair_examples"], 32)
        self.assertEqual(audit["branch_attributes_examples"], 32)
        for policy, counts in branch_audits.items():
            for key, value in counts.items():
                self.assertEqual(audit[f"branch_{policy}:{key}"], value)
        for category in ("role_reversal", "color_only", "shape_only", "binding_swap"):
            self.assertEqual(audit[f"branch_attributes:omitted_{category}"], 8)
            self.assertEqual(audit[f"branch_attributes:requested_{category}"], 24)
        for position in range(4):
            self.assertEqual(audit[f"branch_attributes:answer_position_{position}"], 8)
        for original, copied in zip(batch, output):
            for key in ("text", "image", "_scene_key", "_first_caption"):
                self.assertEqual(original[key], copied[key])

    def test_mixed_results_and_audits_are_independent_of_batch_boundaries(self):
        batch = [self.batch[(i * 19) % len(self.batch)] for i in range(75)]
        whole, whole_audit = make_training_examples(batch, self.catalog, random.Random(9273),
                                                    exposure_offset=11, policy="mixed_blocks")
        rng, partitioned, audit = random.Random(9273), [], {}
        start = 0
        for end in (7, 25, 41, 75):
            rows, counts = make_training_examples(batch[start:end], self.catalog, rng,
                                                  exposure_offset=11 + start, policy="mixed_blocks")
            partitioned.extend(rows)
            for key, value in counts.items():
                audit[key] = audit.get(key, 0) + value
            start = end
        self.assertEqual(partitioned, whole)
        self.assertEqual(audit, whole_audit)
        self.assertEqual(audit["branch_same_pair_examples"] + audit["branch_attributes_examples"], 75)

    def test_mixed_baseline_blocks_leave_negative_rng_untouched_and_defaults_unchanged(self):
        rng = random.Random(9274)
        before = rng.getstate()
        first, audit = make_training_examples(self.batch[:16], self.catalog, rng, policy="mixed_blocks")
        self.assertEqual(first, self.batch[:16])
        self.assertEqual(rng.getstate(), before)
        self.assertEqual(audit["branch_attributes_examples"], 0)
        self.assertEqual(audit["branch_same_pair_examples"], 16)
        expected_rng = random.Random(9274)
        expected, _ = make_training_examples(self.batch[16:32], self.catalog, expected_rng, policy="attributes")
        actual, _ = make_training_examples(self.batch[16:32], self.catalog, rng,
                                           exposure_offset=16, policy="mixed_blocks")
        self.assertEqual(actual, expected)
        self.assertEqual(rng.getstate(), expected_rng.getstate())
        manifest = negative_policy_manifest("mixed_blocks")
        self.assertEqual(manifest["block_exposures"], 16)
        self.assertEqual(manifest["cycle_order"], ["same_pair", "attributes"])
        self.assertEqual(manifest["expected_example_fractions_per_complete_cycle"],
                         {"same_pair": .5, "attributes": .5})
        default, default_audit = make_training_examples(self.batch[:8], self.catalog, random.Random(9275))
        explicit, explicit_audit = make_training_examples(self.batch[:8], self.catalog, random.Random(9275),
                                                        policy="attributes")
        self.assertEqual(default, explicit)
        self.assertEqual(default_audit, explicit_audit)


if __name__ == "__main__":
    unittest.main()
