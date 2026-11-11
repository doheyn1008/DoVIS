import hashlib
import struct
import unittest
from dataclasses import replace

from mutual_lab.config import AgentConfig, ExperimentConfig
from mutual_lab.images import choice_sheet, scene_png
from mutual_lab.world import controlled_caption, make_dataset


class ConfigTests(unittest.TestCase):
    def test_round_trip_and_typo_rejection(self):
        config = ExperimentConfig().validate()
        self.assertEqual(config, ExperimentConfig.from_dict(config.to_dict()))
        with self.assertRaises(ValueError):
            ExperimentConfig.from_dict({"eval_itmes": 24})

    def test_invalid_values_fail_before_execution(self):
        for key, value in (("seed", True), ("max_exchanges", 0), ("eval_every", 0),
                           ("eval_items", 5), ("threshold", float("nan")),
                           ("temperature", float("inf")), ("allow_remote", "false"),
                           ("context_exchanges", 0), ("sustained_checks", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ExperimentConfig.from_dict({key: value})

    def test_endpoint_urls_and_api_key_indirection(self):
        agent = AgentConfig("A", "ko", "endpoint", "model-a", "http://127.0.0.1:8000/v1", "LAB_A_KEY")
        agent.validate(False)
        for url in ("http://user:secret@localhost:8000/v1", "https://example.com/v1",
                    "http://localhost:8000/v1?key=secret", "file:///etc/passwd"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                replace(agent, base_url=url).validate(False)
        replace(agent, base_url="https://example.com/v1").validate(True)

    def test_duplicate_names_and_mixed_backends_rejected(self):
        with self.assertRaises(ValueError):
            replace(ExperimentConfig(), agents=(AgentConfig("A", "en"), AgentConfig("A", "ko"))).validate()
        with self.assertRaises(ValueError):
            replace(ExperimentConfig(), agents=(AgentConfig("A", "en"),
                    AgentConfig("B", "ko", "endpoint", "m", "http://localhost/v1"))).validate()

    def test_request_upper_bound_includes_baseline_frozen_and_final(self):
        agents = (AgentConfig("A", "en", "endpoint", "m", "http://localhost/v1"),
                  AgentConfig("B", "ko", "endpoint", "m", "http://localhost/v1"))
        config = replace(ExperimentConfig(), agents=agents, max_exchanges=11, eval_every=5, eval_items=4)
        # Frozen 8 + four checkpoints of 8 (0,5,10,11) + 22 dialogue requests.
        self.assertEqual(config.request_upper_bound(), 62)


class WorldTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig()
        self.dataset = make_dataset(self.config)

    def test_split_is_reproducible_and_group_disjoint(self):
        self.assertEqual(len(self.dataset.train), 108)
        self.assertEqual(len(self.dataset.holdout), 36)
        self.assertEqual(self.dataset.fingerprint, make_dataset(self.config).fingerprint)
        self.assertNotEqual(self.dataset.fingerprint, make_dataset(replace(self.config, seed=8)).fingerprint)
        self.assertFalse({s.group for s in self.dataset.train} & {s.group for s in self.dataset.holdout})

    def test_options_are_unique_hard_negatives_and_balanced(self):
        positions = [probe.correct_choice for probe in self.dataset.probes]
        self.assertEqual([positions.count(i) for i in range(1, 5)], [6] * 4)
        for probe in self.dataset.probes:
            self.assertEqual(probe.choices[probe.correct_choice - 1], probe.scene)
            self.assertEqual(len(set(probe.choices)), 4)
            self.assertEqual({scene.group for scene in probe.choices}, {probe.scene.group})
            self.assertTrue(set(probe.choices) <= set(self.dataset.holdout))

    def test_images_are_valid_distinct_png_without_semantic_metadata(self):
        images = [scene_png(scene) for scene in (*self.dataset.train, *self.dataset.holdout)]
        self.assertEqual(len({hashlib.sha256(image).digest() for image in images}), 144)
        for image in images:
            self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(struct.unpack(">II", image[16:24]), (256, 176))
            self.assertNotIn(b"tEXt", image)
        self.assertEqual(struct.unpack(">II", choice_sheet(self.dataset.probes[0].choices)[16:24]), (512, 400))

    def test_caption_languages_and_excessive_eval_items(self):
        scene = self.dataset.train[0]
        self.assertIn("The ", controlled_caption(scene, "en"))
        self.assertIn("있다", controlled_caption(scene, "ko"))
        self.assertIn("ある", controlled_caption(scene, "ja"))
        with self.assertRaises(ValueError):
            make_dataset(replace(self.config, eval_items=40))


if __name__ == "__main__":
    unittest.main()
