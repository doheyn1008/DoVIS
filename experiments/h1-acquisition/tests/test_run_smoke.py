import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_smoke as r

HAS_NEURAL = all(importlib.util.find_spec(n) for n in ("torch", "numpy", "PIL"))


class PlanningTests(unittest.TestCase):
    def test_derangement_preserves_marginals_and_is_fixed(self):
        values = r.derangement(32, 119031)
        self.assertEqual(values, r.derangement(32, 119031))
        self.assertEqual(sorted(values), list(range(32)))
        self.assertTrue(all(i != j for i, j in enumerate(values)))
        for bad in (True, 0, 1):
            with self.assertRaises(ValueError):
                r.derangement(bad, 1)

    def test_development_stream_is_fit_only_and_evaluation_is_pair_disjoint(self):
        from mutual_lab.config import ExperimentConfig
        config = ExperimentConfig.load(r.LAB / "configs/neural.example.json")
        split, scenes, permutation, evaluation, pngs, public = r.make_semantic_plan(config, 32, 119003, 119031, 119011)
        self.assertEqual(len(scenes), len({s.key for s in scenes}))
        self.assertTrue(set(scenes) <= set(split["train"]))
        self.assertFalse({s.group for s in scenes} & {s.group for s, _, _ in evaluation})
        self.assertEqual(len(evaluation), 112)
        self.assertEqual(len(pngs), 332)
        self.assertEqual(public["development_manifest"]["protected_pairs_excluded"], 22)
        self.assertFalse(public["protected_images_rendered_or_scored"])
        again = r.make_semantic_plan(config, 32, 119003, 119031, 119011)
        self.assertEqual(r.value_sha(public), r.value_sha(again[-1]))

    def test_prefix_counter_guard_catches_outer_inner_cursor_disagreement(self):
        tokenizer = types.SimpleNamespace(encode=lambda s: [1, *s.encode(), 2])
        learners = {}
        for language in ("ko", "ja"):
            learners[language] = types.SimpleNamespace(model=types.SimpleNamespace(tokenizer=tokenizer))
        stream = [{"utterances": {"ko": {"text": "원"}, "ja": {"text": "円"}}}]
        expected = {"exposures": 1, "updates": 1, "characters": 1, "bytes_received": 3,
                    "input_units": 3, "training_sequence_units": 5, "native_replay_exposures": 0}
        for learner in learners.values():
            learner.counters = lambda: dict(expected)
        r.verify_exchange_counters(learners, stream, [0], "aligned", 1)
        learners["ja"].counters = lambda: {**expected, "exposures": 0}
        with self.assertRaisesRegex(ValueError, "prefix"):
            r.verify_exchange_counters(learners, stream, [0], "aligned", 1)

    def test_evaluation_reports_three_metrics_and_rejects_state_mutation(self):
        from mutual_lab.world import Object, Scene, variants
        scene = Scene(Object("red", "circle"), "left_of", Object("blue", "square"))
        choices = variants(scene.group)
        pngs = {s.key: b"PNG" + s.key.encode() for s in choices}
        class Fake:
            value = 0
            mutate = False
            def state_digest(self): return str(self.value)
            def cache_digest(self): return "fixed"
            def predict_many(self, texts, choices):
                self.value += int(self.mutate)
                return [0] * len(texts)
        learners = {"ko": Fake(), "ja": Fake()}
        frozen = {l: {scene.key: {"text": "peer"}} for l in learners}
        points, rows = r.evaluate_pair(learners, [(scene, choices, choices.index(scene))], pngs, frozen, 0)
        self.assertEqual(len(points), 6)
        self.assertEqual(len(rows), 18)
        self.assertEqual({p["metric"] for p in points}, {"native_authored", "foreign_authored", "foreign_peer_generated"})
        learners["ja"].mutate = True
        with self.assertRaisesRegex(ValueError, "Evaluation updated"):
            r.evaluate_pair(learners, [(scene, choices, 0)], pngs, frozen, 0)


@unittest.skipUnless(HAS_NEURAL, "Optional CPU Torch, NumPy and Pillow")
class CheckpointTests(unittest.TestCase):
    class Fake:
        def __init__(self, exposures=16):
            self.state = {"counters": {"exposures": exposures, "updates": exposures, "native_replay_exposures": 0},
                          "feature_caches": {"frozen": {b"x": [1., 2.]}}}
            self.loaded = False
        def state_dict(self): return copy.deepcopy(self.state)
        def state_digest(self):
            from mutual_lab.neural import digest
            value = copy.deepcopy(self.state)
            value.pop("feature_caches")
            return digest(value)
        def load_state_dict(self, value): self.state = copy.deepcopy(value); self.loaded = True

    def test_pair_scope_and_cursor_are_checked_before_either_learner_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.pt"
            learners = {l: self.Fake() for l in ("ko", "ja")}
            metadata = {"condition": "aligned", "exchange": 16, "stream_sha256": "x"}
            r.save_pair(path, learners, metadata)
            target = {l: self.Fake(0) for l in learners}
            r.restore_pair(path, target, metadata)
            self.assertTrue(all(x.loaded for x in target.values()))
            badmeta = {**metadata, "exchange": 15}
            bad = Path(tmp) / "cursor-mismatch.pt"
            r.save_pair(bad, learners, badmeta)
            target = {l: self.Fake(0) for l in learners}
            with self.assertRaisesRegex(ValueError, "cursor"):
                r.restore_pair(bad, target, badmeta)
            self.assertFalse(any(x.loaded for x in target.values()))
            manifest = json.loads(path.with_suffix(".json").read_text())
            manifest["eligible_for_l2_research"] = True
            path.with_suffix(".json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "scope/hash"):
                r.restore_pair(path, target, metadata)

    def test_failed_new_sidecar_never_overwrites_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, newer = Path(tmp) / "previous.pt", Path(tmp) / "newer.pt"
            learners = {l: self.Fake() for l in ("ko", "ja")}
            metadata = {"condition": "aligned", "exchange": 16}
            r.save_pair(original, learners, metadata)
            before = r.file_sha(original), r.file_sha(original.with_suffix(".json"))
            with patch.object(r, "write_json", side_effect=OSError("sidecar unavailable")):
                with self.assertRaises(OSError): r.save_pair(newer, learners, metadata)
            self.assertEqual(before, (r.file_sha(original), r.file_sha(original.with_suffix(".json"))))
            self.assertFalse(newer.with_suffix(".json").exists())
            r.restore_pair(original, {l: self.Fake(0) for l in learners}, metadata)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                r.save_pair(original, learners, metadata)


if __name__ == "__main__":
    unittest.main()
