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


class CompletedPrefixPersistenceTests(unittest.TestCase):
    """Run the real condition loop while keeping the injected failure independent of neural compute."""

    def assert_prefix_survives(self, failure):
        class Scene:
            def __init__(self, key): self.key = key
            def to_dict(self): return {"key": self.key}

        class Learner:
            def __init__(self, language):
                self.language = language
                self.values = {k: 0 for k in ("exposures", "updates", "characters", "bytes_received", "input_units",
                                             "training_sequence_units", "native_replay_exposures")}
                self.model = types.SimpleNamespace(tokenizer=types.SimpleNamespace(encode=lambda s: [1, *s.encode(), 2]))
                self.optimizer = types.SimpleNamespace(state_dict=lambda: {"updates": self.values["updates"]})
            def warm(self, pngs): pass
            def emit(self, png): return {"text": self.language + ":" + png.decode()}
            def cache_digest(self): return "canonical-cache"
            def counters(self): return dict(self.values)
            def state_digest(self): return r.value_sha({"language": self.language, "counters": self.values})
            def branch_digests(self):
                return {"understanding": str(self.values["updates"]), "producer": "frozen-P", "visual": "frozen-V"}
            def update(self, text, target, negatives, *, updates):
                self.values["exposures"] += 1
                self.values["updates"] += updates
                self.values["characters"] += len(text)
                self.values["bytes_received"] += len(text.encode())
                self.values["input_units"] += len(text.encode())
                self.values["training_sequence_units"] += (len(text.encode()) + 2) * updates
                return {"updates": updates, "exposures": self.values["exposures"]}

        scenes = [Scene("s" + str(i)) for i in range(32)]
        evaluation = [(scene, [scene], 0) for scene in scenes[:3]]
        pngs = {scene.key: scene.key.encode() for scene in scenes}
        grade = lambda trace, scene, language: {"recognized_meaning": scene.to_dict(), "exact_match": True,
            "terminated": True, "invalid_encoding": False, "special_tokens_in_body": 0}
        modules = {}
        for name, attrs in {
            "h1_adapter": {"H1DevelopmentLearner": types.SimpleNamespace(from_checkpoint=lambda *a, **kw: Learner(kw["language"]))},
            "mutual_lab.config": {"ExperimentConfig": types.SimpleNamespace(load=lambda path: None)},
            "mutual_lab.development_world": {"grade_generation": grade},
            "mutual_lab.neural": {"configure_cpu": lambda: None, "digest": r.value_sha},
        }.items():
            module = types.ModuleType(name); module.__dict__.update(attrs); modules[name] = module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / "models.zip"; archive.write_bytes(b"fixture archive")
            output = root / "result"
            texts = {"checkpoint": b"checkpoint", "sidecar": b"checkpoint sidecar", "prior": b"prior",
                     "prior_sidecar": b"prior sidecar", "config": b"blueprint"}
            sha = lambda value: __import__("hashlib").sha256(value).hexdigest()
            plan = {"exposures": 32, "evaluate_at": [0, 8, 16, 32], "conditions": list(r.CONDITIONS),
                "languages": ["ko", "ja"], "optimizer": {"learning_rate": .001, "updates_per_exposure": 1,
                    "native_replay_per_update": 0, "foreign_replay_per_update": 0},
                "excluded_from_research": True, "eligible_for_fresh_experiment": False, "replica": 0,
                "checkpoints": {l: {"sha256": sha(texts["checkpoint"]), "sidecar_sha256": sha(texts["sidecar"])} for l in ("ko", "ja")},
                "prior": {"sha256": sha(texts["prior"]), "sidecar_sha256": sha(texts["prior_sidecar"])},
                "source": {"blueprint_sha256": sha(texts["config"])}, "data_pair_sha256": "fixture-pairs",
                "seeds": {"stream": 1, "shuffle": 2, "evaluation": 3, "learner": 4}}
            plan_path = root / "plan.json"; plan_path.write_text(json.dumps(plan))

            def artifacts(_archive, directory, expected):
                paths = {}
                for language in expected:
                    paths[language] = {}
                    for key, value in texts.items():
                        path = Path(directory) / language / key
                        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
                        paths[language][key] = path
                return paths

            def evaluate(learners, _evaluation, _pngs, _frozen, stop):
                self.assertTrue(all(a.counters()["exposures"] == stop for a in learners.values()))
                rows = [{"learner": l, "exchange": stop, "prediction": 0, "completed_evaluation": True} for l in learners]
                return [{"learner": l, "exchange": stop} for l in learners], rows

            def save(path, learners, metadata):
                if failure == "checkpoint" and metadata["exchange"] == 16:
                    raise OSError("injected checkpoint failure after completed evaluation")
                path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"checkpoint marker")
                return path

            with patch.dict(sys.modules, modules), patch.object(r, "ARCHIVE_SHA256", sha(archive.read_bytes())), \
                    patch.object(r, "verify_core_sources", return_value={"all_unchanged": True}), \
                    patch.object(r, "prepare_artifacts", side_effect=artifacts), \
                    patch.object(r, "make_semantic_plan", return_value=({}, scenes, list(range(1, 32)) + [0], evaluation,
                        pngs, {"development_manifest": {"pair_sha256": "fixture-pairs"}})), \
                    patch.object(r, "_learning_inputs", side_effect=lambda scene, text, pool: (text, pool[scene.key], [b"a", b"b", b"c"])), \
                    patch.object(r, "evaluate_pair", side_effect=evaluate), patch.object(r, "save_pair", side_effect=save), \
                    patch.object(r, "restore_pair", side_effect=RuntimeError("injected final resume failure")):
                with self.assertRaisesRegex((RuntimeError, OSError), "injected"):
                    r.run(archive, plan_path, output, "failure-injection-fixture")

            stops = [0, 8, 16] if failure == "checkpoint" else [0, 8, 16, 32]
            predictions = json.loads((output / "aligned-predictions.json").read_text())
            events = json.loads((output / "aligned-events.json").read_text())
            self.assertEqual([(row["learner"], row["exchange"]) for row in predictions],
                             [(language, stop) for stop in stops for language in ("ko", "ja")])
            self.assertTrue(all(row["completed_evaluation"] for row in predictions))
            self.assertEqual([row["exchange"] for row in events], list(range(1, stops[-1] + 1)))
            self.assertTrue(all(row["updates"]["ko"]["exposures"] == row["exchange"]
                                and row["updates"]["ja"]["exposures"] == row["exchange"] for row in events))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertIn("injected", summary["error"])
            self.assertFalse(summary["all_mechanistic_checks_passed"])
            self.assertFalse((output / "aligned-summary.json").exists())

    def test_checkpoint_failure_preserves_every_completed_prefix_record(self):
        self.assert_prefix_survives("checkpoint")

    def test_final_resume_failure_preserves_all_completed_prediction_and_event_records(self):
        self.assert_prefix_survives("resume")


if __name__ == "__main__":
    unittest.main()
