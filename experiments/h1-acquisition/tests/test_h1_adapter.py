import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
CORE = HERE.parent / "mutual-understanding"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CORE))
HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "PIL"))
from h1_adapter import H1DevelopmentLearner, FORMAT


@unittest.skipUnless(HAS_TORCH, "requires optional CPU torch, NumPy and Pillow")
class H1AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab.neural import torch, configure_cpu
        from mutual_lab.mastery_lab import MasteryPlan
        from mutual_lab.profiles import ModelProfile
        from mutual_lab.producer_context import build_producer_context_model
        from mutual_lab.spatial_prior import pretrain_prior, save_prior_bundle, load_prior_bundle
        from mutual_lab.development_world import make_examples, render_scene
        from mutual_lab.world import Object, variants
        configure_cpu()
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.scenes = variants((Object("blue", "square"), Object("red", "circle")))
        cls.pngs = [render_scene(scene) for scene in cls.scenes]
        cls.prior = save_prior_bundle(pretrain_prior(cls.pngs, dim=128, steps=1, batch_size=4,
                                                    eval_every=1), cls.root / "prior.pt")
        cls.bundle = load_prior_bundle(cls.prior)
        cls.profile = ModelProfile("h1-adapter-test", embedding_dim=64, hidden_dim=128)
        cls.plan = asdict(MasteryPlan(budgets=(2,), b_seed_pair=(79007, 79009), model_profile=cls.profile))
        cls.initial = build_producer_context_model(cls.profile, 79007, cls.bundle)
        model = build_producer_context_model(cls.profile, 79007, cls.bundle)
        examples = make_examples(model, cls.scenes, "ko")
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (model.grounding_loss(examples) + model.production_loss(examples)).backward()
            optimizer.step()
        cls.payload = {"format": "mutual-mastery-development-v1", "kind": "development_only",
            "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "language": "ko", "phase": "C", "condition": "C1", "replica": 0,
            "seed_pair": [79007, 79009], "step": 2, "plan": cls.plan,
            "model_factory": "mutual_lab.producer_context_study.RecordingFactory.H1",
            "model": copy.deepcopy(model.state_dict()), "optimizer": optimizer.state_dict(),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (CORE / "mutual_lab").glob("*.py")},
            "prior_metadata": model.spatial_prior_metadata, "frozen_feature_policy": "per_png_singleton_v1"}
        cls.checkpoint = cls.write_checkpoint("h1", cls.payload)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def write_checkpoint(cls, name, payload):
        from mutual_lab.neural import torch
        path = cls.root / (name + ".pt"); torch.save(payload, path)
        path.with_suffix(".json").write_text(json.dumps({"format": payload["format"],
            "kind": payload["kind"], "excluded_from_research": payload["excluded_from_research"],
            "eligible_for_fresh_experiment": payload["eligible_for_fresh_experiment"], "step": payload["step"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        return path

    def learner(self, **options):
        value = H1DevelopmentLearner.from_checkpoint(self.checkpoint, self.prior,
            allow_development_training=True, seed=99101, **options)
        value.warm(self.pngs)
        return value

    def test_explicit_optin_language_h1_source_and_research_guards(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            H1DevelopmentLearner.from_checkpoint(self.checkpoint, self.prior)
        with self.assertRaisesRegex(ValueError, "language"):
            H1DevelopmentLearner.from_checkpoint(self.checkpoint, self.prior,
                allow_development_training=True, language="ja")
        for field, bad in (("model_factory", "mutual_lab.producer_context_study.RecordingFactory.H0"),
                           ("eligible_for_fresh_experiment", True)):
            payload = copy.deepcopy(self.payload); payload[field] = bad
            path = self.write_checkpoint("bad-source", payload)
            with self.subTest(field=field), self.assertRaises(ValueError):
                H1DevelopmentLearner.from_checkpoint(path, self.prior, allow_development_training=True)
        payload = copy.deepcopy(self.payload); payload["source_sha256"]["producer_context.py"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            H1DevelopmentLearner.from_checkpoint(self.write_checkpoint("bad-hash", payload), self.prior,
                allow_development_training=True)

    def test_real_updates_reach_base_and_attention_but_freeze_every_other_tensor(self):
        from mutual_lab.neural import digest
        learner = self.learner(); before = learner.branch_digests()
        base = digest(learner.model.understanding.base.state_dict())
        attention = digest({n: p for n, p in learner.model.understanding.state_dict().items() if not n.startswith("base.")})
        answer = learner.update("The blue square is left of the red circle.", self.pngs[0], self.pngs[1:], updates=2)
        self.assertEqual(answer["counters"]["exposures"], 1)
        self.assertEqual(answer["counters"]["updates"], 2)
        self.assertTrue(answer["excluded_from_research"])
        self.assertFalse(answer["eligible_for_l2"])
        self.assertNotEqual(base, digest(learner.model.understanding.base.state_dict()))
        self.assertNotEqual(attention, digest({n: p for n, p in learner.model.understanding.state_dict().items() if not n.startswith("base.")}))
        after = learner.branch_digests()
        for branch in ("producer", "visual", "frozen"):
            self.assertEqual(before[branch], after[branch])
        self.assertEqual(len(learner.optimizer.state), len(list(learner.model.understanding.parameters())))

    def test_no_update_records_exposure_without_weights_adam_or_rng_changes(self):
        from mutual_lab.neural import digest
        learner = self.learner(); before = learner.state_dict()
        learner.update("A foreign utterance.", self.pngs[0], self.pngs[1:], updates=0)
        after = learner.state_dict()
        for field in ("weights", "optimizer", "rng", "torch_rng", "feature_caches"):
            self.assertEqual(digest(before[field]), digest(after[field]), field)
        self.assertEqual(after["counters"]["exposures"], 1)
        self.assertEqual(after["counters"]["updates"], 0)

    def test_full_h1_reset_matches_exact_constructor_and_keeps_trained_producer(self):
        from mutual_lab.neural import digest
        normal, reset = self.learner(), self.learner(reset_understanding=True)
        self.assertNotEqual(normal.branch_digests()["understanding"], reset.branch_digests()["understanding"])
        self.assertEqual(digest(self.initial.understanding.state_dict()), reset.branch_digests()["understanding"])
        self.assertEqual(normal.branch_digests()["frozen"], reset.branch_digests()["frozen"])
        self.assertEqual(set(normal.model.understanding.state_dict()), set(reset.model.understanding.state_dict()))

    def test_evaluation_is_readonly_after_update_including_private_and_global_rng(self):
        from mutual_lab.neural import torch, digest
        learner = self.learner()
        learner.update("The blue square is left of the red circle.", self.pngs[0], self.pngs[1:])
        before, cache, rng = learner.state_digest(), learner.cache_digest(), torch.random.get_rng_state().clone()
        trace = learner.emit(self.pngs[0])
        self.assertIn("text", trace)
        batch = learner.predict_many(["A square.", "A circle."], [self.pngs, self.pngs])
        self.assertEqual(len(batch), 2)
        self.assertEqual(learner.predict("A square.", self.pngs), batch[0])
        self.assertEqual(learner.state_digest(), before)
        self.assertEqual(learner.cache_digest(), cache)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        snapshot = learner.state_dict()
        self.assertEqual(digest(snapshot["torch_rng"]), digest(learner.torch_rng))

    def test_safe_serialization_and_exact_continuation(self):
        from mutual_lab.neural import torch
        full, split = self.learner(), self.learner()
        texts = ["The square is left of the circle.", "The circle is right of the square.", "A red circle."]
        for text in texts:
            full.update(text, self.pngs[0], self.pngs[1:])
        split.update(texts[0], self.pngs[0], self.pngs[1:])
        path = self.root / "pilot.pt"; torch.save(split.state_dict(), path)
        restored = self.learner(); restored.load_state_dict(torch.load(path, weights_only=True))
        self.assertEqual(split.state_digest(), restored.state_digest())
        self.assertEqual(split.cache_digest(), restored.cache_digest())
        for text in texts[1:]:
            restored.update(text, self.pngs[0], self.pngs[1:])
        self.assertEqual(full.state_digest(), restored.state_digest())
        self.assertEqual(full.emit(self.pngs[0]), restored.emit(self.pngs[0]))

    def test_snapshot_corruption_and_native_promotion_are_rejected_without_mutation(self):
        from mutual_lab.neural import load_checkpoint, torch
        learner = self.learner(); learner.update("An utterance.", self.pngs[0], self.pngs[1:])
        good = learner.state_dict(); before = learner.state_digest()
        for case in ("eligible", "binding", "options", "frozen", "moment", "missing_state", "cache", "rng"):
            value = copy.deepcopy(good)
            if case == "eligible": value["eligible_for_l2"] = True
            elif case == "binding": value["bindings"]["prior_sha256"] = "f" * 64
            elif case == "options": value["options"]["reset_understanding"] = True
            elif case == "frozen": value["weights"]["producer.embedding.weight"][0, 0] += 1
            elif case == "moment": value["optimizer"]["state"][0]["exp_avg"][0, 0] = float("nan")
            elif case == "missing_state": del value["optimizer"]["state"][0]
            elif case == "cache":
                key = next(iter(value["feature_caches"]["visual_cache"]))
                value["feature_caches"]["visual_cache"][key][0] += .1
            else: value["rng"] = ()
            with self.subTest(case=case), self.assertRaises(ValueError): learner.load_state_dict(value)
            self.assertEqual(learner.state_digest(), before)
        path = self.root / "not-native.pt"; torch.save(good, path)
        path.with_suffix(".json").write_text(json.dumps({"format": FORMAT, "kind": "development_only",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        for kind in ("native", "acquisition"):
            with self.assertRaisesRegex(ValueError, "format"):
                load_checkpoint(path, expected_kind=kind)

    def test_invalid_update_inputs_do_not_change_state(self):
        learner = self.learner(); before = learner.state_digest()
        for text, target, negatives in (("", self.pngs[0], self.pngs[1:]),
                ("A square.", self.scenes[0], self.pngs[1:]), ("A square.", self.pngs[0], self.pngs[:3])):
            with self.assertRaises(ValueError): learner.update(text, target, negatives)
            self.assertEqual(learner.state_digest(), before)


if __name__ == "__main__":
    unittest.main()
