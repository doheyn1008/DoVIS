"""Safe, development-only restoration and read-only actual model inference."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from mutual_lab.mastery_infer import FORMAT, LEGACY_CLI_FACTORIES, infer_development, load_development_model

ROOT = Path(__file__).resolve().parents[1]


class InferenceHeaderTests(unittest.TestCase):
    def test_research_formats_flags_and_corrupt_bytes_are_rejected_before_deserialization(self):
        fake_load = Mock()
        fake_neural = types.ModuleType("mutual_lab.neural")
        fake_neural.torch = types.SimpleNamespace(load=fake_load)
        cases = [
            {"format": "mutual-gru-v1", "kind": "native"},
            {"format": FORMAT, "kind": "acquisition"},
            {"format": FORMAT, "kind": "development_only", "excluded_from_research": False},
            {"format": FORMAT, "kind": "development_only", "eligible_for_fresh_experiment": True},
            {"format": FORMAT, "kind": "development_only", "sha256": "bad"},
        ]
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"mutual_lab.neural": fake_neural}):
            path = Path(temporary) / "checkpoint.pt"
            path.write_bytes(b"not a tensor checkpoint")
            for changed in cases:
                manifest = {"format": FORMAT, "kind": "development_only", "excluded_from_research": True,
                            "eligible_for_fresh_experiment": False, "step": 1,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **changed}
                path.with_suffix(".json").write_text(json.dumps(manifest))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    load_development_model(path)
        fake_load.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional CPU torch")
class InferenceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab.neural import configure_cpu
        from mutual_lab.images import scene_png
        from mutual_lab.spatial_prior import pretrain_prior, save_prior_bundle, load_prior_bundle
        from mutual_lab.profiles import ModelProfile
        from mutual_lab.world import Object, variants
        configure_cpu()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.profile = ModelProfile("mastery-inference-test", embedding_dim=8, hidden_dim=16,
                                   max_sequence_length=128)
        cls.scenes = variants((Object("blue", "square"), Object("red", "circle")))
        cls.pngs = [scene_png(scene) for scene in cls.scenes]
        bundle = pretrain_prior(cls.pngs, dim=16, steps=2, batch_size=4, eval_every=1)
        cls.prior_path = save_prior_bundle(bundle, cls.directory / "prior.pt")
        cls.bundle = load_prior_bundle(cls.prior_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def fixture(self, name, *, phase="C", factory="default", feature_policy=None):
        from mutual_lab.attentive_spatial import build_attentive_spatial_model
        from mutual_lab.development_models import build_development_model
        from mutual_lab.development_world import make_examples
        from mutual_lab.mastery_lab import MasteryPlan
        from mutual_lab.neural import torch
        from mutual_lab.spatial_prior import build_spatial_model
        memory_source = ("objects" if factory.endswith(".F1") else "cnn") if factory.startswith(
            "mutual_lab.memory_study.RecordingFactory.") else None
        attentive = factory.endswith(("build_attentive_spatial_model", ".D1", ".attentive"))
        if phase == "A":
            model = build_development_model(self.profile, 39007)
            prior_metadata = {"kind": "fixed_pixel_random_projection"}
        elif memory_source is not None:
            from mutual_lab.object_memory import build_object_memory_model
            model = build_object_memory_model(self.profile, 39007, self.bundle, "B1", memory=memory_source)
            prior_metadata = model.spatial_prior_metadata
        else:
            builder = build_attentive_spatial_model if attentive else build_spatial_model
            model = builder(self.profile, 39007, self.bundle, "B1")
            prior_metadata = model.spatial_prior_metadata
        if feature_policy is not None:
            from mutual_lab.frozen_features import enforce_singleton_features
            enforce_singleton_features(model)
        # Save one real language-weight update, rather than testing only random initialization.
        examples = make_examples(model, self.scenes[:1], "ko")
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.003)
        optimizer.zero_grad(set_to_none=True)
        (model.grounding_loss(examples) + model.production_loss(examples)).backward()
        optimizer.step()
        payload = {"format": FORMAT, "kind": "development_only", "excluded_from_research": True,
                   "eligible_for_fresh_experiment": False, "language": "ko", "phase": phase,
                   "condition": phase + "1", "seed_pair": [39007, 39009], "step": 1,
                   "plan": asdict(MasteryPlan(budgets=(1,), model_profile=self.profile)),
                   "model_factory": factory, "model": model.state_dict(), "prior_metadata": prior_metadata,
                   "revision": {"commit": "test-fixture"}, "source_sha256": {
                       p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "mutual_lab").glob("*.py")}}
        if feature_policy is not None:
            payload["frozen_feature_policy"] = feature_policy
        path = self.save(name, payload)
        model.eval()
        return path, payload, model

    def save(self, name, payload):
        from mutual_lab.neural import torch
        path = self.directory / (name + ".pt")
        torch.save(payload, path)
        manifest = {"format": FORMAT, "kind": "development_only", "excluded_from_research": True,
                    "eligible_for_fresh_experiment": False, "step": payload["step"],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix(".json").write_text(json.dumps(manifest))
        return path

    def test_standard_and_attentive_models_restore_actual_weights_and_read_only_outputs(self):
        from mutual_lab.neural import digest, torch
        from mutual_lab.world import native_captions
        for name, factory in (("standard", "default"),
                              ("attentive", "mutual_lab.attentive_spatial.build_attentive_spatial_model")):
            with self.subTest(factory=factory):
                path, payload, original = self.fixture(name, factory=factory)
                rng = torch.random.get_rng_state().clone()
                with patch("mutual_lab.neural.torch.load", wraps=torch.load) as deserializer:
                    restored = load_development_model(path, self.prior_path)
                self.assertTrue(all(call.kwargs["weights_only"] for call in deserializer.call_args_list))
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                self.assertEqual(digest(original.state_dict()), digest(restored.state_dict()))
                self.assertFalse(restored.training)
                self.assertFalse(any(p.requires_grad for p in restored.parameters()))
                before = digest(restored.state_dict())
                output = infer_development(restored, self.pngs[0], text=native_captions(self.scenes[0], "ko")[0],
                                           choices_png=self.pngs)
                self.assertEqual(output["generation"], original.generate_with_trace(self.pngs[0]))
                self.assertEqual(output["language"], "ko")
                self.assertTrue(output["read_only_state_verified"])
                self.assertFalse(output["eligible_for_fresh_experiment"])
                self.assertEqual(digest(restored.state_dict()), before)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_explicit_study_factory_aliases_and_projection_model_are_supported(self):
        for index, factory in enumerate(("mutual_lab.decoder_study.RecordingFactory.D0",
                                         "mutual_lab.decoder_study.RecordingFactory.D1",
                                         "mutual_lab.negative_policy_study.FixedLanguageFactory.standard",
                                         "mutual_lab.negative_policy_study.FixedLanguageFactory.attentive")):
            path, _, _ = self.fixture("alias" + str(index), factory=factory)
            model = load_development_model(path, self.prior_path)
            self.assertEqual(model.mastery_inference_metadata["model_factory"], factory)
        path, _, _ = self.fixture("projection", phase="A")
        self.assertEqual(load_development_model(path).mastery_inference_metadata["phase"], "A")

    def test_legacy_ab_without_a_factory_is_supported_only_for_the_standard_tensor_layout(self):
        _, payload, _ = self.fixture("legacy-b-source", phase="B")
        del payload["model_factory"]
        path = self.save("legacy-b", payload)
        model = load_development_model(path, self.prior_path)
        self.assertEqual(model.mastery_inference_metadata["factory_provenance"], "legacy_v1_standard_layout")
        self.assertEqual(model.mastery_inference_metadata["model_factory"], "default")
        _, custom, _ = self.fixture("legacy-custom-source", phase="B",
            factory="mutual_lab.attentive_spatial.build_attentive_spatial_model")
        del custom["model_factory"]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            load_development_model(self.save("legacy-custom", custom), self.prior_path)
        _, c_payload, _ = self.fixture("missing-c-source", phase="C")
        del c_payload["model_factory"]
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_development_model(self.save("missing-c", c_payload), self.prior_path)

    def test_exact_historical_de_cli_aliases_keep_original_provenance_and_strict_layout_source_guards(self):
        for index, (recorded, canonical) in enumerate(LEGACY_CLI_FACTORIES.items()):
            with self.subTest(recorded=recorded):
                path, payload, _ = self.fixture("legacy-cli-" + str(index), factory=recorded)
                model = load_development_model(path, self.prior_path)
                metadata = model.mastery_inference_metadata
                self.assertEqual(metadata["model_factory"], canonical)
                self.assertEqual(metadata["recorded_model_factory"], recorded)
                self.assertEqual(metadata["factory_provenance"], "legacy_cli_alias_verified")
                self.assertIn("core source hashes verified", metadata["factory_compatibility_note"])
                changed = {**payload, "phase": "B", "condition": "B1"}
                with self.assertRaisesRegex(ValueError, "C engine phase"):
                    load_development_model(self.save("legacy-cli-wrong-phase-" + str(index), changed), self.prior_path)
                source = "attentive_spatial.py" if recorded.endswith((".D1", ".attentive")) else "spatial_prior.py"
                changed = {**payload, "source_sha256": {**payload["source_sha256"], source: "wrong"}}
                with self.assertRaisesRegex(ValueError, "source mismatch"):
                    load_development_model(self.save("legacy-cli-wrong-source-" + str(index), changed), self.prior_path)
        _, standard, _ = self.fixture("legacy-cli-standard-layout", factory="__main__.RecordingFactory.D0")
        standard["model_factory"] = "__main__.RecordingFactory.D1"
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            load_development_model(self.save("legacy-cli-wrong-layout", standard), self.prior_path)
        for unknown in ("__main__.RecordingFactory.D2", "__main__.UnknownFactory.attentive",
                        "__main__.mutual_lab.decoder_study.RecordingFactory.D0"):
            with self.subTest(unknown=unknown), self.assertRaisesRegex(ValueError, "Unsupported"):
                load_development_model(self.save("unknown-cli-alias", {**standard, "model_factory": unknown}), self.prior_path)

    def test_relevant_sources_and_factory_consistency_are_checked_but_unrelated_files_are_ignored(self):
        path, payload, _ = self.fixture("metadata")
        for name, changed, message in (
            ("bad-source", {"source_sha256": {**payload["source_sha256"], "neural.py": "wrong"}}, "source mismatch"),
            ("bad-factory", {"model_factory": "untrusted.module.factory"}, "Unsupported"),
            ("bad-seed", {"seed_pair": [12345, 39009]}, "seed pair"),
        ):
            candidate = self.save(name, {**payload, **changed})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                load_development_model(candidate, self.prior_path)
        other = self.save("unrelated-module", {**payload, "source_sha256": {
            **payload["source_sha256"], "development_world.py": "older-evaluation-source"}})
        self.assertIsNotNone(load_development_model(other, self.prior_path))
        _, attention, _ = self.fixture("attention-metadata",
            factory="mutual_lab.attentive_spatial.build_attentive_spatial_model")
        ambiguous = self.save("ambiguous", {**attention, "model_factory": "default"})
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            load_development_model(ambiguous, self.prior_path)
        with self.assertRaisesRegex(ValueError, "required"):
            load_development_model(path)

    def test_wrong_prior_pool_and_changed_frozen_parameters_are_rejected(self):
        from mutual_lab.spatial_prior import pretrain_prior, save_prior_bundle
        path, payload, _ = self.fixture("prior-check")
        wrong = save_prior_bundle(pretrain_prior(self.pngs[:2], dim=16, steps=1, batch_size=2),
                                  self.directory / "wrong-prior.pt")
        with self.assertRaisesRegex(ValueError, "fit image pool|fit PNG pool"):
            load_development_model(path, wrong)
        changed = copy.deepcopy(payload)
        key = next(name for name in changed["model"] if name.startswith("visual_encoder."))
        changed["model"][key] += .1
        candidate = self.save("changed-vision", changed)
        with self.assertRaisesRegex(ValueError, "visual parameters"):
            load_development_model(candidate, self.prior_path)

    def test_cli_generates_png_only_json_without_changing_checkpoint_files(self):
        path, _, _ = self.fixture("cli")
        png = self.directory / "example.png"
        png.write_bytes(self.pngs[0])
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        process = subprocess.run([sys.executable, "-m", "mutual_lab.mastery_infer", "--checkpoint", str(path),
                                  "--prior-bundle", str(self.prior_path), "--image", str(png)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        output = json.loads(process.stdout)
        self.assertTrue(output["excluded_from_research"])
        self.assertIn("text", output["generation"])
        self.assertTrue(output["read_only_state_verified"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_inference_rejects_weights_changed_after_checkpoint_restoration(self):
        from mutual_lab.neural import torch
        path, _, _ = self.fixture("mutated-after-load")
        model = load_development_model(path, self.prior_path)
        with torch.no_grad():
            next(model.producer.parameters()).add_(.1)
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            infer_development(model, self.pngs[0])

    def test_explicit_singleton_policy_restores_canonical_features_and_rejects_unknown_or_changed_helper(self):
        from mutual_lab.frozen_features import FROZEN_FEATURE_POLICY
        from mutual_lab.neural import digest, torch
        for index, factory in enumerate(("default", "mutual_lab.attentive_spatial.build_attentive_spatial_model",
                "mutual_lab.memory_study.RecordingFactory.F0", "mutual_lab.memory_study.RecordingFactory.F1")):
            with self.subTest(factory=factory):
                path, payload, original = self.fixture("singleton-" + str(index), factory=factory,
                                                        feature_policy=FROZEN_FEATURE_POLICY)
                restored = load_development_model(path, self.prior_path)
                self.assertEqual(restored.mastery_inference_metadata["frozen_feature_policy"], FROZEN_FEATURE_POLICY)
                self.assertEqual(digest(original.state_dict()), digest(restored.state_dict()))
                with torch.no_grad():
                    self.assertTrue(torch.equal(original.visual(self.pngs), restored.visual(self.pngs)))
                self.assertEqual(infer_development(restored, self.pngs[0])["generation"],
                                 original.generate_with_trace(self.pngs[0]))
                if factory.endswith((".F0", ".F1")):
                    self.assertEqual(restored.object_memory_source, "objects" if factory.endswith(".F1") else "cnn")
                    changed = {**payload, "source_sha256": {**payload["source_sha256"], "object_memory.py": "wrong"}}
                    with self.assertRaisesRegex(ValueError, "object_memory.py"):
                        load_development_model(self.save("wrong-object-source", changed), self.prior_path)
                for policy in ("unknown", None):
                    with self.assertRaisesRegex(ValueError, "frozen feature policy"):
                        load_development_model(self.save("unknown-policy", {**payload, "frozen_feature_policy": policy}), self.prior_path)
                changed = {**payload, "source_sha256": {**payload["source_sha256"], "frozen_features.py": "wrong"}}
                with self.assertRaisesRegex(ValueError, "frozen_features.py"):
                    load_development_model(self.save("wrong-helper-source", changed), self.prior_path)
        path, _, _ = self.fixture("legacy-no-cache-policy")
        restored = load_development_model(path, self.prior_path)
        self.assertFalse(hasattr(restored, "frozen_feature_policy"))
        self.assertEqual(restored.mastery_inference_metadata["frozen_feature_policy"], "legacy_unspecified")

    def test_cli_preserves_real_unicode_bmp_surrogate_trace_with_strict_utf8_stdout(self):
        from mutual_lab.neural import torch
        from mutual_lab.spatial_prior import build_spatial_model
        _, payload, _ = self.fixture("surrogate-header-source")
        profile = replace(self.profile, tokenizer="unicode-bmp")
        model = build_spatial_model(profile, 39007, self.bundle, "B1")
        surrogate_token = 0xD800 + 3
        # Exercise the actual greedy decoder/tokenizer with a repeatable invalid
        # BMP output, rather than replacing the trace or stdout serializer.
        with torch.no_grad():
            model.producer.output.weight.zero_()
            model.producer.output.bias.zero_()
            model.producer.output.bias[surrogate_token] = 1000
        payload["plan"]["model_profile"] = asdict(profile)
        payload["model"] = model.state_dict()
        path = self.save("surrogate-checkpoint", payload)
        png = self.directory / "surrogate-input.png"
        png.write_bytes(self.pngs[0])
        process = subprocess.run([sys.executable, "-m", "mutual_lab.mastery_infer", "--checkpoint", str(path),
                                  "--prior-bundle", str(self.prior_path), "--image", str(png)],
            cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8:strict"}, capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr.decode("utf-8", errors="strict"))
        output = json.loads(process.stdout.decode("utf-8", errors="strict"))
        trace = output["generation"]
        self.assertTrue(trace["invalid_encoding"])
        self.assertFalse(trace["terminated"])
        self.assertEqual(trace["text"], "\ud800" * (profile.max_sequence_length - 2))
        self.assertEqual(trace["token_ids"], [surrogate_token] * (profile.max_sequence_length - 2))
        self.assertIn(b"\\ud800", process.stdout)
        self.assertTrue(all(value < 128 for value in process.stdout))
        self.assertTrue(output["read_only_state_verified"])


if __name__ == "__main__":
    unittest.main()
