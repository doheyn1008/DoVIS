import importlib.util
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mutual_lab.agents import SharedImage, choose, observe_persistent, speak
from mutual_lab.cli import main
from mutual_lab.config import AgentConfig, ExperimentConfig
from mutual_lab.grid import GridConfig, aggregate_grid, plan_grid
from mutual_lab.images import scene_png
from mutual_lab.metrics import (classify_initial, common_curve_metrics, compare_common_auc,
                                corrected_accuracy, evaluate, raw_threshold, summarize_direction)
from mutual_lab.profiles import AcquisitionProfile, EvaluationProfile, ModelProfile, NativeTrainingProfile
from mutual_lab.world import make_research_dataset, native_captions


def tiny_config():
    return ExperimentConfig(schema_version=2, max_exchanges=2, eval_every=1, eval_items=4,
                            stop_on_threshold=False,
                            model_profile=ModelProfile("test-gru", embedding_dim=8, hidden_dim=16),
                            native_training_profile=NativeTrainingProfile(steps=3, batch_size=2),
                            evaluation_profile=EvaluationProfile(pretest_items=4, final_items=4, native_items=4))


class ProtocolTests(unittest.TestCase):
    def test_correction_thresholds_and_negative_values(self):
        self.assertEqual(corrected_accuracy(.25), 0)
        self.assertEqual(corrected_accuracy(1), 1)
        self.assertLess(corrected_accuracy(0), 0)
        self.assertAlmostEqual(raw_threshold(.9, "corrected"), .925)
        self.assertEqual(raw_threshold(.9, "raw"), .9)
        with self.assertRaises(ValueError):
            raw_threshold(.9, "unknown")

    def test_common_auc_rejects_unequal_early_stop_windows(self):
        points = [{"exchange": 0, "accuracy": 0}, {"exchange": 5, "accuracy": .5},
                  {"exchange": 10, "accuracy": 1}]
        full = common_curve_metrics(points, 10)
        self.assertEqual(full["normalized_auc"], .5)
        self.assertEqual(full["fixed_exposure_accuracy"], 1)
        early = common_curve_metrics(points[:2], 10)
        with self.assertRaises(ValueError):
            compare_common_auc([{"common_curve": full}, {"common_curve": early}])
        self.assertEqual(compare_common_auc([{"common_curve": full}] * 2), [.5, .5])

    def test_failure_cancellation_and_unreached_are_distinct(self):
        point = {"exchange": 0, "accuracy": .25, "source": "A", "listener": "B",
                 "source_language": "ja", "listener_language": "ko", "wilson_low": 0, "wilson_high": 1, "n": 4}
        for status in ("completed", "failed", "cancelled"):
            d = summarize_direction([point], .9, 2, status=status)
            self.assertEqual(d["learner_native_language"], "ko")
            self.assertEqual(d["target_language"], "ja")
            self.assertEqual(d["right_censored"], status == "completed")
            self.assertEqual(d["threshold_outcome"], "right_censored" if status == "completed" else status)

    def test_prescreen_requires_equivalence_and_native_provenance(self):
        p = {"listener_language": "ko", "source_language": "ja", "accuracy": .25,
             "corrected_accuracy": 0, "wilson_low": .1, "wilson_high": .6,
             "screening_low": .1, "screening_high": .6}
        def gate(points, eligible=(True, True), verified=True):
            return classify_initial(points, list(eligible), .1, verified_foreign_exposure=verified)["analysis_group"]
        self.assertEqual(gate([p, p]), "uncertain")
        equivalent = {**p, "screening_low": .2, "screening_high": .3}
        self.assertEqual(gate([equivalent, equivalent]), "primary")
        positive = {**p, "screening_low": .4, "screening_high": .6}
        self.assertEqual(gate([positive, equivalent]), "positive_baseline_extension")
        self.assertEqual(gate([equivalent, equivalent], (False, True)), "ineligible")
        self.assertEqual(gate([equivalent, equivalent], verified=False), "ineligible")

    def test_all_five_splits_are_pair_disjoint(self):
        data = make_research_dataset(tiny_config())
        groups = [{s.group for s in data.train}]
        for split in ("native", "pretest", "curve", "final"):
            probes = data.view(split).probes
            groups.append({p.scene.group for p in probes})
            self.assertTrue(all(all(c.group == p.scene.group for c in p.choices) for p in probes))
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                self.assertFalse(a & b)
        self.assertEqual(data.fingerprint, make_research_dataset(tiny_config()).fingerprint)
        with self.assertRaises(ValueError):
            make_research_dataset(replace(tiny_config(), eval_items=140))

    def test_spanish_and_profiles_validate_end_to_end(self):
        from mutual_lab.prompts import PROMPTS
        data = make_research_dataset(tiny_config())
        for language in ("en", "ko", "ja", "es"):
            captions = native_captions(data.train[0], language)
            self.assertEqual(len(set(captions)), 4)
            self.assertIn(language, PROMPTS)
        for change in ({"model_profile": {"architecture": "unimplemented"}},
                       {"native_training_profile": {"corpus_version": "missing"}},
                       {"acquisition_profile": {"optimizer": "missing"}},
                       {"evaluation_profile": {"id": "missing"}}):
            with self.assertRaises(ValueError):
                ExperimentConfig.from_dict(change)

    def test_grid_combinations_seeds_and_replica_hierarchy(self):
        grid = GridConfig(experiment=tiny_config()).validate()
        plan = plan_grid(grid)
        self.assertEqual(plan["budget"]["pair_runs"], 24)
        self.assertEqual(plan["budget"]["direction_curves"], 48)
        self.assertEqual(len({t["task_id"] for t in plan["tasks"]}), 24)
        self.assertEqual(plan, plan_grid(GridConfig.from_dict(grid.to_dict())))
        paired = [t for t in plan["tasks"] if t["model_replica_id"] == "r0" and t["experience_replica_id"] == "d0"]
        self.assertEqual(len(paired), 6)
        self.assertEqual(len({t["initialization_seed"] for t in paired}), 1)
        self.assertEqual(len({t["dialogue_seed"] for t in paired}), 1)
        with self.assertRaises(ValueError):
            GridConfig(languages=("ko", "missing")).validate()
        with self.assertRaises(ValueError):
            GridConfig(languages=("ko", "ko")).validate()

    def test_grid_aggregate_does_not_promote_dialogues_to_independent_models(self):
        direction = {"learner_native_language": "ko", "target_language": "ja", "threshold_outcome": "right_censored",
                     "sustained_threshold_started_exchange": None}
        rows = [{"model_profile_sha256": p, "analysis_group": "uncertain", "model_replica_id": "r0",
                 "experience_replica_id": f"d{i}", "run_id": str(i), "directions": [direction]}
                for p in ("small", "large") for i in range(3)]
        result = aggregate_grid(rows)["directions"]
        self.assertEqual(len(result), 2)
        self.assertTrue(all(g["n_independent_models"] == 1 for g in result))
        self.assertTrue(all(g["model_replicas"][0]["n_experiences"] == 3 for g in result))


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("torch", "PIL", "numpy")), "optional neural dependencies")
class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab.neural import train_native
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.config = tiny_config()
        cls.paths = {}
        for language in ("ko", "ja", "es", "en"):
            cls.paths[language] = train_native(cls.config, language, cls.root / f"{language}.pt",
                                             initialization_seed=7, native_training_seed=11)
        cls.config = replace(cls.config, agents=tuple(AgentConfig(name, lang, "neural", native_checkpoint=str(cls.paths[lang]))
                             for name, lang in (("A", "ko"), ("B", "ja"))))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def agent(self, config=None):
        from mutual_lab.neural import NeuralAgent
        config = config or self.config
        return NeuralAgent(config.agents[0], config)

    def test_l1_weights_are_reused_and_initialization_is_paired(self):
        from mutual_lab.neural import digest, load_checkpoint
        a = self.agent()
        state = load_checkpoint(self.paths["ko"])
        self.assertEqual(digest(a.model.understanding.state_dict()), digest({
            k.removeprefix("understanding."): v for k, v in state["weights"].items() if k.startswith("understanding.")}))
        self.assertNotEqual(digest(a.model.understanding.state_dict()), digest(state["initial_understanding"]))
        other = load_checkpoint(self.paths["ja"])
        self.assertEqual(state["metadata"]["initial_understanding_sha256"], other["metadata"]["initial_understanding_sha256"])
        self.assertNotEqual(state["metadata"]["native_data_sha256"], other["metadata"]["native_data_sha256"])
        self.assertEqual(state["metadata"]["training_languages"], ["ko"])

    def test_same_target_stream_updates_the_actual_different_l1_receivers(self):
        from mutual_lab.neural import NeuralAgent, digest
        english = AgentConfig("B", "en", "neural", native_checkpoint=str(self.paths["en"]))
        a, b = self.agent(), NeuralAgent(english, replace(self.config, agents=(self.config.agents[0], english)))
        scene = make_research_dataset(self.config).train[0]
        text = native_captions(scene, "ja")[0]
        for agent in (a, b):
            prior = digest(agent.model.understanding.gru.state_dict())
            observe_persistent(agent, scene, text)
            self.assertNotEqual(prior, digest(agent.model.understanding.gru.state_dict()))
            self.assertTrue(all(any(param is p for group in agent.optimizer.param_groups for p in group["params"])
                                for param in agent.model.understanding.gru.parameters()))
        # No assertion about which L1 learns faster: that is a research outcome.

    def test_fixed_tokenization_sensitivity_has_no_fitted_foreign_vocabulary(self):
        from mutual_lab.neural import Tokenizer
        byte = Tokenizer(self.config.model_profile)
        char = Tokenizer(replace(self.config.model_profile, tokenizer="unicode-bmp"))
        for text in ("red circle", "빨간 원", "赤い円", "círculo rojo"):
            self.assertEqual(byte.decode(byte.encode(text)), text)
            self.assertEqual(char.decode(char.encode(text)), text)
        self.assertGreater(len(byte.encode("빨간 원")), len(char.encode("빨간 원")))
        with self.assertRaises(ValueError):
            char.encode("😀")
        with self.assertRaises(ValueError):
            byte.encode("x" * 256)

    def test_persistent_learning_restore_and_evaluation_are_read_only(self):
        from mutual_lab.neural import NeuralAgent
        a = self.agent()
        data = make_research_dataset(self.config)
        scene = data.train[0]
        text = native_captions(scene, "ja")[0]
        producer = speak(a, scene)
        before = a.state_digest()
        observe_persistent(a, scene, text)
        self.assertNotEqual(before, a.state_digest())
        self.assertEqual(producer, speak(a, scene))  # native generation is frozen
        path = a.save(self.root / "acquisition.pt")
        b = NeuralAgent.restore(path, self.config.agents[0], self.config)
        self.assertEqual(a.state_digest(), b.state_digest())
        probe = data.probes[0]
        state = a.state_digest()
        selected = choose(a, text, probe.choices, random.Random(1))
        self.assertEqual(state, a.state_digest())
        # Inference uses weights, not a transcript or replay lookup.
        a.foreign_buffer.clear()
        self.assertEqual(selected, choose(a, text, probe.choices, random.Random(1)))
        a = NeuralAgent.restore(path, self.config.agents[0], self.config)
        observe_persistent(a, data.train[1], text)
        observe_persistent(b, data.train[1], text)
        self.assertEqual(a.state_digest(), b.state_digest())  # optimizer, buffer and RNG restored too

    def test_zero_update_reset_and_label_only_controls(self):
        from mutual_lab.neural import digest, load_checkpoint
        scene = make_research_dataset(self.config).train[0]
        text = native_captions(scene, "ja")[0]
        zero = self.agent(replace(self.config, acquisition_profile=AcquisitionProfile(updates_per_exposure=0)))
        before = digest(zero.model.state_dict())
        observe_persistent(zero, scene, text)
        self.assertEqual(before, digest(zero.model.state_dict()))
        self.assertEqual(zero.updates, 0)
        reset = self.agent(replace(self.config, acquisition_profile=AcquisitionProfile(reset_l1_understanding=True)))
        self.assertEqual(digest(reset.model.understanding.state_dict()), digest(load_checkpoint(self.paths["ko"])["initial_understanding"]))
        a, b = self.agent(), self.agent()
        b.config = replace(b.config, language="en")  # label alone cannot change the learning rule
        observe_persistent(a, scene, text)
        observe_persistent(b, scene, text)
        self.assertEqual(a.state_digest(), b.state_digest())

    def test_incompatible_and_tampered_checkpoints_rejected(self):
        from mutual_lab.neural import NeuralAgent, load_checkpoint
        with self.assertRaises(ValueError):
            NeuralAgent(replace(self.config.agents[0], language="es"), self.config)
        with self.assertRaises(ValueError):
            self.agent(replace(self.config, model_profile=replace(self.config.model_profile, hidden_dim=32)))
        with self.assertRaises(ValueError):
            self.agent(replace(self.config, scene_split_seed=42))
        with self.assertRaises(ValueError):
            self.agent(replace(self.config, model_replica_id="r1"))
        path = self.root / "bad.pt"
        path.write_bytes(self.paths["ko"].read_bytes() + b"tampered")
        path.with_suffix(".json").write_bytes(self.paths["ko"].with_suffix(".json").read_bytes())
        with self.assertRaises(ValueError):
            load_checkpoint(path)

    def test_spanish_run_has_independent_tests_and_ineligible_reasons(self):
        from mutual_lab.runner import run_experiment
        config = replace(self.config, agents=(self.config.agents[0], AgentConfig("B", "es", "neural",
                                  native_checkpoint=str(self.paths["es"]))))
        path = run_experiment(config, self.root / "runs")
        summary = json.loads((path / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["inclusion"]["analysis_group"], "ineligible")
        self.assertEqual(len(summary["pretest"]), 2)
        self.assertEqual(len(summary["final_evaluation"]), 2)
        self.assertEqual(summary["exposure"]["A"]["updates"], 2)
        self.assertTrue((path / "learning_checkpoint.json").exists())
        for d in summary["directions"]:
            self.assertTrue(d["common_curve"]["available"])
            self.assertIn(d["target_language"], {"ko", "es"})

    def test_shuffled_and_zero_update_runs_are_separate_controls(self):
        from mutual_lab.runner import run_experiment
        from mutual_lab.neural import digest, load_checkpoint
        for acquisition in (AcquisitionProfile(association="shuffled"), AcquisitionProfile(updates_per_exposure=0)):
            config = replace(self.config, acquisition_profile=acquisition)
            path = run_experiment(config, self.root / "controls")
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual(summary["inclusion"]["analysis_group"], "control")
            if acquisition.association == "shuffled":
                events = [json.loads(s) for s in (path / "events.jsonl").read_text().splitlines()]
                self.assertTrue(all(e["image"] != e["learning_image"] for e in events if e["type"] == "dialogue"))
            else:
                checkpoint = json.loads((path / "learning_checkpoint.json").read_text())
                state = load_checkpoint(path / checkpoint["agents"]["A"])
                self.assertEqual(digest(state["weights"]), digest(load_checkpoint(self.paths["ko"])["weights"]))
                self.assertEqual(summary["exposure"]["A"]["updates"], 0)

    def test_resume_reproduces_uninterrupted_weights_and_curves(self):
        from mutual_lab.runner import ExperimentFailed, run_experiment
        from mutual_lab.research import run_neural_experiment
        full = run_experiment(self.config, self.root / "runs")
        def stop(event):
            if event.get("exchange") == 1:
                raise KeyboardInterrupt
        with self.assertRaises(ExperimentFailed) as caught:
            run_experiment(self.config, self.root / "runs", stop)
        resumed = run_neural_experiment(self.config, self.root / "runs", resume_from=caught.exception.run_path)
        for name in ("evaluations.json", "final_evaluations.json"):
            self.assertEqual(json.loads((full / name).read_text()), json.loads((resumed / name).read_text()))
        a = json.loads((full / "learning_checkpoint.json").read_text())
        b = json.loads((resumed / "learning_checkpoint.json").read_text())
        self.assertEqual(a["state_sha256"], b["state_sha256"])

    def test_grid_resume_reuses_native_models_but_resets_l2_for_each_experience(self):
        from mutual_lab.grid import run_grid
        config = replace(self.config, max_exchanges=1, eval_every=1)
        grid = GridConfig(languages=("ko", "es"), model_replicas=1, experience_repeats=2,
                          model_profiles=(config.model_profile,), experiment=config)
        root = run_grid(grid, self.root / "gridruns", limit=1)
        outcomes = json.loads((root / "outcomes.json").read_text())
        self.assertEqual(outcomes["completed"], 1)
        self.assertEqual(outcomes["pending"], 1)
        run_grid(grid, self.root / "gridruns", resume_dir=root)
        statuses = json.loads((root / "grid.json").read_text())["statuses"].values()
        self.assertTrue(all(r["status"] == "completed" for r in statuses), list(statuses))
        summaries = [json.loads((Path(r["run_path"]) / "summary.json").read_text()) for r in statuses]
        for index in (0, 1):
            self.assertEqual(summaries[0]["native_models"][index]["native_checkpoint_id"],
                             summaries[1]["native_models"][index]["native_checkpoint_id"])
            self.assertEqual(summaries[1]["exposure"]["A"]["exposures"], 1)


if __name__ == "__main__":
    unittest.main()
