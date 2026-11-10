import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mutual_lab.agents import SymbolicAgent, observe, speak
from mutual_lab.cli import main, sweep
from mutual_lab.config import ExperimentConfig
from mutual_lab.metrics import evaluate, summarize_direction, wilson
from mutual_lab.runner import ExperimentFailed, run_experiment
from mutual_lab.world import make_dataset


class LearningTests(unittest.TestCase):
    def test_evaluation_is_read_only_and_training_generalizes(self):
        config = ExperimentConfig()
        dataset = make_dataset(config)
        agents = [SymbolicAgent(item, config) for item in config.agents]
        captions = {agent.config.name: [speak(agent, p.scene, frozen=True) for p in dataset.probes] for agent in agents}
        before = [agent.state_digest() for agent in agents]
        baseline = evaluate(agents, dataset, captions, 0, config.seed)
        self.assertEqual(before, [agent.state_digest() for agent in agents])
        self.assertTrue(all(point["accuracy"] < .5 for point in baseline))
        # Listeners receive only partner utterances and training scenes, no held-out answers/dictionaries.
        for scene in dataset.train:
            observe(agents[0], scene, speak(agents[1], scene))
            observe(agents[1], scene, speak(agents[0], scene))
        before = [agent.state_digest() for agent in agents]
        learned = evaluate(agents, dataset, captions, len(dataset.train), config.seed)
        self.assertTrue(all(point["accuracy"] >= .9 for point in learned), learned)
        self.assertEqual(before, [agent.state_digest() for agent in agents])
        self.assertEqual(learned, evaluate(agents, dataset, captions, len(dataset.train), config.seed))

    def test_threshold_requires_consecutive_checks_and_preserves_censoring(self):
        def point(exchange, accuracy):
            return {"exchange": exchange, "accuracy": accuracy, "source": "A", "listener": "B",
                    "source_language": "en", "listener_language": "ko", "wilson_low": 0, "wilson_high": 1, "n": 4}
        points = [point(0, .95), point(5, .5), point(10, 1), point(15, .95)]
        result = summarize_direction(points, .9, 2)
        self.assertEqual(result["first_observed_threshold_exchange"], 0)
        self.assertEqual(result["sustained_threshold_started_exchange"], 10)
        self.assertEqual(result["sustained_threshold_confirmed_exchange"], 15)
        self.assertFalse(result["right_censored"])
        censored = summarize_direction(points[:3], .9, 2)
        self.assertTrue(censored["right_censored"])
        self.assertIsNone(censored["sustained_threshold_started_exchange"])
        self.assertEqual(censored["censoring_exchange"], 10)
        self.assertAlmostEqual(summarize_direction([point(0, 0), point(10, 1)], .9, 2)["normalized_auc_observed"], .5)

    def test_wilson_includes_extreme_observations(self):
        low, high = wilson(0, 24)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, .1)
        low, high = wilson(24, 24)
        self.assertLess(low, .9)
        self.assertAlmostEqual(high, 1)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = replace(ExperimentConfig(), max_exchanges=7, eval_every=3, eval_items=8,
                              sustained_checks=100, stop_on_threshold=False)

    def test_run_artifacts_baseline_final_checkpoint_and_reproduction(self):
        path = run_experiment(self.config, self.root)
        other = run_experiment(self.config, self.root)
        self.assertNotEqual(path, other)
        required = {"manifest.json", "summary.json", "config.json", "prompts.json", "dataset.json",
                    "events.jsonl", "metrics.csv", "evaluations.json", "frozen_captions.json", "assets"}
        self.assertTrue(required <= {file.name for file in path.iterdir()})
        summary = json.loads((path / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["exchanges_completed"], 7)
        self.assertEqual(summary["utterances_completed"], 14)
        self.assertEqual(summary["transport"]["requests"], 0)
        self.assertTrue(all(direction["right_censored"] for direction in summary["directions"]))
        self.assertEqual((path / "metrics.csv").read_bytes(), (other / "metrics.csv").read_bytes())
        with (path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            metrics = list(csv.DictReader(stream))
        self.assertEqual([int(row["exchange"]) for row in metrics], [0, 0, 3, 3, 6, 6, 7, 7])
        events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
        dialogue = [event for event in events if event["type"] == "dialogue"]
        self.assertEqual(len(dialogue), 7)
        self.assertEqual([e["messages"][0]["speaker"] for e in dialogue], ["A", "B", "A", "B", "A", "B", "A"])
        for event in dialogue:
            self.assertTrue((path / "assets" / event["image"]).exists())

    def test_failures_preserve_partial_logs_with_failed_status(self):
        with patch("mutual_lab.runner.speak", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(ExperimentFailed) as caught:
                run_experiment(self.config, self.root)
        path = caught.exception.run_path
        summary = json.loads((path / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["exchanges_completed"], 0)
        self.assertIn("fixture failure", summary["error"])
        self.assertTrue((path / "events.jsonl").exists())

    def test_interrupts_are_marked_cancelled(self):
        with patch("mutual_lab.runner.speak", side_effect=KeyboardInterrupt):
            with self.assertRaises(ExperimentFailed) as caught:
                run_experiment(self.config, self.root)
        summary = json.loads((caught.exception.run_path / "summary.json").read_text())
        self.assertEqual(summary["status"], "cancelled")

    def test_seed_sweep_counts_unreached_runs(self):
        path = sweep(self.config, [7, 11], self.root, quiet=True)
        aggregate = json.loads((path / "aggregate.json").read_text())
        for item in aggregate["directions"]:
            self.assertEqual(item["n_runs"], 2)
            self.assertEqual(item["n_reached"] + item["n_right_censored"], 2)
        self.assertTrue((path / "directions.csv").exists())

    def test_cli_validation_does_not_start_a_run(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config.to_dict()))
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(["validate", str(config_path)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stream.getvalue())["request_upper_bound"], 0)
        self.assertEqual(len(list(self.root.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
