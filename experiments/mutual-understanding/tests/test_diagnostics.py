import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mutual_lab.config import AgentConfig, ExperimentConfig
from mutual_lab.diagnostics import (DiagnosticOptions, diagnostic_splits, grade_generation,
                                    semantic_index)
from mutual_lab.metrics import classify_initial, corrected_accuracy, screening_interval
from mutual_lab.profiles import EvaluationProfile, ModelProfile, NativeTrainingProfile
from mutual_lab.screening_plan import (ScreeningPlan, ScreeningScenario, distribution_stats,
                                       hit_distribution, joint_hit_distribution, simulate_screening)
from mutual_lab.world import Object, Scene, make_research_dataset, native_captions


def experiment():
    return ExperimentConfig(schema_version=2, eval_items=4, max_exchanges=2, eval_every=1,
        agents=(AgentConfig("A","ko","neural"),AgentConfig("B","ja","neural")),
        model_profile=ModelProfile("diagnostic-test", embedding_dim=8, hidden_dim=16),
        native_training_profile=NativeTrainingProfile(steps=3, batch_size=2),
        evaluation_profile=EvaluationProfile(pretest_items=4, final_items=4, native_items=4))


class ScreeningPlanTests(unittest.TestCase):
    def test_cluster_distribution_preserves_means_and_icc(self):
        for mean in (.25, .30, .75):
            for correlation in (0, .5, 1):
                p = hit_distribution(mean, correlation)
                self.assertAlmostEqual(sum(p), 1)
                stats = distribution_stats(p)
                self.assertAlmostEqual(stats["raw_mean"], mean)
                self.assertAlmostEqual(stats["within_pair_icc"], correlation)
        self.assertAlmostEqual(distribution_stats((0, 1, 0, 0, 0))["within_pair_icc"], -1/3)

    def test_direction_coupling_preserves_marginals(self):
        a, b = hit_distribution(.25, .5), hit_distribution(.4, 1)
        for coupling in (0, .4, 1):
            joint = joint_hit_distribution(a, b, coupling)
            self.assertAlmostEqual(sum(p for _, _, p in joint), 1)
            for i in range(5):
                self.assertAlmostEqual(sum(p for x, _, p in joint if x == i), a[i])
                self.assertAlmostEqual(sum(p for _, y, p in joint if y == i), b[i])
        matched = joint_hit_distribution(a, a, 1)
        self.assertTrue(all(i == j for i, j, p in matched if p > 1e-12))

    def test_actual_gate_is_reused_and_deterministic_counts_handle_minimum(self):
        fixed = ScreeningScenario("one", cluster_model="custom_hits", hits_a=(0,1,0,0,0), hits_b=(0,1,0,0,0))
        report = simulate_screening(ScreeningPlan(candidate_pairs=(97,98), scenarios=(fixed,), replicates=10))
        self.assertEqual([r["primary_probability"] for r in report["results"]], [0, 1])
        positive = ScreeningScenario("two", cluster_model="custom_hits", hits_a=(0,0,1,0,0), hits_b=(0,1,0,0,0))
        result = simulate_screening(ScreeningPlan(candidate_pairs=(320,), scenarios=(positive,), replicates=10))["results"][0]
        points = []
        for mean in (.5, .25):
            low, high = screening_interval(mean, 320)
            points.append({"listener_language":"a", "source_language":"b", "accuracy":mean,
                "corrected_accuracy":corrected_accuracy(mean), "wilson_low":0, "wilson_high":1,
                "screening_low":low, "screening_high":high})
        group = classify_initial(points, [True,True], .15, verified_foreign_exposure=True)["analysis_group"]
        self.assertEqual(result["analysis_group_counts"][group], 10)
        self.assertFalse(result["within_current_world_pair_count"])
        self.assertFalse(result["within_current_pretest_item_limit"])

    def test_simulation_matches_analytic_cluster_power_and_joint_dependence(self):
        # With rho=1 each pair is all wrong/all correct, hence successes ~ Binomial(m,p).
        independent = ScreeningScenario("independent", within_pair_correlation=1)
        shared = replace(independent, id="shared", direction_shared_probability=1)
        report = simulate_screening(ScreeningPlan(candidate_pairs=(200,), scenarios=(independent,shared), replicates=1200))
        radius = math.sqrt(math.log(80) / 400)
        allowed = [k for k in range(201) if .1 + radius <= k/200 <= .4 - radius]
        single = sum(math.comb(200,k) * .25**k * .75**(200-k) for k in allowed)
        a, b = report["results"]
        self.assertAlmostEqual(a["primary_probability"], single**2, delta=.08)
        self.assertAlmostEqual(b["primary_probability"], single, delta=.06)
        self.assertEqual(b["primary_probability"], b["direction_a_equivalent_probability"])
        self.assertEqual(b["primary_probability"], b["direction_b_equivalent_probability"])
        self.assertAlmostEqual(b["implied_direction_count_correlation"], 1)

    def test_reproducibility_is_not_affected_by_ordering_other_cases(self):
        a, b = ScreeningScenario("a"), ScreeningScenario("b", within_pair_correlation=1)
        plan = ScreeningPlan(candidate_pairs=(98,200), scenarios=(a,b), replicates=50)
        first = simulate_screening(plan)
        other = simulate_screening(replace(plan, candidate_pairs=(200,98), scenarios=(b,a)))
        key = lambda row: (row["scenario_id"],row["pairs"])
        self.assertEqual({key(r):r for r in first["results"]}, {key(r):r for r in other["results"]})
        self.assertEqual(plan, ScreeningPlan.from_dict(plan.to_dict()))

    def test_invalid_assumptions_and_budget_are_rejected(self):
        for data in ({"candidate_pairs":[True]}, {"candidate_pairs":[98,98]},
                     {"max_pair_draws":1}, {"chance_margin":float("nan")},
                     {"replicates":0}, {"schema_version":True}, {"scenarios":[{"id":"x", "means":[.25,2]}]},
                     {"scenarios":[{"id":"x", "within_pair_correlation":-.5}]},
                     {"scenarios":[{"id":"x", "cluster_model":"custom_hits", "hits_a":[1], "hits_b":[1]}]},
                     {"scenarios":[{"id":"x", "cluster_model":"custom_hits", "means":[.25,.25]}]},
                     {"misspelled":1}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                ScreeningPlan.from_dict(data)

    def test_planning_cli_works_without_site_packages_and_is_development_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = ScreeningPlan(candidate_pairs=(98,), scenarios=(ScreeningScenario("one",cluster_model="custom_hits",
                hits_a=(0,1,0,0,0),hits_b=(0,1,0,0,0)),), replicates=10)
            path = root / "plan.json"
            path.write_text(json.dumps(plan.to_dict()))
            result = subprocess.run([sys.executable,"-S","-m","mutual_lab","plan-screening",str(path),
                "--output",str(root / "reports"),"--quiet"], cwd=Path(__file__).resolve().parents[1],
                text=True,capture_output=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            report_path = next((root / "reports").glob("*/report.json"))
            report = json.loads(report_path.read_text())
            self.assertTrue(report["excluded_from_research"])
            self.assertEqual(report["results"][0]["primary_probability"],1)
            self.assertTrue(report_path.with_name("probabilities.csv").exists())
            self.assertFalse(report_path.with_name("summary.json").exists())


class DiagnosticDataTests(unittest.TestCase):
    def test_development_splits_are_disjoint_and_use_only_research_training(self):
        config, options = experiment(), DiagnosticOptions()
        splits = diagnostic_splits(config,options)
        source = make_research_dataset(config)
        self.assertFalse({s.group for s in splits["train"]} & {s.group for s in splits["dev"]})
        self.assertTrue(set(splits["train"] + splits["dev"]) <= set(source.train))
        for name in ("native","pretest","curve","final"):
            protected = {p.scene.group for p in source.view(name).probes}
            self.assertFalse(protected & {s.group for pool in splits.values() for s in pool})
        self.assertEqual(splits,diagnostic_splits(config,options))
        with self.assertRaises(ValueError):
            diagnostic_splits(config,replace(options,train_pairs=35,dev_pairs=35))
        with self.assertRaises(ValueError):
            diagnostic_splits(config,replace(options,initialization_seed=config.seed))
        with self.assertRaises(ValueError):
            diagnostic_splits(ExperimentConfig(),options)

    def test_bounded_semantic_grader_separates_surface_content_and_unknown(self):
        scene = Scene(Object("red","circle"),"left_of",Object("blue","square"))
        trace = lambda text: {"text":text,"terminated":True,"invalid_encoding":False,"special_tokens_in_body":0}
        for language in ("en","ko","ja","es"):
            index = semantic_index(language)
            for text in native_captions(scene,language):
                score = grade_generation(trace(text),scene,language,index)
                self.assertTrue(score["exact_match"])
                self.assertTrue(score["recognized_semantic_correct"])
            text = "  " + native_captions(scene,language)[0].upper().rstrip(".。") + "!  "
            score = grade_generation(trace(text),scene,language,index)
            self.assertFalse(score["exact_match"])
            self.assertTrue(score["recognized_semantic_correct"])
            self.assertIn("surface_only",score["error_flags"])
        index = semantic_index("en")
        literal = {"en":"The blue square is to the right of the red circle.",
                   "ko":"파란 사각형은 빨간 원의 오른쪽에 있다.",
                   "ja":"青い四角は赤い円の右にある。", "es":"El cuadrado azul está a la derecha del círculo rojo."}
        for language, text in literal.items():
            self.assertTrue(grade_generation(trace(text),scene,language,semantic_index(language))["recognized_semantic_correct"])
        for other, flag in ((Scene(scene.object,"left_of",scene.subject),"subject_object_order"),
                            (replace(scene,relation="above"),"relation_error"),
                            (replace(scene,subject=Object("green","circle")),"color_error")):
            score = grade_generation(trace(native_captions(other,"en")[0]),scene,"en",index)
            self.assertFalse(score["recognized_semantic_correct"])
            self.assertIn(flag,score["error_flags"])
        unknown = grade_generation(trace("A perfectly plausible but unsupported paraphrase."),scene,"en",index)
        self.assertIsNone(unknown["recognized_semantic_correct"])
        self.assertIn("unrecognized_surface",unknown["error_flags"])

    def test_decoder_failures_are_retained_separately_from_semantic_unknowns(self):
        scene = Scene(Object("red","circle"),"left_of",Object("blue","square"))
        trace = {"text":"blah blah blah blah ","terminated":False,"invalid_encoding":True,"special_tokens_in_body":1}
        score = grade_generation(trace,scene,"en",semantic_index("en"))
        self.assertTrue({"no_eos","invalid_encoding","special_tokens_in_body","repetition_heuristic"} <= set(score["error_flags"]))
        self.assertIsNone(score["recognized_semantic_correct"])


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("torch","numpy","PIL")),"optional neural dependencies")
class NeuralDiagnosticTests(unittest.TestCase):
    def test_real_small_learning_report_is_isolated_and_evaluation_does_not_change_learning(self):
        from mutual_lab.diagnostics import run_native_diagnostic
        config = experiment()
        before = config.to_dict()
        options = DiagnosticOptions(steps=4,eval_every=2,train_pairs=1,dev_pairs=1)
        with tempfile.TemporaryDirectory() as temporary:
            first = run_native_diagnostic(config,"es",options,temporary)
            other = run_native_diagnostic(config,"es",replace(options,eval_every=1),temporary)
            report = json.loads((first / "report.json").read_text())
            repeated = json.loads((other / "report.json").read_text())
            self.assertEqual(report["status"],"completed")
            self.assertTrue(report["excluded_from_research"])
            self.assertEqual(report["training_steps_completed"],4)
            self.assertNotEqual(report["initial_weights_sha256"],report["final_weights_sha256"])
            self.assertEqual(report["final_weights_sha256"],repeated["final_weights_sha256"])
            self.assertEqual(report["data_sha256"],repeated["data_sha256"])
            self.assertEqual(report["curves"][-2:],repeated["curves"][-2:])
            self.assertEqual([r["step"] for r in report["curves"]],[0,0,2,2,4,4])
            self.assertEqual(config.to_dict(),before)
            self.assertFalse((first / "summary.json").exists())
            self.assertFalse(list(first.glob("*.pt")))
            self.assertTrue((first / "learning_curve.csv").exists())
            self.assertEqual(len(json.loads((first / "generations.json").read_text())),24)
            self.assertIn("unseen_dev_features",report)

    def test_generation_trace_detects_dropped_utf8_and_preserves_generate_text(self):
        from mutual_lab.neural import NativeModel, torch
        from mutual_lab.images import scene_png
        model = NativeModel(experiment().model_profile,19007)
        image = scene_png(next(iter(diagnostic_splits(experiment(),DiagnosticOptions())["train"])))
        def decode_sequence(tokens):
            iterator = iter(tokens)
            def forward(*args,**kwargs):
                logits = torch.zeros(1,1,model.tokenizer.vocab_size)
                logits[0,0,next(iterator)] = 10
                return logits,None
            return forward
        with patch.object(model.producer,"forward",side_effect=decode_sequence([258,ord("A")+3,2])):
            trace = model.generate_with_trace(image)
        self.assertEqual(trace["text"],"A")
        self.assertTrue(trace["invalid_encoding"])
        self.assertTrue(trace["terminated"])
        self.assertEqual(trace["token_ids"],[258,68])
        with patch.object(model.producer,"forward",side_effect=decode_sequence([258,68,2])):
            self.assertEqual(model.generate(image),trace["text"])
        with patch.object(model.producer,"forward",side_effect=decode_sequence([68]*254)):
            self.assertFalse(model.generate_with_trace(image)["terminated"])

    def test_visual_check_detects_collapse_without_using_language(self):
        from mutual_lab.diagnostics import visual_diagnostic
        from mutual_lab.neural import NativeModel, torch
        data = diagnostic_splits(experiment(),DiagnosticOptions(train_pairs=1,dev_pairs=1))
        class Collapsed:
            def visual(self,pngs):
                return torch.zeros(len(pngs),16)
        bad = visual_diagnostic(Collapsed(),data)
        self.assertTrue(all(r["near_collisions_at_1e_6"] == 6 for r in bad["rows"]))
        self.assertTrue(all(r["image_identity_retrieval_accuracy"] == .25 for r in bad["rows"]))
        good = visual_diagnostic(NativeModel(experiment().model_profile,19007),data)
        self.assertTrue(all(r["near_collisions_at_1e_6"] == 0 for r in good["rows"]))

    def test_diagnostic_interrupt_preserves_partial_report_and_invalid_config_creates_nothing(self):
        from mutual_lab.diagnostics import DiagnosticFailed, run_native_diagnostic
        options = DiagnosticOptions(steps=4,eval_every=1,train_pairs=1,dev_pairs=1)
        def stop(event):
            if event["diagnostic_step"] == 1:
                raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(DiagnosticFailed) as caught:
                run_native_diagnostic(experiment(),"en",options,root,stop)
            report = json.loads((caught.exception.report_path / "report.json").read_text())
            self.assertEqual(report["status"],"cancelled")
            self.assertEqual(report["training_steps_completed"],1)
            invalid = root / "invalid"
            with self.assertRaises(ValueError):
                run_native_diagnostic(experiment(),"en",replace(options,steps=-1),invalid)
            self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
