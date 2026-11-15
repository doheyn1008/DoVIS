"""Authenticated-byte guards and actual 96-tensor-checkpoint repair fixtures."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repair_H_aggregation", ROOT / "scripts/repair_H_aggregation.py")
repair = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(repair)
NEURAL = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "PIL"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(repair._json(value))


def bundle_artifacts(root):
    pins = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        output = root / (directory.name + ".zip")
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(directory.rglob("*")):
                if path.is_file(): archive.write(path, path.relative_to(directory).as_posix())
        pins[directory.name] = (len(pins) + 1, repair._sha(output))
    return pins


class RepairPureTests(unittest.TestCase):
    def test_authenticated_zip_blocks_coordinated_extracted_metric_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw = root / "artifact/report.json"
            write(raw, {"hits": 99}); pins = bundle_artifacts(root)
            with patch.object(repair, "ARTIFACTS", pins):
                self.assertEqual(len(repair._authority(root)[0]), 1)
                write(raw, {"hits": 112})
                with self.assertRaisesRegex(ValueError, "Extracted artifact bytes changed"): repair._authority(root)
            root.joinpath("artifact.zip").write_bytes(b"replaced")
            with patch.object(repair, "ARTIFACTS", pins):
                with self.assertRaisesRegex(ValueError, "ZIP digest"): repair._authority(root)

    def test_only_two_exact_initial_leaf_paths_are_authorized(self):
        original = {"memory_preflight": {"feature_tables": {"caches": {"visual_cache": {"sha256": "old"}}, "sha256": "old"}}}
        changed = copy.deepcopy(original); changed["memory_preflight"]["feature_tables"]["caches"]["visual_cache"]["sha256"] = "new"
        self.assertTrue(set(repair._differences(original, changed)) <= repair.ALLOWED_PATHS)
        changed["other_metric"] = 1
        self.assertFalse(set(repair._differences(original, changed)) <= repair.ALLOWED_PATHS)

    def test_path_traversal_symlink_and_output_replacement_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "data").write_bytes(b"original")
            for name in ("../data", "/data", "a\\data"):
                with self.assertRaises(ValueError): repair._safe(root, name)
            (root / "linked").symlink_to(root / "data")
            with self.assertRaises(ValueError): repair._safe(root, "linked")
            output = root / "output.json"; repair._publish(output, b"winner")
            with self.assertRaises(FileExistsError): repair._publish(output, b"replacement")
            self.assertEqual(output.read_bytes(), b"winner")


@unittest.skipUnless(NEURAL, "requires optional torch/numpy/Pillow")
class RepairTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import producer_context_study as study
        from mutual_lab.mastery_lab import MasteryPlan
        from mutual_lab.neural import digest, torch
        cls.study, cls.digest, cls.torch = study, staticmethod(digest), torch
        cls.plan_path = ROOT / "configs/mastery-H-plan.json"
        cls.plan = MasteryPlan.load(cls.plan_path)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.inputs = self.root / "input"; self.inputs.mkdir()
        self.evidence = self.root / "host/host-evidence.json"
        package = Path(self.study.__file__).parent
        self.sources = {path.name: repair._sha(path) for path in package.glob("*.py")}
        context = {"code_revision": repair.SOURCE_REVISION, "run_id": str(repair.SOURCE_RUN), "run_attempt": "1",
                   "repository": "doheyn1008/DoVIS", "source_sha256": self.sources}
        write(self.inputs / 'l1-H-input-43652561578/provenance/run-context.json', context)
        self.make_prior_evidence()
        for language in self.study.LANGUAGES_ORDER:
            for replica in range(3): self.make_suite(language, replica)
        self.pins = bundle_artifacts(self.inputs)

    def table(self, visual=repair.CANONICAL_VISUAL, old=False):
        caches = {name: {"entries": 0 if name == "spatial_grid_cache" else 332, "sha256": value}
                  for name, value in repair.NONVISUAL.items() if not old or name != "producer_context_cache"}
        caches["visual_cache"] = {"entries": 332, "sha256": visual}
        result = {"policy": "per_png_singleton_v1", "caches": caches}
        return {**result, "sha256": self.digest(result)}

    def make_prior_evidence(self):
        evidence = {"schema_version": 1, "kind": "dovis-known-frozen-feature-host-groups", "source_run_id": 43647997707,
            "source_revision": "c65ab2752c117ddf66c98c4e871f553cf9c8f95a", "predates_H_training": True,
            "policy": "per_png_singleton_v1", "cache": "visual_cache", "allowed_sha256": sorted(repair.VISUAL_GROUPS), "evidence_files": []}
        for identity, group in zip((14135346265, 14135337469), sorted(repair.VISUAL_GROUPS)):
            table = self.table(group, old=True); name = "old-" + str(identity)
            path = self.evidence.parent / name / "trial/report.json"
            write(path, {"frozen_feature_policy": "per_png_singleton_v1", "frozen_feature_tables": {"step_0": table, "final": table}})
            evidence["evidence_files"].append({"artifact": {"id": identity, "name": name, "workflow_run": {"id": 43647997707}},
                "report_relative_path": "trial/report.json", "report_sha256": repair._sha(path),
                "feature_table_sha256": table["sha256"], "visual_cache_sha256": group})
        write(self.evidence, evidence); self.evidence_sha = repair._sha(self.evidence)

    def point(self, step, condition, language, replica, table):
        hits = 95 if (condition, language, replica) == ("H0", "ko", 1) else 111
        categories = {}; gates = {}
        for name, (anchors, pairs) in self.study.CONTRAST_COVERAGE.items():
            categories[name] = {"hits": anchors * 4, "items": anchors * 4, "accuracy": 1.0}
            gates[name] = {**categories[name], "available_anchors": anchors, "available_pairs": pairs,
                           "coverage_sufficient": True, "score_target_met": True}
        passed = hits / 112 >= .9
        return {"step": step, "train": {}, "dev": {"comprehension_hits": 448, "comprehension_items": 448,
            "comprehension_accuracy": 1., "valid_generation_hits": hits, "generation_scenes": 112, "valid_generation_accuracy": hits / 112},
            "contrasts": {"categories": categories}, "candidate": {"status": "development_candidate" if passed else "targets_not_met",
                "score_targets_met": passed, "coverage_sufficient": True, "categories": gates,
                "excluded_from_research": True, "eligible_for_fresh_experiment": False}, "frozen_feature_table_sha256": table["sha256"]}

    def make_suite(self, language, replica):
        pair = self.study.DISCOVERY_SEEDS[replica]; engine_plan = replace(self.plan, b_seed_pair=pair)
        table = self.table(next(value for value in repair.VISUAL_GROUPS if value != repair.CANONICAL_VISUAL)
                           if (language, replica) == ("en", 1) else repair.CANONICAL_VISUAL)
        directory = self.inputs / f"l1-H-trials-{language}-r{replica}-{repair.SOURCE_RUN}" / "suite"
        suite = {"schema_version": 1, "kind": self.study.VERSION, "phase": "H", "axis": "producer_global_conditioning",
            "language": language, "replica": replica, "seed_pair": list(pair), "plan": asdict(self.plan),
            "source_rule": self.study.SOURCE_RULE, "study_request": {"fixture": True}, "implementation_source_sha256": {},
            "capacity_matched": True, "excluded_from_research": True, "eligible_for_fresh_experiment": False,
            "status": "completed", "reports": []}
        for condition, context in self.study.CONDITIONS.items():
            trial = directory / "raw-engine" / condition; trial.mkdir(parents=True, exist_ok=True)
            def model(step):
                return {"understanding.base.weight": self.torch.tensor([replica + 1. + step / 100000]),
                    "producer.weight": self.torch.tensor([.1 + step / (10000 if condition == "H0" else 5000)]),
                    "visual_encoder.weight": self.torch.tensor([.5]), "visual_projection": self.torch.tensor([replica + .25])}
            initial = model(0); subset = lambda values, prefix: {name.removeprefix(prefix): value for name, value in values.items() if name.startswith(prefix)}
            u0, p0, prior = (self.digest(subset(initial, prefix)) for prefix in ("understanding.", "producer.", "visual_encoder."))
            last = model(9600)
            raw = {"kind": "mastery-lab-v1-trial", "language": language, "replica": replica, "seed_pair": list(pair),
                "plan": asdict(engine_plan), "plan_sha256": self.digest(asdict(engine_plan)), "split": {"same": True},
                "data_sha256": "data-" + language, "config_sha256": "config", "source_sha256": self.sources,
                "model_profile": asdict(self.plan.model_profile), "parameter_count": 3, "trainable_parameter_count": 2,
                "batch_size": 16, "learning_rate": .003, "policy": "mixed_blocks", "negative_sampling_seed": pair[1] + 100003,
                "negative_exposures": {"examples": 153600}, "optimization": "independent-branch-clipping-adam-v1", "clip_norm_per_branch": 1,
                "choice_count": 4, "initial_understanding_sha256": u0, "final_understanding_sha256": self.digest(subset(last, "understanding.")),
                "initial_producer_sha256": p0, "final_producer_sha256": self.digest(subset(last, "producer.")),
                "initial_prior_sha256": prior, "final_prior_sha256": prior, "meaning_stream_sha256": "meanings",
                "comprehension_caption_stream_sha256": "captions", "environment_sha256": "host-" + language,
                "frozen_feature_policy": "per_png_singleton_v1", "frozen_feature_tables": {"step_0": table, "final": table},
                "phase": "C", "condition": "C1", "status": "completed", "training_languages": [language], "foreign_training_languages": [],
                "protected_splits_used": [], "excluded_from_research": True, "eligible_for_fresh_experiment": False,
                "training_steps_completed": 9600, "model_factory": "mutual_lab.producer_context_study.RecordingFactory." + condition,
                "revision": {"commit": repair.SOURCE_REVISION}, "curves": [], "checkpoints": []}
            u_hashes = {}
            for step in (0, 2400, 4800, 9600):
                path = trial / f"step-{step:06d}.pt"
                payload = {key: raw[key] for key in ("language", "replica", "phase", "condition", "revision", "data_sha256", "split", "frozen_feature_policy", "seed_pair", "source_sha256", "model_factory", "frozen_feature_tables")}
                payload.update(format="mutual-mastery-development-v1", kind="development_only", excluded_from_research=True,
                    eligible_for_fresh_experiment=False, step=step, plan=asdict(engine_plan), model=model(step))
                self.torch.save(payload, path)
                item = {"format": payload["format"], "kind": "development_only", "excluded_from_research": True,
                    "eligible_for_fresh_experiment": False, "step": step, "path": path.name, "sha256": repair._sha(path)}
                write(path.with_suffix(".json"), item); raw["checkpoints"].append(item)
                u_hashes[str(step)] = self.digest(subset(model(step), "understanding."))
                raw["curves"].append(self.point(step, condition, language, replica, table))
            write(trial / "report.json", raw)
            record = {"context_condition": condition, "producer_context": context, "status": "completed", "engine_report": raw,
                "raw_report_path": f"raw-engine/{condition}/report.json", "understanding_checkpoints": u_hashes,
                "initial_state": {"full_state_sha256": self.digest(initial), "understanding_sha256": u0, "producer_sha256": p0,
                    "prior_sha256": prior, "parameter_count": 3, "trainable_parameter_count": 2, "initialization_seed": pair[0],
                    "parameter_flags": [["understanding.base.weight", True], ["producer.weight", True], ["visual_encoder.weight", False]],
                    "understanding_specification": {"understanding": "objects"}, "producer_memory_specification": {"memory_source": "objects"},
                    "context_specification": {"context": context, "nn_parameters_changed": False},
                    "memory_preflight": {"status": "completed", "input_pngs": 332, "successful_pngs": 332, "unique_pngs": 332,
                        "fallback_count": 0, "model_state_unchanged": True, "rng_unchanged": True, "png_pool_sha256": "same", "feature_tables": table}}}
            suite["reports"].append(record)
        suite["pair_audit"] = self.study.context_pair_audit(suite["reports"])
        assert suite["pair_audit"]["status"] == "matched", suite["pair_audit"]
        write(directory / "producer-context-study.json", suite)

    def execute(self):
        with patch.object(repair, "ARTIFACTS", self.pins), patch.object(repair, "EVIDENCE_SHA", self.evidence_sha):
            return repair.repair_aggregation(self.inputs, self.plan_path, self.evidence)

    def test_all_96_actual_tensor_checkpoints_and_only_two_metadata_paths_repair(self):
        with patch.object(self.torch, "load", wraps=self.torch.load) as loads:
            summary, manifest = self.execute()
        self.assertEqual(loads.call_count, 96)
        self.assertEqual(summary["selection"]["condition"], "H1")
        self.assertEqual(len(summary["rows"]), 96)
        self.assertEqual(len(manifest["actual_checkpoint_audits"]), 96)
        self.assertEqual(len(manifest["raw_pair_audits"]), 12)
        self.assertEqual(len(manifest["normalizations"]), 2)
        self.assertEqual(manifest["original_files_sha256_before"], manifest["original_files_sha256_after"])
        self.assertTrue(summary["posthoc_metadata_repair"]["applied"])

    def test_96th_checkpoint_corruption_and_metric_tampering_fail_archive_authority(self):
        checkpoint = sorted(self.inputs.rglob("*.pt"))[-1]; checkpoint.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Extracted artifact bytes"): self.execute()

    def test_unknown_host_or_other_cache_changes_fail_even_with_re_authenticated_fixture(self):
        table = self.table(); table["caches"]["visual_cache"]["sha256"] = "0" * 64
        table["sha256"] = self.digest({key: value for key, value in table.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "Unknown host"): repair._table(table, self.digest)
        table = self.table(); table["caches"]["object_memory_cache"]["sha256"] = "0" * 64
        table["sha256"] = self.digest({key: value for key, value in table.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "non-visual-cache"): repair._table(table, self.digest)

    def test_prior_evidence_byte_change_fails_immutable_pin(self):
        self.evidence.write_bytes(self.evidence.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "prior-690"): self.execute()


if __name__ == "__main__": unittest.main()
