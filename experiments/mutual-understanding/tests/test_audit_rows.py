from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "scripts"))
from mutual_lab.config import ExperimentConfig
from mutual_lab.development_world import (build_development_contrasts, captions,
    development_splits, grade_generation)
from mutual_lab.world import variants
from audit_rows import audit_rows, exact_counter


def gold_fixture():
    config = ExperimentConfig.load(PACKAGE / 'configs/neural.example.json')
    splits = development_splits(config, 59003)
    probes = build_development_contrasts(config, splits)
    language = 'en'
    generations, native, curves = [], [], []
    for step in (0, 2400, 4800, 9600):
        point = {'step': step}
        for split, scenes in splits.items():
            for scene in scenes:
                text = captions(scene, language)[0]
                ids = [b + 3 for b in text.encode('utf8')]
                trace = {'text': text, 'token_ids': ids, 'terminated': True,
                         'invalid_encoding': False, 'special_tokens_in_body': 0,
                         'output_units': len(ids), 'unit_limit': 254}
                generations.append({'step': step, 'split': split, **grade_generation(trace, scene, language)})
                alternatives = variants(scene.group)
                answer = alternatives.index(scene)
                for form, text in enumerate(captions(scene, language)):
                    native.append({'step': step, 'split': split, 'scene': scene.to_dict(),
                        'template_index': form, 'text': text, 'answer': answer, 'prediction': answer,
                        'correct': True, 'predicted_scene': scene.to_dict()})
            n = len(scenes)
            point[split] = {'comprehension_loss': 0., 'production_loss': 0.,
                'teacher_forced_token_accuracy': 1., 'comprehension_hits': n * 4,
                'comprehension_items': n * 4, 'comprehension_accuracy': 1.,
                'generation_scenes': n, 'generation_exact_hits': n, 'generation_exact_accuracy': 1.,
                'valid_generation_hits': n, 'valid_generation_accuracy': 1.,
                'recognized_semantic_correct': n, 'semantic_success_lower_bound': 1.,
                'semantic_unknown_count': 0, 'generation_error_counts': {}}
        rows, counts = [], {k: {'hits': 0, 'items': 0} for k in probes['availability']}
        for probe in probes['probes']:
            for form, text in enumerate(captions(probe.scene, language)):
                rows.append({'category': probe.category, 'scene': probe.scene.to_dict(), 'text': text,
                    'template_index': form, 'answer': probe.correct_choice,
                    'prediction': probe.correct_choice, 'correct': True,
                    'predicted_scene': probe.scene.to_dict()})
                counts[probe.category]['hits'] += 1
                counts[probe.category]['items'] += 1
        for count in counts.values():
            count.update(accuracy=1., corrected_accuracy=1.)
        point['contrasts'] = {'version': probes['version'], 'excluded_from_research': True,
            'language': language, 'choice_count': 2, 'chance_accuracy': .5,
            'availability': probes['availability'], 'categories': counts, 'predictions': rows}
        details = {}
        for category, values in counts.items():
            a = probes['availability'][category]
            details[category] = {'hits': values['hits'], 'items': values['items'], 'accuracy': 1.,
                'available_anchors': a['available'], 'available_pairs': a['available_object_pairs'],
                'coverage_sufficient': True, 'score_target_met': True}
        point['candidate'] = {'status': 'development_candidate', 'score_targets_met': True,
            'coverage_sufficient': True, 'categories': details,
            'eligible_for_fresh_experiment': False, 'excluded_from_research': True}
        curves.append(point)
    raw = {'language': language, 'plan': {'comprehension_min': .9, 'production_min': .9,
        'contrast_min': .9, 'min_contrast_anchors': 8, 'min_contrast_pairs': 2,
        'model_profile': {'tokenizer': 'utf8', 'max_sequence_length': 256}}, 'curves': curves}
    return config, splits, raw, generations, native


class RawRowsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = gold_fixture()

    def setUp(self):
        self.config, self.splits, raw, gs, us = self.fixture
        self.raw, self.gs, self.us = copy.deepcopy(raw), copy.deepcopy(gs), copy.deepcopy(us)
        self.temporary = tempfile.TemporaryDirectory(prefix='H-rows-unit-')
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def run_audit(self):
        (self.directory/'generations.json').write_text(json.dumps(self.gs))
        (self.directory/'comprehension_predictions.json').write_text(json.dumps(self.us))
        return audit_rows(self.directory, self.raw, self.splits, 'en', config=self.config)

    def test_positive_all_four_budgets_have_exact_counts(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'pass')
        self.assertEqual((result['generation_rows'], result['native_rows'], result['binary_rows']), (1328, 5312, 6272))
        self.assertTrue(result['fixed_final_qualified'])

    def test_generation_duplicate_with_same_count_is_rejected(self):
        self.gs[-1] = copy.deepcopy(self.gs[-2])
        with self.assertRaisesRegex(ValueError, 'multiset'):
            self.run_audit()

    def test_native_duplicate_with_same_count_is_rejected(self):
        self.us[-1] = copy.deepcopy(self.us[-2])
        with self.assertRaisesRegex(ValueError, 'multiset'):
            self.run_audit()

    def test_native_wrong_step_is_rejected(self):
        self.us[0]['step'] = 1
        with self.assertRaisesRegex(ValueError, 'multiset'):
            self.run_audit()

    def test_native_false_grade_is_rejected(self):
        self.us[0]['correct'] = False
        with self.assertRaisesRegex(ValueError, 'correctness'):
            self.run_audit()

    def test_generation_trace_not_tokens_is_rejected(self):
        self.gs[0]['token_ids'][0] = 0
        with self.assertRaisesRegex(ValueError, 'trace differs'):
            self.run_audit()

    def test_claimed_valid_generation_score_is_rejected(self):
        self.raw['curves'][-1]['dev']['valid_generation_hits'] -= 1
        with self.assertRaisesRegex(ValueError, 'valid_generation_hits'):
            self.run_audit()

    def test_binary_duplicate_with_same_count_is_rejected(self):
        rows = self.raw['curves'][2]['contrasts']['predictions']
        rows[0] = copy.deepcopy(rows[1])
        with self.assertRaisesRegex(ValueError, 'multiset'):
            self.run_audit()

    def test_binary_false_grade_is_rejected(self):
        self.raw['curves'][0]['contrasts']['predictions'][0]['prediction'] ^= 1
        with self.assertRaisesRegex(ValueError, 'Binary correct'):
            self.run_audit()

    def test_boolean_metric_cannot_impersonate_integer_hits(self):
        self.raw['curves'][0]['dev']['comprehension_hits'] = True
        with self.assertRaisesRegex(ValueError, 'comprehension_hits'):
            self.run_audit()

    def mark_invalid_training_trace(self):
        row = self.gs[0]
        row['token_ids'].append(258)  # Invalid trailing UTF-8 byte; ignored by the saved decoder.
        row['output_units'] += 1
        row['invalid_encoding'] = True
        row['error_flags'] = ['invalid_encoding']
        metrics = self.raw['curves'][0]['train']
        metrics['valid_generation_hits'] -= 1
        metrics['valid_generation_accuracy'] = 219 / 220
        metrics['generation_error_counts'] = {'invalid_encoding': 1}
        return metrics

    def test_honest_invalid_utf8_trace_is_retained_but_not_valid_generation(self):
        self.mark_invalid_training_trace()
        result = self.run_audit()
        self.assertEqual(result['points']['0']['train']['generation_exact_hits'], 220)
        self.assertEqual(result['points']['0']['train']['valid_generation_hits'], 219)

    def test_honest_failed_final_model_is_retained_as_not_qualified(self):
        target_rows = [row for row in self.gs if row['step'] == 9600 and row['split'] == 'dev']
        for row in target_rows[:12]:
            scene = next(scene for scene in self.splits['dev'] if scene.to_dict() == row['scene'])
            wrong = next(other for other in self.splits['dev'] if other != scene)
            text = captions(wrong, 'en')[0]
            ids = [b + 3 for b in text.encode('utf8')]
            trace = {'text': text, 'token_ids': ids, 'terminated': True, 'invalid_encoding': False,
                     'special_tokens_in_body': 0, 'output_units': len(ids), 'unit_limit': 254}
            row.update(grade_generation(trace, scene, 'en'))
        metrics = self.raw['curves'][-1]['dev']
        for name in ('generation_exact_hits', 'valid_generation_hits', 'recognized_semantic_correct'):
            metrics[name] = 100
        for name in ('generation_exact_accuracy', 'valid_generation_accuracy', 'semantic_success_lower_bound'):
            metrics[name] = 100 / 112
        metrics['generation_error_counts'] = dict(Counter(flag for row in target_rows for flag in row['error_flags']))
        self.raw['curves'][-1]['candidate'].update(status='targets_not_met', score_targets_met=False)
        result = self.run_audit()
        self.assertEqual(result['status'], 'pass')
        self.assertFalse(result['fixed_final_qualified'])
        self.assertEqual(result['points']['9600']['dev']['valid_generation_hits'], 100)

    def test_candidate_boolean_substitutes_are_rejected(self):
        self.raw['curves'][-1]['candidate']['score_targets_met'] = 1
        with self.assertRaisesRegex(ValueError, 'Candidate flags'):
            self.run_audit()

    def test_nested_boolean_substitutes_are_rejected(self):
        self.raw['curves'][-1]['candidate']['categories']['color_only']['coverage_sufficient'] = 1
        with self.assertRaisesRegex(ValueError, 'Candidate flags'):
            self.run_audit()

    def test_error_count_boolean_or_float_substitutes_are_rejected(self):
        metrics = self.mark_invalid_training_trace()
        for invalid_count in (True, 1.0):
            with self.subTest(invalid_count=invalid_count):
                metrics['generation_error_counts'] = {'invalid_encoding': invalid_count}
                with self.assertRaisesRegex(ValueError, 'generation_error_counts'):
                    self.run_audit()


if __name__ == '__main__':
    unittest.main()
