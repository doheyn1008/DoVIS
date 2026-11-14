"""H confirmation contract fixtures, never evidence of model proficiency."""
import copy
import contextlib
import hashlib
import importlib.util
import json
import io
from dataclasses import asdict, replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mutual_lab
PACKAGE = Path(mutual_lab.__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "tests"))
from mutual_lab import cohort_context_validation as cohort
from mutual_lab import cohort_validation as old
from mutual_lab.mastery_lab import MasteryPlan
from mutual_lab.native_preparation import LANGUAGES_ORDER
from test_candidate_validation import audit_fixture


def plans():
    reference = cohort._reference_plan()
    return (replace(reference, seed_pairs=cohort.SOURCE_SEED_PAIRS, b_seed_pair=cohort.SOURCE_SEED_PAIRS[0]),
            replace(reference, seed_pairs=cohort.SEED_PAIRS, b_seed_pair=cohort.SEED_PAIRS[0]))


def source_summary(plan):
    rows = []
    for condition in cohort.CATALOG:
        for replica, pair in enumerate(cohort.SOURCE_SEED_PAIRS):
            for language in LANGUAGES_ORDER:
                point = audit_fixture(language, replica, plan)['engine_report']['curves'][-1]
                rows.append({'condition': condition, 'replica': replica, 'seed_pair': list(pair),
                             'language': language, 'step': plan.budgets[-1], 'status': 'completed',
                             'dev': point['dev'], 'contrasts': point['contrasts']['categories'],
                             'candidate': point['candidate']})
    return {'kind': 'producer-context-study-v1-comparison', 'status': 'completed', 'phase': 'H',
            'plan': asdict(plan), 'rows': rows, 'failed_trials': [], 'missing_trials': [],
            'source_rule': cohort.SELECTION_RULE, 'capacity_matched': True,
            'producer_memory': 'objects', 'understanding': 'objects',
            'frozen_feature_policy': cohort.FROZEN_FEATURE_POLICY,
            'discovery_seed_pairs': [list(pair) for pair in cohort.SOURCE_SEED_PAIRS],
            'confirmation_seed_pairs': [list(pair) for pair in cohort.SEED_PAIRS],
            'fixed_prior_condition': 'B1', 'policy': 'mixed_blocks', 'excluded_from_research': True,
            'eligible_for_fresh_experiment': False, 'selection': {'status': 'selected', 'condition': 'H0',
                'rule': cohort.SELECTION_RULE, 'tie_break': 'H0', 'fixed_final_step': plan.budgets[-1]}}


def request_record(source_plan, plan):
    condition = 'H0'
    content = {'schema_version': 1, 'phase': 'H-confirmation', 'source_run_id': 12345, 'source_run_attempt': 1,
        'source_revision': 'a'*40, 'source_condition': condition, 'source_summary_kind': 'producer-context-study-v1-comparison',
        'source_summary_sha256': '1'*64, 'source_plan_sha256': '2'*64, 'prior_manifest_sha256': '3'*64,
        'prior_checkpoint_sha256': '4'*64, 'validation_plan_sha256': '5'*64,
        'confirmation_implementation_sha256': hashlib.sha256(Path(cohort.__file__).read_bytes()).hexdigest(),
        'fixed_prior_condition': 'B1', 'policy': 'mixed_blocks', 'validation_run_attempt': 1,
        'frozen_feature_policy': cohort.FROZEN_FEATURE_POLICY, 'fixed_final_step': plan.budgets[-1],
        'data_version': cohort.base.DATA_VERSION, 'rationale': 'Authored contract fixture; no activation or measured outcomes.',
        'candidate_spec': {'candidate_kind': 'producer_context', 'producer_memory': 'objects',
                           'understanding': 'attentive', 'producer_context': 'cnn'}}
    content['candidate_source_sha256'] = cohort._implementation_hashes(content)
    return {'sha256': '6'*64, 'content': content, 'source_summary': source_summary(source_plan),
            'prior_evidence': {'manifest_sha256': content['prior_manifest_sha256'],
                'checkpoint_sha256': content['prior_checkpoint_sha256'], 'trained_prior_sha256': '7'*64}}


def suite_fixture(language, replica, plan, request):
    source = audit_fixture(language, replica, plan)
    raw, initial = source['engine_report'], source['initial_state']
    content, pair = request['content'], cohort.SEED_PAIRS[replica]
    _, spec, factory = cohort._selection(content)
    initial.pop('model')
    initial.update(prior_sha256=request['prior_evidence']['trained_prior_sha256'],
        producer_memory_specification={'memory_source': 'objects', 'memory_normalization': 'l2_first32_raw_yx_v1'},
        understanding_specification={'understanding': 'objects', 'producer_memory': 'objects'},
        context_specification={'context': spec['producer_context'], 'nn_parameters_changed': False},
        memory_preflight={'status': 'completed', 'input_pngs': 332, 'unique_pngs': 332,
                          'successful_pngs': 332, 'invalid_pngs': 0, 'fallback_count': 0})
    raw.update(seed_pair=list(pair), plan=asdict(replace(plan, b_seed_pair=pair)),
        model_factory=factory, negative_sampling_seed=pair[1]+plan.negative_seed_offset,
        source_sha256=content['candidate_source_sha256'],
        initial_prior_sha256=initial['prior_sha256'], final_prior_sha256=initial['prior_sha256'])
    raw['prior_metadata'].update(prior_sha256=initial['prior_sha256'], source_bundle_sha256=content['prior_checkpoint_sha256'])
    raw['split']['protected_pairs_excluded'] = 22
    table = {'policy': cohort.FROZEN_FEATURE_POLICY, 'caches': {'visual_cache': {'entries': 332}}, 'sha256': '8'*64}
    raw['frozen_feature_policy'] = cohort.FROZEN_FEATURE_POLICY
    raw['frozen_feature_tables'] = {'step_0': table, 'final': copy.deepcopy(table)}
    raw['benchmark']['image_caches_cleared'] = {'visual_cache': 332}
    for point in raw['curves']:
        point['frozen_feature_table_sha256'] = table['sha256']
    return {'schema_version': 1, 'kind': cohort.VERSION, 'language': language, 'replica': replica,
        'seed_pair': list(pair), 'plan': asdict(plan), 'status': 'completed', 'candidate_request': copy.deepcopy(request),
        'candidate_spec': spec, 'source_condition': content['source_condition'], 'fixed_prior_condition': 'B1',
        'policy': 'mixed_blocks', 'negative_policy_specification': cohort.negative_policy_manifest('mixed_blocks'),
        'fresh_initialization': True, 'scratch_language_training': True, 'shared_visual_prior': True,
        'resumed': False, 'independent_visual_replications': False, 'excluded_from_research': True,
        'eligible_for_fresh_experiment': False, 'implementation_source_sha256': content['candidate_source_sha256'],
        'cohort_source_sha256': content['confirmation_implementation_sha256'], 'initial_state': initial,
        'engine_report': raw, 'execution': {'workflow_run_id': '123', 'workflow_run_attempt': '1',
            'code_revision': None, 'trial_id': f'{language}-{replica}'},
        'initial_checkpoint_evidence': {'sha256': '9'*64, **{key: True for key in
            ('step_zero', 'optimizer_empty', 'main_rng_fresh', 'negative_rng_fresh', 'initial_weights_match', 'feature_policy_matches')}}}


class HConfirmationTests(unittest.TestCase):
    @classmethod
    def actual_image_prior(cls):
        if not hasattr(cls, '_actual_prior'):
            from mutual_lab import development_world as world
            from mutual_lab.config import ExperimentConfig
            from mutual_lab.spatial_prior import pretrain_prior
            config = ExperimentConfig.load(PACKAGE/'configs/neural.example.json')
            scenes = world.development_splits(config, 59003)['train']
            pngs = tuple(world.render_scene(scene) for scene in scenes)
            cls._actual_prior = pretrain_prior(pngs, seed=88001, dim=128, steps=1,
                batch_size=2, max_seconds=30, eval_every=1)
            cls._actual_scenes = scenes
        return cls._actual_prior

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ('producer_context.py', 'producer_context_study.py'):
            (self.root/name).write_text('# Known-name interface fixture only; not model evidence.\n')
        self.module = SimpleNamespace(__file__=str(self.root/'producer_context_study.py'))
        self.patch = patch.object(cohort, '_h_module', return_value=self.module)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.source_plan, self.plan = plans()
        self.request = request_record(self.source_plan, self.plan)

    def suites(self):
        return [suite_fixture(language, replica, self.plan, self.request)
                for replica in range(3) for language in LANGUAGES_ORDER]

    def test_separate_seed_grids_do_not_change_old_v1_or_allow_cross_grid_retry(self):
        before = old.SEED_PAIRS
        cohort._validate_plan(self.plan); cohort._validate_source_plan(self.source_plan)
        for wrong in (replace(self.plan, seed_pairs=before, b_seed_pair=before[0]), self.source_plan):
            with self.assertRaises(ValueError): cohort._validate_plan(wrong)
        with self.assertRaises(ValueError): cohort._validate_source_plan(self.plan)
        self.assertEqual(old.SEED_PAIRS, before)
        self.assertEqual(before[0], (69007, 69009))

    def test_source_requires_all_24_final_cells_and_selection_minimum_over_all_12(self):
        content = self.request['content']; summary = self.request['source_summary']
        verified = cohort._source_checks(summary, content, self.plan)
        self.assertEqual(verified['selected_condition'], 'H0')
        self.assertEqual(verified['verified_source_cells'], 24)
        for mutation in ('missing', 'duplicate', 'wrong_seed', 'bool_replica', 'wrong_winner', 'failed_control'):
            bad = copy.deepcopy(summary)
            if mutation == 'missing': bad['rows'].pop()
            elif mutation == 'duplicate': bad['rows'].append(copy.deepcopy(bad['rows'][0]))
            elif mutation == 'wrong_seed': bad['rows'][0]['seed_pair'] = [69007, 69009]
            elif mutation == 'bool_replica': bad['rows'][0]['replica'] = False
            elif mutation == 'wrong_winner': bad['selection']['condition'] = 'H1'
            else: bad['failed_trials'] = [{'status': 'pair_mismatch'}]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cohort._source_checks(bad, content, self.plan)

    def test_one_ineligible_source_cell_cannot_be_rescued_by_other_languages_or_early_scores(self):
        summary = self.request['source_summary']
        row = summary['rows'][0]
        row['dev'].update(valid_generation_hits=99, valid_generation_accuracy=99/112)
        row['candidate'].update(status='targets_not_met', score_targets_met=False)
        with self.assertRaisesRegex(ValueError, 'winner'):
            cohort._source_checks(summary, self.request['content'], self.plan)
        self.request['content']['source_condition'] = summary['selection']['condition'] = 'H1'
        self.request['content']['candidate_spec']['producer_context'] = 'object_moments'
        verified = cohort._source_checks(summary, self.request['content'], self.plan)
        self.assertEqual(verified['eligible_conditions'], ['H1'])

    def test_confirmation_retains_failed_missing_duplicate_and_final_target_outcomes(self):
        suites = self.suites()
        result = cohort.aggregate_cohort_validations(suites, self.plan, candidate_request=self.request)
        self.assertTrue(result['robust_development_candidate'], result['failed_trials'])
        self.assertEqual(result['final_candidate_count'], 12)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            cohort.aggregate_cohort_validations(suites+suites[:1], self.plan, candidate_request=self.request)
        missing = cohort.aggregate_cohort_validations(suites[:-1], self.plan, candidate_request=self.request)
        self.assertEqual(len(missing['missing_trials']), 1)
        final = suites[-1]['engine_report']['curves'][-1]
        final['dev'].update(valid_generation_hits=99, valid_generation_accuracy=99/112)
        final['candidate'].update(status='targets_not_met', score_targets_met=False)
        outcome = cohort.aggregate_cohort_validations(suites, self.plan, candidate_request=self.request)
        self.assertTrue(outcome['execution_completed']); self.assertTrue(outcome['provenance_valid'])
        self.assertFalse(outcome['robust_development_candidate']); self.assertEqual(outcome['final_candidate_count'], 11)
        suites[-1]['status'] = suites[-1]['engine_report']['status'] = 'time_limit'
        suites[-1]['error'] = 'preserved stop'
        outcome = cohort.aggregate_cohort_validations(suites, self.plan, candidate_request=self.request)
        self.assertFalse(outcome['provenance_valid']); self.assertEqual(outcome['failed_trials'][-1]['error'], 'preserved stop')

    def test_known_factory_and_exact_parameter_source_metadata_reject_arbitrary_requests(self):
        for mutation in ('factory', 'context', 'source', 'wrapper'):
            request = copy.deepcopy(self.request)
            if mutation == 'factory': request['content']['source_condition'] = 'os.system'
            elif mutation == 'context': request['content']['candidate_spec']['producer_context'] = 'arbitrary'
            elif mutation == 'source': request['content']['candidate_source_sha256']['unknown.py'] = 'a'*64
            else: request['content']['confirmation_implementation_sha256'] = '0'*64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cohort._validate_request(request, self.plan)

    def test_real_cli_request_and_prior_loaders_aggregate_and_training_dispatch(self):
        from mutual_lab.spatial_prior import save_prior_bundle
        def save(path, value): path.write_text(json.dumps(value))
        source_path, plan_path = self.root/'source-plan.json', self.root/'plan.json'
        save(source_path, asdict(self.source_plan)); save(plan_path, asdict(self.plan))
        summary_path = self.root/'source-summary.json'; save(summary_path, self.request['source_summary'])
        prior_path = self.root/'prior.pt'
        save_prior_bundle(self.actual_image_prior(), prior_path)
        content = self.request['content']
        for key, path in (('source_plan_sha256', source_path), ('validation_plan_sha256', plan_path),
                          ('source_summary_sha256', summary_path), ('prior_checkpoint_sha256', prior_path),
                          ('prior_manifest_sha256', prior_path.with_suffix('.json'))):
            content[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        request_path = self.root/'request.json'; save(request_path, content)
        record = cohort.load_candidate_request(request_path, self.plan, plan_path=plan_path,
            prior_path=prior_path, source_summary_path=summary_path, source_plan_path=source_path)
        self.request = record
        suite_root = self.root/'cells'; suite_root.mkdir()
        for index, suite in enumerate(self.suites()):
            directory = suite_root/str(index); directory.mkdir()
            save(directory/'cohort-context-validation.json', suite)
        output = self.root/'cohort-summary.json'
        with patch.object(cohort, 'run_cohort_validation') as training, contextlib.redirect_stdout(io.StringIO()):
            result = cohort.main(['--plan', str(plan_path), '--candidate-request', str(request_path),
                '--source-plan', str(source_path), '--source-summary', str(summary_path), '--prior-bundle', str(prior_path),
                '--aggregate-dir', str(suite_root), '--output', str(output)])
        self.assertEqual(result, 0); training.assert_not_called()
        self.assertTrue(json.loads(output.read_text())['robust_development_candidate'])
        produced = self.root/'mock-training-result'; produced.mkdir()
        save(produced/'cohort-context-validation.json', {'status': 'completed'})
        with patch.object(cohort, 'run_cohort_validation', return_value=produced) as training, \
                contextlib.redirect_stdout(io.StringIO()):
            result = cohort.main(['--plan', str(plan_path), '--candidate-request', str(request_path),
                '--source-plan', str(source_path), '--source-summary', str(summary_path), '--prior-bundle', str(prior_path),
                '--language', 'ko', '--replica', '0', '--output', str(self.root/'not-created')])
        self.assertEqual(result, 0); training.assert_called_once()
        self.assertEqual(training.call_args.args[1:3], ('ko', 0))
        self.assertEqual(training.call_args.kwargs['prior_bundle']['_bundle_sha256'],
                         content['prior_checkpoint_sha256'])
        self.assertEqual(training.call_args.kwargs['candidate_request'], record)
        prior_path.write_bytes(b'changed bytes')
        with self.assertRaisesRegex(ValueError, 'Pinned file differs'):
            cohort.load_candidate_request(request_path, self.plan, plan_path=plan_path,
                prior_path=prior_path, source_summary_path=summary_path, source_plan_path=source_path)

    def test_actual_h128_factories_start_two_fresh_990xx_models_and_adam_without_l1_load(self):
        from mutual_lab import development_world as world
        from mutual_lab.neural import digest, torch
        bundle = self.actual_image_prior()
        path = Path(cohort.__file__).with_name('producer_context_study.py')
        spec = importlib.util.spec_from_file_location('H_context_study_actual_fixture', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        initial_u, initial_p = [], []
        pngs = tuple(world.render_scene(scene) for scene in self._actual_scenes[:2])
        with patch.object(cohort, '_h_module', return_value=module), patch.object(torch, 'load') as deserialize:
            for pair in cohort.SEED_PAIRS[:2]:
                factory = cohort._factory(self.request['content'], pngs)
                model = factory(self.plan.model_profile, pair[0], bundle, condition='B1')
                self.assertEqual(factory.__module__+'.'+factory.__qualname__, cohort.CATALOG['H0'][4])
                initial = factory.initial_state
                self.assertEqual(initial['initialization_seed'], pair[0])
                self.assertEqual(initial['parameter_count'], 477443)
                self.assertEqual(initial['full_state_sha256'], digest(model.state_dict()))
                initial_u.append(initial['understanding_sha256']); initial_p.append(initial['producer_sha256'])
                optimizer = torch.optim.Adam(model.parameters(), lr=.003)
                self.assertFalse(optimizer.state)
                examples = world.make_examples(model, self._actual_scenes[:1], 'ko')[:2]
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.grounding_loss(examples) + model.production_loss(examples)
                    loss.backward(); optimizer.step()
                self.assertTrue(optimizer.state)
                self.assertNotEqual(initial['full_state_sha256'], digest(model.state_dict()))
                self.assertEqual(digest(model.visual_encoder.state_dict()), bundle['report']['trained_prior_sha256'])
            deserialize.assert_not_called()
        self.assertEqual(len(set(initial_u)), 2); self.assertEqual(len(set(initial_p)), 2)


if __name__ == '__main__': unittest.main()
