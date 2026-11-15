"""Unactivated cohort protocol and tiny actual fresh-seed fixtures."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict,replace
from pathlib import Path
from unittest.mock import patch

import mutual_lab
ROOT=Path(mutual_lab.__file__).resolve().parents[1]
from test_candidate_validation import audit_fixture
from mutual_lab import cohort_validation as cohort
from mutual_lab.config import ExperimentConfig
from mutual_lab.frozen_features import FROZEN_FEATURE_POLICY
from mutual_lab.mastery_lab import MasteryPlan
from mutual_lab.native_preparation import LANGUAGES_ORDER
from mutual_lab.profiles import ModelProfile


def declared_plan():return replace(cohort._reference_plan(),seed_pairs=cohort.SEED_PAIRS,b_seed_pair=cohort.SEED_PAIRS[0])


def source_summary(plan,condition='F1'):
    kind=cohort.CATALOG[condition][3];rows=[]
    for language in LANGUAGES_ORDER:
        point=audit_fixture(language,0,plan)['engine_report']['curves'][-1]
        rows.append({'language':language,'condition':condition,'step':plan.budgets[-1],'status':'completed',
                     'dev':point['dev'],'contrasts':point['contrasts']['categories'],'candidate':point['candidate'],
                     'memory_source':'cnn' if condition=='F0' else 'objects'})
    return {'kind':kind,'status':'completed','missing_languages':[],'failed_trials':[],'fixed_prior_condition':'B1',
            'policy':'mixed_blocks','model':'attentive','producer_memory':'objects','excluded_from_research':True,'eligible_for_fresh_experiment':False,'rows':rows,
            'selection':{'rule':'all_languages_minimum_six_final_scores_tie_g0_v1','status':'selected','condition':condition,'fixed_final_step':plan.budgets[-1],'tie_break':'G0'}}


def request_record(plan,condition='F1',actual_sources=False,prior_sha='b'*64,prior_state='d'*64):
    kind,memory,u,summary_kind,_=cohort.CATALOG[condition]
    spec={'candidate_kind':kind,'producer_memory':memory or 'objects','understanding':u}
    content={'schema_version':1,'source_run_id':123,'source_run_attempt':1,'source_revision':'a'*40,
        'source_summary_kind':summary_kind,'source_condition':condition,'source_summary_sha256':'1'*64,
        'source_plan_sha256':'2'*64,'prior_manifest_sha256':'3'*64,'prior_checkpoint_sha256':prior_sha,
        'validation_plan_sha256':'4'*64,'policy':'mixed_blocks','fixed_prior_condition':'B1','validation_run_attempt':1,
        'fixed_final_step':plan.budgets[-1],'data_version':cohort.base.DATA_VERSION,'frozen_feature_policy':FROZEN_FEATURE_POLICY,
        'candidate_spec':spec,'rationale':'One fixed synthetic source candidate for an unactivated test.'}
    content['candidate_source_sha256']=cohort._implementation_hashes(content) if actual_sources else {name:'e'*64 for name in cohort._source_paths(content)}
    return {'sha256':'f'*64,'content':content,'source_summary':source_summary(plan,condition),
            'prior_evidence':{'manifest_sha256':content['prior_manifest_sha256'],'checkpoint_sha256':prior_sha,'trained_prior_sha256':prior_state}}


def suite_fixture(language,replica,plan,request):
    source=audit_fixture(language,replica,plan);raw=source['engine_report'];initial=source['initial_state']
    content=request['content'];_,spec,factory=cohort._selection(content)
    memory={'memory_source':spec['producer_memory'],'memory_normalization':'l2_first32_raw_yx_v1'}
    initial.pop('model',None);initial['memory_specification']=memory
    initial['memory_preflight']={'status':'completed','input_pngs':332,'unique_pngs':332,'successful_pngs':332,'invalid_pngs':0,'fallback_count':0}
    initial['prior_sha256']=request['prior_evidence']['trained_prior_sha256']
    raw['initial_prior_sha256']=raw['final_prior_sha256']=initial['prior_sha256']
    raw['prior_metadata'].update(prior_sha256=initial['prior_sha256'],source_bundle_sha256=content['prior_checkpoint_sha256'])
    raw['model_factory']=factory;raw['source_sha256']=content['candidate_source_sha256']
    raw['split']['protected_pairs_excluded']=22
    tables={'policy':FROZEN_FEATURE_POLICY,'caches':{'visual_cache':{'entries':332,'sha256':'a'*64}},'sha256':'b'*64}
    raw['frozen_feature_policy']=FROZEN_FEATURE_POLICY;raw['frozen_feature_tables']={'step_0':tables,'final':copy.deepcopy(tables)}
    raw['benchmark']['image_caches_cleared']={'visual_cache':220}
    for point in raw['curves']:point['frozen_feature_table_sha256']=tables['sha256']
    if spec['candidate_kind']=='e_attentive':initial.pop('memory_specification');initial['model']='attentive'
    if spec['candidate_kind']=='grounding':
        initial['producer_memory_specification']=initial.pop('memory_specification')
        initial['understanding_specification']={'understanding':'global' if spec['understanding']=='base' else 'objects','producer_memory':spec['producer_memory'],
                                               'initial_base_understanding_sha256':f'U-{replica}'}
    return {'schema_version':1,'kind':cohort.VERSION,'language':language,'replica':replica,'seed_pair':list(cohort.SEED_PAIRS[replica]),
        'plan':asdict(plan),'status':'completed','candidate_request':copy.deepcopy(request),'candidate_spec':spec,'source_condition':content['source_condition'],
        'fixed_prior_condition':'B1','policy':'mixed_blocks','negative_policy_specification':cohort.negative_policy_manifest('mixed_blocks'),
        'fresh_initialization':True,'resumed':False,'scratch_language_training':True,'shared_visual_prior':True,'independent_visual_replications':False,
        'excluded_from_research':True,'eligible_for_fresh_experiment':False,'implementation_source_sha256':content['candidate_source_sha256'],
        'initial_state':initial,'engine_report':raw,'cohort_source_sha256':'f'*64,'execution':{'workflow_run_id':'456','workflow_run_attempt':'1','code_revision':None,'trial_id':f'{language}-{replica}'},
        'initial_checkpoint_evidence':{'sha256':'a'*64,**{key:True for key in ('step_zero','optimizer_empty','main_rng_fresh','negative_rng_fresh','initial_weights_match','feature_policy_matches')}}}


class CohortProtocolTests(unittest.TestCase):
    def setUp(self):self.plan=declared_plan();self.request=request_record(self.plan)
    def suites(self,request=None):
        request=request or self.request
        return [suite_fixture(language,replica,self.plan,request) for replica in range(3) for language in LANGUAGES_ORDER]
    def test_twelve_fixed_final_outcomes_required_and_target_failure_is_completed_execution(self):
        suites=self.suites();result=cohort.aggregate_cohort_validations(suites,self.plan,candidate_request=self.request)
        self.assertTrue(result['robust_development_candidate']);self.assertTrue(result['execution_completed']);self.assertTrue(result['provenance_valid'])
        self.assertEqual(result['final_candidate_count'],12)
        final=suites[-1]['engine_report']['curves'][-1];final['dev'].update(valid_generation_hits=100,valid_generation_accuracy=100/112)
        final['candidate'].update(status='targets_not_met',score_targets_met=False)
        result=cohort.aggregate_cohort_validations(suites,self.plan,candidate_request=self.request)
        self.assertEqual(result['status'],'completed');self.assertEqual(result['assessment'],'targets_not_met');self.assertEqual(result['final_candidate_count'],11)
        self.assertFalse(result['robust_development_candidate']);self.assertTrue(result['execution_completed'])
    def test_known_catalog_covers_e_f_g_without_arbitrary_factory_strings(self):
        for condition in cohort.CATALOG:
            request=request_record(self.plan,condition);suites=self.suites(request)
            result=cohort.aggregate_cohort_validations(suites,self.plan,candidate_request=request)
            self.assertTrue(result['robust_development_candidate'],condition)
        for field,value in (('source_condition','os.system'),('source_summary_kind','raw-engine')):
            request=copy.deepcopy(self.request);request['content'][field]=value
            with self.assertRaises(ValueError):cohort._validate_request(request,self.plan)
        request=copy.deepcopy(self.request);request['content']['candidate_spec']['producer_memory']='os.system'
        with self.assertRaises(ValueError):cohort._validate_request(request,self.plan)
    def test_unavailable_or_malformed_benchmark_cannot_qualify_and_retains_failure_context(self):
        for status in ('completed','failed','cancelled','time_limit'):
            for variant in ('missing',None,False,7,'unavailable',[]):
                with self.subTest(status=status,benchmark=variant):
                    suites=self.suites();suite=suites[0];raw=suite['engine_report']
                    suite['status']=raw['status']=status
                    suite['error']='benchmark did not complete'
                    if variant=='missing':raw.pop('benchmark')
                    else:raw['benchmark']=variant
                    result=cohort.aggregate_cohort_validations(suites,self.plan,candidate_request=self.request)
                    self.assertEqual(result['status'],'incomplete')
                    self.assertFalse(result['robust_development_candidate']);self.assertFalse(result['provenance_valid'])
                    self.assertFalse(result['trial_checks'][0]['checks']['benchmark_reset'])
                    failure=result['failed_trials'][0]
                    self.assertEqual((failure['language'],failure['replica'],failure['status']),
                                     (suite['language'],suite['replica'],status))
                    self.assertEqual(failure['error'],'benchmark did not complete')
                    self.assertIn('benchmark_reset',failure['failed_checks'])
    def test_grounding_catalog_preserves_four_distinct_original_factory_ids(self):
        names=set()
        for condition in ('G0','G1'):
            for memory in ('cnn','objects'):
                request=request_record(self.plan,condition)
                request['content']['candidate_spec']['producer_memory']=memory
                request['source_summary']['producer_memory']=memory
                _,_,expected=cohort._selection(request['content'])
                factory=cohort._factory(request['content'],[])
                actual=factory.__module__+'.'+factory.__qualname__
                self.assertEqual(actual,expected)
                self.assertEqual(actual,'mutual_lab.grounding_study.RecordingFactory.'+condition+'_'+memory)
                names.add(actual)
        self.assertEqual(len(names),4)

    def test_invalid_discovery_comparison_or_cross_language_source_selection_is_rejected(self):
        for mutation in ('incomplete','failed_control','missing_language','different_condition','final_target'):
            request=copy.deepcopy(self.request);summary=request['source_summary']
            if mutation=='incomplete':summary['status']='incomplete'
            elif mutation=='failed_control':summary['failed_trials']=[{'status':'pair_mismatch'}]
            elif mutation=='missing_language':summary['rows'].pop()
            elif mutation=='different_condition':summary['rows'][0]['condition']='F0'
            else:summary['rows'][0]['candidate']['score_targets_met']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):cohort._validate_request(request,self.plan)
        overridden=request_record(self.plan,'G1');overridden['source_summary']['selection']['condition']='G0'
        with self.assertRaisesRegex(ValueError,'prespecified G source winner'):cohort._validate_request(overridden,self.plan)
        wrong_model=request_record(self.plan,'E1');wrong_model['source_summary']['model']='standard'
        with self.assertRaises(ValueError):cohort._validate_request(wrong_model,self.plan)
        invalid_run=request_record(self.plan,'E1');invalid_run['content']['source_run_id']=43634532997
        with self.assertRaises(ValueError):cohort._validate_request(invalid_run,self.plan)
        original=request_record(self.plan,'E1');original['source_summary']['status']='incomplete'
        with self.assertRaises(ValueError):cohort._validate_request(original,self.plan)
    def test_missing_duplicate_changed_seed_representation_and_feature_mutation_cannot_qualify(self):
        suites=self.suites()
        with self.assertRaisesRegex(ValueError,'Duplicate'):cohort.aggregate_cohort_validations(suites+suites[:1],self.plan,candidate_request=self.request)
        result=cohort.aggregate_cohort_validations(suites[:-1],self.plan,candidate_request=self.request);self.assertEqual(len(result['missing_trials']),1)
        for mutation in ('seed','weights','stream','prior','cache','optimizer','representation','attempt','source','count'):
            suites=self.suites();s=suites[0];raw=s['engine_report']
            if mutation=='seed':raw['seed_pair']=[59007,59009]
            elif mutation=='weights':raw['initial_producer_sha256']='wrong'
            elif mutation=='stream':raw['meaning_stream_sha256']='wrong'
            elif mutation=='prior':raw['final_prior_sha256']='wrong'
            elif mutation=='cache':raw['frozen_feature_tables']['final']['sha256']='c'*64
            elif mutation=='optimizer':s['initial_checkpoint_evidence']['optimizer_empty']=False
            elif mutation=='representation':s['initial_state']['memory_specification']['memory_normalization']='different'
            elif mutation=='attempt':s['execution']['workflow_run_attempt']='2'
            elif mutation=='source':raw['source_sha256']={**raw['source_sha256'],'changed.py':'a'*64}
            else:raw['curves'][-1]['dev']['valid_generation_hits']=111
            result=cohort.aggregate_cohort_validations(suites,self.plan,candidate_request=self.request)
            with self.subTest(mutation=mutation):self.assertFalse(result['robust_development_candidate']);self.assertFalse(result['provenance_valid'])
    def test_cross_host_feature_hash_variation_is_recorded_without_relaxing_within_trial_immutability(self):
        suites=self.suites()
        for i,s in enumerate(suites):
            tables=s['engine_report']['frozen_feature_tables'];tables['step_0']['sha256']=tables['final']['sha256']=f'{i:064x}'
            for point in s['engine_report']['curves']:point['frozen_feature_table_sha256']=f'{i:064x}'
        result=cohort.aggregate_cohort_validations(suites,self.plan,candidate_request=self.request)
        self.assertTrue(result['robust_development_candidate']);self.assertFalse(result['cross_host_feature_table_bit_equality_required'])
    def test_byte_pins_and_known_implementation_are_checked_before_training(self):
        with tempfile.TemporaryDirectory() as dirname:
            root=Path(dirname);plan_path=root/'plan.json';prior=root/'prior.pt';summary=root/'source-summary.json';source_plan=root/'source-plan.json';request_path=root/'request.json'
            plan_path.write_text(json.dumps(asdict(self.plan)));source_plan.write_text(json.dumps(asdict(cohort._reference_plan())));prior.write_bytes(b'only-test-prior-bytes')
            request=request_record(self.plan,actual_sources=True);content=request['content'];summary.write_text(json.dumps(request['source_summary']))
            manifest={'kind':'development_only','sha256':hashlib.sha256(prior.read_bytes()).hexdigest(),'report':{'status':'completed','trained_prior_sha256':'d'*64}}
            prior.with_suffix('.json').write_text(json.dumps(manifest))
            for key,path in (('validation_plan_sha256',plan_path),('prior_manifest_sha256',prior.with_suffix('.json')),('prior_checkpoint_sha256',prior),('source_summary_sha256',summary),('source_plan_sha256',source_plan)):content[key]=hashlib.sha256(path.read_bytes()).hexdigest()
            request_path.write_text(json.dumps(content))
            loaded=cohort.load_candidate_request(request_path,self.plan,plan_path=plan_path,prior_path=prior,source_summary_path=summary,source_plan_path=source_plan)
            self.assertEqual(loaded['sha256'],hashlib.sha256(request_path.read_bytes()).hexdigest())
            prior.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'prior_checkpoint_sha256'):cohort.load_candidate_request(request_path,self.plan,plan_path=plan_path,prior_path=prior,source_summary_path=summary,source_plan_path=source_plan)
    def test_cancelled_attempt_survives_failed_suite_persistence(self):
        from mutual_lab.composition_suite import ArmCancelled
        config=ExperimentConfig.load(ROOT/'configs/neural.example.json');request=request_record(self.plan,actual_sources=True)
        with tempfile.TemporaryDirectory() as dirname:
            interrupt=ArmCancelled(Path(dirname)/'unavailable.json');interrupt.report={'status':'cancelled','sentinel':True};snapshots=[]
            def persist(path,value):
                snapshots.append(copy.deepcopy(value))
                if value['status']=='cancelled':raise OSError('disk failed')
            with patch.object(cohort,'run_trial',side_effect=interrupt) as execute,patch.object(cohort,'safe_json',side_effect=persist):
                with self.assertRaises(KeyboardInterrupt) as caught:cohort.run_cohort_validation(config,'en',0,self.plan,dirname,prior_bundle={'_bundle_sha256':'b'*64},candidate_request=request)
            self.assertIs(caught.exception,interrupt);execute.assert_called_once();self.assertEqual(snapshots[-1]['status'],'cancelled');self.assertTrue(snapshots[-1]['engine_report']['sentinel'])


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires optional CPU torch')
class CohortNeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mutual_lab import development_world as world
        from mutual_lab.neural import configure_cpu
        from mutual_lab.spatial_prior import load_prior_bundle,pretrain_prior,save_prior_bundle
        configure_cpu();cls.temporary=tempfile.TemporaryDirectory()
        config=ExperimentConfig.load(ROOT/'configs/neural.example.json');cls.config=replace(config,native_training_profile=replace(config.native_training_profile,batch_size=4))
        tiny=ModelProfile('cohort-test',embedding_dim=8,hidden_dim=16,max_sequence_length=128)
        cls.plan=replace(declared_plan(),budgets=(1,2),model_profile=tiny,benchmark_steps=0)
        reference=replace(cohort._reference_plan(),budgets=(1,2),model_profile=tiny,benchmark_steps=0)
        full=world.development_splits(config,59003);fixture={'train':full['train'][:4]+full['train'][100:104]+full['train'][-4:],'dev':full['dev'][:4]}
        prior=save_prior_bundle(pretrain_prior([world.render_scene(s) for s in fixture['train']],dim=16,steps=1,batch_size=4,eval_every=1),Path(cls.temporary.name)/'prior.pt');bundle=load_prior_bundle(prior)
        cls.prior_path,cls.bundle,cls.fixture,cls.reference_plan=prior,bundle,fixture,reference
        cls.request=request_record(cls.plan,actual_sources=True,prior_sha=bundle['_bundle_sha256'],prior_state=bundle['report']['trained_prior_sha256'])
        cls.suites={};cls.paths={}
        with patch.object(cohort,'_reference_plan',return_value=reference),patch.object(world,'development_splits',return_value=fixture),patch.object(world,'development_manifest',return_value={'fixture':'tiny-cohort'}):
            for replica in range(3):
                for language in ('en','ko'):
                    path=cohort.run_cohort_validation(cls.config,language,replica,cls.plan,cls.temporary.name,progress=None,prior_bundle=bundle,candidate_request=cls.request)
                    cls.paths[(replica,language)]=path;cls.suites[(replica,language)]=json.loads((path/'cohort-validation.json').read_text())
    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()
    def test_actual_two_languages_three_fresh_seed_blocks_use_original_known_factory_and_fresh_adam(self):
        initial_u=[];initial_p=[]
        for replica,pair in enumerate(cohort.SEED_PAIRS):
            a,b=(self.suites[(replica,language)] for language in ('en','ko'))
            self.assertEqual(a['initial_state'],b['initial_state'])
            initial_u.append(a['initial_state']['understanding_sha256']);initial_p.append(a['initial_state']['producer_sha256'])
            self.assertEqual(a['engine_report']['meaning_stream_sha256'],b['engine_report']['meaning_stream_sha256'])
            for suite in (a,b):
                self.assertEqual(suite['status'],'completed',suite.get('error'));raw=suite['engine_report']
                self.assertEqual(raw['seed_pair'],list(pair));self.assertEqual(raw['plan']['b_seed_pair'],list(pair))
                self.assertEqual(raw['model_factory'],'mutual_lab.memory_study.RecordingFactory.F1')
                self.assertEqual(raw['frozen_feature_policy'],FROZEN_FEATURE_POLICY);self.assertEqual(raw['frozen_feature_tables']['step_0'],raw['frozen_feature_tables']['final'])
                self.assertTrue(all(value for key,value in suite['initial_checkpoint_evidence'].items() if key!='sha256'))
                self.assertFalse(suite['resumed']);self.assertFalse(suite['eligible_for_fresh_experiment'])
        self.assertEqual(len(set(initial_u)),3);self.assertEqual(len(set(initial_p)),3)
    def test_actual_checkpoint_preserves_loadable_factory_and_optimizer_rng_provenance(self):
        from mutual_lab.neural import torch
        for key,path in self.paths.items():
            suite=self.suites[key];raw=suite['engine_report'];trial=(path/suite['raw_report_path']).parent
            state=torch.load(trial/'step-000002.pt',map_location='cpu',weights_only=True)
            self.assertTrue(state['optimizer']['state']);self.assertEqual(state['model_factory'],'mutual_lab.memory_study.RecordingFactory.F1')
            self.assertEqual(state['seed_pair'],list(cohort.SEED_PAIRS[key[0]]));self.assertEqual(state['frozen_feature_policy'],FROZEN_FEATURE_POLICY)
            self.assertEqual(state['negative_exposure_offset'],8)

    def test_actual_g1_cohort_trains_fresh_seed_blocks_and_restores_token_conditioned_inference(self):
        from mutual_lab import development_world as world
        from mutual_lab.mastery_infer import infer_development,load_development_model
        from mutual_lab.neural import torch
        request=request_record(self.plan,'G1',actual_sources=True,
            prior_sha=self.bundle['_bundle_sha256'],prior_state=self.bundle['report']['trained_prior_sha256'])
        suites,paths={},{}
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                cohort,'_reference_plan',return_value=self.reference_plan),patch.object(
                world,'development_splits',return_value=self.fixture),patch.object(
                world,'development_manifest',return_value={'fixture':'tiny-fresh-G1-cohort'}):
            # The only loads during fresh training inspect each NEW step-zero
            # checkpoint. The already loaded image-only prior is shared.
            with patch('mutual_lab.neural.torch.load',wraps=torch.load) as deserialize:
                for replica in (0,1):
                    for language in ('en','ko'):
                        path=cohort.run_cohort_validation(self.config,language,replica,self.plan,
                            temporary,progress=None,prior_bundle=self.bundle,candidate_request=request)
                        paths[(replica,language)]=path
                        suites[(replica,language)]=json.loads((path/'cohort-validation.json').read_text())
            self.assertEqual(deserialize.call_count,4)
            for call in deserialize.call_args_list:
                loaded_path=Path(call.args[0]).resolve()
                self.assertTrue(loaded_path.is_relative_to(Path(temporary).resolve()))
                self.assertEqual(loaded_path.name,'step-000000.pt')
                self.assertIs(call.kwargs['weights_only'],True)
            base_u,producer=[],[]
            for replica in (0,1):
                en,ko=(suites[(replica,language)] for language in ('en','ko'))
                self.assertEqual(en['initial_state'],ko['initial_state'])
                for key in ('base_understanding_sha256','producer_sha256','prior_sha256'):
                    self.assertEqual(en['initial_state'][key],ko['initial_state'][key])
                base_u.append(en['initial_state']['base_understanding_sha256'])
                producer.append(en['initial_state']['producer_sha256'])
                self.assertEqual(en['engine_report']['meaning_stream_sha256'],ko['engine_report']['meaning_stream_sha256'])
                self.assertNotEqual(en['engine_report']['comprehension_caption_stream_sha256'],ko['engine_report']['comprehension_caption_stream_sha256'])
                for language in ('en','ko'):
                    suite=suites[(replica,language)];raw=suite['engine_report']
                    self.assertEqual(suite['status'],'completed',suite.get('error'))
                    self.assertTrue(suite['fresh_initialization']);self.assertFalse(suite['resumed'])
                    self.assertEqual(raw['model_factory'],'mutual_lab.grounding_study.RecordingFactory.G1_objects')
                    self.assertEqual(raw['seed_pair'],list(cohort.SEED_PAIRS[replica]))
                    self.assertEqual(raw['plan']['b_seed_pair'],list(cohort.SEED_PAIRS[replica]))
                    self.assertTrue(all(value for key,value in suite['initial_checkpoint_evidence'].items() if key!='sha256'))
                    self.assertEqual(raw['frozen_feature_tables']['step_0'],raw['frozen_feature_tables']['final'])
                    trial=(paths[(replica,language)]/suite['raw_report_path']).parent
                    checkpoint=trial/'step-000002.pt'
                    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    self.assertTrue(payload['optimizer']['state'])
                    self.assertGreater(float(payload['model']['understanding.residual_output.weight'].abs().sum()),0)
                    model=load_development_model(checkpoint,self.prior_path)
                    self.assertTrue(model.understanding.object_context)
                    self.assertEqual(model.object_memory_source,'objects')
                    example=world.make_examples(model,self.fixture['dev'][:1],language)[0]
                    with patch.object(model,'understanding_objects',wraps=model.understanding_objects) as tokens:
                        actual=infer_development(model,example['image'],text=example['text'],choices_png=example['choices'])
                    tokens.assert_called()
                    self.assertEqual(tuple(model.understanding_objects([example['image']]).shape),(1,2,34))
                    generated=next(row for row in json.loads((trial/'generations.json').read_text()) if row['step']==2 and row['split']=='dev')
                    predicted=next(row for row in json.loads((trial/'comprehension_predictions.json').read_text()) if row['step']==2 and row['split']=='dev')
                    self.assertEqual(actual['generation'],{key:generated[key] for key in actual['generation']})
                    self.assertEqual(actual['comprehension']['prediction'],predicted['prediction'])
                    self.assertTrue(actual['read_only_state_verified'])
            self.assertEqual(len(set(base_u)),2);self.assertEqual(len(set(producer)),2)

if __name__=='__main__':unittest.main()
