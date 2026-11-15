"""Execute the actual H workflow admission-to-data request block without training."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent if HERE.name=='tests' else HERE.parent/'DoVIS-f12bb57-review/experiments/mutual-understanding'
WORKFLOW=EXPERIMENT.parents[1]/'.github/workflows/l1-producer-context-H.yml' if HERE.name=='tests' else HERE/'l1-producer-context-H.yml'
PLANS=EXPERIMENT/'configs' if HERE.name=='tests' else HERE
sys.path.insert(0,str(EXPERIMENT))

def blocks():
    return [textwrap.dedent(value) for value in re.findall(r"          python - <<'PY'\n(.*?)          PY",WORKFLOW.read_text(),re.S)]

class HDiscoveryWorkflowTests(unittest.TestCase):
    def request(self):
        return {'schema_version':1,'phase':'H','model':'attentive','fixed_prior_condition':'B1','policy':'mixed_blocks',
            'producer_memory':'objects','understanding':'objects','hidden_dim':128,'frozen_feature_policy':'per_png_singleton_v1',
            'memory_normalization':'l2_first32_raw_yx_v1','fixed_final_step':9600,'data_version':'five-colors-three-shapes-v1',
            'source_rule':'all_discovery_cells_minimum_final_production_tie_h0_v1',
            'source_cohort_run_id':43647997707,'source_cohort_run_attempt':1,
            'source_cohort_revision':'c65ab2752c117ddf66c98c4e871f553cf9c8f95a',
            'source_cohort_summary_sha256':'31154a7910a2653d7c9a6043af111f1eacaf050ae0fed8d21391f84ea89d48a5',
            'source_plan_sha256':'bfc2900fb2153c4f267a87270d91b8ef2f05313304071b03480ea79ec526fd47',
            'plan_sha256':'a2622ee9bada8ce5f79365fa790d90bf2902a806c8af4d4a389b5081009a962c',
            'prior_manifest_sha256':'3ef0e0ce056c4c553468a0572f42098b09d3b3d7f9d68fe9021a5832a7e272a6',
            'prior_checkpoint_sha256':'862eed3ec93a4567019075adcce59faf75295987a5cf6b816cace607da926e6d',
            'rationale':'Whole producer-global conditioning discovery, never resume the failed native cohort.'}
    def execute(self,changes=None,*,attempt='1',revision='refs/heads/codex/l1-memory',mutation=None):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'configs').mkdir();(root/'mutual_lab').mkdir()
            for name in ('neural.example.json','mastery-C-plan.json'):shutil.copyfile(EXPERIMENT/'configs'/name,root/'configs'/name)
            shutil.copyfile(PLANS/'mastery-H-plan.json',root/'configs/mastery-H-plan.json')
            request=self.request()
            for name,key in (('producer_context.py','context_implementation_sha256'),('producer_context_study.py','study_implementation_sha256'),('mastery_infer.py','inference_compatibility_sha256')):
                content=('# fixture pin '+name+'\n').encode();(root/'mutual_lab'/name).write_bytes(content);request[key]=hashlib.sha256(content).hexdigest()
            request.update(changes or {});(root/'configs/mastery-H-run.request.json').write_text(json.dumps(request))
            if mutation:mutation(root)
            old=os.getcwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ,{'GITHUB_REF':revision,'GITHUB_RUN_ATTEMPT':attempt}):
                    exec(compile(blocks()[0],'<actual-H-workflow-request>','exec'),{'__name__':'__main__'})
                return json.loads((root/'runs/producer-context-H-input/provenance/data-manifest.json').read_text())
            finally:os.chdir(old)
    def test_actual_full_request_and_native_data_gate(self):
        data=self.execute();self.assertEqual(data['scene_counts'],{'train':220,'dev':112});self.assertEqual(data['protected_pairs_excluded'],22)
    def test_missing_phase_and_wrong_architecture_rejected(self):
        for changes in ({'phase':None},{'hidden_dim':256},{'producer_memory':'cnn'},{'understanding':'base'},{'policy':'same_pair'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.execute(changes)
    def test_prior_and_source_identity_pins_are_required(self):
        for key in ('source_cohort_summary_sha256','source_cohort_revision','source_cohort_run_id','prior_manifest_sha256','prior_checkpoint_sha256'):
            with self.subTest(key=key),self.assertRaises(ValueError):self.execute({key:'0'*64})
    def test_integer_controls_reject_booleans_and_floats(self):
        for changes in ({'source_cohort_run_attempt':True},{'hidden_dim':128.0},{'fixed_final_step':9600.0},{'schema_version':True}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.execute(changes)
    def test_rerun_and_wrong_branch_rejected(self):
        with self.assertRaises(ValueError):self.execute(attempt='2')
        with self.assertRaises(ValueError):self.execute(revision='refs/heads/main')
    def test_changed_implementation_and_changed_plan_rejected(self):
        with self.assertRaises(ValueError):self.execute({'context_implementation_sha256':'0'*64})
        def mutate(root):
            path=root/'configs/mastery-H-plan.json';value=json.loads(path.read_text());value['seed_pairs'][0]=[69007,69009];path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.execute(mutation=mutate)
    def test_all_inline_blocks_compile_and_literal_commands_have_only_actual_cli_flags(self):
        self.assertEqual(len(blocks()),5)
        for index,code in enumerate(blocks()):compile(code,'<H-block-'+str(index)+'>','exec')
        commands=[line.strip() for line in WORKFLOW.read_text().splitlines() if 'python -m mutual_lab.producer_context_study ' in line]
        self.assertEqual(len(commands),2)
        self.assertIn("--replica '${{ matrix.replica }}'",commands[0]);self.assertNotIn('--policy',commands[0]);self.assertNotIn('--model',commands[0])
        self.assertNotIn('--prior-bundle',commands[1]);self.assertIn('--aggregate-dir',commands[1])
    def test_actual_cell_guard_rejects_rerun_context_and_source_drift(self):
        for fault in (None,'attempt','code','run','request','source','admission'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);(root/'configs').mkdir();(root/'mutual_lab').mkdir()
                (root/'configs/mastery-H-plan.json').write_bytes(b'fixed plan')
                request={'plan_sha256':hashlib.sha256(b'fixed plan').hexdigest()}
                request_path=root/'configs/mastery-H-run.request.json';request_path.write_text(json.dumps(request))
                request_sha=hashlib.sha256(request_path.read_bytes()).hexdigest()
                (root/'mutual_lab/native.py').write_bytes(b'unchanged source')
                context={'run_attempt':'1','code_revision':'a'*40,'run_id':'900','request':request,'request_sha256':request_sha,
                    'source_sha256':{'native.py':hashlib.sha256(b'unchanged source').hexdigest()}}
                admission={'status':'no_prior_admission','request_sha256':request_sha,'current_run':{'id':900}}
                if fault=='code':context['code_revision']='b'*40
                if fault=='run':context['run_id']='901'
                if fault=='request':context['request_sha256']='0'*64
                if fault=='source':(root/'mutual_lab/native.py').write_bytes(b'changed source')
                if fault=='admission':admission['status']='previously_admitted'
                provenance=root/'runs/producer-context-H-input/provenance';provenance.mkdir(parents=True)
                (provenance/'run-context.json').write_text(json.dumps(context));(provenance/'admission.json').write_text(json.dumps(admission))
                env={'GITHUB_RUN_ATTEMPT':'2' if fault=='attempt' else '1','GITHUB_SHA':'a'*40,'GITHUB_RUN_ID':'900'}
                code=blocks()[4].replace('${{ matrix.language }}','ja').replace('${{ matrix.replica }}','1')
                with patch.dict(os.environ,env):
                    old=os.getcwd()
                    try:
                        os.chdir(root)
                        if fault:
                            with self.assertRaises(ValueError):exec(compile(code,'<actual-H-cell-guard>','exec'),{'__name__':'__main__'})
                        else:
                            exec(compile(code,'<actual-H-cell-guard>','exec'),{'__name__':'__main__'})
                            self.assertTrue((root/'runs/producer-context-H/provenance-ja-r1/run-context.json').is_file())
                    finally:os.chdir(old)
    def test_global_discovery_lock_and_all_twelve_matrix_cells_are_preserved(self):
        source=WORKFLOW.read_text();self.assertIn('group: l1-producer-context-H\n  cancel-in-progress: false',source)
        self.assertIn('language: [ko, ja, en, es]',source);self.assertIn('replica: [0, 1, 2]',source)
        self.assertIn('retention-days: 30',source);self.assertIn('if: always()',source)

if __name__=='__main__':unittest.main()
