"""Exercise the future loader in an isolated assembled package, never the live one."""

import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FUTURE_SOURCE = ROOT / "mutual_lab"

H_SCRIPT = r'''
from dataclasses import asdict
import copy,hashlib,json
from pathlib import Path
from mutual_lab.neural import configure_cpu,digest,nn,torch
from mutual_lab.mastery_lab import MasteryPlan
from mutual_lab.mastery_infer import load_development_model,infer_development
from mutual_lab.producer_context import build_producer_context_model
from mutual_lab.profiles import ModelProfile
from mutual_lab.spatial_prior import pretrain_prior,save_prior_bundle,load_prior_bundle
from mutual_lab.world import Object,variants
from mutual_lab.development_world import make_examples,render_scene
configure_cpu()
root=Path('artifacts');root.mkdir()
profile=ModelProfile('h-loader-h128',embedding_dim=64,hidden_dim=128,max_sequence_length=128)
scenes=variants((Object('blue','square'),Object('red','circle')))
pngs=[render_scene(scene) for scene in scenes]
prior=save_prior_bundle(pretrain_prior(pngs,dim=128,steps=1,batch_size=4,eval_every=1),root/'prior.pt')
bundle=load_prior_bundle(prior)
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('mutual_lab').glob('*.py')}
plan=asdict(MasteryPlan(budgets=(2,),data_seed=59003,b_seed_pair=(79007,79009),
    seed_pairs=((79007,79009),(79017,79019),(79027,79029)),model_profile=profile))

def save(name,payload):
    path=root/(name+'.pt');torch.save(payload,path)
    path.with_suffix('.json').write_text(json.dumps({'format':'mutual-mastery-development-v1','kind':'development_only',
      'excluded_from_research':True,'eligible_for_fresh_experiment':False,'step':2,
      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    return path

results={};models=[]
for condition,context in (('H0','cnn'),('H1','object_moments')):
    model=build_producer_context_model(profile,79007,bundle,context=context);models.append(model)
    results[condition]={'initial_nn_sha256':digest(model.state_dict())}
assert results['H0']['initial_nn_sha256']==results['H1']['initial_nn_sha256']
for (condition,context),model in zip((('H0','cnn'),('H1','object_moments')),models):
    examples=make_examples(model,scenes[:1],'ko')
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model.grounding_loss(examples)+model.production_loss(examples)).backward()
        nn.utils.clip_grad_norm_(model.understanding.parameters(),1.,error_if_nonfinite=True)
        nn.utils.clip_grad_norm_(model.producer.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    ids,_=model.tokenizer.batch([examples[0]['text']])
    with torch.no_grad(): expected_logits=model.teacher_forced_logits(ids[:,:-1],[pngs[0]])
    expected_trace=model.generate_with_trace(pngs[0]);expected_u=model.predict(examples[0]['text'],pngs)
    factory='mutual_lab.producer_context_study.RecordingFactory.'+condition
    payload={'format':'mutual-mastery-development-v1','kind':'development_only','excluded_from_research':True,
      'eligible_for_fresh_experiment':False,'language':'ko','phase':'C','condition':'C1','replica':0,
      'seed_pair':[79007,79009],'step':2,'plan':plan,'model_factory':factory,'model':model.state_dict(),
      'optimizer':optimizer.state_dict(),'source_sha256':sources,'prior_metadata':model.spatial_prior_metadata,
      'frozen_feature_policy':'per_png_singleton_v1'}
    path=save(condition,payload);restored=load_development_model(path,prior)
    assert restored.producer_context_source==context
    assert restored.object_memory_source=='objects' and restored.understanding.object_context
    assert digest(restored.state_dict())==digest(model.state_dict())
    with torch.no_grad(): assert torch.equal(expected_logits,restored.teacher_forced_logits(ids[:,:-1],[pngs[0]]))
    result=infer_development(restored,pngs[0],text=examples[0]['text'],choices_png=pngs)
    assert result['generation']==expected_trace and result['comprehension']['prediction']==expected_u
    assert result['read_only_state_verified'] and result['excluded_from_research'] and not result['eligible_for_fresh_experiment']
    assert result['producer_global_context']['context']==context and result['model_factory']==factory
    results[condition].update(context=context,factory=factory,teacher_logits_exact=True,greedy_exact=True,understanding_exact=True)

negative=0
for name in ('producer_context.py','producer_context_study.py','frozen_features.py'):
    altered={**payload,'source_sha256':{**sources,name:'wrong'}}
    try: load_development_model(save('bad-source',altered),prior)
    except ValueError as error: assert name in str(error);negative+=1
    else: raise AssertionError('Missing source guard: '+name)
for factory in ('mutual_lab.producer_context_study.RecordingFactory.H2','__main__.RecordingFactory.H1'):
    try: load_development_model(save('bad-factory',{**payload,'model_factory':factory}),prior)
    except ValueError: negative+=1
    else: raise AssertionError('Unknown H factory accepted')
altered=copy.deepcopy(payload);del altered['model']['understanding.base.embedding.weight']
try: load_development_model(save('bad-state',altered),prior)
except ValueError: negative+=1
else: raise AssertionError('Missing G1 base tensor accepted')
altered={key:value for key,value in payload.items() if key!='frozen_feature_policy'}
try: load_development_model(save('missing-policy',altered),prior)
except ValueError: negative+=1
else: raise AssertionError('H missing canonical policy accepted')
print(json.dumps({'models':results,'negative_cases':negative,'scope':'DEVELOPMENT_ONLY_H128_SHORT_CHECKPOINT_SMOKE'}))
'''


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "numpy", "PIL")),
                     "requires optional CPU torch, NumPy and Pillow")
class FutureHLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        package = cls.root / "mutual_lab"
        shutil.copytree(ROOT / "mutual_lab", package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # This assembles exactly the future package layout in temporary space;
        # the actual frozen project and its loader are never overwritten.
        for name in ("mastery_infer.py", "producer_context.py", "producer_context_study.py"):
            shutil.copyfile(FUTURE_SOURCE / name, package / name)
        tests = cls.root / "tests"; tests.mkdir()
        shutil.copyfile(ROOT / "tests/test_mastery_infer.py", tests / "test_mastery_infer.py")
        (cls.root / "h_checkpoint_smoke.py").write_text(H_SCRIPT)
        cls.environment = {**os.environ, "PYTHONPATH": str(cls.root), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_existing_loader_formats_and_factory_aliases_remain_supported(self):
        process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=self.root, env=self.environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("OK", process.stderr)

    def test_saved_h0_h1_actual_h128_short_checkpoints_restore_every_route_and_strict_provenance(self):
        process = subprocess.run([sys.executable, "h_checkpoint_smoke.py"], cwd=self.root,
            env=self.environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["negative_cases"], 7)
        self.assertEqual(result["models"]["H0"]["context"], "cnn")
        self.assertEqual(result["models"]["H1"]["context"], "object_moments")


if __name__ == "__main__":
    unittest.main()
