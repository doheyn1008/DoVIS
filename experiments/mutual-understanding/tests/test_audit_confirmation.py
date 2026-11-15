import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "scripts"))
import audit_confirmation as a
HAS_TORCH = all(importlib.util.find_spec(k) is not None for k in ("torch", "numpy", "PIL"))


class PureTests(unittest.TestCase):
    def official(self):
        record={'sha256':'a'*64};plan={'budgets':[2400,4800,9600]};spec={'context':'H1'}
        value={'kind':'cohort-context-validation-v2-comparison','candidate_request':record,'plan':plan,'candidate_spec':spec,
            'source_condition':'H1','fixed_final_step':9600,'shared_visual_prior':True,'independent_visual_replications':False,
            'excluded_from_research':True,'eligible_for_fresh_experiment':False,
            'rows':[{'language':l,'replica':r,'step':s,'status':'completed','seed_pair':list(a.PAIRS[r])}
                for l in a.LANGUAGES for r in range(3) for s in a.STEPS]}
        return value,record,plan,spec

    def test_official_requires_all_exact48_native_identities_and_source_labels(self):
        value,record,plan,spec=self.official()
        self.assertEqual(len(a.official_rows_index(value,record,plan,spec)),48)
        for field,bad in [('replica',False),('step',False),('status','cancelled'),('seed_pair',[69007,69009])]:
            with self.subTest(field=field):
                fake=copy.deepcopy(value);fake['rows'][0][field]=bad
                with self.assertRaises(ValueError):a.official_rows_index(fake,record,plan,spec)

    def test_official_rejects_duplicate_rows_even_with_total48(self):
        value,record,plan,spec=self.official();value['rows'][0]=copy.deepcopy(value['rows'][1])
        with self.assertRaisesRegex(ValueError,'identities'):a.official_rows_index(value,record,plan,spec)

    def test_official_cannot_change_scope_candidate_or_plan(self):
        value,record,plan,spec=self.official()
        for field,bad in [('candidate_request',{}),('plan',{}),('candidate_spec',{}),('source_condition','H0'),
                ('excluded_from_research',False),('eligible_for_fresh_experiment',True),('fixed_final_step',4800),
                ('shared_visual_prior',False),('independent_visual_replications',True)]:
            with self.subTest(field=field):
                fake=copy.deepcopy(value);fake[field]=bad
                with self.assertRaises(ValueError):a.official_rows_index(fake,record,plan,spec)

    def test_payload_and_manifest_scope_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            a.check_payload({}, {}, {}, 0, a.PAIRS[0], {}, {}, {}, "0"*40, None)

    def test_canonical_plan_normalizes_tuple_list_without_bool_alias(self):
        self.assertEqual(a.canon(((1, 2),)), a.canon([[1, 2]]))
        self.assertNotEqual(a.canon(False), a.canon(0))

    def test_zip_authentication_rejects_coordinated_extracted_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); name = "test-artifact"; artifact = root / name; artifact.mkdir()
            (artifact / "report.json").write_text('{"score":1}')
            archive = root / (name + ".zip")
            with zipfile.ZipFile(archive, "w") as z:
                z.write(artifact / "report.json", "report.json")
            meta = {"id": 1, "name": name, "expired": False, "size_in_bytes": archive.stat().st_size,
                    "digest": "sha256:"+a.sha(archive), "workflow_run": {"id": 2, "head_sha": "a"*40}}
            (root / (name + ".artifact.json")).write_text(json.dumps(meta))
            self.assertTrue(a.authenticate(artifact, name, 2, "a"*40)["verified"])
            (artifact / "report.json").write_text('{"score":0}')
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                a.authenticate(artifact, name, 2, "a"*40)

    def test_zip_rejects_extra_files_and_wrong_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); name = "artifact"; d = root / name; d.mkdir(); p = d / "x.json"; p.write_text('{}')
            zpath = root / (name+".zip")
            with zipfile.ZipFile(zpath, "w") as z: z.write(p, "x.json")
            (root/(name+".artifact.json")).write_text(json.dumps({"id":1,"name":name,"expired":False,
                "size_in_bytes":zpath.stat().st_size,"digest":"sha256:"+a.sha(zpath),"workflow_run":{"id":2,"head_sha":"a"*40}}))
            with self.assertRaisesRegex(ValueError, "source run"):
                a.authenticate(d,name,3,"a"*40)
            (d/"extra.json").write_text('{}')
            with self.assertRaisesRegex(ValueError,"file set"):
                a.authenticate(d,name,2,"a"*40)

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for value in ("../x", "/x", "x\\y"):
                with self.subTest(value=value), self.assertRaises(ValueError):a.local(Path(temp), value)


@unittest.skipUnless(HAS_TORCH, "Optional CPU Torch, NumPy and Pillow")
class TensorTests(unittest.TestCase):
    def fixture(self):
        from mutual_lab.neural import torch, digest
        weights = {"understanding.w": torch.tensor([1.,2.]), "producer.w":torch.tensor([3.]), "visual_encoder.w":torch.tensor([4.])}
        branches = {b:digest({n[len(b)+1:]:v for n,v in weights.items() if n.startswith(b+'.')}) for b in ('understanding','producer','visual_encoder')}
        group = [{"lr":.001,"params":[0,1,2]}]
        torch.manual_seed(99009); tr=torch.random.get_rng_state().clone()
        ref = {"full_state_sha256":digest(weights),"prior_sha256":branches['visual_encoder'],
            "shapes":{n:(list(v.shape),str(v.dtype)) for n,v in weights.items()},"optimizer_groups":group,
            "torch_rng_sha256":hashlib.sha256(tr.numpy().tobytes()).hexdigest(),"torch_rng_shape":list(tr.shape)}
        raw = {"language":"ko","phase":"C","condition":"C1","replica":0,"seed_pair":[99007,99009],"model_factory":a.FACTORY,
            "source_sha256":{"x":"a"*64},"data_sha256":"b"*64,"split":{"x":1},"revision":{"commit":"c"*40,"working_tree_dirty":False},
            "data_version":"five-colors-three-shapes-v1","prior_metadata":{"x":1},"plan":{"seed_pairs":((99007,99009),)},
            "batch_size":16,"initial_prior_sha256":branches['visual_encoder'],"final_prior_sha256":branches['visual_encoder'],
            "frozen_feature_tables":{"step_0":{"x":1},"final":{"x":1}}}
        rng,neg=random.Random(99009).getstate(),random.Random(199012).getstate()
        stream={"snapshots":{0:{"training_rng_sha256":hashlib.sha256(a.canon(rng).encode()).hexdigest(),
            "negative_rng_sha256":hashlib.sha256(a.canon(neg).encode()).hexdigest()}}}
        p={k:copy.deepcopy(v) for k,v in raw.items() if k not in ('batch_size','initial_prior_sha256','final_prior_sha256')}
        p.update(format="mutual-mastery-development-v1",kind="development_only",excluded_from_research=True,
            eligible_for_fresh_experiment=False,step=0,model=weights,optimizer={"state":{},"param_groups":group},
            training_rng_state=rng,negative_rng_state=neg,torch_rng_state=tr,negative_exposure_offset=0,frozen_feature_policy="per_png_singleton_v1")
        m={"format":p['format'],"kind":p['kind'],"excluded_from_research":True,"eligible_for_fresh_experiment":False,"step":0,"sha256":"d"*64}
        return p,m,raw,ref,stream,digest

    def checked(self,p,m,r,ref,stream,digest):
        return a.check_payload(p,m,r,0,a.PAIRS[0],ref,stream,r['source_sha256'],'c'*40,digest)

    def test_actual_tensor_payload_plan_rng_and_initial_nn_are_verified(self):
        p,m,r,ref,s,d=self.fixture(); r['plan']=json.loads(json.dumps(r['plan']))
        self.assertTrue(self.checked(p,m,r,ref,s,d)['actual_rng_and_optimizer_verified'])

    def test_payload_or_manifest_actual_step_mismatch_fails(self):
        for payload_step,manifest_step in ((1,0),(0,1),(0,False),(False,0)):
            with self.subTest(payload_step=payload_step,manifest_step=manifest_step):
                p,m,r,ref,s,d=self.fixture();p['step']=payload_step;m['step']=manifest_step
                with self.assertRaisesRegex(ValueError,'step mismatch'):self.checked(p,m,r,ref,s,d)

    def test_initial_weight_optimizer_rng_and_shape_tampering_fail(self):
        cases=['weights','optimizer','main_rng','negative_rng','torch_rng','shape','scope']
        for case in cases:
            with self.subTest(case=case):
                p,m,r,ref,s,d=self.fixture()
                if case=='weights':p['model']['understanding.w'][0]+=1
                if case=='optimizer':p['optimizer']['state']={0:{'step':1}}
                if case=='main_rng':p['training_rng_state']=random.Random(1).getstate()
                if case=='negative_rng':p['negative_rng_state']=random.Random(2).getstate()
                if case=='torch_rng':p['torch_rng_state'][0]+=1
                if case=='shape':p['model']['understanding.w']=p['model']['understanding.w'].view(1,2)
                if case=='scope':m['eligible_for_fresh_experiment']=True
                with self.assertRaises(ValueError):self.checked(p,m,r,ref,s,d)

    def test_cache_integrity_requires_same_own_tables_not_cross_host_groups(self):
        from mutual_lab.neural import digest
        caches={n:{'entries':count,'sha256':'a'*64} for n,count in {'object_memory_cache':332,'producer_context_cache':332,
            'spatial_grid_cache':0,'understanding_object_cache':332,'visual_cache':332}.items()}
        table={'policy':'per_png_singleton_v1','caches':caches};table['sha256']=digest(table)
        pre={'status':'completed','input_pngs':332,'unique_pngs':332,'successful_pngs':332,'fallback_count':0,
            'fallback_policy':'raise; no fallback','model_state_unchanged':True,'rng_unchanged':True,'feature_tables':table}
        self.assertEqual(a.validate_cache({'step_0':table,'final':table},{'memory_preflight':pre},[{'frozen_feature_table_sha256':table['sha256']}],digest),table['sha256'])
        broken=copy.deepcopy(table);broken['caches']['visual_cache']['sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'composite digest'):
            a.validate_cache({'step_0':broken,'final':broken},{'memory_preflight':pre},[],digest)
        for field in ('fallback_count','rng_unchanged','model_state_unchanged'):
            fake=copy.deepcopy(pre);fake[field]=1 if field=='fallback_count' else False
            with self.assertRaises(ValueError):a.validate_cache({'step_0':table,'final':table},{'memory_preflight':fake},[],digest)


if __name__=='__main__':unittest.main()
