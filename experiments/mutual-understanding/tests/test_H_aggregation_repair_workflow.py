"""Execute the real aggregation-only workflow guards; no native training/network."""
import contextlib,hashlib,io,json,os
from pathlib import Path
import re,stat,tempfile,textwrap,unittest,urllib.error,urllib.parse,urllib.request,zipfile
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
WORKFLOW=HERE/'l1-H-aggregation-repair.yml'
if not WORKFLOW.is_file():
 for root in HERE.parents:
  candidate=root/'.github/workflows/l1-H-aggregation-repair.yml'
  if candidate.is_file():WORKFLOW=candidate;break
SOURCE=736097
REVISION='736097e5298fb71037a420ec8bd635c73a0154dd'
RUN=43652561578
FAILURE=132631927039
REPOSITORY='doheyn1008/DoVIS'

def block():
 source=WORKFLOW.read_text();return textwrap.dedent(re.findall(r"          python - <<'PY'\n(.*?)          PY",source,re.S)[0])
def source_run():return {'id':RUN,'head_sha':REVISION,'run_attempt':1,'repository':{'full_name':REPOSITORY},'path':'.github/workflows/l1-producer-context-H.yml','head_branch':'codex/l1-mastery','event':'push','status':'completed','conclusion':'failure'}
def jobs():
 values=[{'id':100,'name':'verify','status':'completed','conclusion':'success'},{'id':FAILURE,'name':'aggregate_h','status':'completed','conclusion':'failure'}]
 values.extend({'id':200+i,'name':f'phase_h ({language}, {replica})','status':'completed','conclusion':'success'} for i,(language,replica) in enumerate((l,r) for l in ('ko','ja','en','es') for r in range(3)))
 return values

class RepairWorkflowTests(unittest.TestCase):
 def execute_source_guard(self,*,run=None,records=None,repeat=False):
  run=run or source_run();records=records or jobs();calls=[]
  def github(path):
   calls.append(path)
   if path==f'/actions/runs/{RUN}/attempts/1':return run
   page=int(urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)['page'][0])
   return {'jobs':records[:7] if page==1 or repeat else records[7:] if page==2 else []}
  code=block().split('run=github(',1)[1];code='run=github('+code.split("catalog=request.get('source_artifacts')",1)[0]
  scope={'github':github,'source_id':RUN,'source_revision':REVISION,'failure_id':FAILURE,'repository':REPOSITORY,'urllib':__import__('urllib'),'json':json}
  exec(compile(code,'<actual-H-repair-source-job-guard>','exec'),scope);return calls
 def test_actual_known_failure_and_all_twelve_native_success_paginated(self):self.assertEqual(len(self.execute_source_guard()),4)
 def test_any_other_source_state_or_native_failure_is_rejected(self):
  for fault in ('overall_success','wrong_revision','attempt2','verify','native','aggregate_cancelled','aggregate_id','missing','duplicate','extra'):
   run,records=source_run(),jobs()
   if fault=='overall_success':run['conclusion']='success'
   if fault=='wrong_revision':run['head_sha']='0'*40
   if fault=='attempt2':run['run_attempt']=2
   if fault=='verify':records[0]['conclusion']='failure'
   if fault=='native':records[-1]['conclusion']='failure'
   if fault=='aggregate_cancelled':records[1]['conclusion']='cancelled'
   if fault=='aggregate_id':records[1]['id']=FAILURE+1
   if fault=='missing':records.pop()
   if fault=='duplicate':records[-1]=dict(records[-2])
   if fault=='extra':records.append({'id':999,'name':'new_training','status':'completed','conclusion':'success'})
   with self.subTest(fault=fault),self.assertRaises(ValueError):self.execute_source_guard(run=run,records=records)
 def test_stalled_job_pagination_fails_closed(self):
  with self.assertRaises(ValueError):self.execute_source_guard(repeat=True)
 def execute_archive(self,*,old_host=False,unsafe=False):
  memory=io.BytesIO()
  with zipfile.ZipFile(memory,'w') as output:
   output.writestr('../escape.json' if unsafe else 'report.json','{"frozen":true}')
   output.writestr('never-load-old.pt',b'not a model')
  content=memory.getvalue();name='fixture-original';pin={'id':123,'name':name,'digest':'sha256:'+hashlib.sha256(content).hexdigest(),'size_in_bytes':len(content)}
  artifact={**pin,'expired':False,'workflow_run':{'id':43647997707 if old_host else RUN,'head_sha':'c65ab2752c117ddf66c98c4e871f553cf9c8f95a' if old_host else REVISION}}
  class Opener:
   def open(self,query,timeout):return io.BytesIO(content)
  with tempfile.TemporaryDirectory() as temporary:
   directory=Path(temporary);provenance=directory/'provenance';provenance.mkdir();target=directory/'original'/name
   code=block();code=code[code.index('def fetch_artifact('):code.index('verified=[]')]
   scope={'github':lambda path:artifact,'source_id':RUN,'source_revision':REVISION,'re':re,'opener':Opener(),'api_request':lambda path:path,'directory':directory,'provenance':provenance,'urllib':__import__('urllib'),'shutil':__import__('shutil'),'hashlib':hashlib,'zipfile':zipfile,'PurePosixPath':__import__('pathlib').PurePosixPath,'stat':stat,'json':json}
   exec(compile(code,'<actual-H-repair-artifact-fetch>','exec'),scope)
   scope['fetch_artifact'](pin,target,old_host=old_host,only_member='report.json' if old_host else None)
   self.assertTrue((target/'report.json').is_file())
   self.assertEqual((target/'never-load-old.pt').exists(),not old_host)
   self.assertEqual((target.parent/(name+'.zip')).exists(),not old_host)
 def test_original_artifact_zip_is_retained_and_bound_to_digest(self):self.execute_archive()
 def test_old_evidence_extracts_only_report_and_never_old_tensor(self):self.execute_archive(old_host=True)
 def test_unsafe_zip_member_fails_closed(self):
  with self.assertRaises(ValueError):self.execute_archive(unsafe=True)
 def test_literal_workflow_has_original_source_checkout_and_no_native_training_cli(self):
  text=WORKFLOW.read_text();self.assertIn('ref: '+REVISION,text)
  self.assertIn('PYTHONPATH: ${{ github.workspace }}/source/experiments/mutual-understanding',text)
  self.assertIn('scripts/repair_H_aggregation.py" --input',text)
  for invalid in ('python -m mutual_lab.producer_context_study','--language','--replica','--resume'):self.assertNotIn(invalid,text)
  for number,code in enumerate(re.findall(r"          python - <<'PY'\n(.*?)          PY",text,re.S)):compile(textwrap.dedent(code),'<actual-H-repair-block-'+str(number)+'>','exec')
if __name__=='__main__':unittest.main()
