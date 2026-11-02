"""Exclusive supplement artifacts; original rolling design/tensors remain read-only."""
import os,sys,json,csv,math
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=HERE.parents[5]
FULLRUN=ROOT/'revision_work/outputs/fullrun'
OLD=ROOT/'revision_work/outputs/fullrun/sessions/14c_B2_20261017T101438Z_1a8d5041'
sys.path.insert(0,str(OLD/'scripts'))
import b2_io as original_io
from fullrun14a_io import target,jsonout,textout,csvout,utc,unique,sha,rel,profile_environment,ENV_KEYS,PYTHON,table,WEIGHTS
META=json.loads((EXP/'experiment.json').read_text());PLAN=META['plan'];NEW=META['new_plan']
KEYS=original_io.KEYS;OLD_IDS={s['job_id'] for s in original_io.PLAN}
design=original_io.design;counts=original_io.counts;artifact=original_io.artifact;sound=original_io.sound
def jobs():return [{k:s[k] for k in KEYS} for s in PLAN]
def spec_for(jobid):return next(s for s in PLAN if s['job_id']==jobid)
def extra_dependencies():
 return [artifact(p) for p in [EXP/'experiment.json',EXP/'ORIGINAL_REUSE.json',EXP/'W3_provenance.json']+sorted(HERE.rglob('*.py'))]
def verify_completion(path,expected_job=None):
 r=json.loads(Path(path).read_text());spec=spec_for(r['job_id'])
 if spec['job_id'] in OLD_IDS:return original_io.verify_completion(path,expected_job)
 if expected_job:
  for k in KEYS:assert r[k]==expected_job[k],(k,path)
 assert r['seed'] in [21,3407] and spec['block']=='neural'
 assert r['validated'] and r['row_counts']==counts(spec['window'])
 assert r['thread_counts_start']==r['thread_counts_end']==[12,12] and not r['thread_counts_changed']
 for env in [r['environment_start'],r['environment_end']]:assert env['OMP_NUM_THREADS'] is None and env['OMP_DYNAMIC']==env['MKL_DYNAMIC']=='FALSE'
 assert r['loader_generator_seed']==42 and r['training_entrypoint']=='experiment4.train_and_validate_cell'
 for a in r['artifacts']+r['dependencies']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
 assert r['checkpoint'] in [a['path'] for a in r['artifacts']]
 return r
def completed(spec):
 if spec['job_id'] in OLD_IDS:
  p,r=original_io.completed(spec)
  assert r,('Original completed cell failed verification; MUST NOT be retrained',spec)
  return p,r
 for p in sorted((EXP/'runs'/spec['job_id']).glob('attempt_*/COMPLETE.json')):
  try:return p,verify_completion(p,spec)
  except (AssertionError,OSError,KeyError,ValueError):continue
 return None,None
