"""Immutable rolling-origin bindings for the imported14a scheduler."""
import os,sys,json,csv,pickle,math
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=HERE.parents[5];FULLRUN=ROOT/'revision_work/outputs/fullrun'
sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'))
from fullrun14a_io import target,jsonout,textout,csvout,utc,unique,sha,rel,profile_environment,ENV_KEYS,PYTHON,table,WEIGHTS
PLAN=json.loads((EXP/'experiment.json').read_text())['plan'];KEYS=['job_id','input_set','mechanism','horizon','seed','queue']
def jobs():return [{k:j[k] for k in KEYS} for j in PLAN]
def spec_for(jobid):return next(j for j in PLAN if j['job_id']==jobid)
def design():return json.loads((EXP/'design.json').read_text())
def counts(w):return design()['windows'][w]['counts']
def artifact(p):return dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p))
def extra_dependencies():return [artifact(p) for p in [EXP/'experiment.json',EXP/'design.json',EXP/'tensor_manifest.json']+sorted(HERE.rglob('*.py'))]
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,float) and not math.isfinite(v):return None
 if hasattr(v,'item'):return clean(v.item())
 return v
def verify_completion(path,expected_job=None):
 path=Path(path);r=json.loads(path.read_text());spec=spec_for(r['job_id'])
 if expected_job:
  for k in KEYS:assert r[k]==expected_job[k],(k,path)
 assert r['validated'] and r['row_counts']==counts(spec['window'])
 assert r['thread_counts_start']==r['thread_counts_end']==[12,12] and not r['thread_counts_changed']
 for env in [r['environment_start'],r['environment_end']]:assert env['OMP_NUM_THREADS'] is None and env['OMP_DYNAMIC']==env['MKL_DYNAMIC']=='FALSE'
 assert r['loader_generator_seed']==(42 if spec['block']=='neural' else None)
 for a in r['artifacts']+r['dependencies']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
 assert r['checkpoint'] in [a['path'] for a in r['artifacts']]
 if spec['block']=='reference':assert r['test_replay']['classification']=='EXACT'
 return r
def completed(spec):
 for p in sorted((EXP/'runs'/spec['job_id']).glob('attempt_*/COMPLETE.json')):
  try:return p,verify_completion(p,spec)
  except (AssertionError,OSError,KeyError,ValueError):continue
 return None,None
def sound(ok=True):
 import subprocess
 path='/System/Library/Sounds/Glass.aiff' if ok else '/System/Library/Sounds/Basso.aiff';r=subprocess.run(['/usr/bin/afplay',path],capture_output=True,text=True)
 return dict(utc=utc(),path=path,returncode=r.returncode,stderr=r.stderr)
