"""Step14c-day adapter for the imported14a queue and immutable contracts."""
import os,sys,json,csv,pickle
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=HERE.parents[5]
sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'))
from fullrun14a_io import (target,jsonout,textout,csvout,utc,unique,sha,rel,profile_environment,ENV_KEYS,COUNTS,SEEDS,HORIZONS,PYTHON,table)
FULLRUN=ROOT/'revision_work/outputs/fullrun'
PLAN=json.loads((EXP/'experiment.json').read_text())['plan']
SPEC_KEYS=['job_id','input_set','mechanism','horizon','seed','queue']
SAME_WEIGHTS=[f'daily_expert{i}_patch10' for i in range(1,5)]+[f'weekly_expert{i}_patch4' for i in range(1,5)]
DWEIGHTS=['daily_w_5','daily_w_10','daily_w_20','daily_w_30']
WWEIGHTS=['weekly_w_2','weekly_w_4','weekly_w_8','weekly_w_13']
def phase_jobs(phase):
 if phase=='b1':rs=[j for j in PLAN if j['block']=='B1']
 elif phase=='pilot':rs=PLAN[30:31]
 elif phase=='b4':rs=PLAN[31:]
 else:raise ValueError(phase)
 return [{k:j[k] for k in SPEC_KEYS} for j in rs]
def spec_for(jobid):return next(j for j in PLAN if j['job_id']==jobid)
def verify_completion(path,expected_job=None):
 path=Path(path);r=json.loads(path.read_text());assert r['validated']
 if expected_job:
  for k in SPEC_KEYS:assert r[k]==expected_job[k],(k,path)
 spec=spec_for(r['job_id'])
 assert r['row_counts']==COUNTS
 assert r['thread_counts_start']==r['thread_counts_end']==[12,12] and not r['thread_counts_changed']
 assert r['environment_start']['OMP_NUM_THREADS'] is None
 assert r['environment_start']['OMP_DYNAMIC']==r['environment_start']['MKL_DYNAMIC']=='FALSE'
 assert r['loader_generator_seed']==(42 if spec['block']=='B1' or spec['mechanism']=='adaptive' else None)
 for dep in r['artifacts']+r['dependencies']:
  p=ROOT/dep['path'];assert sha(p)==dep['sha256'],p
 assert r['checkpoint'] in [a['path'] for a in r['artifacts']]
 if spec['block']=='B4':assert r['test_replay']['classification']=='EXACT' and r['causal_input_check']['passed']
 return r
def completed(spec):
 for p in sorted((EXP/'runs'/spec['job_id']).glob('attempt_*/COMPLETE.json')):
  try:return p,verify_completion(p,spec)
  except (AssertionError,OSError,ValueError,KeyError):continue
 return None,None
def extra_dependencies():
 ps=[EXP/'experiment.json']+sorted(HERE.glob('*.py'))
 return [dict(path=rel(p),sha256=sha(p)) for p in ps]
def artifact(p):return dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p))
def sound(success=True):
 import subprocess
 p='/System/Library/Sounds/Glass.aiff' if success else '/System/Library/Sounds/Basso.aiff'
 r=subprocess.run(['/usr/bin/afplay',p],capture_output=True,text=True)
 return dict(utc=utc(),sound=p,returncode=r.returncode,stderr=r.stderr)
