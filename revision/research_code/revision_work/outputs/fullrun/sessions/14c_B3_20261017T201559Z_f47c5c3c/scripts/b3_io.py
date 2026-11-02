"""Immutable B3 design/job metadata and original14a exclusive output helpers."""
import sys,os,json,csv,math
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=HERE.parents[5];FULLRUN=ROOT/'revision_work/outputs/fullrun'
sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'))
from fullrun14a_io import target,jsonout,textout,csvout,sha,rel,utc,unique,table,profile_environment,ENV_KEYS,COUNTS,PYTHON,WEIGHTS
PLAN=json.loads((EXP/'experiment.json').read_text())['plan'];KEYS=['job_id','input_set','mechanism','horizon','seed','queue']
def artifact(p):return dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p))
def spec_for(jobid):return next(s for s in PLAN if s['job_id']==jobid)
def jobs():return [{k:s[k] for k in KEYS} for s in PLAN]
