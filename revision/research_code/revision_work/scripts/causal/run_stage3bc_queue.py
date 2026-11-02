"""User-authorized Step4 Stage3b + Stage3c queue; starts after Step3B h10 ends."""
import os,sys,time,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'revision_work/outputs/causal_features';OUT=BASE/'stage3bc_queue';REPRO=ROOT/'revision_work/outputs/repro_threads'
def utc():return datetime.now(timezone.utc).isoformat()
def main():
 assert (REPRO/'runs/adaptive_dw_h10_s42_t12.json').is_file()
 OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();stamp=utc();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
 with (OUT/'feature_build.log').open('x') as f:code=subprocess.call([str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'revision_work/scripts/causal/build_stage3c_asof.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 if code:raise SystemExit('As-of feature builder failed; see retained log')
 jobs=[dict(stage='stage3b',mechanism=m,horizon=h,seed=s) for s in [0,1,21,3407] for h in [5,10,20] for m in ['static','adaptive']]+[dict(stage='stage3c',mechanism=m,horizon=h,seed=42) for h in [5,10,20] for m in ['static','adaptive']];assert len(jobs)==30;active=[];records=[]
 for job in jobs:
  name=f"{job['stage']}_{job['mechanism']}_h{job['horizon']}_s{job['seed']}";p=OUT/f'{name}.log'
  if p.exists():raise FileExistsError(p)
 while jobs or active:
  while jobs and len(active)<4:
   job=jobs.pop(0);name=f"{job['stage']}_{job['mechanism']}_h{job['horizon']}_s{job['seed']}";log=(OUT/f'{name}.log').open('x');cmd=[str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'revision_work/scripts/causal/stage3bc_worker.py')]
   for k,v in job.items():cmd+=['--'+k,str(v)]
   p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);row=dict(**job,pid=p.pid,started_utc=utc(),start=time.perf_counter(),log=str(Path(log.name).relative_to(ROOT)));active.append((p,log,row));print('LAUNCHED',row,flush=True)
  for p,log,row in list(active):
   code=p.poll()
   if code is not None:row.update(returncode=code,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-row.pop('start'));records.append(row);log.close();active.remove((p,log,row));print('EXIT',row,flush=True)
  if active:time.sleep(2)
 with (OUT/'processes.json').open('x') as f:json.dump(dict(started_utc=stamp,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-start,max_workers=4,processes=records),f,indent=2)
 if any(r['returncode'] for r in records):raise SystemExit('Stage3b/c worker failure; inspect retained logs')
 print('QUEUE COMPLETE',utc(),len(records),flush=True)
if __name__=='__main__':main()
