"""Bounded parallel Stage 3a only; refuses existing logs or artifacts."""
import os,sys,time,subprocess,json
from pathlib import Path
sys.dont_write_bytecode=True
from core import OUT,ROOT,target,jsonout,utc

def main():
 assert json.loads((OUT/'precision_gate_decision.json').read_text())['stage3a_authorized_by_diagnostic']
 jobs=[('static',5),('adaptive',5),('static',10),('adaptive',10),('static',20),('adaptive',20),('ridge',None)]
 # Ridge is a brief fourth worker; fill released slots with remaining neural cells.
 jobs.insert(3,jobs.pop())
 for m,h in jobs:target(OUT/f'logs/{m}_{h if h else "all"}.log')
 target(OUT/'stage3a_processes.json');start=time.perf_counter();started=utc();active=[];finished=[]
 while jobs or active:
  while jobs and len(active)<4:
   m,h=jobs.pop(0);cmd=[str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'revision_work/scripts/causal/stage3a_worker.py'),'--mechanism',m]
   if h:cmd+=['--horizon',str(h)]
   log=target(OUT/f'logs/{m}_{h if h else "all"}.log').open('x');env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',PYTHONDONTWRITEBYTECODE='1');p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   row=dict(mechanism=m,horizon=h,pid=p.pid,started_utc=utc(),monotonic_start=time.perf_counter(),log=str(Path(log.name).relative_to(ROOT)));active.append((p,log,row));print('LAUNCHED',row,flush=True)
  for p,log,row in list(active):
   code=p.poll()
   if code is not None:
    row.update(returncode=code,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-row.pop('monotonic_start'));finished.append(row);log.close();active.remove((p,log,row));print('EXIT',row,flush=True)
  if active:time.sleep(2)
 jsonout('stage3a_processes.json',dict(started_utc=started,completed_utc=utc(),parallel_wall_clock_seconds=time.perf_counter()-start,max_processes=4,processes=finished,stage3b='NOT RUN',stage3c='NOT RUN'))
 if any(x['returncode'] for x in finished):raise SystemExit('One or more Stage 3a workers failed; inspect logs')
if __name__=='__main__':main()
