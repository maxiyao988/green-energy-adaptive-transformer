"""Retry only the two interrupted Static cells after the first bounded batch has exited."""
import os,sys,time,subprocess,json
sys.dont_write_bytecode=True
from core import *
def main():
 while not (OUT/'stage3a_processes.json').exists():time.sleep(5)
 first=json.loads((OUT/'stage3a_processes.json').read_text());failed={(x['mechanism'],x['horizon']) for x in first['processes'] if x['returncode']!=0};assert failed=={('static',5),('static',10)},failed
 active=[];start=time.perf_counter();started=utc();records=[]
 for h in [5,10]:
  assert not (OUT/f'checkpoints/static_dw_h{h}_s42.pt').exists()
  log=target(OUT/f'logs/static_{h}_retry.log').open('x');env=dict(os.environ,OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',PYTHONDONTWRITEBYTECODE='1');p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'revision_work/scripts/causal/stage3a_worker.py'),'--mechanism','static','--horizon',str(h)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);active.append((p,log,dict(mechanism='static',horizon=h,pid=p.pid,started_utc=utc(),start=time.perf_counter(),log=rel(log.name))));print('RETRY',h,p.pid,flush=True)
 while active:
  for p,log,r in list(active):
   code=p.poll()
   if code is not None:r.update(returncode=code,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-r.pop('start'));records.append(r);log.close();active.remove((p,log,r));print('EXIT',r,flush=True)
  if active:time.sleep(2)
 jsonout('stage3a_retry_processes.json',dict(started_utc=started,completed_utc=utc(),parallel_wall_clock_seconds=time.perf_counter()-start,processes=records,max_processes=2))
 if any(r['returncode'] for r in records):raise SystemExit('Retry failed')
if __name__=='__main__':main()
