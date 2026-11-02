"""Launch exactly three separate formal recovery processes concurrently."""
import os,sys,subprocess,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'revision_work/outputs'
script=ROOT/'revision_work/scripts/recover_adaptive42.py'
(OUT/'logs').mkdir(parents=True,exist_ok=True)
children=[];start=time.perf_counter()
env=os.environ.copy();env.update(OMP_NUM_THREADS='3',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
for h in [5,10,20]:
 log=(OUT/'logs'/f'recovery_h{h}.log').open('x')
 child=subprocess.Popen([sys.executable,'-B',str(script),'--horizon',str(h)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 children.append((h,child,log));print('LAUNCHED horizon',h,'pid',child.pid,flush=True)
statuses=[]
for h,child,log in children:
 status=child.wait();log.close();statuses.append(dict(horizon=h,pid=child.pid,returncode=status))
 print('EXIT horizon',h,'status',status,flush=True)
with (OUT/'task_a_processes.json').open('x') as f:json.dump(dict(processes=statuses,parallel_wall_clock_seconds=time.perf_counter()-start,OMP_NUM_THREADS=3,torch_num_threads_per_process=3,torch_num_interop_threads_per_process=1),f,indent=2)
raise SystemExit(0 if all(s['returncode']==0 for s in statuses) else 1)
