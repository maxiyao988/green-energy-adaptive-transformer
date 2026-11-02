"""One training worker at a time; branch only on the Adaptive h10 comparison."""
import os,sys,time,json,subprocess,argparse,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revision_work/outputs/repro_threads'

def utc():
 from datetime import datetime,timezone
 return datetime.now(timezone.utc).isoformat()
def exclusive(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(obj,f,indent=2)
def main():
 p=argparse.ArgumentParser();p.add_argument('--isolation',required=True,choices=['other_user_work_stopped','resident_apps_allowed']);a=p.parse_args();exclusive('isolation_condition.json',dict(utc=utc(),condition=a.isolation,training_concurrency=1,notes='No applications are terminated or suspended by this runner. OS/services and agent remain present; background activity is sampled.'))
 env=os.environ.copy();env.pop('OMP_NUM_THREADS',None);env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE');records=[];start=time.perf_counter();started=utc()
 def run(mechanism,h):
  stem=f'{mechanism}_dw_h{h}_s42_t12';logpath=OUT/f'logs/{stem}.log';logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open('x');command=[str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'revision_work/scripts/causal/repro_threads_worker.py'),'--mechanism',mechanism,'--horizon',str(h)];beg=time.perf_counter();stamp=utc();child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  print('LAUNCHED',utc(),mechanism,h,child.pid,flush=True);loadpath=OUT/f'logs/{stem}_background_processes.csv'
  with loadpath.open('x',newline='') as f:
   writer=csv.writer(f);writer.writerow(['utc','training_pid','pid','ppid','cpu_pct','command']);nextsample=0
   while child.poll() is None:
    if time.perf_counter()>=nextsample:
     snapshot=subprocess.run(['ps','-axo','pid,ppid,pcpu,comm'],capture_output=True,text=True,check=True).stdout.splitlines()[1:]
     parsed=[]
     for line in snapshot:
      fields=line.strip().split(None,3)
      if len(fields)==4:parsed.append(fields)
     for pid,ppid,cpu,comm in sorted(parsed,key=lambda row:float(row[2]),reverse=True)[:20]:writer.writerow([utc(),child.pid,pid,ppid,cpu,comm])
     f.flush();nextsample=time.perf_counter()+30
    time.sleep(2)
  code=child.wait();log.close();row=dict(mechanism=mechanism,horizon=h,pid=child.pid,returncode=code,started_utc=stamp,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-beg,log=str(logpath.relative_to(ROOT)));records.append(row);print('EXIT',row,flush=True)
  if code:exclusive('processes_failed.json',dict(records=records));raise SystemExit('Worker failed; retained artifacts; do not overwrite/retry blindly')
  return json.loads((OUT/f'runs/{stem}.json').read_text())
 first=run('adaptive',10)
 queue_start=utc();queue_code=subprocess.call([str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'revision_work/scripts/causal/run_stage3bc_queue.py')],cwd=ROOT,env=env)
 exclusive('stage3bc_queue_status.json',dict(started_utc=queue_start,completed_utc=utc(),returncode=queue_code))
 if queue_code:raise SystemExit('Stage3b/3c queue failed; retain outputs and resolve before final recovery')
 if first['classification'] in ['EXACT','NEAR']:
  branch='Adaptive h10 EXACT/NEAR -> Stage3b/3c queue -> sequential h5, h20';run('adaptive',5);run('adaptive',20)
 else:
  branch='Adaptive h10 DIFFERENT -> Stage3b/3c queue -> Static h10 control';run('static',10)
 exclusive('processes.json',dict(started_utc=started,completed_utc=utc(),wall_clock_seconds=time.perf_counter()-start,branch=branch,training_concurrency=1,processes=records));print('COMPLETE',branch,flush=True)
if __name__=='__main__':main()
