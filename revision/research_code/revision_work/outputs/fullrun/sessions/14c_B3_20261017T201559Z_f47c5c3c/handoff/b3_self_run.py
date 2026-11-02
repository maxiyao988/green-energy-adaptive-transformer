#!/usr/bin/env python3
"""User-facing B3 launcher/status/verification; never duplicates or kills a queue."""
import argparse,csv,fcntl,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
from datetime import datetime,timezone,timedelta
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=HERE.parents[5]
PYTHON=ROOT/'.venv/bin/python';QUEUE=EXP/'scripts/b3_queue.py'
SGT=timezone(timedelta(hours=8));PLAN=json.loads((EXP/'experiment.json').read_text())['plan']

def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
 return digest.hexdigest()

def queue_locked():
 p=EXP/'queue.lock'
 if not p.exists():return False
 with p.open('rb') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  fcntl.flock(f,fcntl.LOCK_UN);return False

def workers():
 r=subprocess.run(['ps','-axo','pid,ppid,command'],capture_output=True,text=True,check=True)
 out=[]
 for line in r.stdout.splitlines()[1:]:
  parts=line.split(None,2)
  if len(parts)!=3:continue
  cmd=parts[2]
  if str(EXP) in cmd and 'fullrun14a_worker_v2.py' in cmd and '--job-id ' in cmd:
   job=cmd.split('--job-id ',1)[1].split()[0];out.append(dict(pid=int(parts[0]),job_id=job))
 return out

def snapshot():
 saved={};failures=[]
 for s in PLAN:
  for p in sorted((EXP/'runs'/s['job_id']).glob('attempt_*/COMPLETE.json')):
   try:r=json.loads(p.read_text())
   except (OSError,ValueError):continue
   if r.get('validated'):saved[s['job_id']]=r;break
  if s['job_id'] not in saved:
   failures.extend(str(p.relative_to(ROOT)) for p in (EXP/'runs'/s['job_id']).glob('attempt_*/FAILED.json'))
 active=workers();v=EXP/'FINAL_VALIDATION.json';final=json.loads(v.read_text()) if v.exists() else None
 return dict(singapore_time=datetime.now(SGT).isoformat(timespec='seconds'),saved_complete_jobs=len(saved),total=len(PLAN),completion_counts_by_seed={str(seed):sum(r['seed']==seed for r in saved.values()) for seed in [42,0,1,21,3407]},queue_running=queue_locked(),active_workers=active,unfinished_failure_records=failures,final_validation=final,session=str(EXP))

def print_status():
 d=snapshot();print(f"[{d['singapore_time']}] B3 已保存 {d['saved_complete_jobs']}/{d['total']}；队列运行={d['queue_running']}；训练进程={len(d['active_workers'])}",flush=True)
 for w in d['active_workers']:print(f"  PID {w['pid']}: {w['job_id']}",flush=True)
 if d['unfinished_failure_records']:print('待处理失败记录：'+str(d['unfinished_failure_records']),flush=True)
 if d['final_validation']:print('最终报告：'+d['final_validation']['report'],flush=True)
 return d

def final_present():
 p=EXP/'FINAL_VALIDATION.json'
 if not p.exists():return False
 d=json.loads(p.read_text());assert d['cells']==180 and d['all_cells_verified'];assert sha(ROOT/d['report'])==d['report_sha256']
 return True

def verify():
 # Imports only stdlib queue contracts; no Torch/model inference or training.
 sys.path.insert(0,str(EXP/'scripts'))
 from b3_contracts import completed
 from b3_io import jobs
 found=[];missing=[]
 for spec in jobs():
  path,r=completed(spec)
  (found if r else missing).append(spec['job_id'])
 result=dict(singapore_time=datetime.now(SGT).isoformat(),verified_complete=len(found),total=180,missing=missing,final_report_verified=final_present())
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
 return 0 if len(found)==180 and result['final_report_verified'] else 2

def run():
 assert PYTHON.exists() and QUEUE.exists()
 preflight=json.loads((EXP/'preflight.json').read_text());assert preflight['passed'] and preflight['strict_policy_checks_passed']
 assert preflight['causal_reference_policy']['policy']=='completed_week_close'
 if final_present():print_status();print('B3 已完成，无需重跑。',flush=True);return 0
 lockpath=HERE/'selfrun.lock'
 # The lock is immutable after exclusive creation; subsequent opens are read-only.
 try:
  with lockpath.open('x') as f:f.write('B3 user-launcher duplicate guard.\n')
 except FileExistsError:pass
 with lockpath.open('rb') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print('已有手动启动器在监控；请运行 status 查看。',flush=True);return 0
  if queue_locked():print_status();print('现有 B3 队列仍在运行，未启动第二份。',flush=True);return 0
  orphaned=workers()
  if orphaned:
   print_status();print('调度器已退出，但仍有训练进程在运行。先等待这些单元结束，再执行 run；本工具不会终止它们。',flush=True);return 3
  for a in preflight['dependencies']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
  env=dict(os.environ)
  for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env.pop(k,None)
  env.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
  stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
  folder=HERE/'launches'/stamp;folder.mkdir(parents=True,exist_ok=False)
  logfile=folder/'queue.log'
  with logfile.open('x') as log:
   process=subprocess.Popen([str(PYTHON),'-B',str(QUEUE)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  with (folder/'launch.json').open('x') as f:json.dump(dict(started_singapore=datetime.now(SGT).isoformat(),pid=process.pid,log=str(logfile),command=[str(PYTHON),'-B',str(QUEUE)],profile='CPU12/12 x3; OMP unset; dynamic controls FALSE; original trainers and original14a scheduler',detached_process_session=True),f,indent=2)
  print(f'已启动/续跑 B3，PID={process.pid}\n日志：{logfile}\n每60秒显示状态；Ctrl+C 仅退出此状态显示，独立进程中的队列继续运行。',flush=True)
  try:
   while process.poll() is None:
    print_status()
    try:process.wait(timeout=60)
    except subprocess.TimeoutExpired:pass
  except KeyboardInterrupt:
   print('\n已退出状态显示，未发送停止信号给队列。可用 status 查看。',flush=True);return 0
  print_status()
  if process.returncode==0 and final_present():print('B3 完成，报告和完成提示音由原队列生成。运行 verify 后把结果告诉 Codex。',flush=True);return 0
  print(f'队列退出码={process.returncode}。保留日志并发给 Codex；不要删除或覆盖已有产物。',flush=True);return process.returncode or 1

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['status','run','verify']);a=p.parse_args()
 if a.action=='status':print_status();return 0
 if a.action=='verify':return verify()
 return run()
if __name__=='__main__':sys.exit(main())
