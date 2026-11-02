#!/usr/bin/env python3
"""Read-only B5FIX status; no Torch imports or training."""
import json,fcntl,subprocess,shlex,hashlib
from datetime import datetime,timezone,timedelta
from pathlib import Path
EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[4]
config=json.loads((EXP/'experiment.json').read_text());done=[]
for spec in config['plan']:
 for p in sorted((EXP/'runs'/spec['job_id']).glob('attempt_*/COMPLETE.json')):
  r=json.loads(p.read_text())
  if r.get('validated'):done.append(r);break
print(datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'))
print(f"B5FIX 已保存 {len(done)}/9；已复用 {sum(r['training_reused'] for r in done)}/1；新训练完成 {sum(not r['training_reused'] for r in done)}/8")
for name in ['supervisor.lock','queue.lock']:
 p=EXP/name;held=False
 if p.exists():
  with p.open('rb') as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
   except BlockingIOError:held=True
 print(name+'运行='+str(held))
ps=subprocess.run(['ps','-axo','pid=,command='],capture_output=True,text=True,check=True).stdout
for line in ps.splitlines():
 parts=line.split(None,1)
 if len(parts)!=2:continue
 try:args=shlex.split(parts[1])
 except ValueError:continue
 if '--job-id' in args and any(a.startswith(str(EXP)) and a.endswith('/fullrun14a_worker_v2.py') for a in args):
  job=args[args.index('--job-id')+1];print(f'活动 PID {parts[0]}：{job}')
for p in (EXP/'control').glob('*/FAILED.json'):print('失败记录：'+str(p))
p=EXP/'FINAL_VALIDATION.json'
if p.exists():
 r=json.loads(p.read_text());assert hashlib.sha256((ROOT/r['report']).read_bytes()).hexdigest()==r['report_sha256']
 print('最终报告：'+r['report']);print('历史归档比较 EXACT='+str(r['archive_EXACT'])+'；DIFFERENT='+str(r['archive_DIFFERENT']))
