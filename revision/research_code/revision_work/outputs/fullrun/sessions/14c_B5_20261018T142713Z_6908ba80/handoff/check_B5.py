#!/usr/bin/env python3
"""Lightweight B3/B5 status; no Torch imports or training."""
import sys,json,fcntl,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
sys.dont_write_bytecode=True
EXP=Path(__file__).resolve().parents[1]
CONFIG=json.loads((EXP/'experiment.json').read_text());ROOT=EXP.parents[4];B3=ROOT/CONFIG['wait_for_b3']
def lock_held(path):
 if not path.exists():return False
 with path.open('rb') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  fcntl.flock(f,fcntl.LOCK_UN);return False
def done(folder):
 count=0
 if not folder.exists():return 0
 for run in folder.iterdir():
  if any(json.loads(p.read_text()).get('validated',False) for p in run.glob('attempt_*/COMPLETE.json')):count+=1
 return count
print(datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'))
print(f'B3: {done(B3/"runs")}/180; 队列运行={lock_held(B3/"queue.lock")}')
print(f'B3 验收通过={(EXP/"B3_ACCEPTANCE.json").exists()}')
pilot=CONFIG['pilot']['job_id'];nd=done(EXP/'runs');pilotdone=done(EXP/'runs'/pilot) if False else any((EXP/'runs'/pilot).glob('attempt_*/COMPLETE.json'))
print(f'B5: {nd-int(pilotdone)}/72; seed42检查完成={bool(pilotdone)}; supervisor运行={lock_held(EXP/"supervisor.lock")}; 训练队列运行={lock_held(EXP/"queue.lock")}')
if (EXP/'FINAL_VALIDATION.json').exists():print('最终报告：'+json.loads((EXP/'FINAL_VALIDATION.json').read_text())['report'])
elif not (EXP/'B3_ACCEPTANCE.json').exists():print('阶段：等待 B3 完成／验收；尚未开始 B5 训练。')
failures=list((EXP/'control').glob('*/FAILED.json'))
for f in failures:print('历史失败记录：'+str(f))
ps=subprocess.run(['ps','-axo','pid,command'],capture_output=True,text=True,check=True).stdout
for line in ps.splitlines():
 if str(EXP) in line and 'fullrun14a_worker_v2.py' in line and '--job-id ' in line:print('活动 worker：'+line.strip())
