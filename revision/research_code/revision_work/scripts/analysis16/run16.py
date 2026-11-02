"""Reproducible resume entrypoint. Executes no model training."""
import sys,os,json,csv,hashlib,subprocess,uuid,datetime,argparse
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'revision_work/outputs/analysis16';HERE=Path(__file__).parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def guard():
 p=OUT/'registry.csv'
 if p.exists():
  with p.open() as f:rows=list(csv.DictReader(f))
  for src,digest in {(r['source'],r['sha256']) for r in rows}:assert sha(ROOT/src)==digest,src

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 OUT.mkdir(exist_ok=True,parents=True);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8];logdir=OUT/'logs'/stamp;logdir.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');events=[]
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
 try:
  for file in ['registry16.py','interventions16.py','statistics16_v2.py','ensembles16.py','portfolio16_v2.py','intervention_stats16_v2.py','reconcile16_v2.py','verify16_v2.py','report16_v2.py']:
   guard();start=datetime.datetime.now(datetime.timezone.utc).isoformat();print('STEP16',file,start,flush=True)
   with (logdir/(file+'.log')).open('xb') as log:
    result=subprocess.run([sys.executable,'-B',str(HERE/file)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   events.append(dict(script=file,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=result.returncode,log=str((logdir/(file+'.log')).relative_to(ROOT))))
   if result.returncode:raise RuntimeError(f'{file} failed; see {logdir}')
  guard();subprocess.run(['/usr/bin/afplay','/System/Library/Sounds/Glass.aiff'],check=False);print('STEP16 FINISHED',OUT/'REPORT_STEP16.md',flush=True)
 finally:
  with (logdir/'events.json').open('x') as f:json.dump(events,f,indent=2)
  awake.terminate();awake.wait()
if __name__=='__main__':main()
