"""Read-only, stdlib-only workflow status; no Torch import or inference."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'revision_work/outputs/causal_features';R=ROOT/'revision_work/outputs/repro_threads'
def main():
 total=0;active=[]
 for stage in ['stage3b','stage3c']:
  done=list((BASE/f'{stage}/runs').glob('*.json'));total+=len(done)
  for log in sorted((BASE/'stage3bc_queue').glob(stage+'_*.log')):
   m=re.fullmatch(r'(stage3[bc])_(static|adaptive)_h(5|10|20)_s(0|1|21|42|3407)',log.stem)
   if not m:continue
   _,mech,h,seed=m.groups();meta=BASE/f'{stage}/runs/{mech}_dw_h{h}_s{seed}.json'
   if meta.exists():continue
   text=log.read_text();phases=[line for line in text.splitlines() if line.startswith('PHASE ')]
   if 'Traceback' in text:active.append('ERROR '+log.name)
   elif phases:active.append(phases[-1])
   else:active.append('STARTING '+log.name)
 print('Stage3b/c completed',total,'/30')
 print('Active:', '; '.join(' '.join(line.split()[2:]) if line.startswith('PHASE ') else line for line in active))
 for log in sorted((R/'logs').glob('*_t12.log')):
  meta=R/f'runs/{log.stem}.json'
  if meta.exists():
   d=json.loads(meta.read_text());print('T12 DONE',d['mechanism'],d['horizon'],d['classification'])
  else:
   lines=[line for line in log.read_text().splitlines() if line.startswith('PHASE ')];print('T12 ACTIVE',lines[-1] if lines else log.name)
 print('Queue complete ledger exists:',(BASE/'stage3bc_queue/processes.json').exists())
if __name__=='__main__':main()
