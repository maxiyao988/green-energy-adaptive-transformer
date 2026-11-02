"""Recorded histories, optimizer settings, TRAIN scales and protocol hashes. No Torch/training import."""
from review24_common import *
import ast
R=ROOT/'revision_work/outputs';checks=[];runs=[];groups=[]
def ck(name,yes,**meta):checks.append(dict(check=name,passed=bool(yes),**meta))
for step in ['step22a','step22b','step22b2','step22d/session_v3','step22e','step22f','step22g']:
 folder=R/step;j=js(folder/'RNG_MANIFEST.json');protocol=next(folder.glob('PROTOCOL*.md'));ck('Immutable protocol hash',hashlib.sha256(read(protocol)).hexdigest()==j['protocol_sha256'],step=step)
 for p in sorted((folder/'fits').glob('**/COMPLETE.json')):
  rec=js(p);status=rec.get('status');job=rec.get('job',{});h=job.get('h',rec.get('h'));seed=job.get('seed',rec.get('seed'));histpath=p.parent/'history.csv'
  if not histpath.exists():continue
  hist=df(histpath);best=int(rec.get('best_epoch',0));n=int(rec.get('epochs',len(hist)));profile=rec.get('profile',{});sp=rec.get('start_profile',profile)
  intra=sp.get('torch_num_threads',sp.get('intra',sp.get('num_threads')));inter=sp.get('torch_num_interop_threads',sp.get('inter',sp.get('interop_threads')))
  if isinstance(sp.get('profile'),dict):sp=sp['profile'];intra=sp.get('intra',sp.get('torch_num_threads'));inter=sp.get('inter',sp.get('torch_num_interop_threads'))
  if intra is not None:ck('Recorded CPU intra/inter12',intra==12 and inter==12,step=step,run=str(p.relative_to(ROOT)))
  if best and len(hist) and status=='COMPLETE':
   # Trainer accepts >1e-12 improvement, as archived; ignored tiny plateaus are permitted.
   chosen=float(hist.loc[hist.epoch.eq(best),'val_loss'].iloc[0]);minimum=float(hist.val_loss.min());ck('Selected epoch has minimum validation training loss',chosen<=minimum+1e-12,step=step,run=str(p.relative_to(ROOT)),difference=chosen-minimum)
   last=int(hist.epoch.iloc[-1]);limit=50 if step=='step22a' else100
   ck('Patience10 unless max epoch reached',last-best==10 or last==limit,step=step,run=str(p.relative_to(ROOT)),best=best,last=last)
  if 'lr' in hist and len(hist):
   vals=hist.lr.unique().tolist();delta=hist.huber_delta.unique().tolist() if 'huber_delta' in hist else [];setting=job.get('setting',rec.get('setting'))
   if setting in ['T0','T1','T2','T3']:
    lr=3e-5 if setting in ['T1','T3'] else1e-4;ck('Recorded effective LR '+setting,all(v==lr for v in vals),step=step,run=str(p.relative_to(ROOT)))
  if 'frozen_unchanged' in rec:ck('Stored frozen hash unchanged',rec['frozen_unchanged'],step=step,run=str(p.relative_to(ROOT)))
  runs.append(dict(step=step,record=str(p.relative_to(ROOT)),status=status,model=job.get('model',rec.get('variant')),setting=job.get('setting',rec.get('setting')),h=h,seed=seed,best_epoch=best,epochs=n,profile_intra=intra,profile_inter=inter))
# Stored actual Adam groups for R10/R100: inspect every saved start/end group.
for p in sorted((R/'step22f/fits').glob('**/OPTIMIZER_GROUPS*.json')):
 j=js(p);g=j['groups'];job=j['job'];fast=g[1];base=g[0];ok=base['options']['lr']==1e-4 and fast['options']['lr']==1e-4*job['multiplier'];ck('Stored Adam parameter groups apply R multiplier',ok,record=str(p.relative_to(ROOT)))
 ck('Stored Adam groups disjoint/exhaustive',set(base['parameter_names']).isdisjoint(fast['parameter_names']) and base['parameter_count']==107777 and fast['parameter_count']==(8 if job['model']=='static' else8520),record=str(p.relative_to(ROOT)))
 groups.append(dict(path=str(p.relative_to(ROOT)),variant=job['variant'],h=job['h'],seed=job['seed'],base_lr=base['options']['lr'],accelerated_lr=fast['options']['lr'],fast_parameters=fast['parameter_count']))
# Independent TRAIN-only sigma calculations (same target byte values, float64 and ddof1).
B2=R/'fullrun/sessions/14c_B2_20261017T101438Z_1a8d5041';sg=js(R/'step22g/SIGMAS_CS.json');sd=js(R/'step22d/session_v3/SIGMAS.json');scale=[]
for w in ['W1','W2','W3']:
 meta=df(B2/'tensors'/w/'sample_index.csv');mask=meta.split.eq('train').to_numpy()
 for h in [5,10,20]:
  p=B2/'tensors'/w/f'y_h{h}.npy';read(p);y=np.load(p).astype(float)
  if w!='W3':
   val=float(y[mask].std(ddof=1));old=float(sd['values'][w][str(h)]);ck('TRAIN sigma ddof1 rolling',abs(val-old)<=1e-10,window=w,h=h,delta=val-old);scale.append(dict(window=w,h=h,target='raw',independent_sigma=val,stored_sigma=old))
  else:
   f=pd.DataFrame(dict(date=meta.anchor_date,y=y));demeaned=f.y-f.groupby('date').y.transform('mean');val=float(demeaned.to_numpy()[mask].std(ddof=1));old=float(sg['values'][str(h)]);ck('TRAIN sigma_cs ddof1 after date demeaning',abs(val-old)<=1e-10,window=w,h=h,delta=val-old);scale.append(dict(window=w,h=h,target='cs',independent_sigma=val,stored_sigma=old))
# Evidence for best-state restoration/eval and the exact DataLoader seed in formal paths.
code=[]
for p in ['scripts/python/panel_pathformer_main_v2.py','scripts/python/panel_pathformer_experiment3.py','scripts/python/panel_pathformer_experiment4_multiseed.py','scripts/python/panel_baseline_lstm.py']:
 lines=read(p).decode().splitlines()
 for i,line in enumerate(lines):
  if any(x in line for x in ['best_state =','load_state_dict(best_state)','model.eval()','manual_seed(42)','manual_seed(SEED)','clip_grad_norm_', 'generator=g','HuberLoss','optimizer = torch.optim.Adam']):code.append(dict(path=p,line=i+1,source=line.strip()))
write('TRAINING_HISTORY_CHECKS.csv',pd.DataFrame(checks));write('TRAINING_RECORDS_REVIEWED.csv',pd.DataFrame(runs));write('ADAM_GROUPS_REVIEWED.csv',pd.DataFrame(groups));write('TRAIN_ONLY_SIGMA_CHECKS.csv',pd.DataFrame(scale));write('FORMAL_TRAINING_LINE_EVIDENCE.csv',pd.DataFrame(code));source_manifest('SOURCES_TRAINING.csv');print('TRAINING records',len(runs),'checks',len(checks),'failures',sum(not c['passed'] for c in checks));print([c for c in checks if not c['passed']][:10])
