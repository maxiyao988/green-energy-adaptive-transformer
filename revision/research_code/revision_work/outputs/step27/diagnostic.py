"""Frozen-label diagnostics and stored-checkpoint inference only; exclusive outputs."""
import os, sys, csv, json, hashlib, inspect, warnings, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import torch
from torch import nn
torch.set_num_threads(12)
torch.set_num_interop_threads(12)
SOURCES = {}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def rel(p):
    p=Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def source(p,purpose,location=None):
    p=Path(p)
    if not p.is_absolute():p=ROOT/p
    key=rel(p)
    if key not in SOURCES:SOURCES[key]=dict(source_file=key,sha256_before=sha(p),bytes=p.stat().st_size,locations=set(),purposes=set())
    if location is None:
        location='lines 1-'+str(len(p.read_text().splitlines())) if p.suffix in ['.py','.csv','.md','.json','.txt'] else 'binary: no text lines; complete stored array/state; retained row mapping in sample_provenance.csv'
    SOURCES[key]['locations'].add(location);SOURCES[key]['purposes'].add(purpose)
    return p
def readcsv(p,purpose):
    with source(p,purpose).open(newline='') as f:return [dict(r,_line=i) for i,r in enumerate(csv.DictReader(f),2)]
def readjson(p,purpose):return json.loads(source(p,purpose).read_text())
def loadarray(p,purpose):return np.load(source(p,purpose),allow_pickle=False)
def fmt(x):return format(x,'.6g') if isinstance(x,(float,np.floating)) else str(x)
def csvout(name,rows):
    assert rows
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:fmt(v) for k,v in r.items()} for r in rows)
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,(str,bytes,os.PathLike)) and isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            p=Path(os.fsdecode(path)).resolve()
            if not p.is_relative_to(OUT) or not flags&os.O_EXCL:raise RuntimeError('READ-ONLY/EXCLUSIVE WRITE GUARD: '+str(p))
sys.addaudithook(guard)
from scripts.python import panel_common as common
from scripts.python import panel_pathformer_main_v2 as formal
code_ranges={
'scripts/python/panel_common.py':[(1,20),(65,130),(134,165)],
'scripts/python/panel_pathformer_main_v2.py':[(65,78),(117,135),(208,250),(410,613),(613,715),(1335,1376),(1641,1735)],
'scripts/python/panel_pathformer_experiment3.py':[(224,249),(542,575),(635,701),(713,733)],
'scripts/python/panel_pathformer_experiment4_multiseed.py':[(380,381),(411,419),(912,980),(991,1014)],
'revision_work/scripts/causal/fullrun14a_worker.py':[(139,154),(225,278)],
'revision_work/scripts/causal/fullrun14a_worker_v2.py':[(139,154),(225,278)],
'revision_work/scripts/causal/core.py':[(45,47)],
'revision_work/scripts/causal/build_and_audit.py':[(145,149)],
'revision_work/outputs/manuscript19/prepare_step19_v2.py':[(99,148),(274,277)]}
for p,ranges in code_ranges.items():
    for a,b in ranges:source(p,'formal data/model/history provenance',f'lines {a}-{b}')
base='revision_work/outputs/causal_features/tensors'
meta_rows=readcsv(base+'/sample_index.csv','authoritative retained samples and partitions')
meta=pd.DataFrame(meta_rows).drop(columns='_line')
assert len(meta)==43571 and meta.ticker.nunique()==17 and meta.anchor_date.nunique()==2563
assert set(meta.ticker)==set(common.FROZEN_TICKERS)
assert not meta.duplicated(['ticker','anchor_date']).any()
assert meta.groupby('anchor_date').size().eq(17).all() and meta.groupby('anchor_date').split.nunique().eq(1).all()
counts={'train':30634,'val':6290,'test':6647}
dates={'train':(1802,'2016-01-29','2023-03-27'),'val':(370,'2023-04-26','2024-10-14'),'test':(391,'2024-11-12','2026-06-05')}
for sp,n in counts.items():
    m=meta[meta.split.eq(sp)];assert len(m)==n and (m.anchor_date.nunique(),m.anchor_date.min(),m.anchor_date.max())==dates[sp]
y32={h:loadarray(f'{base}/y_h{h}_frozen.npy','formal frozen labels') for h in [5,10,20]}
assert all(y.shape==(43571,) and y.dtype==np.float32 and np.isfinite(y).all() for y in y32.values())
original_d=[];original_w=[];sample_prov=[]
for ticker in sorted(meta.ticker.unique()):
    prefix=f'dataset/multiscale_dataset/panel/{ticker}'
    tr=readcsv(prefix+'/meta.csv','original formal anchor alignment')
    lookup={str(pd.Timestamp(r['anchor_date']).date()):(j,r['_line']) for j,r in enumerate(tr)}
    group=meta[meta.ticker.eq(ticker)];ix=np.array([lookup[d][0] for d in group.anchor_date])
    for h in [5,10,20]:
        a=loadarray(f'{prefix}/y_{h}d.npy','bitwise verification of saved formal labels')
        assert np.array_equal(a[ix].astype(np.float32),y32[h][group.index]),(ticker,h)
    for freq,dest in [('daily',original_d),('weekly',original_w)]:
        a=loadarray(f'{prefix}/X_{freq}.npy','original formal features for checkpoint inference');dest.append(a[ix].astype(np.float32))
    for idx,d,j in zip(group.index,group.anchor_date,ix):
        sample_prov.append(dict(sample_index=int(idx),ticker=ticker,anchor_date=d,partition=meta.loc[idx,'split'],frozen_index_file=base+'/sample_index.csv',frozen_index_csv_line=int(idx)+2,frozen_label_array_index=int(idx),original_meta_file=prefix+'/meta.csv',original_meta_csv_line=lookup[d][1],original_array_index=int(j),original_h5_file=prefix+'/y_5d.npy',original_h10_file=prefix+'/y_10d.npy',original_h20_file=prefix+'/y_20d.npy'))
assert meta.ticker.tolist()==sorted(meta.ticker.tolist())
O=common.PanelDataset(meta.ticker.to_numpy(),pd.to_datetime(meta.anchor_date).to_numpy(),meta.split.to_numpy(),np.concatenate(original_d),np.concatenate(original_w),y32)
source(formal.DATASET_MANIFEST_PATH,'formal dataset contract manifest');formal._formal_check_dataset_contract(O)
csvout('sample_provenance.csv',sample_prov)
Y={h:v.astype(np.float64) for h,v in y32.items()};mu={h:Y[h][meta.split.eq('train')].mean() for h in Y}
def huber(e):return np.where(np.abs(e)<=1,0.5*e*e,np.abs(e)-0.5)
def stats(h,sp,mask,year=None):
    y=Y[h][mask];e=y-mu[h];row=dict(horizon=h,partition=sp)
    if year is not None:row['year']=int(year)
    row.update(n=len(y),mean_y=y.mean(),sd_y_ddof0=y.std(ddof=0),mean_y_squared=np.mean(y*y),mu_train=mu[h],mse_train_mean=np.mean(e*e),huber_train_mean=np.mean(huber(e)),mse_zero=np.mean(y*y),huber_zero=np.mean(huber(y)))
    return row
A=[stats(h,sp,meta.split.eq(sp)) for h in [5,10,20] for sp in counts]
AY=[stats(20,sp,meta.split.eq(sp)&meta.anchor_date.str.startswith(yr),yr) for sp in counts for yr in sorted(meta.loc[meta.split.eq(sp),'anchor_date'].str[:4].unique())]
S2=readcsv('revision_work/outputs/manuscript19/S2_table.csv','manuscript S2 sanity check')
expected_train=['0.00322776','0.00657869','0.0130738'];expected_val=['-0.00909077','-0.0179949','-0.036443'];expected_test=['0.00999955','0.0192278','0.0365213'];checks=[]
for i,h in enumerate([5,10,20]):
    row={r['partition']:r for r in A if r['horizon']==h}
    for name,value,expected,field in [('train_mean',mu[h],expected_train[i],'train_mean'),('val_mean',row['val']['mean_y'],expected_val[i],'val_mean'),('test_gtm_mse',row['test']['mse_train_mean'],expected_test[i],'gtm_test_mse')]:
        ok=fmt(value)==expected and S2[i][field]==expected
        checks.append(dict(horizon=h,check=name,actual=fmt(value),expected=expected,status='PASS' if ok else 'FAIL'))
        if not ok:raise RuntimeError(f'BLOCKING SANITY CHECK FAILED: h={h}, {name}, {value!r} != {expected} at six significant digits')
csvout('A_partition_dispersion.csv',A);csvout('A_by_year_h20.csv',AY);csvout('sanity_checks.csv',checks)
print('PART A PASS: 9/9 sanity checks; exact retained keys and frozen labels verified',flush=True)
H=readcsv('revision_work/outputs/manuscript19/learning_history_cells.csv','Step19 history-to-source mapping')
H=[r for r in H if r['horizon']=='20' and r['mechanism'] in ['static','adaptive']]
runs=[];epoch1=[];C=[];summaries={}
for name in ['experiment3_pathformer_ablation/experiment3','experiment4_multiseed/experiment4']:
    summaries[name]=readcsv(f'dataset/audit/{name}_training_summary.csv','original validation-selected checkpoint path/hash')
for inp in ['O','C']:
 for mechanism in ['static','adaptive']:
  for seed in [0,1,21,42,3407]:
    hr=[r for r in H if r['input_set']==inp and r['mechanism']==mechanism and int(r['seed'])==seed]
    assert hr and len({r['source_file'] for r in hr})==1
    f=hr[0]['source_file'];raw=readcsv(f,'full-precision source history');best=int(hr[0]['best_epoch']);selected={}
    for r in hr:
        line=int(r['source_line']);q=raw[line-2];assert int(q['epoch'])==int(r['epoch'])
        tr=float(q.get('train_huber',q['train_loss']));va=float(q.get('val_huber',q['val_loss']))
        assert fmt(tr)==r['train_huber'] and fmt(va)==r['val_huber']
        if int(r['epoch']) in [1,best]:selected[int(r['epoch'])]=(tr,va,line)
    tr,va,line=selected[1]
    epoch1.append(dict(input_set=inp,mechanism=mechanism,horizon=20,seed=seed,epoch=1,train_huber=tr,val_huber=va,train_minus_val=tr-va,source_file=f,source_line=line))
    if inp=='C':
        jp=Path(f).parent/'COMPLETE.json';j=readjson(jp,'14a completion binding exact history and checkpoint')
        assert j['best_epoch']==best and j['mechanism']==mechanism and j['seed']==seed
        checkpoint=j['checkpoint'];expected_sha=j['checkpoint_sha256'];status='STORED';checkpoint_prov=str(jp)+'#/best_epoch,checkpoint,checkpoint_sha256'
        for dep in j['dependencies']:
            dp=source(dep['path'],'14a stored dependency verification');assert sha(dp)==dep['sha256'],dep['path']
    elif mechanism=='adaptive' and seed==42:
        jp='revision_work/outputs/repro_threads/runs/adaptive_dw_h20_s42_t12.json';j=readjson(jp,'previously stored EXACT recovery provenance')
        assert j['classification']=='EXACT' and j['history_exact_equal'] and j['best_epoch']==best and j['test_mse_difference']==0
        checkpoint=j['checkpoint'];expected_sha=j['checkpoint_sha256'];status='STORED_RECOVERED_EXACT';checkpoint_prov=jp+'#/checkpoint,checkpoint_sha256,classification,history_exact_equal'
    else:
        name='experiment3_pathformer_ablation/experiment3' if seed==42 else 'experiment4_multiseed/experiment4'
        sr=[r for r in summaries[name] if r['horizon']=='20d' and r['mechanism']==mechanism and int(r['seed'])==seed]
        assert len(sr)==1 and int(sr[0]['best_epoch'])==best
        checkpoint=sr[0]['checkpoint_path'];expected_sha=sr[0]['checkpoint_sha256'];status='STORED';checkpoint_prov=f'dataset/audit/{name}_training_summary.csv:'+str(sr[0]['_line'])
    cp=ROOT/checkpoint
    if cp.exists():source(cp,'selected stored h20 checkpoint','binary state_dict; '+checkpoint_prov);assert sha(cp)==expected_sha,checkpoint
    else:status='NOT STORED'
    tr,va,line=selected[best]
    runs.append(dict(input_set=inp,mechanism=mechanism,horizon=20,seed=seed,best_epoch=best,checkpoint_status=status,checkpoint_file=checkpoint,checkpoint_sha256=expected_sha,checkpoint_provenance=checkpoint_prov,history_source_file=f,history_source_line_best=line,logged_train_huber_best=tr,logged_val_huber_best=va))
csvout('epoch1_logged_losses.csv',epoch1)
print('CHECKPOINTS: verified',len(runs),'selected cells',flush=True)
Craw=[loadarray(f'{base}/X_{freq}_causal_raw.npy','formal C features for stored-checkpoint inference') for freq in ['daily','weekly']]
Cds=common.PanelDataset(O.ticker,O.anchor_date,O.split,*Craw,y32);formal._formal_check_dataset_contract(Cds);features={}
for inp,ds in [('O',O),('C',Cds)]:
    features[inp]=common.apply_normalizer(ds,common.fit_train_only_normalizer(ds),'daily_weekly')
    if inp=='C':
        for freq,actual in zip(['daily','weekly'],features[inp]):
            saved=loadarray(f'{base}/X_{freq}_causal_standardized.npy','bitwise normalization verification');assert np.array_equal(actual,saved),freq
def forbidden(*a,**k):raise RuntimeError('Training/backward is forbidden in Step27')
torch.Tensor.backward=forbidden;torch.autograd.backward=forbidden
for run in runs:
    row=dict(run)
    if run['checkpoint_status']=='NOT STORED':
        row.update(eval_train_huber='NOT STORED',eval_val_huber='NOT STORED',eval_test_huber='NOT STORED',eval_minus_logged_train='NOT STORED',eval_minus_logged_val='NOT STORED',parameters_unchanged='NOT STORED')
    else:
        start=time.perf_counter();model=formal._make_model('daily_weekly',run['mechanism'])
        state=torch.load(ROOT/run['checkpoint_file'],map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True);model.eval();assert not any(m.training for m in model.modules())
        before={k:v.detach().clone() for k,v in model.state_dict().items()};xd,xw=features[run['input_set']]
        for sp in counts:
            idx=np.flatnonzero(O.split==sp);preds=[]
            with torch.no_grad():
                for off in range(0,len(idx),32):
                    b=idx[off:off+32];pred=model(torch.from_numpy(xd[b]),torch.from_numpy(xw[b]));assert not pred.requires_grad
                    preds.append(pred.numpy().astype(np.float64))
            p=np.concatenate(preds);assert len(p)==counts[sp] and np.isfinite(p).all()
            row['eval_'+sp+'_huber']=np.mean(huber(p-Y[20][idx]))
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
        row['eval_minus_logged_train']=row['eval_train_huber']-run['logged_train_huber_best'];row['eval_minus_logged_val']=row['eval_val_huber']-run['logged_val_huber_best'];row['parameters_unchanged']='PASS'
        assert abs(row['eval_minus_logged_val'])<1e-7,row
        print('EVAL',run['input_set'],run['mechanism'],run['seed'],{s:fmt(row['eval_'+s+'_huber']) for s in counts},'seconds',fmt(time.perf_counter()-start),flush=True)
    C.append(row)
csvout('C_eval_mode_losses.csv',C)
for cls in [nn.HuberLoss,nn.MultiheadAttention,nn.Dropout]:
    p=Path(inspect.getsourcefile(cls));start=inspect.getsourcelines(cls)[1]
    source(p,'installed PyTorch reduction/dropout semantics',f'class {cls.__name__}, starts line {start}')
for module in list(sys.modules.values()):
    p=getattr(module,'__file__',None)
    if p and str(p).startswith(str(ROOT/'scripts')) and Path(p).suffix=='.py':source(p,'imported local formal dependency; definitions only')
manifest=[]
for k,r in sorted(SOURCES.items()):
    p=Path(k) if Path(k).is_absolute() else ROOT/k;after=sha(p);assert after==r['sha256_before'],k
    manifest.append(dict(source_file=k,source_location='; '.join(sorted(r['locations'])),purpose='; '.join(sorted(r['purposes'])),sha256=r['sha256_before'],sha256_after=after,unchanged='PASS',bytes=r['bytes']))
csvout('SOURCE_MANIFEST.csv',manifest)
# Retain full calculation precision for the report builder; presentation CSVs use six significant digits.
with (OUT/'calculation_state.json').open('x') as f:
    json.dump(dict(A=A,AY=AY,C=C,epoch1=epoch1,checks=checks,manifest=manifest,train2020_n=int(np.sum(meta.split.eq('train')&meta.anchor_date.str.startswith('2020'))),train2020_y2_share=float(np.sum(Y[20][meta.split.eq('train')&meta.anchor_date.str.startswith('2020')]**2)/np.sum(Y[20][meta.split.eq('train')]**2))),f,indent=2)
print('COMPLETE DIAGNOSTICS',len(manifest),'inputs; 20 checkpoint evaluations',flush=True)
