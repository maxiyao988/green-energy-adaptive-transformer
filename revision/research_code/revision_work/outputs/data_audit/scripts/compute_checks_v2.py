"""Step 13 C/D/F: existing builders and metric paths, no model training/inference."""
import sys,os,time,ast,types,importlib.util,json
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import *
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'));sys.path.insert(0,str(ROOT/'revision_work/scripts/portfolio'));sys.path.insert(0,str(ROOT/'revision_work/scripts/skill'))
import core
import numpy as np,pandas as pd,torch
import engine,a1_benchmarks
from numpy.lib.stride_tricks import sliding_window_view
torch.set_num_threads(1);torch.set_num_interop_threads(1)
# Import the original builder's unmodified function ASTs, excluding plotting/main side effects.
source=ROOT/'revision_work/scripts/causal/build_and_audit.py'
original_ast=ast.parse(source.read_text(),filename=str(source))
builder_module=types.ModuleType('step13_stage3a_builder');builder_module.__dict__.update(core.__dict__);builder_module.__dict__['sliding_window_view']=sliding_window_view
exec(compile(ast.Module(body=[n for n in original_ast.body if isinstance(n,ast.FunctionDef) and n.name in ['load_prices','windows','transform']],type_ignores=[]),str(source),'exec'),builder_module.__dict__)
reads=set()
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if 'revision_work/outputs/fullrun' in str(p):raise RuntimeError('Forbidden fullrun access')
  flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
  if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT):
   if not p.is_relative_to(OUT):raise RuntimeError('Write outside Step 13: '+str(p))
   if not flags & os.O_EXCL:raise RuntimeError('Output must use exclusive creation: '+str(p))
  elif p.is_relative_to(ROOT) and p.is_file() and not p.is_relative_to(OUT):reads.add(p)
sys.addaudithook(audit)
start=utc();clock=time.perf_counter();ds=core.common.load_frozen_panel();stats=core.common.fit_train_only_normalizer(ds)
cds=core.load_causal();cstats=core.common.fit_train_only_normalizer(cds)
assert np.array_equal(ds.ticker,cds.ticker) and np.array_equal(pd.to_datetime(ds.anchor_date).to_numpy(dtype='datetime64[ns]'),pd.to_datetime(cds.anchor_date).to_numpy(dtype='datetime64[ns]')) and np.array_equal(ds.split,cds.split)
for h in [5,10,20]:assert np.array_equal(ds.y[h],cds.y[h])
# Source digests at first use; no existing writer/trainer is called.
input_paths=list(reads)+[source,ROOT/'revision_work/scripts/portfolio/engine.py',ROOT/'revision_work/scripts/skill/a1_benchmarks.py',ROOT/'revision_work/scripts/causal/core.py']
protected={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(input_paths))}
events=pd.read_csv(ROOT/'revision_audit/corporate_action_support/corporate_action_events.csv');events=events[events.action=='splits'].copy()
records=[];coverage=[]
for _,ev in events.iterrows():
 t=ev.ticker;date=ev.Date;old=builder_module.load_prices(t);cal=pd.DatetimeIndex(old.datetime);ei=cal.get_indexer([pd.Timestamp(date)])[0]
 if ei<0:coverage.append(dict(ticker=t,event_date=date,status='NOT FOUND',reason='event date absent from frozen original price file'));continue
 weekly=core.builder._reconstruct_weekly_from_daily(old)
 valid=[]
 for off in range(-6,6):
  ii=ei+off
  if ii<0 or ii>=len(old):continue
  a=old.datetime.iloc[ii];wi=np.searchsorted(weekly.available_date.to_numpy(),np.datetime64(a),side='right')-1
  if ii<89 or wi<25:
   if off>=-5:coverage.append(dict(ticker=t,event_date=date,anchor_date=str(a.date()),session_offset=off,status='NOT FOUND',reason=f'insufficient original window history: daily position={ii}, weekly position={wi}; requires >=89 and >=25'))
  else:valid.append((off,ii,a))
 if not valid:continue
 anchors=np.array([np.datetime64(v[2],'ns') for v in valid]);(xd,xw),_,_,anchor_close,positions=builder_module.windows(old,anchors)
 od=xd.astype(np.float32);ow=xw.astype(np.float32);cd=builder_module.transform(xd,anchor_close).astype(np.float32);cw=builder_module.transform(xw,anchor_close).astype(np.float32)
 temp=lambda d,w:core.common.PanelDataset(ticker=np.array([t]*len(valid)),anchor_date=anchors,split=np.array(['diagnostic']*len(valid)),X_daily=d,X_weekly=w,y={})
 oz=core.common.apply_normalizer(temp(od,ow),stats,'daily_weekly');cz=core.common.apply_normalizer(temp(cd,cw),cstats,'daily_weekly')
 for res,k in [('daily',0),('weekly',1)]:
  for ch,channel in [(3,'Close'),(4,'Volume')]:
   for j,(off,ii,a) in enumerate(valid):
    if off<-5:continue
    rec=dict(ticker=t,event_date=date,vendor_ratio=ev.value,event_type='special/non-cash distribution encoded as split' if t=='BEP' and date=='2020-07-30' else ('reverse split' if ev.value<1 else 'split'),anchor_date=str(a.date()),session_offset=off,resolution=res,channel=channel,original_last=float(oz[k][j,-1,ch]),original_window_mean=float(oz[k][j,:,ch].mean(dtype=np.float64)),causal_last=float(cz[k][j,-1,ch]),causal_window_mean=float(cz[k][j,:,ch].mean(dtype=np.float64)),original_close_anchor=float(old.close.iloc[ii]),original_volume_anchor=float(old.volume.iloc[ii]))
    for family,z in [('original',oz[k]),('causal',cz[k])]:
     for measure in ['last','window_mean']:
      now=z[j,-1,ch] if measure=='last' else z[j,:,ch].mean(dtype=np.float64)
      before=(z[j-1,-1,ch] if measure=='last' else z[j-1,:,ch].mean(dtype=np.float64)) if j>0 and valid[j-1][1]==ii-1 else np.nan
      delta=float(now-before);rec[f'{family}_{measure}_one_session_change']=delta;rec[f'{family}_{measure}_flag_gt_1sd']=bool(abs(delta)>1) if np.isfinite(delta) else False
    for h in [5,10,20]:
     rec[f'label_log_return_h{h}']=float(np.float32(np.log(old.close.iloc[ii+h]/old.close.iloc[ii]))) if ii+h<len(old) else np.nan
    rec['explanation']='No >1 SD flag' if not any(v for k_,v in rec.items() if k_.endswith('flag_gt_1sd')) else ('Event-coincident; nominal/split-normalized volume convention is not established by a jump alone' if channel=='Volume' and off==0 else 'NOT EXPLAINED mechanically by a split in split-adjusted Close; economic/vendor movement or normalization sensitivity remains unverified')
    rec['explained_by_event']='NOT APPLICABLE' if rec['explanation']=='No >1 SD flag' else 'UNVERIFIED'
    records.append(rec)
 coverage.append(dict(ticker=t,event_date=date,status='COMPUTED',n_anchors=sum(v[0]>=-5 for v in valid),first_anchor=str(valid[0][2].date()),last_anchor=str(valid[-1][2].date())))
c=rows('C_event_window_features.csv',records);rows('C_event_window_coverage.csv',coverage)
flags=[cc for cc in c if cc.endswith('flag_gt_1sd')]
rows('C_event_window_flags.csv',c[c[flags].any(axis=1)])
rows('C_event_window_summary.csv',c.groupby(['ticker','event_date','resolution','channel'],as_index=False).agg(n_anchors=('anchor_date','nunique'),original_last_max_change=('original_last_one_session_change',lambda x:np.abs(x).max()),original_mean_max_change=('original_window_mean_one_session_change',lambda x:np.abs(x).max()),causal_last_max_change=('causal_last_one_session_change',lambda x:np.abs(x).max()),causal_mean_max_change=('causal_window_mean_one_session_change',lambda x:np.abs(x).max()),n_original_last_flags=('original_last_flag_gt_1sd','sum'),n_causal_last_flags=('causal_last_flag_gt_1sd','sum'),n_original_mean_flags=('original_window_mean_flag_gt_1sd','sum'),n_causal_mean_flags=('causal_window_mean_flag_gt_1sd','sum')))
print('C complete',len(c),len(c[c[flags].any(axis=1)]),flush=True)
# Step 8 decisions, including the separate risk-parity run: unique split/h/date cells.
rebs=[]
for run in ['original','original_rp']:
 p=ROOT/f'revision_work/outputs/portfolio/{run}/rebalances.csv'
 x=pd.read_csv(p,usecols=['split','h','decision_date']).drop_duplicates();x['source_run']=run;rebs.append(x)
dec=pd.concat(rebs).groupby(['split','h','decision_date'],as_index=False).source_run.agg(lambda x:','.join(sorted(set(x))))
rows('D_decision_dates.csv',dec)
C,R=engine.load_prices();allC=pd.DataFrame({t:pd.read_csv(ROOT/f'dataset/finance/panel_raw/{t}_daily.csv').assign(date=lambda x:pd.to_datetime(x.datetime).dt.strftime('%Y-%m-%d')).set_index('date').close for t in engine.TICKERS}).dropna().sort_index()
P=allC[engine.TICKERS].to_numpy(np.float64);all_dates=list(allC.index);positions={d:i for i,d in enumerate(all_dates)}
factors=pd.read_csv(ROOT/'revision_work/outputs/causal_features/adjustment_factors_by_date.csv');factors=factors.pivot(index='Date',columns='ticker',values='c')[engine.TICKERS]
anchor_factors=pd.read_csv(ROOT/'revision_work/outputs/causal_features/stage3c/asof_anchor_factors.csv');anchor_factors=anchor_factors.pivot(index='anchor_date',columns='ticker',values='c')[engine.TICKERS]
for d in dec.decision_date:assert np.array_equal(factors.loc[d].to_numpy(),anchor_factors.loc[d].to_numpy())
# Calls the existing signal helper for original signals; unchanged SIGNALS lambdas evaluate rebased inputs.
keys=pd.DataFrame([(t,d) for d in sorted(set(dec.decision_date)) for t in engine.TICKERS],columns=['ticker','anchor_date']);formal_sig=a1_benchmarks.signal_table(keys).set_index(['ticker','anchor_date'])
rowdiff=[];covs=[];returns=[];summ=[]
def compare(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);dd=np.abs(a-b);nz=a!=0
 return dict(max_abs_difference=float(dd.max()),max_relative_difference=float(np.max(dd[nz]/np.abs(a[nz]))) if nz.any() else 0.,max_relative_difference_floor_1e_12=float(np.max(dd/np.maximum(np.abs(a),1e-12))),n_values=int(a.size),zero_reference_count=int((~nz).sum()),zero_reference_nonzero_difference=int(((~nz)&(dd!=0)).sum()))
for row in dec.itertuples(index=False):
 d=row.decision_date;h=int(row.h);sp=row.split;i=positions[d];fac=factors.loc[d].to_numpy(np.float64);assert np.isfinite(fac).all() and (fac>0).all()
 # A single anchor-specific c_t per ticker is applied to all past prices AND the label endpoints.
 # It is deliberately NOT a time-spliced P_s/c_s series.
 window=P[i-252:i+h+1];A=window/fac[None,:];ai=252
 refR=R.loc[:d].tail(252)[engine.TICKERS].to_numpy(np.float64);asofR=A[1:253]/A[:252]-1
 assert np.array_equal(refR,window[1:253]/window[:252]-1)
 rc=compare(refR,asofR);returns.append(dict(split=sp,h=h,decision_date=d,**rc))
 refCov=engine.LedoitWolf().fit(refR).covariance_*h;newCov=engine.LedoitWolf().fit(asofR).covariance_*h
 summ.append(dict(split=sp,h=h,decision_date=d,quantity='covariance',**compare(refCov,newCov)))
 for j,t in enumerate(engine.TICKERS):
  for k,u in enumerate(engine.TICKERS):covs.append(dict(split=sp,h=h,decision_date=d,ticker=t,other_ticker=u,original=float(refCov[j,k]),asof=float(newCov[j,k]),abs_difference=float(abs(refCov[j,k]-newCov[j,k]))))
 for name,fun in a1_benchmarks.SIGNALS.items():
  original=fun(P,i);asof=fun(A,ai)
  assert np.array_equal(original,np.array([formal_sig.loc[(t,d),name] for t in engine.TICKERS]))
  summ.append(dict(split=sp,h=h,decision_date=d,quantity=name,**compare(original,asof)))
  for j,t in enumerate(engine.TICKERS):rowdiff.append(dict(split=sp,h=h,decision_date=d,ticker=t,quantity=name,original=float(original[j]),asof=float(asof[j]),abs_difference=float(abs(original[j]-asof[j]))))
 for lh in [5,10,20]:
  original=np.log(P[i+lh]/P[i]);future=P[i+lh]/fac;asof=np.log(future/A[ai]);summ.append(dict(split=sp,h=h,decision_date=d,quantity=f'label_h{lh}_float64',**compare(original,asof)));summ.append(dict(split=sp,h=h,decision_date=d,quantity=f'label_h{lh}_float32',**compare(original.astype(np.float32),asof.astype(np.float32))))
  for j,t in enumerate(engine.TICKERS):rowdiff.append(dict(split=sp,h=h,decision_date=d,ticker=t,quantity=f'label_h{lh}_float64',original=float(original[j]),asof=float(asof[j]),abs_difference=float(abs(original[j]-asof[j]))))
rows('D_input_comparisons.csv',rowdiff);rows('D_covariance_entries.csv',covs);rows('D_return_window_comparisons.csv',returns);detail=rows('D_per_decision_summary.csv',summ)
agg=detail.groupby('quantity',as_index=False).agg(max_abs_difference=('max_abs_difference','max'),max_relative_difference=('max_relative_difference','max'),max_relative_difference_floor_1e_12=('max_relative_difference_floor_1e_12','max'),n_values=('n_values','sum'),n_decisions=('decision_date','size'),zero_reference_count=('zero_reference_count','sum'),zero_reference_nonzero_difference=('zero_reference_nonzero_difference','sum'));rows('D_timing_summary.csv',agg)
print('D complete',len(dec),agg[['quantity','max_abs_difference']].to_dict('records'),flush=True)
# Seed provenance: exact quotations only. Absence of a rationale for the particular numbers is explicit.
quotes=[]
for path,first,last,interpretation in [('docs/experiment4_multiseed_protocol.md',77,85,'pre-specified set; no reason for individual numeric values stated'),('docs/experiment4_multiseed_protocol.md',93,103,'seed42 reuse provenance'),('project_progress.md',727,727,'at least five seeds and planned values'),('project_progress.md',4066,4066,'documented motivation for at least five seeds and mean/std')]:
 lines=(ROOT/path).read_text().splitlines()
 for n in range(first,last+1):
  if lines[n-1].strip():quotes.append(dict(path=path,line=n,quote=lines[n-1],scope=interpretation))
rows('F_seed_provenance_quotes.csv',quotes)
js('F_seed_provenance_conclusion.json',dict(search_scope=['docs/','project_progress.md'],rationale_for_particular_values='NOT FOUND',rationale_for_five_seeds='FOUND: project_progress.md:4066',selection_policy='pre-specified: docs/experiment4_multiseed_protocol.md:77-85',seed42_policy='reuse frozen MAIN V2 and Experiment3 evidence'))
for p in sorted(reads):
 key=str(p.relative_to(ROOT))
 if key not in protected:protected[key]=sha(p)
rows('protected_inputs_CD.csv',[dict(path=k,sha256_before=v,sha256_after=sha(ROOT/k),unchanged=v==sha(ROOT/k)) for k,v in sorted(protected.items())])
js('computation_telemetry.json',dict(started_utc=start,ended_utc=utc(),wall_seconds=time.perf_counter()-clock,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']},torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),nice=os.getpriority(os.PRIO_PROCESS,0),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,load_average=os.getloadavg(),training_runs=0,inference_runs=0,stage3a_function_import='exact unmodified AST definitions; plotting/main side effects excluded',source_sha256=sha(source),fullrun_accesses=0))
