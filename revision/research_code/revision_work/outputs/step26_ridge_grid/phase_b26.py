"""Frozen original run_one selection prefix; no TEST deserialization."""
import os,sys,time,json,csv,ast,copy,hashlib,pickle,traceback,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import torch
torch.set_num_threads(12);torch.set_num_interop_threads(12)
import numpy as np,pandas as pd,scipy,sklearn
from threadpoolctl import threadpool_info
from scripts.python import panel_common as common,panel_baseline_ridge as ridge
B2=ROOT/'revision_work/outputs/fullrun/sessions/14c_B2_20261017T101438Z_1a8d5041'
HS=[5,10,20];READS=[];REPRO=[];SELECT=[];TIMES=[]

def js(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2,default=str,allow_nan=False)
def csvout(name,data):
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def check_protocol():
 assert sha(OUT/'PROTOCOL_STEP26.md')==json.loads((OUT/'PROTOCOL_SHA256.json').read_text())['sha256']
def install_guard():
 tmp=OUT/'tmp_phase_b';tmp.mkdir();tempfile.tempdir=str(tmp)
 # All empirical CSV/parquet reads must be known TRAIN/VAL files or index metadata.
 inv=list(csv.DictReader((OUT/'INVENTORY_STEP26.csv').open()))
 allow={str((ROOT/r[k]).resolve()) for r in inv for k in ['train','val','history','alpha_source'] if r[k]}
 def allowed(p):
  return p.is_relative_to(OUT) or str(p) in allow or (p.name=='sample_index.csv' and p.is_relative_to(B2/'tensors'))
 def audit(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2] if len(args)>2 else 0
  writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing:
   if not p.is_relative_to(OUT):raise RuntimeError('Source write forbidden '+str(p))
   if not (flags & os.O_EXCL):raise RuntimeError('Nonexclusive file creation '+str(p))
  elif p.suffix.lower() in ['.csv','.parquet','.npz'] and p.is_relative_to(ROOT) and not allowed(p):
   raise RuntimeError('TEST locked / non-allowlisted empirical read '+str(p))
 sys.addaudithook(audit)
 for name in ['read_csv','read_parquet']:
  orig=getattr(pd,name)
  def wrapped(p,*a,_orig=orig,**kw):
   if not allowed(Path(p).resolve()):raise RuntimeError('TEST deserializer locked '+str(p))
   READS.append(dict(path=str(Path(p).resolve().relative_to(ROOT)),kind=_orig.__name__,TEST_lock='LOCKED'))
   return _orig(p,*a,**kw)
  setattr(pd,name,wrapped)
 orig=np.load
 def loadguard(path,*a,**kw):
  p=Path(path).resolve()
  if p.suffix=='.npy' and not p.is_relative_to(OUT):
   assert sys._getframe(1).f_code is selected.__code__ and kw.get('mmap_mode')=='r','Only TRAIN/VAL selector can map frozen pack'
  return orig(path,*a,**kw)
 np.load=loadguard
 return inv

def selected(path,keep):
 mapped=np.load(path,mmap_mode='r',allow_pickle=False)
 assert mapped.dtype==np.float32
 a=mapped[keep].copy()
 READS.append(dict(path=str(path.relative_to(ROOT)),kind='read-only mmap, TRAIN/VAL slice only',rows=len(a),TEST_lock='LOCKED'))
 return a

def dataset(window,inp,target):
 folder=B2/'tensors'/window;meta=pd.read_csv(folder/'sample_index.csv');keep=meta.split.isin(['train','val']).to_numpy();keys=meta.loc[keep].reset_index(drop=True)
 ds=common.PanelDataset(keys.ticker.to_numpy(),pd.to_datetime(keys.anchor_date).to_numpy(),keys.split.to_numpy(),selected(folder/inp/'X_daily_raw.npy',keep),selected(folder/inp/'X_weekly_raw.npy',keep),{h:selected(folder/f'y_h{h}.npy',keep) for h in HS})
 assert not (ds.split=='test').any()
 if target=='cross_sectional':
  dates,inverse=np.unique(ds.anchor_date,return_inverse=True);assert (np.bincount(inverse)==17).all()
  for h in HS:
   y=ds.y[h].astype(np.float64);z=np.empty_like(y)
   for code in range(len(dates)):
    m=inverse==code;z[m]=y[m]-np.mean(y[m],dtype=np.float64)
   ds.y[h]=z.astype(np.float32)
 return ds

def selection_prefix():
 source=Path(ridge.__file__);tree=ast.parse(source.read_text());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_one'))
 pos=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='test_pred' for t in n.targets))
 original=node.body[:pos];node.body=original+ast.parse('return best_alpha,best_val_mse,best_model,X').body;node.name='selection_only'
 tree=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));code=compile(tree,str(source),'exec')
 provenance=dict(source=str(source.relative_to(ROOT)),sha256=sha(source),selection_prefix_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=original,type_ignores=[]),include_attributes=False).encode()).hexdigest(),modifications='remove suffix beginning test_pred; return selection locals; bind ALPHA_GRID and capture Ridge constructor only')
 with (OUT/'SELECTION_PREFIX_AST.py').open('x') as f:f.write(ast.unparse(tree)+'\n')
 js('CODE_PROVENANCE.json',provenance)
 return code

def fit(code,ds,stats,cfg,h,grid):
 models=[]
 def capture(*a,**kw):
  m=ridge.Ridge(*a,**kw);models.append(m);return m
 env=dict(ridge.__dict__,ALPHA_GRID=grid,Ridge=capture);exec(code,env)
 a,mse,m,X=env['selection_only'](ds,stats,cfg,h)
 trials=[dict(alpha=float(o.alpha),val_mse=float(np.mean((o.predict(X[ds.split=='val'])-ds.y[h][ds.split=='val'])**2))) for o in models]
 return a,mse,m,X,trials

def comparison(r,ds,m,X):
 out=dict(cell_id=r['cell_id'],old_alpha=float(r['old_alpha']),reproduced_alpha=float(m.alpha),alpha_equal=float(r['old_alpha'])==m.alpha,old_val_mse=float(r['old_val_mse']),reproduced_val_mse=float(np.mean((m.predict(X[ds.split=='val'])-ds.y[int(r['h'])][ds.split=='val'])**2)))
 for sp in ['train','val']:
  if not r[sp]:out[sp+'_max_abs_delta']=None;out[sp+'_class']='UNAVAILABLE';continue
  p=ROOT/r[sp];frame=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
  f=frame[['ticker','anchor_date','y_pred','y_true']].copy();f.anchor_date=pd.to_datetime(f.anchor_date)
  mask=ds.split==sp;g=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=ds.anchor_date[mask],pred=m.predict(X[mask]),truth=ds.y[int(r['h'])][mask]))
  both=g.merge(f,on=['ticker','anchor_date'],validate='one_to_one');assert len(both)==len(g)==len(f)
  assert np.array_equal(both.truth.to_numpy(np.float32),both.y_true.to_numpy(np.float32)),'labels differ'
  d=float(np.max(np.abs(both.pred.to_numpy(float)-both.y_pred.to_numpy(float))));out[sp+'_max_abs_delta']=d;out[sp+'_class']='EXACT' if d<1e-7 else 'NEAR' if d<1e-5 else 'DIFFERENT'
 return out

def main():
 assert (OUT/'SNAPSHOT_COMPLETE.json').exists();check_protocol();start=time.perf_counter();inv=install_guard();code=selection_prefix()
 js('ENVIRONMENT_STEP26.json',dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,torch_threads=[torch.get_num_threads(),torch.get_num_interop_threads()],environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC']},threadpools=threadpool_info(),fitting_seeds=None))
 current=None
 for i,r in enumerate(inv):
  t=time.perf_counter();key=r['window'],r['input_set'],r['target']
  if current!=key:ds=dataset(*key);stats=common.fit_train_only_normalizer(ds);current=key
  h=int(r['h']);cfg=r['config'];folder=OUT/'cells'/r['cell_id'];folder.mkdir(parents=True)
  a,mse,m,X,trials=fit(code,ds,stats,cfg,h,[.1,1.,10.,100.]);comp=comparison(r,ds,m,X);REPRO.append(comp)
  csvout(folder.relative_to(OUT)/'original_grid_trials.csv',trials)
  with (folder/'original_model.pkl').open('xb') as f:pickle.dump(m,f,protocol=5)
  if comp['val_class']=='DIFFERENT':raise RuntimeError('BLOCKING reproduction DIFFERENT '+r['cell_id'])
  if not comp['alpha_equal']:raise RuntimeError('BLOCKING original-grid alpha mismatch '+r['cell_id'])
  grid=[10.**k for k in range(-2,7)];alltrials=[];extensions=[]
  a,mse,m,X,trials=fit(code,ds,stats,cfg,h,grid);alltrials.extend(trials)
  # Add only genuinely new candidate models; preserve strict original comparison by ordered trial selection.
  while (a==grid[0] and grid[0]>1e-6) or (a==grid[-1] and grid[-1]<1e12):
   lower=a==grid[0];edge=int(round(np.log10(grid[0] if lower else grid[-1])))
   new=sorted(set(10.**max(-6,min(12,edge+sign)) for sign in ([-2,-1] if lower else [1,2]))-set(grid));assert new
   aa,mm,model,XX,tr=fit(code,ds,stats,cfg,h,new);alltrials.extend(tr);grid=sorted(grid+new);extensions.append(dict(side='LOWER' if lower else 'UPPER',added=new))
   if mm<mse or (mm==mse and aa<a):a,mse,m=aa,mm,model
  interior=grid[0]<a<grid[-1];row={k:r[k] for k in ['cell_id','window','input_set','config','h','target']};row.update(old_alpha=float(r['old_alpha']),selected_alpha=a,val_mse=mse,number_extensions=len(extensions),interior=interior,status='INTERIOR' if interior else 'BOUNDARY_AT_CAP',extension_log=json.dumps(extensions),val_mse_by_alpha=json.dumps(sorted(alltrials,key=lambda x:x['alpha'])))
  SELECT.append(row);csvout(folder.relative_to(OUT)/'extended_grid_trials.csv',sorted(alltrials,key=lambda x:x['alpha']))
  with (folder/'selected_model.pkl').open('xb') as f:pickle.dump(m,f,protocol=5)
  for sp in ['train','val']:
   mask=ds.split==sp;frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),y_true=ds.y[h][mask],y_pred=m.predict(X[mask])))
   with (folder/(sp+'.csv')).open('x') as f:frame.to_csv(f,index=False)
  TIMES.append(dict(cell_id=r['cell_id'],seconds=time.perf_counter()-t,cumulative_seconds=time.perf_counter()-start));js(folder.relative_to(OUT)/'PHASE_B_COMPLETE.json',dict(selection=row,reproduction=comp,timing=TIMES[-1],protocol_sha256=sha(OUT/'PROTOCOL_STEP26.md')))
  print('SELECTED',i+1,len(inv),r['cell_id'],r['old_alpha'],'->',a,comp['val_class'],'seconds',round(TIMES[-1]['seconds'],2),flush=True)
  if i==2:
   projected=(time.perf_counter()-start)/3*len(inv);js('TIMING_GATE.json',dict(first_three=TIMES,projection_seconds=projected,limit_seconds=7200,passed=projected<=7200))
   if projected>7200:raise RuntimeError('BLOCKING projected runtime exceeds two hours')
 csvout('REPRODUCTION_STEP26.csv',REPRO);csvout('TIMING_STEP26.csv',TIMES)
 csvout('SELECTION_LOCK.csv',SELECT)
 js('SELECTION_LOCK_SHA256.json',dict(sha256=sha(OUT/'SELECTION_LOCK.csv'),n_cells=len(SELECT),protocol_sha256=sha(OUT/'PROTOCOL_STEP26.md'),phase_b_seconds=time.perf_counter()-start,TEST_lock='released only after this record',test_reads=0))
 js('PHASE_B_READ_LOG.json',READS)
 print('PHASE_B_LOCK_SEALED',len(SELECT),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  js('PHASE_B_BLOCKING.json',dict(error=repr(e),traceback=traceback.format_exc(),TEST_lock='LOCKED',completed_cells=len(SELECT),reproductions=REPRO,timings=TIMES,read_events=READS))
  if REPRO:csvout('REPRODUCTION_AT_STOP.csv',REPRO)
  print(traceback.format_exc(),flush=True)
