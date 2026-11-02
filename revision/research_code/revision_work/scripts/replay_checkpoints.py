"""Inference only, using original model factories, normalization/loaders and batched helpers."""
import os,sys,time,argparse
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='3'
from step3_common import *
from torch.utils.data import DataLoader
from scripts.python import panel_pathformer_main_v2 as formal
from scripts.python import panel_pathformer_experiment3 as exp3
from scripts.python import panel_pathformer_experiment4_multiseed as exp4

def cells_existing():
 s=pd.read_csv(E3/'experiment3_single_selection.csv');e3=pd.read_csv(E3/'experiment3_training_summary.csv');e4=pd.read_csv(E4/'experiment4_training_summary.csv');cells=[]
 for h in [5,10,20]:
  chosen=s[s.horizon.map(horizon)==h].iloc[0]
  for mechanism in ['single','fixed','static']:
   frame=e3[(e3.mechanism==mechanism)&(e3.horizon.map(horizon)==h)]
   dp=wp=None
   if mechanism=='single':
    dp=int(chosen.selected_daily_patch);wp=int(chosen.selected_weekly_patch)
    frame=frame[(frame.daily_patch==dp)&(frame.weekly_patch==wp)]
   row=frame.iloc[0]
   cells.append(dict(mechanism=mechanism,horizon=h,seed=42,daily_patch=dp,weekly_patch=wp,checkpoint=ROOT/row.checkpoint_path,expected_sha256=row.checkpoint_sha256,helper='experiment3',source_predictions=E3/'experiment3_test_predictions.csv'))
 for _,row in e4.iterrows():
  cells.append(dict(mechanism=row.mechanism,horizon=horizon(row.horizon),seed=int(row.seed),daily_patch=None,weekly_patch=None,checkpoint=ROOT/row.checkpoint_path,expected_sha256=row.checkpoint_sha256,helper='experiment4',source_predictions=E4/'experiment4_test_predictions.csv'))
 assert len(cells)==33
 return cells

def cells_recovered():
 cells=[];excluded=[]
 for h in [5,10,20]:
  path=OUT/'task_a'/f'recovery_h{h}.json'
  if not path.is_file():excluded.append(dict(horizon=h,reason='recovery result NOT FOUND'));continue
  m=json.loads(path.read_text())
  if m['classification'] not in ['EXACT','NEAR']:
   excluded.append(dict(horizon=h,reason='recovery classification '+m['classification']));continue
  cells.append(dict(mechanism='adaptive',horizon=h,seed=42,daily_patch=None,weekly_patch=None,checkpoint=ROOT/m['checkpoint'],expected_sha256=m['checkpoint_sha256'],helper='experiment4',source_predictions=MAIN/'panel_pathformer_test_predictions.csv'))
 return cells,excluded

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['existing','recovered'],required=True);args=p.parse_args();environment=threads(3)
 exp3._verify_frozen_dependencies();exp4._verify_frozen_dependencies()
 cells=cells_existing() if args.stage=='existing' else cells_recovered()[0]
 if args.stage=='recovered':write_json(OUT/'task_b_recovered_exclusions.json',cells_recovered()[1])
 ds=formal.load_frozen_panel();formal._formal_check_dataset_contract(ds)
 rows=[];files=[];start=time.perf_counter()
 print('START',utc(),args.stage,'cells',len(cells),environment,flush=True)
 for h in [5,10,20]:
  group=[c for c in cells if c['horizon']==h]
  if not group:continue
  train,val,test=formal._build_training_loaders('daily_weekly',h,ds,batch_size=formal.BATCH_SIZE)
  if train.generator.initial_seed()!=42:raise RuntimeError('Original loader generator changed')
  loaders={'train':DataLoader(train.dataset,batch_size=formal.BATCH_SIZE,shuffle=False),'val':val,'test':test}
  for c in group:
   elapsed=time.perf_counter();mechanism=c['mechanism'];seed=c['seed'];checkpoint=c['checkpoint']
   if sha(checkpoint)!=c['expected_sha256']:raise RuntimeError(f'Checkpoint hash mismatch {checkpoint}')
   model=exp3._make_experiment3_model(mechanism,c['daily_patch'],c['weekly_patch']) if c['helper']=='experiment3' else exp4._make_experiment4_model(mechanism)
   model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.to(formal._resolve_device());model.eval()
   for split,loader in loaders.items():
    mask=ds.split==split;n=COUNTS[split]
    if len(loader.dataset)!=n:raise RuntimeError(f'Row count mismatch {split}')
    if c['helper']=='experiment3':
     prediction,dw,ww=exp3._batched_inference(model,loader,mechanism,c['daily_patch'],c['weekly_patch'],n,formal._resolve_device())
    else:prediction,dw,ww=exp4._batched_inference(model,loader,n,formal._resolve_device())
    frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=split,y_true=ds.y[h][mask],y_pred=prediction))
    if mechanism in ['static','adaptive']:
     for i,col in enumerate(WEIGHTS):frame[col]=(dw if i<4 else ww)[:,i%4]
     if not np.allclose(dw.sum(1),1,rtol=0,atol=1e-6) or not np.allclose(ww.sum(1),1,rtol=0,atol=1e-6):raise ValueError('Invalid branch weight simplex')
     if mechanism=='static' and (np.max(np.ptp(dw,axis=0))>1e-7 or np.max(np.ptp(ww,axis=0))>1e-7):raise ValueError('Static weights vary by sample')
    if len(frame)!=n or frame.duplicated(['ticker','anchor_date']).any() or not np.isfinite(frame[['y_true','y_pred']]).all().all():raise ValueError('Invalid prediction table')
    path=OUT/'preds'/f'{mechanism}_dw_h{h}_s{seed}_{split}.parquet';write_parquet(path,frame)
    # Round-trip the serialized artifact, not merely the in-memory table.
    checked=pd.read_parquet(path)
    if len(checked)!=n or list(checked.columns)!=list(frame.columns) or not checked.equals(frame):raise ValueError(f'Parquet round-trip mismatch {path}')
    files.append(dict(mechanism=mechanism,input_config='daily_weekly',horizon=h,seed=seed,split=split,n_rows=n,n_dates=frame.anchor_date.nunique(),n_tickers=frame.ticker.nunique(),path=rel(path),sha256=sha(path),checkpoint_path=rel(checkpoint),checkpoint_sha256=sha(checkpoint),columns='|'.join(frame.columns)))
    if split=='test':
     old=formal_subset(c['source_predictions'],mechanism,h,seed)
     result=compare(checked,old)
     rows.append(dict(mechanism=mechanism,input_config='daily_weekly',horizon=h,seed=seed,checkpoint_path=rel(checkpoint),source_predictions=rel(c['source_predictions']),**result))
   rows[-1]['wall_clock_seconds']=time.perf_counter()-elapsed
   print('REPLAY',utc(),mechanism,h,seed,'max_delta',rows[-1]['max_abs_delta_pred'],'MSE_delta',rows[-1]['test_mse_difference'],'seconds',rows[-1]['wall_clock_seconds'],flush=True)
  del train,val,test,loaders
 write_csv(OUT/f'task_b_replay_{args.stage}.csv',pd.DataFrame(rows) if rows else pd.DataFrame(columns=['mechanism','input_config','horizon','seed','checkpoint_path','source_predictions','n_rows','max_abs_delta_pred','replayed_test_mse','formal_test_mse','test_mse_difference','classification','wall_clock_seconds']))
 write_csv(OUT/f'task_b_files_{args.stage}.csv',pd.DataFrame(files) if files else pd.DataFrame(columns=['mechanism','input_config','horizon','seed','split','n_rows','n_dates','n_tickers','path','sha256','checkpoint_path','checkpoint_sha256','columns']))
 write_json(OUT/f'task_b_runtime_{args.stage}.json',dict(environment=environment,stage=args.stage,n_cells=len(rows),n_files=len(files),wall_clock_seconds=time.perf_counter()-start,completed_utc=utc()))
 print('COMPLETE',args.stage,'cells',len(rows),'files',len(files),flush=True)
if __name__=='__main__':main()
