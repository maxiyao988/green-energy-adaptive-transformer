"""Independent post-B5FIX artifact/table acceptance; no training or new forward pass."""
import os,sys,json,statistics
from pathlib import Path
EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[4]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
sys.path[:0]=[str(EXP/'scripts'),str(ROOT),str(ROOT/'revision_work/outputs/_deps_pyarrow15')]
import fx_io as io
import torch
torch.set_num_threads(12);torch.set_num_interop_threads(12)
import numpy as np,pandas as pd
from scripts.python import panel_common as common
from scripts.python import panel_baseline_lstm as lstm
from scripts.python import panel_baseline_vanilla_transformer as vanilla
from scripts.python import panel_baseline_swim_cpu as swim

def close(a,b):assert abs(float(a)-float(b))<1e-12,(a,b)
def canonical(f):
 f=f.copy();f['anchor_date']=pd.to_datetime(f.anchor_date).dt.strftime('%Y-%m-%d');return f.sort_values(['ticker','anchor_date']).reset_index(drop=True)
def main():
 final=json.loads((EXP/'FINAL_VALIDATION.json').read_text());assert final['cells']==9 and final['newly_trained']==8 and final['reused']==1
 assert final['worker_launches']==8 and final['all_cells_verified'] and io.sha(ROOT/final['report'])==final['report_sha256']
 ds=common.load_frozen_panel();checks=[];score_map={}
 for spec in io.PLAN:
  cp,r=io.completed(spec);assert r
  model={'lstm':lstm.make_model,'vanilla':vanilla.make_model,'swim':swim.formal.make_model}[spec['mechanism']]('daily_weekly',5,5)
  state=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);model.eval()
  assert all(torch.isfinite(v).all().item() for v in state.values())
  assert sum(p.numel() for p in model.parameters() if p.requires_grad)==r['parameter_count']
  hist=pd.read_csv(cp.parent/'history.csv',float_precision='round_trip');assert len(hist)==r['epochs'] and int(hist.loc[hist.val_loss.idxmin(),'epoch'])==r['best_epoch']
  assert np.isfinite(hist[['train_huber','val_huber','val_mse']]).all().all()
  for split,n in io.COUNTS.items():
   f=pd.read_parquet(cp.parent/f'{split}.parquet');mask=ds.split==split
   assert len(f)==n and (f.split==split).all() and not f.duplicated(['ticker','anchor_date']).any()
   assert np.array_equal(f.ticker.to_numpy(),ds.ticker[mask]) and np.array_equal(pd.to_datetime(f.anchor_date).to_numpy(),pd.to_datetime(ds.anchor_date[mask]).to_numpy())
   assert np.array_equal(f.y_true.to_numpy(),ds.y[spec['horizon']][mask]) and np.isfinite(f.select_dtypes(include='number')).all().all()
   if split=='test':
    m=common.evaluate_predictions(f.y_true.to_numpy(),f.y_pred.to_numpy(),f.ticker.to_numpy(),f.anchor_date.to_numpy())
    score_map[(spec['mechanism'],spec['horizon'])]=(m,canonical(f))
  telemetry=[json.loads(x) for x in (cp.parent/'telemetry.jsonl').read_text().splitlines() if x]
  for t in telemetry:
   assert t['thread_counts']==[12,12] and t['environment']['OMP_NUM_THREADS'] is None
   assert t['environment']['OMP_DYNAMIC']==t['environment']['MKL_DYNAMIC']=='FALSE'
  checks.append(dict(job_id=spec['job_id'],training_reused=r['training_reused'],checkpoint_strict_load=True,all_three_split_keys_labels=True,history_best_epoch=True,profile_12_12=True,same_run_recorded_delta=r['test_replay']['max_abs_delta_pred']))
 primary=pd.read_csv(EXP/'O_uniform12_per_seed.csv',float_precision='round_trip');reference=pd.read_csv(EXP/'O_archived42_reference_per_seed.csv',float_precision='round_trip')
 assert len(primary)==len(reference)==45 and set(primary.profile)=={'CPU12/12'}
 keys=['model','horizon','seed']
 pother=primary[primary.seed!=42].sort_values(keys).reset_index(drop=True)
 rother=reference[reference.seed!=42].sort_values(keys).reset_index(drop=True)
 pd.testing.assert_frame_equal(pother,rother,check_exact=True);assert len(pother)==36
 comparison=pd.read_csv(EXP/'seed42_archive_comparison.csv',float_precision='round_trip');assert len(comparison)==9
 refs=json.loads((EXP/'reference_manifest.json').read_text())['references'];refmap={(r['model'],r['horizon']):r for r in refs}
 for _,r in comparison.iterrows():
  m,f=score_map[(r.model,int(r.horizon))];old=canonical(pd.read_parquet(ROOT/refmap[(r.model,int(r.horizon))]['local_copy']['path']))
  pd.testing.assert_frame_equal(f[['ticker','anchor_date','split','y_true']],old[['ticker','anchor_date','split','y_true']],check_exact=True)
  oldm=common.evaluate_predictions(old.y_true.to_numpy(),old.y_pred.to_numpy(),old.ticker.to_numpy(),old.anchor_date.to_numpy())
  d=float(np.max(np.abs(f.y_pred.to_numpy(np.float64)-old.y_pred.to_numpy(np.float64))))
  close(d,r.max_abs_delta_pred);close(m['pooled_mse']-oldm['pooled_mse'],r.delta_mse)
  assert r.classification==('EXACT' if d<1e-7 else 'DIFFERENT')
  for data,metrics in [(primary,m),(reference,oldm)]:
   x=data[(data.model==r.model)&(data.horizon==r.horizon)&(data.seed==42)];assert len(x)==1
   close(x.iloc[0].test_mse,metrics['pooled_mse']);close(x.iloc[0].mean_rank_ic,metrics['cross_sectional_rank_ic'])
 for name,data in [('O_uniform12_summary_5seed.csv',primary),('O_archived42_reference_summary_5seed.csv',reference)]:
  saved=pd.read_csv(EXP/name,float_precision='round_trip');assert len(saved)==9
  for (model,h),g in data.groupby(['model','horizon']):
   assert len(g)==5 and set(g.seed)==set(io.SEEDS)
   r=saved[(saved.model==model)&(saved.horizon==h)].iloc[0]
   for metric in ['test_mse','mean_rank_ic']:
    close(r[metric+'_mean'],statistics.mean(g[metric]));close(r[metric+'_sd'],statistics.stdev(g[metric]))
 side=pd.read_csv(EXP/'O_summary_side_by_side.csv',float_precision='round_trip')
 for _,r in side.iterrows():
  c=comparison[(comparison.model==r.model)&(comparison.horizon==r.horizon)].iloc[0]
  close(r.delta_five_seed_test_mse_mean,c.delta_mse/5);close(r.delta_five_seed_mean_rank_ic,c.delta_mean_rank_ic/5)
 original=json.loads((io.B5/'FINAL_VALIDATION.json').read_text());assert io.sha(ROOT/original['report'])==original['report_sha256']
 # Prove the new queue never launched the reused pilot or any other model/seed/input cell.
 launches=[]
 for session in (EXP/'sessions').iterdir():
  q=session/'queue_result.json'
  if not q.exists():continue
  result=json.loads(q.read_text());assert not result['failures'] and not result['blocked']
  assert json.loads((session/'protected_inputs_verification.json').read_text())['passed']
  launches += [json.loads(x) for x in (session/'events.jsonl').read_text().splitlines() if x and json.loads(x)['event']=='LAUNCH']
 assert len(launches)==8 and all(x['job_id']!=io.REUSE_ID and x['input_set']=='O' and x['seed']==42 for x in launches)
 assert max(x['active_workers'] for x in launches)==3
 out=EXP/'postrun_checks'/io.unique();out.mkdir(parents=True,exist_ok=False)
 io.csvout(out/'verified_cells.csv',checks)
 io.jsonout(out/'ACCEPTANCE.json',dict(utc=io.utc(),passed=True,cells=9,strict_checkpoint_reloads=9,prediction_files_verified=27,new_training_launches=8,reused_pilot=1,no_duplicate_h5_training=True,uniform_summary_cells=45,reference_summary_cells=45,identical_other_seed_cells=36,archive_comparisons_verified=9,summary_mean_and_ddof1_sd_verified=True,mean_changes_equal_seed42_changes_divided_by_5=True,old_B5_report_unchanged=True,archive_EXACT=final['archive_EXACT'],archive_DIFFERENT=final['archive_DIFFERENT'],report=io.artifact(ROOT/final['report'])))
 print('B5FIX_POSTRUN_ACCEPTANCE_PASS',out,flush=True)
if __name__=='__main__':main()
