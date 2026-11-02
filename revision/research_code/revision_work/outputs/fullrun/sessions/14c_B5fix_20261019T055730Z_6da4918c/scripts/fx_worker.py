"""Use the exact imported B5 worker; seed42/O only; no model or trainer rewrite."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
from fx_io import *
sys.path.insert(0,str(B5/'scripts'))
import b5_worker_v2 as b5
import b5_io as oldio
import torch,numpy as np,pandas as pd,argparse,copy,shutil,time
from contextlib import ExitStack
assert Path(b5.__file__).resolve()==B5/'scripts/b5_worker_v2.py'

def dependencies(inp='O'):
 assert inp=='O'
 rows=b5.ORIGINAL_DEPS('O')+[artifact(EXP/'experiment.json')]+[artifact(p) for p in sorted(HERE.rglob('*.py'))]
 rows += [artifact(p) for p in [Path(b5.__file__),Path(oldio.__file__),B5/'experiment.json',B5/'preflight.json',B5/'FINAL_VALIDATION.json',B5/'seed42_metrics.csv',B5/'reference_provenance.json']]
 return list({a['path']:a for a in rows}.values())

def bind(stack,obj,key,value):
 before=getattr(obj,key);setattr(obj,key,value);stack.callback(setattr,obj,key,before)

def run(spec,attempt):
 assert spec['input_set']=='O' and spec['seed']==42 and spec['job_id']!=REUSE_ID,'h5 LSTM must reuse its verified B5 pilot; no duplicate training'
 original_save=b5.day.jsonout
 def save(path,value):
  if Path(path).name=='COMPLETE.json':value=dict(value,training_reused=False)
  return original_save(path,value)
 # B5.run installs its own completion wrapper while retaining this exclusive writer.
 with ExitStack() as stack:
  bind(stack,b5,'EXP',EXP);bind(stack,b5,'HERE',HERE);bind(stack,b5,'dependencies',dependencies)
  bind(stack,b5,'verify_completion',verify_completion);bind(stack,b5.day,'jsonout',save)
  b5.run(spec,attempt)

def verify_keys(frame,ds,h,split):
 mask=ds.split==split
 assert len(frame)==COUNTS[split] and not frame.duplicated(['ticker','anchor_date']).any()
 assert (frame.split==split).all() and np.array_equal(frame.ticker.to_numpy(),ds.ticker[mask])
 assert np.array_equal(pd.to_datetime(frame.anchor_date).to_numpy(),pd.to_datetime(ds.anchor_date[mask]).to_numpy())
 assert np.array_equal(frame.y_true.to_numpy(),ds.y[h][mask])
 assert np.isfinite(frame.select_dtypes(include='number')).all().all()

def preflight():
 b5.day.base.check_profile();ds=b5.day.common.load_frozen_panel();b5.day.formal._formal_check_dataset_contract(ds)
 check=b5.input_check(ds,'O');prior=json.loads((B5/'preflight.json').read_text())
 old=next(x for x in prior['input_checks'] if x['input_set']=='O');assert check['normalized']==old['normalized']
 final=json.loads((B5/'FINAL_VALIDATION.json').read_text());assert final['all_cells_verified']
 assert sha(ROOT/final['report'])==final['report_sha256']
 source_spec=oldio.PILOT;source,r=oldio.completed(source_spec);assert r and source_spec['job_id']==CONFIG['reuse_source_job']
 model=b5.day.lstm.make_model('daily_weekly',5,5)
 model.load_state_dict(torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True),strict=True);model.eval()
 frames,metrics=b5.day.inference(model,ds,spec_for(REUSE_ID))
 delta=0.
 for split in COUNTS:
  oldframe=pd.read_parquet(source.parent/f'{split}.parquet');verify_keys(oldframe,ds,5,split)
  pd.testing.assert_frame_equal(frames[split],oldframe,check_exact=True)
  if split=='test':delta=float(np.max(np.abs(frames[split].y_pred.to_numpy(np.float64)-oldframe.y_pred.to_numpy(np.float64))))
 assert delta<1e-7
 attempt=EXP/'runs'/REUSE_ID/('attempt_reused_'+unique());attempt.mkdir(parents=True,exist_ok=False)
 copied=[]
 for a in r['artifacts']:
  src=ROOT/a['path'];dst=attempt/src.name
  with src.open('rb') as inp,target(dst).open('xb') as out:shutil.copyfileobj(inp,out)
  assert sha(dst)==a['sha256'];copied.append(dst)
 reuse=dict(utc=utc(),source_completion=artifact(source),source_job_id=source_spec['job_id'],test_replay_max_abs_delta_pred=delta,all_three_split_replays_bitwise_equal=True,training_repeated=False,source_training_times_retained=True)
 jsonout(attempt/'reuse_provenance.json',reuse);copied.append(attempt/'reuse_provenance.json')
 record=copy.deepcopy(r);record.update(spec_for(REUSE_ID));record.update(training_reused=True,reuse_source=artifact(source),reuse_verification=reuse,checkpoint=rel(attempt/Path(r['checkpoint']).name),checkpoint_sha256=sha(attempt/Path(r['checkpoint']).name),artifacts=[artifact(p) for p in copied])
 record['dependencies']=list({a['path']:a for a in r['dependencies']+r['artifacts']+dependencies()+[artifact(source)]}.values())
 jsonout(attempt/'COMPLETE.json',record);verify_completion(attempt/'COMPLETE.json',spec_for(REUSE_ID))
 # Freeze nine archived MAIN V2 references, preserving the original float32 neural predictions.
 references=pd.read_csv(B5/'seed42_metrics.csv',float_precision='round_trip');references=references[references.input_set=='O'];assert len(references)==9
 refrows=[]
 for _,x in references.iterrows():
  sourceframe=ROOT/x.prediction;assert sha(sourceframe)==x.prediction_sha256 and sha(ROOT/x.source)==x.source_sha256
  archive=pd.read_parquet(sourceframe);mask=ds.split=='test'
  assert len(archive)==COUNTS['test'] and np.isfinite(archive[['y_true','y_pred']]).all().all()
  expected=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),y_true=ds.y[int(x.horizon)][mask])).sort_values(['ticker','anchor_date']).reset_index(drop=True)
  actual=archive.sort_values(['ticker','anchor_date']).reset_index(drop=True)
  assert np.array_equal(actual[['ticker','anchor_date']].to_numpy(),expected[['ticker','anchor_date']].to_numpy()) and np.array_equal(actual.y_true.to_numpy(),expected.y_true.to_numpy())
  dst=EXP/'references'/f'archived_O_{x.model}_dw_h{int(x.horizon)}_s42_test.parquet'
  with sourceframe.open('rb') as inp,target(dst).open('xb') as out:shutil.copyfileobj(inp,out)
  refrows.append(dict(model=x.model,horizon=int(x.horizon),seed=42,original_csv=x.source,original_csv_sha256=x.source_sha256,source_copy=artifact(sourceframe),local_copy=artifact(dst),archived_test_mse=float(x.test_mse),archived_mean_rank_ic=float(x.mean_rank_ic)))
 jsonout(EXP/'reference_manifest.json',dict(utc=utc(),references=refrows,source_csv=artifact(B5/'seed42_metrics.csv'),precision='original float32 neural predictions reconstructed from archived CSV; same as B5 reference comparisons'))
 jsonout(EXP/'preflight.json',dict(utc=utc(),passed=True,dependencies=dependencies(),input_check=check,original_B5_input_binding_bitwise_equal=True,reused_pilot=rel(attempt/'COMPLETE.json'),reuse_source_replay_max_delta=delta,new_training_cells=8,reused_cells=1,thread_counts=[torch.get_num_threads(),torch.get_num_interop_threads()],environment={k:os.environ.get(k) for k in ENV_KEYS}))
 print('B5FIX_PREFLIGHT_PASS; reused h5 LSTM; all three replay splits bitwise EXACT; new training=8',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--job-id');ap.add_argument('--attempt');a=ap.parse_args()
 if a.preflight:return preflight()
 spec=spec_for(a.job_id);attempt=Path(a.attempt).absolute();assert attempt.is_relative_to(EXP/'runs'/spec['job_id'])
 try:run(spec,attempt)
 except BaseException as e:
  if not (attempt/'FAILED.json').exists():jsonout(attempt/'FAILED.json',dict(utc=utc(),job_id=spec['job_id'],error=repr(e)))
  raise
if __name__=='__main__':main()
