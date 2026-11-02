"""Verify all existing cells/tensors and add only the missing W3 reference seeds."""
from supp_io import *
from b2_data import base,common,load_panel,window_contract
from report_stage3bc_final import old_predictions
import fullrun14a_io as formal_io
import numpy as np,pandas as pd

def prepare():
 assert len(NEW)==48 and len(PLAN)==156 and len(OLD_IDS)==108
 base.check_profile();base.exp4._verify_frozen_dependencies();base.exp4._verify_frozen_training_constants()
 old_validation=json.loads((OLD/'FINAL_VALIDATION.json').read_text());assert old_validation['all_new_cells_verified'] and old_validation['protected_files_unchanged']
 reused=[]
 for spec in original_io.PLAN:
  cp,r=original_io.completed({k:spec[k] for k in KEYS});assert r,spec
  reused.append(dict(job_id=spec['job_id'],completion=artifact(cp),block=spec['block'],seed=spec['seed'],checkpoint_sha256=r['checkpoint_sha256']))
 assert len(reused)==108
 print('ORIGINAL_108_COMPLETIONS_VERIFIED',utc(),flush=True)
 # Full original B2 tree (including its report, logs, status snapshots and tensors) is frozen.
 frozen=[artifact(p) for p in sorted(OLD.rglob('*')) if p.is_file() and not p.is_symlink()]
 for p in [FULLRUN/'REPORT_STEP14C_B2.md']:
  frozen.append(artifact(p))
 jsonout(EXP/'ORIGINAL_REUSE.json',dict(utc=utc(),completed_cells=reused,original_tree_files=frozen,original_report=artifact(FULLRUN/'REPORT_STEP14C_B2.md'),no_original_cell_retraining=True,deterministic_cells_reused=36))
 for w in ['W1','W2','W3']:
  for inp in ['O','C']:window_contract(load_panel(w,inp),w)
 for d in json.loads((OLD/'tensor_manifest.json').read_text())['files']:assert sha(ROOT/d['path'])==d['sha256']
 for d in json.loads((OLD/'preflight.json').read_text())['dependencies']:assert sha(ROOT/d['path'])==d['sha256']
 ds=load_panel('W3','O');mask=ds.split=='test'
 keys=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'))).sort_values(['ticker','anchor_date']).reset_index(drop=True)
 previous=json.loads((OLD/'W3_provenance.json').read_text());entries=list(previous['entries'])
 with (OLD/'W3_reference_metrics.csv').open(newline='') as f:metrics=list(csv.DictReader(f))
 with (OLD/'global_train_mean_references.csv').open(newline='') as f:denom={int(r['horizon']):float(r['test_mse_global_train_mean']) for r in csv.DictReader(f) if r['window']=='W3'}
 old=old_predictions()
 def add(inp,model,h,seed,frame,source,provenance):
  x=frame.copy();x['anchor_date']=pd.to_datetime(x.anchor_date).dt.strftime('%Y-%m-%d');x=x.sort_values(['ticker','anchor_date']).reset_index(drop=True)
  expected=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),y_true=ds.y[h][mask])).sort_values(['ticker','anchor_date']).reset_index(drop=True)
  assert len(x)==6647 and x[['ticker','anchor_date']].equals(keys)
  y=x.y_true.to_numpy(np.float32);p=x.y_pred.to_numpy(np.float32);assert np.array_equal(y,expected.y_true.to_numpy(np.float32)) and np.isfinite(p).all()
  out=pd.DataFrame(dict(ticker=x.ticker,anchor_date=x.anchor_date,split='test',y_true=y,y_pred=p));dest=EXP/'W3_reused_predictions'/f'W3_{inp}_{model}_h{h}_s{seed}_test.parquet'
  with target(dest).open('xb') as f:f.write(out.to_parquet(index=False,engine='pyarrow'))
  pd.testing.assert_frame_equal(out,pd.read_parquet(dest))
  m=common.evaluate_predictions(y,p,out.ticker.to_numpy(),out.anchor_date.to_numpy())
  metrics.append(dict(window='W3',input_set=inp,model=model,horizon=h,seed=seed,test_mse=m['pooled_mse'],mean_rank_ic=m['cross_sectional_rank_ic'],mse_relative_train_mean=m['pooled_mse']/denom[h],R2_OS_train_mean=1-m['pooled_mse']/denom[h],provenance=provenance,predictions=rel(dest)))
  entries.append(dict(window='W3',input_set=inp,model=model,horizon=h,seed=seed,source=rel(source),source_sha256=sha(source),copied_test_predictions=artifact(dest),n_rows=6647,keys_labels_identical=True,provenance=provenance,no_retraining=True))
 for seed in [21,3407]:
  for h in [20,10,5]:
   for model in ['static','adaptive']:
    source=ROOT/'dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv'
    add('O',model,h,seed,old[(model,h,seed)],source,'archived formal Experiment4 W3 predictions; no training or inference')
    cp,r=formal_io.completed(formal_io.job('C',model,h,seed,'Q1'));assert r,(model,h,seed)
    source=cp.parent/'test.parquet';add('C',model,h,seed,pd.read_parquet(source),source,'14a12/12 C W3 predictions; no training or inference')
 assert len(entries)==len(metrics)==78
 for e in entries:
  assert sha(ROOT/e['source'])==e['source_sha256'];a=e['copied_test_predictions'];assert sha(ROOT/a['path'])==a['sha256']
 jsonout(EXP/'W3_provenance.json',dict(utc=utc(),n_cells=78,no_retraining=True,entries=entries,original_reused=54,additional_seed_references=24))
 csvout(EXP/'W3_reference_metrics.csv',metrics)
 jsonout(EXP/'preflight.json',dict(utc=utc(),passed=True,dependencies=extra_dependencies()+[artifact(OLD/n) for n in ['design.json','tensor_manifest.json','FINAL_VALIDATION.json','global_train_mean_references.csv','window_summary.csv']],original_completed_cells_verified=108,new_cells=48,no_reference_retraining=True,windows_rebuilt=False,original_tensor_hashes_verified=True,W3_reference_cells=78,thread_counts=[12,12]))
 textout(EXP/'DESIGN_REUSE.md','# BLOCKING\n\nNone.\n\nThe original B2 window design and tensors are reused exactly, without rebuilding or changing standardization. All108 original cells and the complete original B2 tree have been hashed. The supplementary plan contains48 new neural cells only (seeds21/3407); each uses Experiment4 and DataLoader generator42. All36 deterministic/Ridge cells are reused. W3 now has78 reference cells (60 neural five-seed cells and18 reference cells), with exact keys/labels verified.\n\nOriginal pre-training design: '+rel(OLD/'DESIGN_REPORT_STEP14C_B2.md')+'\nOriginal tensor manifest SHA256: '+sha(OLD/'tensor_manifest.json')+'\n')
 print('SUPPLEMENT_PREFLIGHT_PASS',utc(),flush=True)
if __name__=='__main__':prepare()
