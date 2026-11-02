"""New Step 13 router entry point: original|causal binding, I0 smoke test only.
Imports the existing intervention wrapper and forward loop; never calls its experiment main().
"""
import sys,os,argparse,importlib.util,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import *
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'revision_work/scripts/causal'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input-set',choices=['original','causal'],required=True);ap.add_argument('--horizon',type=int,default=5);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--checkpoint');ap.add_argument('--reference');ap.add_argument('--run-label',default='causal_i0_smoke');a=ap.parse_args()
 src=ROOT/'revision_work/scripts/interventions/router_interventions.py'
 spec=importlib.util.spec_from_file_location('step13_original_interventions',src);iv=importlib.util.module_from_spec(spec);spec.loader.exec_module(iv)
 import core
 import torch,numpy as np,pandas as pd
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
 assert torch.get_num_threads()<=2 and torch.get_num_interop_threads()<=2
 h=a.horizon;seed=a.seed
 if a.input_set=='causal':
  ds=core.load_causal();cp=ROOT/f'revision_work/outputs/causal_features/checkpoints/adaptive_dw_h{h}_s{seed}.pt';ref=ROOT/f'revision_work/outputs/causal_features/preds/adaptive_dw_h{h}_s{seed}_test.parquet'
 else:
  ds=iv.formal.load_frozen_panel();cp=iv.E4/f'checkpoints/adaptive_seed{seed}_{h}d.pt';ref=iv.PREDS/f'adaptive_dw_h{h}_s{seed}_test.parquet'
 cp=Path(a.checkpoint) if a.checkpoint else cp;ref=Path(a.reference) if a.reference else ref
 assert 'fullrun' not in cp.parts and 'fullrun' not in ref.parts
 before={str(p):sha(p) for p in [src,cp,ref,ROOT/'revision_work/scripts/causal/core.py']}
 iv.formal._formal_check_dataset_contract(ds)
 _,_,test=iv.formal._build_training_loaders('daily_weekly',h,ds,batch_size=iv.formal.BATCH_SIZE)
 model=iv.exp4._make_experiment4_model('adaptive');model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True),strict=True);model.cpu().eval()
 t0=time.perf_counter();pr,dw,ww=iv.exp4._batched_inference(model,test,6647,torch.device('cpu'))
 ov=[iv.RouterOverride(model.daily_branch.aggregator,'daily'),iv.RouterOverride(model.weekly_branch.aggregator,'weekly')]
 own=iv.run_forward(model,test,ov,None)
 mask=ds.split=='test';frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split='test',y_true=ds.y[h][mask],y_pred=own))
 for j,c in enumerate(iv.DAILY+iv.WEEKLY):frame[c]=(dw if j<4 else ww)[:,j%4]
 saved=pd.read_parquet(ref);saved['anchor_date']=pd.to_datetime(saved.anchor_date).dt.strftime('%Y-%m-%d')
 keys=['ticker','anchor_date'];mine=frame.sort_values(keys).reset_index(drop=True);saved=saved.sort_values(keys).reset_index(drop=True)
 same=mine[keys].equals(saved[keys]);labels=np.array_equal(mine.y_true.to_numpy(np.float32),saved.y_true.to_numpy(np.float32));delta=float(np.max(np.abs(mine.y_pred.to_numpy(float)-saved.y_pred.to_numpy(float))));dloop=float(np.max(np.abs(own.astype(float)-pr.astype(float))))
 cols=iv.DAILY+iv.WEEKLY;dwref=float(np.max(np.abs(mine[cols].to_numpy(float)-saved[cols].to_numpy(float))))
 mse=lambda p:float(np.mean((p-mine.y_true.to_numpy(float))**2))
 result=dict(input_set=a.input_set,intervention='I0_learned',horizon=h,seed=seed,n_test=len(frame),checkpoint=str(cp.relative_to(ROOT)),reference=str(ref.relative_to(ROOT)),max_abs_delta_prediction=delta,max_abs_delta_router_weights=dwref,max_abs_delta_override_vs_formal_helper=dloop,delta_mse=mse(mine.y_pred.to_numpy(float))-mse(saved.y_pred.to_numpy(float)),identical_keys=same,identical_labels=labels,classification='EXACT' if same and labels and delta<1e-7 and dloop<1e-7 else 'DIFFERENT',exact_tolerance=1e-7,wall_seconds=time.perf_counter()-t0,torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),nice=os.getpriority(os.PRIO_PROCESS,0),utc=utc(),input_binding='core.load_causal (same binding as stage3bc_worker.py)' if a.input_set=='causal' else 'formal.load_frozen_panel')
 rows(a.run_label+'/verification.csv',[result]);js(a.run_label+'/verification.json',result)
 with target(a.run_label+'/I0_test.parquet').open('xb') as f:f.write(frame.to_parquet(index=False))
 js(a.run_label+'/protected_inputs.json',[dict(path=p,sha256_before=v,sha256_after=sha(p),unchanged=v==sha(p)) for p,v in before.items()])
 print(result,flush=True)
 if result['classification']!='EXACT':raise SystemExit(2)
if __name__=='__main__':main()
