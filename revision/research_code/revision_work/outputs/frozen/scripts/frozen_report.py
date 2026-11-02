"""Step15 report and paired seed x common-calendar moving-block bootstrap."""
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ.pop(k,None)
os.environ.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
import torch
torch.set_num_threads(12);torch.set_num_interop_threads(12)
from frozen_io_v3 import *
import argparse,ast,types,math,time,platform
import numpy as np,pandas as pd
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'revision_work/scripts/skill'));sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
# Import the original bootstrap helpers without letting its module defaults replace this run's profile.
klass=type(os.environ);envset=klass.__setitem__
def preserved(self,k,v):
 if k in ENV_KEYS[:5]:return
 return envset(self,k,v)
klass.__setitem__=preserved
try:from skill_common import mbb_index,holm
finally:klass.__setitem__=envset
# Import the existing statistics function verbatim, omitting the old intervention module's thread setters.
source=ROOT/'revision_work/scripts/interventions/router_interventions.py';tree=ast.parse(source.read_text());ns={'np':np};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='boot_stats'],type_ignores=[]),str(source),'exec'),ns);boot_stats=ns['boot_stats']
from scripts.python import panel_common as common
B=20000;BOOT_SEED=20261019

def dataframe_table(df):
 def cell(v):
  if isinstance(v,(float,np.floating)):
   if not np.isfinite(v):return 'NOT FOUND'
   return f'{v:.9g}'
  return str(v).replace('|','/').replace(chr(10),' ')
 return chr(10).join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])+chr(10)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--session',required=True);a=ap.parse_args();session=Path(a.session);clock=time.perf_counter();registry=json.loads((OUT/'backbone_registry.json').read_text());allrows=[];filelist=[];replay=[];history=[];objects={};dep=[]
 for spec in registry:
  assert sha(ROOT/spec['checkpoint'])==spec['checkpoint_sha256'];dep.append(dict(path=spec['checkpoint'],sha256_expected=spec['checkpoint_sha256'],sha256_actual=sha(ROOT/spec['checkpoint']),unchanged=True,provenance=spec['provenance']))
  cachepath,cache=finished(OUT/'backbones'/spec['id']/'cache');assert cachepath
  for row in cache['replay']:replay.append(dict(backbone_id=spec['id'],input_set=spec['input_set'],backbone=spec['backbone'],horizon=spec['horizon'],seed=spec['seed'],provenance=spec['provenance'],**row))
  assert all(r['classification']=='EXACT' for r in cache['replay'])
  for setting in (['head_frozen','head_retrained'] if spec['backbone']=='S1' else ['head_frozen']):
   for variant in ['fixed','static','constant_gate','adaptive']:
    actual_setting='head_frozen' if setting=='head_retrained' and variant=='fixed' else setting
    path,j=finished(OUT/'backbones'/spec['id']/actual_setting/variant);assert path,(spec['id'],setting,variant)
    objects[(spec['input_set'],spec['backbone'],setting,spec['horizon'],spec['seed'],variant)]=(path.parent,j)
    allrows.append(dict(input_set=spec['input_set'],backbone=spec['backbone'],setting=setting,horizon=spec['horizon'],seed=spec['seed'],variant=variant,trainable_parameters=j['trainable_parameters'],best_epoch=j['best_epoch'],epochs=j['epochs'],wall_seconds=j['wall_seconds'],provenance=spec['provenance'],fixed_baseline_reused=actual_setting!=setting,**j['metrics']))
    for ar in j['artifacts']:
     if ar['path'].endswith('.parquet'):filelist.append(dict(input_set=spec['input_set'],backbone=spec['backbone'],setting=setting,variant=variant,horizon=spec['horizon'],seed=spec['seed'],split=Path(ar['path']).stem,**ar))
    hfile=path.parent/'history.csv';hd=pd.read_csv(hfile)
    if len(hd):
     for key,value in dict(input_set=spec['input_set'],backbone=spec['backbone'],setting=setting,horizon=spec['horizon'],seed=spec['seed'],variant=variant).items():hd[key]=value
     history.append(hd)
   c=objects[(spec['input_set'],spec['backbone'],setting,spec['horizon'],spec['seed'],'constant_gate')][1];d=objects[(spec['input_set'],spec['backbone'],setting,spec['horizon'],spec['seed'],'adaptive')][1]
   assert c['initial_router_hash']==d['initial_router_hash'],'Constant/adaptive initialization not paired'
 M=pd.DataFrame(allrows);assert len(M)==360 and M.groupby(['input_set','backbone','setting','horizon','variant']).seed.nunique().eq(5).all()
 #360 includes30 explicit references to primary Fixed for the secondary baseline; unique fitted/evaluated cells330.
 cout(OUT/'cell_metrics.csv',M);cout(OUT/'backbone_cache_replay.csv',pd.DataFrame(replay));cout(OUT/'prediction_files.csv',pd.DataFrame(filelist));cout(OUT/'learning_history.csv',pd.concat(history,ignore_index=True));cout(OUT/'protected_checkpoints.csv',pd.DataFrame(dep))
 grouped=M.groupby(['input_set','backbone','setting','horizon','variant'],as_index=False)
 S=grouped.agg(n_seeds=('seed','nunique'),mse_mean=('test_mse','mean'),mse_sd=('test_mse','std'),rank_ic_mean=('mean_rank_ic','mean'),rank_ic_sd=('mean_rank_ic','std'),direction_accuracy_mean=('direction_accuracy','mean'),direction_accuracy_sd=('direction_accuracy','std'),best_epoch_mean=('best_epoch','mean'),trainable_parameters=('trainable_parameters','first'))
 cout(OUT/'five_seed_summary.csv',S)
 param=M[['setting','variant','trainable_parameters']].drop_duplicates().sort_values(['setting','variant']);cout(OUT/'trainable_parameters.csv',param)
 # Date-level arms are paired on identical seeds and original TEST calendars; do not pool stocks before preserving date means.
 datepanels={};missing=[]
 for key,(folder,j) in objects.items():
  x=pd.read_csv(folder/'test_date_metrics.csv',float_precision='round_trip').sort_values('anchor_date');assert len(x)==391 and x.n_tickers.eq(17).all()
  datepanels[key]=x
 common_dates=None;bootrows=[];loss_panels=[];dist={};wins=[];drawmeta=[]
 contrasts=[('adaptive','static'),('adaptive','constant_gate'),('static','fixed')]
 families=M[['input_set','backbone','setting']].drop_duplicates().itertuples(index=False,name=None)
 # Save the exact resampling draws so joint uncertainty calculations are independently reviewable.
 draws={}
 for h in HORIZONS:
  rng=np.random.default_rng(np.random.SeedSequence([BOOT_SEED,h,h,1]));seeds=sorted(SEEDS);sidx=rng.integers(0,5,size=(B,5));didx=mbb_index(391,h,B,rng)
  wt=np.zeros((B,5));np.add.at(wt,(np.repeat(np.arange(B),5),sidx.ravel()),.2);draws[h]=(wt,didx)
  p=OUT/'bootstrap'/f'draws_h{h}.npz'
  with target(p).open('xb') as f:np.savez_compressed(f,seed_indices=sidx,date_indices=didx)
  drawmeta.append(dict(horizon=h,B=B,L=h,block_type='non-circular overlapping moving blocks, ceil(T/L), truncate391',bootstrap_seed=BOOT_SEED,seed_sequence=json.dumps([BOOT_SEED,h,h,1]),seed_order=json.dumps(seeds),path=rel(p),sha256=sha(p)))
 for inp,bk,setting in families:
  for h in HORIZONS:
   wt,didx=draws[h]
   for left,right in contrasts:
    contrast=f'{left}_minus_{right}'
    for metric,col,positive_good in [('mse','mse_date',False),('rank_ic','rank_ic',True),('direction_accuracy','da_date',True)]:
     mat=[]
     for seed in sorted(SEEDS):
      A=datepanels[(inp,bk,setting,h,seed,left)];R=datepanels[(inp,bk,setting,h,seed,right)];assert A.anchor_date.equals(R.anchor_date)
      dates=A.anchor_date.to_numpy();common_dates=dates if common_dates is None else common_dates;assert np.array_equal(dates,common_dates)
      delta=A[col].to_numpy(float)-R[col].to_numpy(float);mat.append(delta)
      wins.append(dict(input_set=inp,backbone=bk,setting=setting,horizon=h,contrast=contrast,metric=metric,seed=seed,mean_delta=float(np.nanmean(delta)),left_wins=bool(np.nanmean(delta)>0 if positive_good else np.nanmean(delta)<0)))
      loss_panels.extend(dict(input_set=inp,backbone=bk,setting=setting,horizon=h,contrast=contrast,metric=metric,seed=seed,anchor_date=date,paired_difference=float(v) if np.isfinite(v) else None) for date,v in zip(dates,delta))
     X=np.asarray(mat);n_missing=int((~np.isfinite(X)).sum());theta=float(np.nanmean(X,axis=1).mean())
     if n_missing:
      # Keep the391-session calendar. Mean paired-valid outcomes within each sampled seed, then average paired seed draws.
      per=np.stack([np.nanmean(X[s][didx],axis=1) for s in range(5)],axis=1);boot=(wt*per).sum(1)
      missing.append(dict(input_set=inp,backbone=bk,setting=setting,horizon=h,contrast=contrast,metric=metric,n_missing_seed_dates=n_missing,rule='paired valid dates within each seed, calendar retained in MBB'))
     else:boot=np.take_along_axis(wt@X,didx,axis=1).mean(1)
     assert np.isfinite(boot).all()
     bootrows.append(dict(input_set=inp,backbone=bk,setting=setting,horizon=h,contrast=contrast,metric=metric,n_seeds=5,n_dates=391,L=h,B=B,n_missing_seed_dates=n_missing,**boot_stats(boot,theta)))
     name=f'{inp}_{bk}_{setting}_h{h}_{contrast}_{metric}';dist[name]=boot
  print('BOOTSTRAP_COMPLETE',inp,bk,setting,flush=True)
 J=pd.DataFrame(bootrows);J['p_holm_over_h']=np.nan
 for _,g in J.groupby(['input_set','backbone','setting','contrast','metric']):
  assert len(g)==3
  for ix,pv in holm(dict(zip(g.index,g.p_centered))).items():J.loc[ix,'p_holm_over_h']=pv
 assert len(J)==162
 cout(OUT/'paired_seed_date_bootstrap.csv',J);cout(OUT/'paired_seed_date_differences.csv',pd.DataFrame(loss_panels));cout(OUT/'seed_wins.csv',pd.DataFrame(wins));cout(OUT/'bootstrap_draws.csv',pd.DataFrame(drawmeta));cout(OUT/'missing_rank_ic_support.csv',pd.DataFrame(missing,columns=['input_set','backbone','setting','horizon','contrast','metric','n_missing_seed_dates','rule']))
 with target(OUT/'bootstrap'/'paired_distributions.npz').open('xb') as f:np.savez_compressed(f,**dist)
 # Verify every saved prediction row count and key/label identity against each source backbone.
 predchecks=[]
 for spec in registry:
  for setting in (['head_frozen','head_retrained'] if spec['backbone']=='S1' else ['head_frozen']):
   for variant in ['fixed','static','constant_gate','adaptive']:
    folder,j=objects[(spec['input_set'],spec['backbone'],setting,spec['horizon'],spec['seed'],variant)]
    cache=ROOT/j['cache_complete'];cache=cache.parent
    for sp,n in COUNTS.items():
     p=folder/f'{sp}.parquet';d=pd.read_parquet(p);ref=pd.read_parquet(cache/f'backbone_{sp}.parquet');assert len(d)==n and d[['ticker','anchor_date','split','y_true']].equals(ref[['ticker','anchor_date','split','y_true']]);assert np.isfinite(d[['y_true','y_pred']+WEIGHTS]).all().all()
     predchecks.append(dict(backbone_id=spec['id'],setting=setting,variant=variant,split=sp,n_rows=n,identical_keys_labels=True,frozen_parameters_unchanged=j['frozen_parameters_unchanged']))
 cout(OUT/'prediction_integrity.csv',pd.DataFrame(predchecks))
 for spec in registry:assert sha(ROOT/spec['checkpoint'])==spec['checkpoint_sha256']
 assert torch.get_num_threads()==torch.get_num_interop_threads()==12 and os.environ.get('OMP_NUM_THREADS') is None
 # Readable report: every displayed numeric table has a full-precision CSV counterpart.
 display=S[['input_set','backbone','setting','horizon','variant','n_seeds']].copy()
 for name,mu,sd in [('TEST MSE','mse_mean','mse_sd'),('mean Rank IC','rank_ic_mean','rank_ic_sd'),('direction accuracy','direction_accuracy_mean','direction_accuracy_sd')]:display[name]=[f'{m:.9g} +/- {s:.9g}' for m,s in zip(S[mu],S[sd])]
 cout(OUT/'summary_display.csv',display)
 replaydf=pd.DataFrame(replay);nonstrict=replaydf[replaydf.comparison=='cache_vs_full_checkpoint'];non14a=replaydf[(replaydf.provenance!='14a')&(replaydf.comparison=='checkpoint_vs_formal_TEST')]
 blocking='None. All60 backbones loaded strictly and passed replay; all330 unique routing/evaluation cells completed; all five-seed groups and requested paired bootstrap families are present.'
 if missing:blocking='Rank IC has undefined seed/date cells in the comparisons listed in missing_rank_ic_support.csv. The report uses paired-valid support within the original calendar; these are not zero-imputed.'
 report=['# BLOCKING',blocking,'# REPORT STEP15 — Frozen-representation routing comparison',f'Completed {utc()} UTC. O=original frozen V2 inputs; C=the imported Stage3a causal binding. All new code, caches, routing/head states and predictions are under `revision_work/outputs/frozen/`, created exclusively. Existing files and source checkpoints were read only. No backbone was retrained.',
 '## Backbone provenance and qualification',
 'The user authorized17 sources outside14a to complete the60-backbone design. Source counts:14a43, Experiment3 two O Fixed seed42 h10/20, Experiment4 twelve O Adaptive seeds0/1/21/3407, Step3B three EXACT O Adaptive seed42 h5/10/20. The registry binds paths and expected SHA256 to all TRAIN/VAL/TEST references and formal TEST files. Each non14a source was loaded with strict=True and compared to its archived/formal TEST before any routing fit. Non-EXACT replay stops the entire worker; no replacement backbone is trained. All17 passed.',
 dataframe_table(pd.DataFrame([dict(provenance=k,n_backbones=sum(r['provenance']==k for r in registry)) for k in ['14a','Exp3','Exp4','Step3B']])),
 'The complete path/hash/source table is [backbone_provenance.csv](backbone_provenance.csv); machine-readable split references and hashes are [backbone_registry.json](backbone_registry.json). The newly executed qualification uses CPU12/12. The source comparability/profile requirement follows the user-approved source selection, with existing recorded12/12 evidence retained for14a and Step3B.',
 f'Every source loaded strictly. Maximum |delta prediction| against saved split predictions and archived/formal TEST={replaydf[replaydf.comparison!="cache_vs_full_checkpoint"].max_abs_delta_pred.max():.12g}; all EXACT (<1e-7). Cache-versus-full-model max delta={nonstrict.max_abs_delta_pred.max():.12g}; {int(nonstrict.bitwise_predictions.fillna(False).sum())}/180 split comparisons are bitwise equal. [Every replay](backbone_cache_replay.csv).',
 '## Cached representations and frozen boundaries',
 'Each backbone produces one float32 (N,8,32) expert cache per split:30634 TRAIN,6290 VAL,6647 TEST rows. The original backbone runs in eval/no_grad mode, and aggregator pre-hooks capture its actual four Daily and four Weekly expert outputs. Cache replay invokes the original imported aggregators and fusion head, replacing only the branch input with those representations. S1 replays equal weighting; S2 replays its original trained router. TRAIN/VAL/TEST caches and original backbone predictions/router weights are retained with hashes; no expert runs during a routing fit.',
 'Factories, ScaleAggregator, fusion head, metrics and the entire train_one_cell training loop are imported from the formal V2 code. A process-local loader/factory adapter gives that loop cached representations and the selected aggregation modules. Input projections/position encoding/all8 experts are frozen and held in eval; primary fusion/head is also frozen/eval. All variants use the same cached values. The frozen-parameter fingerprint is identical before/after each fit. The secondary S1 head starts at the backbone head and is jointly fine-tuned, rather than randomly reinitialized; its architecture has no dropout.',
 '## Variants and parameter counts',dataframe_table(param),
 'Fixed uses two equal-weight branches and has no training. Static initializes its two four-logit vectors at0. Constant-input gate and Adaptive use the same two imported Linear(128,32)-ReLU-Linear(32,4) routers and identical seeded initial weights. Constant-input feeds each branch the fixed TRAIN-mean expert vector (float64 reduction on TRAIN cache, cast to float32), with no VAL/TEST contribution. Adaptive feeds the sample vector. Static and Constant-input weights are checked constant across samples to numerical tolerance. The secondary Fixed reference is the primary no-training Fixed result; only b/c/d retrain the head. Thus secondary b-a includes head adaptation as well as Static routing.',
 'CPU12/12 throughout, OMP_NUM_THREADS unset, OMP_DYNAMIC=FALSE and MKL_DYNAMIC=FALSE, configured before Torch work. All fits retain formal Adam lr1e-4, batch32, Huber delta1, clip1, max100, patience10, and val-Huber improvement>1e-12 best-state selection/restoration. Model/router initialization seed is the backbone seed; independent TRAIN DataLoader generator is42 for every variant. Only routing and (in the secondary setting) head parameters require gradients. Validation MSE is observed without extra training forwards.',
 '## Five-seed TEST metrics',
 'Mean +/- sample SD (ddof1), all five seeds42/0/1/21/3407. TEST MSE, mean within-date cross-sectional Spearman Rank IC, and sign-based direction accuracy use the formal metric implementation. Fixed secondary rows reuse the primary baseline explicitly; no missing seed is silently summarized. [Numeric summary](five_seed_summary.csv), [per-seed metrics](cell_metrics.csv), [histories](learning_history.csv).',
 dataframe_table(display),
 '## Paired seed x date joint bootstrap',
 'For each input x backbone x setting, compare d-b (Adaptive-Static), d-c (Adaptive-Constant-input), b-a (Static-Fixed), for MSE, Rank IC and direction accuracy. First form per-date means over17 stocks for MSE and direction accuracy, and formal date-level Rank IC. Resample5 paired seeds with replacement and one shared sequence of391 date positions from non-circular overlapping moving blocks (L=h), using identical draws for both arms and all seeds. B=20000, RNG base20261019 with SeedSequence([base,h,h,1]); within each horizon draws are shared across all comparisons. The existing mbb_index, Holm function and boot_stats implementation are reused. CIs are2.5/97.5 percentiles and p-values are the original two-sided centered bootstrap p=(1+count(abs(draw-theta)>=abs(theta)))/(B+1). Holm adjusts over the3 horizons separately within each input/backbone/setting/contrast/metric family, without combining unrelated families.',
 'MSE differences are accumulated in float64 from saved float32 predictions; tiny differences from subtracting the displayed formal float32 pooled-MSE means are rounding only. Negative MSE differences favor the first arm; positive Rank IC/direction differences favor it. '+('All paired Rank IC seed/date cells are finite.' if not missing else 'Undefined Rank IC dates use paired-valid per-seed means while retaining the original calendar and common blocks; see the BLOCKING note.'),
 '[Full162-row bootstrap table](paired_seed_date_bootstrap.csv), [paired seed/date differences](paired_seed_date_differences.csv), [seed wins](seed_wins.csv), [exact resampling indices](bootstrap_draws.csv), and `bootstrap/paired_distributions.npz` retain every requested statistic and draws.',
 dataframe_table(J[['input_set','backbone','setting','horizon','contrast','metric','theta','ci_lo','ci_hi','p_centered','p_holm_over_h']]),
 '## Artifacts, execution and verification',
 'There are60 expert-cache sets,330 unique Fixed/routing/head cells and990 unique prediction Parquets; each contains ticker,anchor_date,split,y_true,y_pred plus8 Daily/Weekly weight columns. The display includes30 additional secondary references to existing primary Fixed outputs. [File list](prediction_files.csv), [row/key/label checks](prediction_integrity.csv), [source checkpoint protection](protected_checkpoints.csv), [training parameter names and completion hashes](backbone_registry.json). Routing/head checkpoints contain the restored best trainable tensors, both constant-input branch vectors where applicable, and a SHA256 binding to the immutable backbone and cache. They do not duplicate or alter the source backbones.',
 'Exactly one numerical worker was used; the supervisor and caffeinate are control processes. The job used the intended fourth slot and did not launch or change14c. The initial process snapshot records that no14c training worker was observed at this job start; later background load is telemetry, not a claimed numerical cause. Telemetry records12/12 counts, environments, power and load at start/every5minutes/end. caffeinate -i covers the queue and reporting. Immutable completion records and per-result status snapshots support resume by hash verification. All new scripts are in this output namespace. No new point-in-time data claim is made.'
 ]
 text(OUT/'REPORT_STEP15.md',(chr(10)*2).join(report)+chr(10))
 files=[artifact(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ['.log','.jsonl']]
 jout(OUT/'FINAL_VALIDATION.json',dict(utc=utc(),backbones=60,non14a_replay_exact=17,unique_cells=330,summary_rows=len(S),five_seed_groups_complete=True,bootstrap_rows=len(J),B=B,L='h',source_checkpoints_unchanged=True,prediction_integrity_rows=len(predchecks),paired_router_initializations=True,worker_profile='12/12',report_profile=[torch.get_num_threads(),torch.get_num_interop_threads()],OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),report_wall_seconds=time.perf_counter()-clock,n_missing_rank_ic_support_families=len(missing)))
 jout(OUT/'MANIFEST_STEP15.json',dict(utc=utc(),files=files,versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__),report='REPORT_STEP15.md',validation='FINAL_VALIDATION.json',active_logs_and_manifest_self_excluded=True))
 print('REPORT_COMPLETE',len(S),len(J),time.perf_counter()-clock,flush=True)
if __name__=='__main__':main()
