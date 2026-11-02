"""Step17 TEST evaluation under protocol+explicit pre-TEST amendment; no CW evaluation."""
from common17 import *
import ast,inspect,textwrap
# Import/reuse only Step5's original scoring prefix. Stop before the CW block: no CW is re-evaluated.
source=inspect.getsource(cell_metrics);fun=ast.parse(textwrap.dedent(source)).body[0];prefix=[]
for node in fun.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='b' for t in node.targets):break
 prefix.append(node)
SCORE_CODE=compile(ast.Module(body=prefix,type_ignores=[]),inspect.getsourcefile(cell_metrics),'exec')
SCORE_TEXT='\n'.join(ast.get_source_segment(source,n) for n in prefix)
def scores(y,p,m):
 refs={'global_train_mean':np.full(len(y),m),'zero':np.zeros(len(y))};env=dict(np=np,y=y,p=p,refs=refs,REFS=['global_train_mean','zero']);exec(SCORE_CODE,env);return env['r']
def read(name):return pd.read_csv(name,float_precision='round_trip')
def cw_registry(cells):
 BF=read(A16/'clarified/B_forecast_skill_final.csv');EN=read(A16/'clarified/B_ensemble_skill.csv');RC=read(A16/'C_rolling_CW.csv');PC=read(A16/'C_rolling_pooled_CW.csv');rows=[]
 for r in cells.itertuples():
  z=None;src='';scope='';rawnw=rawmbb=qnw=qmbb=np.nan
  if r.window=='W3' and r.member!='ensemble':
   seed=42 if r.model=='ridge' else int(r.member);g=BF[(BF.window==r.window)&(BF.input_set==r.input_set)&(BF.model==r.model)&(BF.config=='daily_weekly')&(BF.h==r.h)&(BF.seed==seed)&BF.group.isin(['core','neural','ridge'])];assert len(g)==1,r;z=g.iloc[0];src=rel(A16/'clarified/B_forecast_skill_final.csv');scope='Step16 single-run all738 cells, separate original CW-NW/MBB families'
  elif r.window=='W3' and r.member=='ensemble':
   g=EN[(EN.input_set==r.input_set)&(EN.model==r.model)&(EN.h==r.h)]
   if len(g)==1:z=g.iloc[0];src=rel(A16/'clarified/B_ensemble_skill.csv');scope='Step16 ensemble all30 cells, separate original CW-NW/MBB families'
  if z is not None:
   rawnw=z.cw_p_nw_one_sided;rawmbb=z.cw_p_mbb_one_sided;qnw=z.cw_p_nw_one_sided_holm_all_cells;qmbb=z.cw_p_mbb_one_sided_holm_all_cells
  elif r.member=='ensemble' and r.model in ['static','adaptive']:
   g=PC[(PC.window==r.window)&(PC.input_set==r.input_set)&(PC.model==r.model)&(PC.h==r.h)&(PC.reference=='global_train_mean')];nw=g[g.method=='Newey_West'];mb=g[g.method=='date_only'];assert len(nw)==len(mb)==1,r
   rawnw=nw.iloc[0].p_value;qnw=nw.iloc[0].p_holm_3h;rawmbb=mb.iloc[0].p_value;qmbb=mb.iloc[0].p_holm_3h;src=rel(A16/'C_rolling_pooled_CW.csv');scope='Step16 pooled-seed CW (linear in forecast, hence ensemble CW); original3h families perwindow/input/model/method; conditional date-only MBB'
  elif r.window!='W3' and r.member!='ensemble':
   seed=42 if r.model=='ridge' else int(r.member);g=RC[(RC.window==r.window)&(RC.input_set==r.input_set)&(RC.model==r.model)&(RC.h==r.h)&(RC.seed==seed)&(RC.reference=='global_train_mean')];assert len(g)==1,r;z=g.iloc[0];rawnw=z.p_nw;rawmbb=z.p_mbb;qnw=z.p_nw_holm_3h;qmbb=z.p_mbb_holm_3h;src=rel(A16/'C_rolling_CW.csv');scope='Step16 rolling per-seed CW; original3h families perwindow/input/model/seed/reference/method'
  status='copied_Step16' if src else 'NOT FOUND in Step16: no O Single/Fixed ensemble CW reference; not recomputed'
  rows.append(dict(cell_id=r.cell_id,cw_p_nw_raw=rawnw,cw_p_mbb_raw=rawmbb,cw_p_nw_holm_Step16=qnw,cw_p_mbb_holm_Step16=qmbb,cw_reference_source=src,cw_original_family=scope,cw_reference_status=status))
 return pd.DataFrame(rows)
@lru_cache(None)
def boot_indices(w,h):
 entropy=[BASE,17,int(w[1:]),int(h)];ix=mbb_index(391,h,B,np.random.default_rng(np.random.SeedSequence(entropy))).astype(np.int32);buf=io.BytesIO();np.savez_compressed(buf,date_indices=ix);put(f'rng/{w}_h{h}.npz',buf.getvalue());return ix

def run():
 if verified('COMPUTE_COMPLETE.json'):return
 assert verified('SELECTION_COMPLETE.json');amend=OUT/'PROTOCOL_STEP17_AMENDMENT_01.md';assert amend.exists()
 for r in read(OUT/'sources.csv').itertuples():assert sha(ROOT/r.path)==r.sha256,r.path
 cw_paths=[A16/'clarified/B_forecast_skill_final.csv',A16/'clarified/B_ensemble_skill.csv',A16/'C_rolling_CW.csv',A16/'C_rolling_pooled_CW.csv'];cw_before={str(p):sha(p) for p in cw_paths};cells=read(OUT/'base_cells.csv').fillna('');selections=read(OUT/'selected_lambdas.csv');registry=read(OUT/'prediction_registry.csv');ref=cw_registry(cells);assert len(ref)==414
 # Freeze amendment, unchanged validation choice and all copied reference results before TEST skill arithmetic.
 cwpath=csv('CW_REFERENCE_REGISTRY.csv',ref)
 gate=dict(created_utc=utc(),protocol_sha256=sha(OUT/'PROTOCOL_STEP17.md'),amendment_sha256=sha(amend),selection_sha256=sha(OUT/'selected_lambdas.csv'),curves_sha256=sha(OUT/'validation_curves.csv'),CW_reference_registry_sha256=sha(cwpath),no_STEP17_TEST_metric_computed_yet=True,PRIMARY_family='414loss-MBBp; lambda0/missing rawNA, adjustment1',SECONDARY_family='414loss-MBBp; rawNA onlyif degenerate, adjustment1',new_CW_tests=0,CW_correction='copy originalStep16 fields only; noStep17 recorrection')
 if not (OUT/'PRETEST_GATE_FINAL.json').exists():js('PRETEST_GATE_FINAL.json',gate)
 else:
  g=json.loads((OUT/'PRETEST_GATE_FINAL.json').read_text());assert all(g[k]==gate[k] for k in ['protocol_sha256','amendment_sha256','selection_sha256','curves_sha256','CW_reference_registry_sha256'])
 if not (OUT/'TEST_COMPUTE_START.json').exists():js('TEST_COMPUTE_START.json',dict(started_utc=utc(),pretest_gate_sha256=sha(OUT/'PRETEST_GATE_FINAL.json'),no_training=True))
 CW=ref.set_index('cell_id');files=[cwpath];rows=[];dateparts=[];checks=[];baseline=[];selection_map=selections.set_index('group_id');pred_cache={};vlcurves=read(OUT/'validation_curves.csv')
 for r in cells[cells.member!='ensemble'].itertuples():pred_cache[r.cell_id]=pd.read_parquet(ROOT/r.canonical).y_pred.to_numpy(float)
 for gid,g in cells.groupby('group_id',sort=True):
  s=selection_map.loc[gid];w=s.window;h=int(s.h);inp=s.input_set;model=s.model;m=float(s.training_mean);d=split_labels(w,h,'test');y=d.y_true.to_numpy(float);Y=y.reshape(391,17);mse_m=scores(y,np.full(len(y),m),m)['mse'];members={r.member:r for r in g.itertuples()}
  if model!='ridge':
   pr=ensemble(np.stack([pred_cache[gid+'_'+str(seed)] for seed in SEEDS]),y);pred_cache[gid+'_ensemble']=pr;files.append(pq(f'inputs/test_ensembles/{gid}.parquet',d[['ticker','anchor_date','y_true']].assign(y_pred=pr)))
   p16=A16/f'F_ensembles/{inp}_{model}_h{h}_test.parquet'
   if w=='W3' and p16.exists():assert np.max(np.abs(pr-pd.read_parquet(p16).y_pred.to_numpy(float)))<1e-12,gid
  ix=boot_indices(w,h);l_primary=float(s.lambda_selected) if pd.notna(s.lambda_selected) else None
  for member,r in members.items():
   p=pred_cache[r.cell_id];base_ic=rank_ic_rows(p.reshape(Y.shape),Y);base_m=scores(y,p,m);baseline.append(dict(cell_id=r.cell_id,group_id=gid,window=w,input_set=inp,model=model,h=h,member=member,training_mean=m,original_mse=base_m['mse'],original_rank_ic=float(np.nanmean(base_ic)) if np.isfinite(base_ic).any() else np.nan,reference_mse_train_mean=mse_m))
   for regime,l in [('PRIMARY',l_primary),('SECONDARY',.5)]:
    cr=CW.loc[r.cell_id];meta=dict(regime=regime,cell_id=r.cell_id,group_id=gid,window=w,input_set=inp,model=model,h=h,member=member,seed=r.seed,lambda_selected_PRIMARY=l_primary,lambda_used=l,training_mean=m,n_test=6647,n_dates=391,n_model_seeds=1 if model=='ridge' else 5,validation_curve_file='validation_curves.csv',validation_curve_filter=r.cell_id,selection_status=s.status,cw_reference_source=cr.cw_reference_source,cw_original_family=cr.cw_original_family,L=h,B=B)
    if l is None:
     rows.append(dict(**meta,status='BLOCKING: missing exact-source VAL forecast; PRIMARY not evaluated',test_mse=np.nan,r2os_vs_global_train_mean=np.nan,r2os_vs_zero=np.nan,mean_rank_ic=np.nan,rank_ic_check='not evaluated',loss_difference=np.nan,loss_ci_lower_95=np.nan,loss_ci_upper_95=np.nan,loss_p_raw=np.nan,loss_p_for_Holm=1.,loss_test_status='not evaluated; missing validation',test_performed=False,cw_p_nw_raw=np.nan,cw_p_mbb_raw=np.nan,cw_p_nw_holm_Step16=np.nan,cw_p_mbb_holm_Step16=np.nan,cw_status='not applicable: lambda unavailable'));continue
    pr=combine(p,m,l);score=scores(y,pr,m);ic=rank_ic_rows(pr.reshape(Y.shape),Y);valid=np.isfinite(ic)&np.isfinite(base_ic);rankdelta=float(np.max(np.abs(ic[valid]-base_ic[valid]))) if valid.any() else np.nan
    same_pattern=np.array_equal(np.isfinite(ic),np.isfinite(base_ic));rankcheck='N/A: lambda0 constant Train-Mean' if l==0 else ('PASS' if same_pattern and (not valid.any() or rankdelta<1e-12) else 'FAILED numerical rank invariance')
    if l==0:
     loss=np.zeros(391);theta=0.;lo=hi=pv=np.nan;performed=False;loss_status='undefined: identical to Train-Mean; placeholder1 retained';cw_values={k:np.nan for k in ['cw_p_nw_raw','cw_p_mbb_raw','cw_p_nw_holm_Step16','cw_p_mbb_holm_Step16']};cwstatus='N/A: lambda0 constant Train-Mean'
    else:
     loss,_,_,_=ens_series(y,Y.shape,pr[None,:],np.full((1,len(y)),m),Y,np.array([1.]));theta=float(loss.mean())
     if np.all(loss==0):lo=hi=pv=np.nan;performed=False;loss_status='undefined: identical losses; placeholder1 retained'
     else:
      bs=series_stats(loss[ix].mean(1),theta);lo=bs['ci_lower_95'];hi=bs['ci_upper_95'];pv=bs['centered_p_two_sided'];performed=True;loss_status='two-sided date-level MBB, conditional on frozen forecasts/VAL selection'
     cw_values={k:cr[k] for k in ['cw_p_nw_raw','cw_p_mbb_raw','cw_p_nw_holm_Step16','cw_p_mbb_holm_Step16']};cwstatus='unchanged by shrinkage; corrected in Step16' if cr.cw_reference_status=='copied_Step16' else cr.cw_reference_status
    valrow=vlcurves[(vlcurves.cell_id==r.cell_id)&(vlcurves.lambda_value==l)];assert len(valrow)==1,(r.cell_id,l);val_mse=valrow.iloc[0].val_mse
    rows.append(dict(**meta,status='evaluated',val_mse_at_used_lambda=val_mse,test_mse=score['mse'],r2os_vs_global_train_mean=score['r2os_vs_global_train_mean'],r2os_vs_zero=score['r2os_vs_zero'],mean_rank_ic=float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan,original_rank_ic=float(np.nanmean(base_ic)) if np.isfinite(base_ic).any() else np.nan,rank_ic_max_abs_delta=rankdelta,rank_ic_check=rankcheck,loss_difference=theta,loss_ci_lower_95=lo,loss_ci_upper_95=hi,loss_p_raw=pv,loss_p_for_Holm=pv if np.isfinite(pv) else 1.,loss_test_status=loss_status,test_performed=performed,**cw_values,cw_status=cwstatus))
    path=pq(f'predictions/{regime}_{r.cell_id}_test.parquet',d[['ticker','anchor_date','y_true']].assign(split='test',y_pred=pr));files.append(path)
    dateparts.append(pd.DataFrame(dict(regime=regime,cell_id=r.cell_id,anchor_date=sorted(d.anchor_date.unique()),loss_difference=loss,rank_ic=ic,original_rank_ic=base_ic)))
    checks.append(dict(regime=regime,cell_id=r.cell_id,lambda_used=l,rank_ic_check=rankcheck,rank_ic_max_abs_delta=rankdelta,loss_mse_identity_abs_error=abs(theta-(score['mse']-mse_m)),lambda0_exact_prediction=bool(np.array_equal(pr,np.full(len(pr),m))) if l==0 else None,ensemble_weighting='importedStep7B ens_series; equalweight5 seeds' if member=='ensemble' else 'individual forecast'))
  print('STEP17 TEST',gid,'lambda',l_primary,flush=True)
 T=pd.DataFrame(rows);assert len(T)==828 and T.groupby('regime').size().eq(414).all();T['loss_p_Holm']=np.nan
 for regime,g in T.groupby('regime'):
  for index,pv in holm(dict(zip(g.index,g.loss_p_for_Holm))).items():T.loc[index,'loss_p_Holm']=pv
 files += [csv('test_results.csv',T),csv('original_forecast_scores.csv',pd.DataFrame(baseline)),csv('invariance_checks.csv',pd.DataFrame(checks)),csv('date_level_results.csv',pd.concat(dateparts,ignore_index=True))]
 ledger=T[['regime','cell_id','window','input_set','model','h','member','lambda_used','loss_p_raw','loss_p_for_Holm','loss_p_Holm','test_performed','loss_test_status']].copy();ledger['family']=ledger.regime+' all414loss-MBBcells';files.append(csv('TEST_LEDGER.csv',ledger))
 counts=[]
 for regime,g in T.groupby('regime'):
  counts.append(dict(regime=regime,planned_cells=len(g),evaluated_cells=int(g.status.eq('evaluated').sum()),blocked_cells=int(g.status.ne('evaluated').sum()),lambda0_cells=int(g.lambda_used.eq(0).sum()),valid_loss_MBB_tests=int(g.test_performed.sum()),placeholder_p1_cells=int(g.loss_p_raw.isna().sum()),positive_R2_vs_TrainMean=int(g.r2os_vs_global_train_mean.gt(0).sum()),positive_R2_vs_Zero=int(g.r2os_vs_zero.gt(0).sum()),loss_Holm_below_005=int(g.loss_p_Holm.lt(.05).sum()),CW_references_available=int(g.cw_status.eq('unchanged by shrinkage; corrected in Step16').sum()),CW_NOT_FOUND=int(g.cw_status.str.startswith('NOT FOUND').sum()),new_CW_tests=0,new_CW_corrections=0))
 files.append(csv('COUNTS.csv',pd.DataFrame(counts)))
 rng=[]
 for w in ['W1','W2','W3']:
  for h in HORIZONS:
   p=OUT/f'rng/{w}_h{h}.npz';files.append(p);rng.append(dict(window=w,h=h,B=B,L=h,entropy=[BASE,17,int(w[1:]),h],draws=rel(p),sha256=sha(p),index_dtype='int32, same integer values as imported mbb_index',block='overlapping noncircular, ceil(391/h), truncate391'))
 files.append(js('RNG_MANIFEST.json',dict(created_utc=utc(),bootstrap_rng=rng,model_seeds=SEEDS,statistics_threads=1,new_training_runs=0,new_inference_runs=0,new_CW_tests=0)))
 refsources=[]
 for p in [A16/'clarified/B_forecast_skill_final.csv',A16/'clarified/B_ensemble_skill.csv',A16/'C_rolling_CW.csv',A16/'C_rolling_pooled_CW.csv']:
  assert sha(p)==cw_before[str(p)],p;refsources.append(dict(path=rel(p),sha256=sha(p)))
 files.append(csv('CW_reference_sources.csv',pd.DataFrame(refsources)));put('SCORING_SOURCE_REUSE.py.txt',SCORE_TEXT)
 complete('COMPUTE_COMPLETE.json',files,dict(rows=828,pretest_gate_sha256=sha(OUT/'PRETEST_GATE_FINAL.json'),new_CW_tests=0));print('STEP17 COMPUTE COMPLETE',pd.DataFrame(counts).to_dict('records'),flush=True)
if __name__=='__main__':run()
