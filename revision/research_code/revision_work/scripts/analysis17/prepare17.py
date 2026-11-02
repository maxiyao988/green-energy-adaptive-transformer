"""Source/key/label verification and VAL-only choice; never evaluates TEST skill."""
from common17 import *
def validation_source(r):
 p=ROOT/r['source'];inp,model,h,seed=r['input_set'],r['model'],int(r['h']),int(r['seed'])
 if r['window']!='W3':return p.with_name('val.parquet'),p,'B2/B2-supplement saved split pair'
 if inp=='O' and r['group']=='core' and p.suffix=='.csv':
  if model=='adaptive' and seed==42:
   val=ROOT/f'revision_work/outputs/repro_threads/preds/adaptive_dw_h{h}_s42_t12_val.parquet';test=val.with_name(f'adaptive_dw_h{h}_s42_t12_test.parquet')
  else:
   val=ROOT/f'revision_work/outputs/preds/{model}_dw_h{h}_s{seed}_val.parquet';test=val.with_name(f'{model}_dw_h{h}_s{seed}_test.parquet')
  return val,test,'Archived checkpoint split replay; Step3B12/12 recovery for O Adaptive42'
 if r['group']=='ridge' and inp=='O':return None,None,'NOT FOUND: archived O W3 Ridge exports TEST only; no fitted checkpoint or saved VAL forecast'
 if r['group']=='neural' and inp=='C' and seed==42:
  candidates=list((DAY/'runs'/f'B4_C_{model}_daily_weekly_h{h}_s42').glob('attempt_*/COMPLETE.json'));assert len(candidates)==1,(r,candidates);q=candidates[0];j=json.loads(q.read_text());assert j['validated'];return q.parent/'val.parquet',q.parent/'test.parquet','14c-day12/12 source of B5 C seed42 copy'
 return p.with_name('val.parquet'),p,'Same saved B5FIX/B5/14a/14c-day split pair'
def protect(rows,p):
 p=Path(p);rows[str(p)]=dict(path=rel(p),sha256=sha(p))
def check(d,ref,name):
 assert len(d)==len(ref) and not d.duplicated(['ticker','anchor_date']).any(),name
 assert d[['ticker','anchor_date']].equals(ref[['ticker','anchor_date']]),name+' keys'
 assert np.array_equal(d.y_true.to_numpy(np.float32),ref.y_true.to_numpy(np.float32)),name+' labels'
 assert np.isfinite(d[['y_true','y_pred']]).all().all(),name+' finite'
def prepare():
 if verified('SELECTION_COMPLETE.json'):
  for r in pd.read_csv(OUT/'sources.csv').itertuples():assert sha(ROOT/r.path)==r.sha256,r.path
  return
 assert (OUT/'PROTOCOL_STEP17.md').exists();protocol_sha=sha(OUT/'PROTOCOL_STEP17.md');inventory=pd.read_csv(A16/'registry.csv').fillna('');R=inventory[((inventory.window=='W3')&(inventory.group.isin(['core','neural','ridge'])))|((inventory.window.isin(['W1','W2']))&(inventory.group=='rolling')&(inventory.model.isin(['static','adaptive','ridge'])))];R=R[R.config=='daily_weekly'].copy();assert len(R)==348
 protected={};checks=[];rows=[];files=[];issues=[]
 protect(protected,A16/'registry.csv');protect(protected,OUT/'PROTOCOL_STEP17.md')
 for r0 in R.to_dict('records'):
  r=dict(r0);canonical=ROOT/r['canonical'];assert sha(ROOT/r['source'])==r['sha256'];protect(protected,ROOT/r['source']);protect(protected,canonical)
  d=canon(pd.read_parquet(canonical));ref=split_labels(r['window'],int(r['h']),'test');check(d,ref,r['id']);val,test,note=validation_source(r);status='ready';delta=None
  if val is None or not val.exists() or test is None or not test.exists():status='missing_validation';issues.append(dict(id=r['id'],window=r['window'],input_set=r['input_set'],model=r['model'],h=int(r['h']),seed=int(r['seed']),reason=note))
  else:
   v=canon(pd.read_parquet(val));t=canon(pd.read_parquet(test));check(v,split_labels(r['window'],int(r['h']),'val'),r['id']+' VAL');check(t,ref,r['id']+' source TEST');delta=float(np.max(np.abs(t.y_pred.to_numpy(float)-d.y_pred.to_numpy(float))));assert delta<1e-7,(r['id'],'source mismatch',delta)
   protect(protected,val);protect(protected,test);files.append(pq(f'inputs/val/{r["id"]}.parquet',v[['ticker','anchor_date','y_true','y_pred']]))
  r.update(val_source=rel(val) if val is not None else '',paired_test_source=rel(test) if test is not None else '',source_note=note,status=status,max_abs_delta_paired_test_vs_Prompt16=delta);rows.append(r);checks.append(dict(id=r['id'],window=r['window'],input_set=r['input_set'],model=r['model'],h=r['h'],seed=r['seed'],test_rows=len(d),val_rows=6290 if status=='ready' else None,keys_labels_finite_passed=True,val_status=status,max_abs_delta_paired_test=delta))
 R=pd.DataFrame(rows);groups=[];curves=[];selections=[];means=[];basecells=[]
 for w in ['W1','W2','W3']:
  for h in HORIZONS:
   lab=labels(w,h);tr=lab[lab.split=='train'];m=float(np.mean(tr.y_true.to_numpy(float)));means.append(dict(window=w,h=h,n_train=len(tr),training_mean=m,arithmetic='Prompt16 frozen TRAIN labels mean in float64'));protect(protected,A16/f'labels/{w}_h{h}.parquet')
 for (w,inp,model,h),g in R.groupby(['window','input_set','model','h'],sort=True):
  h=int(h);gid=group_id(w,inp,model,h);expected=[42] if model=='ridge' else SEEDS;assert set(g.seed)==set(expected);g=g.sort_values('seed');n=len(g);m=next(x['training_mean'] for x in means if x['window']==w and x['h']==h);vl=split_labels(w,h,'val');yv=vl.y_true.to_numpy(float);ready=g.status.eq('ready').all();preds={}
  for r in g.itertuples():
   member='deterministic' if model=='ridge' else str(r.seed);basecells.append(dict(cell_id=gid+'_'+member,group_id=gid,window=w,input_set=inp,model=model,h=h,member=member,seed=int(r.seed),canonical=r.canonical,source_id=r.id,val_canonical=rel(OUT/f'inputs/val/{r.id}.parquet') if r.status=='ready' else '',status=r.status))
   if r.status=='ready':preds[member]=pd.read_parquet(OUT/f'inputs/val/{r.id}.parquet').y_pred.to_numpy(float)
  if model!='ridge':
   basecells.append(dict(cell_id=gid+'_ensemble',group_id=gid,window=w,input_set=inp,model=model,h=h,member='ensemble',seed=-1,canonical='',source_id='',val_canonical='',status='ready' if ready else 'missing_validation'))
   if ready:preds['ensemble']=ensemble(np.stack([preds[str(s)] for s in SEEDS]),yv)
  chosen_member='deterministic' if model=='ridge' else 'ensemble';objective=[]
  for member in [str(s) for s in SEEDS]+['ensemble'] if model!='ridge' else ['deterministic']:
   for l in GRID:
    val_mse=float(np.mean((yv-combine(preds[member],m,float(l)))**2)) if member in preds else np.nan
    curves.append(dict(group_id=gid,cell_id=gid+'_'+member,window=w,input_set=inp,model=model,h=h,member=member,lambda_value=float(l),val_mse=val_mse,used_for_selection=member==chosen_member,training_mean=m,status='ready' if member in preds else 'missing_validation'))
    if member==chosen_member:objective.append(val_mse)
  if ready:
   ix=int(np.argmin(np.asarray(objective)));chosen=float(GRID[ix]);best=float(objective[ix]);ties=[float(GRID[i]) for i,x in enumerate(objective) if x==best]
  else:chosen=None;best=None;ties=[]
  selections.append(dict(group_id=gid,window=w,input_set=inp,model=model,h=h,lambda_selected=chosen,val_mse_at_selected=best,selection_forecast=chosen_member,n_model_seeds=n,training_mean=m,exact_tied_lambdas=json.dumps(ties),status='ready' if ready else 'missing_validation'))
 assert len(basecells)==414 and len(selections)==84 and len(curves)==4554
 for p in [ROOT/'scripts/python/panel_baseline_ridge.py',ROOT/'revision_work/scripts/skill/a2_skill.py',ROOT/'revision_work/scripts/skill/skill_common.py',ROOT/'revision_work/scripts/skill/a3_hierarchical.py',ROOT/'revision_work/scripts/ensemble/ensemble_robust.py']:
  protect(protected,p)
 files += [csv('prediction_registry.csv',R),csv('preflight.csv',pd.DataFrame(checks)),csv('base_cells.csv',pd.DataFrame(basecells)),csv('training_means.csv',pd.DataFrame(means)),csv('validation_curves.csv',pd.DataFrame(curves)),csv('selected_lambdas.csv',pd.DataFrame(selections)),csv('BLOCKING_inputs.csv',pd.DataFrame(issues)),csv('sources.csv',pd.DataFrame(protected.values()))]
 for r in protected.values():assert sha(ROOT/r['path'])==r['sha256'],r
 files.append(js('PRETEST_GATE.json',dict(created_utc=utc(),no_STEP17_TEST_metric_computed=True,protocol_path=rel(OUT/'PROTOCOL_STEP17.md'),protocol_sha256=protocol_sha,selection_sha256=sha(OUT/'selected_lambdas.csv'),val_curves_sha256=sha(OUT/'validation_curves.csv'),planned_base_cells=414,selection_groups=84,ready_selection_groups=int(pd.DataFrame(selections).status.eq('ready').sum()),blocked_selection_groups=int(pd.DataFrame(selections).status.ne('ready').sum()),missing_validation_cells=len(issues),all_available_keys_labels_counts_verified=True,primary_multiplicity='all3methods together;1242planned entries, unavailable raw p remainNA with correction placeholder1',RNG='not yet generated',new_training_runs=0)))
 complete('SELECTION_COMPLETE.json',files,dict(protocol_sha256=protocol_sha,only_validation_selection=True));print('PRE-TEST GATE WRITTEN',len(basecells),'forecasts',len(issues),'missing VAL',flush=True)
if __name__=='__main__':prepare()
