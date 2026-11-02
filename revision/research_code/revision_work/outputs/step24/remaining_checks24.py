"""Read-only intervention, diagnostics, DA, volatility, tuning, shrinkage and bound audit."""
from review24_common import *
import independent24 as ind
import ast,itertools
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
C=[]
def ck(name,a,b,tol=1e-10,**meta):
 a=np.asarray(a,float);b=np.asarray(b,float);same=a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b));d=float(np.nanmax(np.abs(a-b))) if np.isfinite(a-b).any() else 0.;C.append(dict(check=name,passed=bool(same and d<=tol),max_abs_difference=d,tolerance=tol,**meta))
A=ROOT/'revision_work/outputs/analysis16';R=df(A/'registry.csv');bindings=js('revision_work/outputs/step22a/BINDINGS.json')
# Isolate the pure intervention transformations, without importing the production runner.
p='revision_work/scripts/interventions/router_interventions.py';tr=ast.parse(read(p));fn=next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='make_fns');D=['daily_w_'+str(i) for i in [5,10,20,30]];W=['weekly_w_'+str(i) for i in [2,4,8,13]];ns=dict(np=np,pd=pd,torch=torch,DAILY=D,WEEKLY=W);exec(compile(ast.Module(body=[fn],type_ignores=[]),p,'exec'),ns)
rng=np.random.default_rng(20261022);learn={b:rng.dirichlet(np.ones(4),size=12).astype('float32') for b in ['daily','weekly']};ti=np.tile(np.arange(3),4);di=np.repeat(np.arange(4),3);means={b:learn[b].mean(0) for b in learn};tm={b:np.stack([learn[b][ti==i].mean(0) for i in range(3)]) for b in learn};F=ns['make_fns'](learn,ti,di,means,tm);ix=np.arange(12);ck('I0 returns unmodified function sentinel',int(F['I0_learned'] is None),1,0)
for b in learn:
 w=torch.from_numpy(learn[b])
 for name in F:
  if F[name] is None:continue
  actual=F[name](w,ix,b).numpy()
  if name=='I1_equal':expect=np.full_like(learn[b],.25)
  elif name=='I2_global_train_mean':expect=np.broadcast_to(means[b],(12,4))
  elif name=='I3_ticker_train_mean':expect=tm[b][ti]
  elif name=='I4_date_mean':expect=np.stack([learn[b][di==d].mean(0) for d in di])
  elif name.startswith('I6'):expect=np.full_like(learn[b],.25) if b in name else learn[b]
  else:
   cols=D if b=='daily' else W;short=[x.replace('_w_','') for x in cols];s=name[8:];expect=learn[b].copy()
   if s in short:expect[:,short.index(s)]=0;expect/=expect.sum(1,keepdims=True)
  ck('Intervention transform '+name,actual,expect,tol=1e-7,branch=b)
  ck('Intervention simplex '+name,actual.sum(1),np.ones(12),tol=2e-7,branch=b)
# Every I0 saved forecast, checked directly against registered formal predictions.
for spec in [b for b in bindings if b['backbone']=='S2']:
 ident=spec['id'];v=js(A/f'H/{ident}/I0_verification.json');p=canon(pq(A/f'H/{ident}/predictions/I0_learned_test.parquet'));q=R[R.group.eq('core')&R.model.eq('adaptive')&R.input_set.eq(spec['input_set'])&R.h.eq(spec['horizon'])&R.seed.eq(spec['seed'])];assert len(q)==1;z=canon(pq(ROOT/q.iloc[0].canonical))
 ck('I0 formal replay all rows',p.y_pred,z.y_pred,tol=1e-7,id=ident);ck('I0 recorded full/cache delta',[v['max_abs_delta_pred'],v['cache_i0_max_delta']],[0,0],tol=1e-7,id=ident)
 ck('I0 key equality',int(p[['anchor_date','ticker']].equals(z[['anchor_date','ticker']])),1,0,id=ident)
# Independent entropy / TV / effective-scale diagnostics on the original stored weights.
base='dataset/audit/experiment5_router_diagnostics/';samples=df(base+'experiment5_router_sample_metrics.csv');summ=df(base+'experiment5_router_cell_summary.csv');cross=df(base+'experiment5_cross_seed_stability_by_sample.csv') if (ROOT/base/'experiment5_cross_seed_stability_by_sample.csv').exists() else None
for (h,seed,b),g in samples.groupby(['horizon','seed','frequency']):
 w=g[['weight_1','weight_2','weight_3','weight_4']].to_numpy(float);center=w.mean(0);tv=.5*np.abs(w-center).sum(1);entropy=-(w*np.log(w)).sum(1);eff=np.exp(entropy);q=summ[summ.horizon.eq(h)&summ.seed.eq(seed)&summ.frequency.eq(b)].iloc[0]
 for key,a in [('sample_dependence_index',tv.mean()),('normalized_entropy_mean',(entropy/np.log(4)).mean()),('effective_scales_mean',eff.mean()),('max_weight_mean',w.max(1).mean())]:
  if key in q:ck('Router '+key,a,q[key],h=h,seed=seed,branch=b)
 for key,group in [('within_ticker_temporal_tv','ticker'),('within_date_crosssectional_tv','anchor_date')]:
  mu=g.groupby(group)[['weight_1','weight_2','weight_3','weight_4']].transform('mean').to_numpy(float);val=(.5*np.abs(w-mu).sum(1)).mean();ck('Router '+key,val,q[key],h=h,seed=seed,branch=b)
 ck('Sample entropy',entropy/np.log(4),g.normalized_entropy,h=h,seed=seed,branch=b);ck('Sample effective scales',eff,g.effective_scales,h=h,seed=seed,branch=b)
# All Step21 DA rows, including S1 head-retrained sensitivity; no zero-return substitution.
DA=df('revision_work/outputs/step21/all_da_per_seed.csv')
for r in DA.itertuples():
 f=pq(ROOT/r.prediction_file) if str(r.prediction_file).endswith('.parquet') else df(ROOT/r.prediction_file);y=f.y_true.to_numpy(float);p=f.y_pred.to_numpy(float);ok=y!=0;correct=int(np.sum(np.sign(p[ok])==np.sign(y[ok])))
 ck('DA zero exclusion and sign accuracy',[len(y),ok.sum(),(~ok).sum(),correct,correct/ok.sum()],[r.n_test_rows,r.n_valid_pairs,r.n_zero_realized_excluded,r.n_correct,r.da],cell=r.id)
# Volatility cutoffs use only TRAIN anchors, independently rebuilt from original daily closes.
prices={}
for ticker in sorted(R.ticker.unique()) if 'ticker' in R else ['AES','BEP','BLDP','BLNK','CSIQ','CWEN','DQ','ENPH','FCEL','FSLR','HASI','JKS','NEE','ORA','PLUG','RUN','SEDG']:
 f=df(ROOT/f'dataset/finance/panel_raw/{ticker}_daily.csv');f.datetime=pd.to_datetime(f.datetime);prices[ticker]=f.set_index('datetime').close
price=pd.DataFrame(prices).sort_index();vol=price.pct_change(fill_method=None).rolling(60,min_periods=60).std(ddof=1);cuts=df(A/'C_volatility_cutpoints.csv');labels={}
for r in cuts.itertuples():
 key=(r.window,r.h)
 if key not in labels:labels[key]=pq(A/f'labels/{r.window}_h{r.h}.parquet')
 f=labels[key];dates=pd.to_datetime(sorted(f[f.split.eq('train')&f.ticker.eq(r.ticker)].anchor_date.unique()));v=vol.loc[dates,r.ticker].to_numpy();lo,hi=np.quantile(v,[1/3,2/3]);ck('TRAIN-fixed volatility terciles',[lo,hi,len(v)],[r.q33,r.q67,r.n_train],window=r.window,h=r.h,ticker=r.ticker)
# Grid choice is validation-only; three complete seed-0 rates, selected minimum, ties lower rate.
trials=df('revision_work/outputs/step22a/tuning_trials.csv');sel=df('revision_work/outputs/step22a/selected_learning_rates.csv')
for r in sel.itertuples():
 q=trials[trials.input_set.eq(r.input_set)&trials.backbone.eq(r.backbone)&trials.h.eq(r.h)&trials.variant.eq(r.variant)];assert len(q)==3;best=q.sort_values(['validation_mse','lr']).iloc[0];ck('Frozen lr selection from seed0 VAL only',[r.lr,r.validation_mse],[best.lr,best.validation_mse],cell=str((r.input_set,r.backbone,r.h,r.variant)));ck('Tuning trial scope',[q.seed.max(),q.seed.min(),q.test_reads.max()],[0,0,0],cell=str(r.Index))
# Step17 selector: ensemble only (or deterministic Ridge), minimum VAL MSE, exact ties lower lambda.
curves=df('revision_work/outputs/analysis17/validation_curves.csv');sel=df('revision_work/outputs/analysis17/selected_lambdas.csv')
for r in sel.itertuples():
 q=curves[curves.group_id.eq(r.group_id)&curves.used_for_selection.eq(True)].sort_values(['val_mse','lambda_value']);z=q.iloc[0];ck('Shrinkage VAL minimum and smaller-lambda tie rule',[r.lambda_selected,r.val_mse_at_selected],[z.lambda_value,z.val_mse],cell=r.group_id)
tests=df('revision_work/outputs/analysis17/test_results.csv')
for regime,g in tests.groupby('regime'):
 ck('Shrinkage Holm full family',ind.holm(g.loss_p_for_Holm),g.loss_p_Holm,regime=regime);q=g[g.lambda_used.eq(0)];ck('Lambda0 retained conservative placeholders',q.loss_p_for_Holm,np.ones(len(q)),regime=regime)
 ck('Positive shrinkage IC invariance',g[g.lambda_used.gt(0)].mean_rank_ic,g[g.lambda_used.gt(0)].original_rank_ic,regime=regime)
# Every saved effect-bound replicate vector; signed q95 retains negative values.
bd=df('revision_work/outputs/step22c/runs/run_20261020T224500895768Z_52dab353/all_effect_bounds.csv')
for r in bd.itertuples():
 p=ROOT/r.replicate_file;read(p);z=np.load(p);lo,hi=np.quantile(z,[.025,.975]);up=np.quantile(z*r.adaptive_improvement_direction,.95);ck('Effect bounds signed percentile',[lo,hi,up],[r.ci95_lower_raw,r.ci95_upper_raw,r.adaptive_improvement_upper95_raw],cell=str(r.Index));ck('Effect bounds replicate count',len(z),20000,0,cell=str(r.Index))
 if r.metric=='mse':ck('Effect bounds fixed second-arm denominator',100*np.array([r.estimate_raw,lo,hi,up])/r.second_arm_mean_test_mse,[r.estimate_pct_second_mse,r.ci95_lower_pct_second_mse,r.ci95_upper_pct_second_mse,r.adaptive_mse_improvement_upper95_pct_second_mse],cell=str(r.Index))
write('REMAINING_CHECKS.csv',pd.DataFrame(C));source_manifest('SOURCES_REMAINING.csv');print('REMAINING',len(C),'failed',sum(not c['passed'] for c in C));print([c for c in C if not c['passed']][:12])
