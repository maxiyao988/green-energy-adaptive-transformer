"""No training: AST model counts, stored cross-seed metrics and effective Huber settings."""
from review24_common import *
import independent24 as ind
import ast,itertools,math
from scipy.stats import spearmanr
import torch
from torch import nn
torch.set_num_threads(1);torch.set_num_interop_threads(1)
C=[];P=[]
def ck(n,a,b,**meta):
 a=np.asarray(a,float);b=np.asarray(b,float);d=float(np.nanmax(abs(a-b))) if np.isfinite(a-b).any() else 0.;C.append(dict(check=n,passed=bool(a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)) and d<=1e-10),max_abs_difference=d,**meta))
ns=dict(torch=torch,nn=nn,np=np,D_MODEL=64,NHEAD=4,NUM_ENCODER_LAYERS=2,DIM_FEEDFORWARD=128,DROPOUT=.1,ACTIVATION='relu',FIXED_MAX_LEN=512,HIDDEN_SIZE=64,NUM_LAYERS=1,NUM_BLOCKS=2,WINDOW_CONFIG={'daily':{'window_size':10,'shift_size':5},'weekly':{'window_size':4,'shift_size':2}})
for p in ['scripts/python/panel_baseline_lstm.py','scripts/python/panel_baseline_vanilla_transformer.py','scripts/python/panel_baseline_swim.py']:
 tr=ast.parse(read(p));classes=[n for n in tr.body if isinstance(n,ast.ClassDef)];exec(compile(ast.Module(body=classes,type_ignores=[]),p,'exec'),ns)
for model,cls,expected in [('LSTM','DualFrequencyLSTM',44673),('Vanilla','DualFrequencyTransformerRegressor',142977),('SWiM','DualFrequencySWiMRegressor',142977)]:
 m=ns[cls](5,5);n=sum(p.numel() for p in m.parameters() if p.requires_grad);ck('Neural model parameter count',n,expected,model=model)
 if model=='SWiM':ck('SWiM branch window sizes',[m.daily_branch.window_size,m.daily_branch.shift_size,m.weekly_branch.window_size,m.weekly_branch.shift_size],[10,5,4,2],model=model)
# Cross-seed pairwise TV and modal-dominant agreement, all39882 sample/branch contexts.
base='dataset/audit/experiment5_router_diagnostics/';sample=df(base+'experiment5_router_sample_metrics.csv');stored=df(base+'experiment5_cross_seed_stability_by_sample.csv');assoc=df(base+'experiment5_loss_association_by_seed.csv');wc=['weight_1','weight_2','weight_3','weight_4']
for (h,b),g in sample.groupby(['horizon','frequency']):
 seeds=sorted(g.seed.unique());arrays=[g[g.seed.eq(s)].sort_values(['anchor_date','ticker'])[wc].to_numpy(float) for s in seeds];W=np.stack(arrays);tv=np.stack([.5*np.abs(W[i]-W[j]).sum(1) for i,j in itertools.combinations(range(5),2)]);dominant=W.argmax(2);agreement=np.stack([(dominant==k).sum(0) for k in range(4)]).max(0)/5;z=stored[stored.horizon.eq(h)&stored.frequency.eq(b)].sort_values(['anchor_date','ticker'])
 ck('Cross-seed all10 pair TV',tv.mean(0),z.mean_pairwise_seed_tv,h=h,branch=b);ck('Cross-seed maximum pair TV',tv.max(0),z.max_pairwise_seed_tv,h=h,branch=b);ck('Dominant modal agreement',agreement,z.dominant_agreement_share,h=h,branch=b);ck('Dominant unanimous share sample',(agreement==1).astype(int),z.dominant_unanimous,h=h,branch=b)
for (h,s,b),g in sample.groupby(['horizon','seed','frequency']):
 z=assoc[assoc.horizon.eq(h)&assoc.seed.eq(s)&assoc.frequency.eq(b)].iloc[0];rho=spearmanr(g.tv_to_static,g.loss_diff).statistic;date=g.groupby('anchor_date')[['tv_to_static','loss_diff']].mean();rd=spearmanr(date.tv_to_static,date.loss_diff).statistic;ck('Sample/date Spearman TV vs loss difference',[rho,rd],[z.sample_level_spearman_rho,z.date_level_spearman_rho],h=h,seed=s,branch=b)
# Effective Huber delta actually saved per epoch, compared to TRAIN-only source scales.
R=ROOT/'revision_work/outputs';sg=js(R/'step22g/SIGMAS_CS.json');sw=js(R/'step22d/session_v3/SIGMAS.json');fixed={5:.098083,10:.138578,20:.195560}
records=df(OUT/'TRAINING_RECORDS_REVIEWED.csv')
for r in records.itertuples():
 p=ROOT/r.record;j=js(p);job=j.get('job',{});hp=p.parent/'history.csv';f=df(hp)
 if 'huber_delta' not in f:continue
 h=int(r.h) if np.isfinite(r.h) else int(job.get('h',j.get('horizon')));setting=job.get('setting',j.get('setting'));step=r.step
 if step=='step22a':expect=1.
 elif step=='step22g':expect=float(sg['values'][str(h)])
 elif step in ['step22d/session_v3','step22e']:
  w=job.get('window',j.get('window'));expect=1. if setting=='T0' else float(sw['values'][w][str(h)])
 elif step=='step22f':expect=1.
 else:expect=1. if setting in ['T0','T1'] else fixed[h]
 ck('Stored effective Huber delta each epoch',f.huber_delta,np.full(len(f),expect),record=r.record)
# Stored H routing SD and between-date share, with float64 alternative precision explicitly disclosed.
diag=df('revision_work/outputs/analysis16/H_router_diagnostics.csv');bind=js('revision_work/outputs/step22a/BINDINGS.json');specs={b['id']:b for b in bind};cols={'daily':['daily_w_'+str(n) for n in [5,10,20,30]],'weekly':['weekly_w_'+str(n) for n in [2,4,8,13]]};cache={}
for r in diag.itertuples():
 if r.id not in cache:cache[r.id]=pq(ROOT/specs[r.id]['cache']['test']['keys'])
 f=cache[r.id];w=f[cols[r.branch]].to_numpy(np.float32);mean=pd.DataFrame(w).groupby(f.anchor_date.to_numpy()).transform('mean').to_numpy();share=float(pd.DataFrame(mean).var().sum()/pd.DataFrame(w).var().sum());sd=float(w.std(0).mean());ck('H routing SD recorded float32',sd,r.test_mean_sd_across_samples,id=r.id,branch=r.branch);ck('H between-date variance share',share,r.share_variance_between_dates,id=r.id,branch=r.branch)
 alt=float(w.astype(float).std(0).mean());P.append(dict(id=r.id,branch=r.branch,stored_float32_sd=sd,alternative_float64_sd=alt,absolute_difference=abs(alt-sd),exceeds_1e10=abs(alt-sd)>1e-10,interpretation='different accumulation precision; disclosed, not a formula change'))
write('DIAGNOSTICS_MODELS_SETTINGS_CHECKS.csv',pd.DataFrame(C));write('FLOAT32_ROUTER_PRECISION_DISCLOSURE.csv',pd.DataFrame(P));source_manifest('SOURCES_DIAGNOSTICS_MODELS_SETTINGS.csv');print('DIAGNOSTICS',len(C),'failed',sum(not c['passed'] for c in C));print([c for c in C if not c['passed']][:10])
