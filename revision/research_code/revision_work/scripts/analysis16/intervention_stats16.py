"""Step9 joint/per-seed bootstrap plus conditional date-only results; no inference here."""
from statistics16_v2 import s
from statistics16 import *
def run():
 if verified('H.COMPLETE.json'):return
 D=[];M=[];V=[];W=[]
 for inp in ['O','C']:
  for h in HS:
   for seed in SEEDS:
    folder=OUT/f'H/S2_{inp}_h{h}_s{seed}';assert verified(str(folder.relative_to(OUT)/'COMPLETE.json'))
    D.append(pd.read_csv(folder/'date_differences.csv'));M.append(pd.read_csv(folder/'metrics.csv'));W.append(pd.read_csv(folder/'routers.csv'));V.append(json.loads((folder/'I0_verification.json').read_text()))
 D=pd.concat(D,ignore_index=True);M=pd.concat(M,ignore_index=True);rows=[]
 for (inp,h,iv),g in D[D.intervention.ne('I0_learned')].groupby(['input_set','h','intervention']):
  for metric,col in [('squared_loss','loss_difference'),('rank_ic','rank_ic_difference')]:
   X=g.pivot(index='seed',columns='anchor_date',values=col).sort_index().to_numpy(float);assert X.shape==(5,391)
   rows+=bootrow(X,h,section='H',input_set=inp,contrast=iv+'−I0',metric=metric)
   for i,seed in enumerate(sorted(SEEDS)):
    v=X[i];bb=np.nanmean(v[dates(h)],axis=1);rows.append(dict(section='H',input_set=inp,h=h,contrast=iv+'−I0',metric=metric,method='per_seed',seed=seed,n_seeds=1,n_dates=391,L=h,B=B,seed_wins=int(np.nanmean(v)>0 if metric=='rank_ic' else np.nanmean(v)<0),seed_delta_mean=float(np.nanmean(v)),seed_delta_sd=np.nan,**series_stats(bb,float(np.nanmean(v)))))
 T=pd.DataFrame(rows);T['seed_family']=T.seed.fillna(-1) if 'seed' in T else -1
 T=corrected(T,['input_set','contrast','metric','method','seed_family']);summary=M.groupby(['input_set','split','h','intervention'],as_index=False).agg(n_seeds=('seed','nunique'),mse_mean=('mse','mean'),mse_sd=('mse','std'),rank_ic_mean=('mean_rank_ic','mean'),rank_ic_sd=('mean_rank_ic','std'),delta_mse_mean=('delta_mse','mean'),delta_mse_sd=('delta_mse','std'),mse_seed_wins=('delta_mse',lambda v:int((v<0).sum())),delta_rank_ic_mean=('delta_rank_ic','mean'),delta_rank_ic_sd=('delta_rank_ic','std'),rank_ic_seed_wins=('delta_rank_ic',lambda v:int((v>0).sum())))
 files=[csv('H_I0_verification.csv',pd.DataFrame(V)),csv('H_intervention_metrics.csv',M),csv('H_seed_summary.csv',summary),csv('H_bootstrap_tests.csv',T),csv('H_date_differences.csv',D),csv('H_router_diagnostics.csv',pd.concat(W,ignore_index=True))];complete('H.COMPLETE.json',files,dict(tests=len(T),I0_EXACT=len(V)))
if __name__=='__main__':run()
