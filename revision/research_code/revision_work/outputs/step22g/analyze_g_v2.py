"""Post-barrier CS evaluation: imported Step22B2 metrics/Step16 inference and engine."""
import sys
sys.dont_write_bytecode=True
import io_g as io
sys.path.insert(0,str(io.ROOT/'revision_work/scripts/analysis16'));sys.path.append(str(io.ROOT/'revision_work/outputs/_deps_pyarrow15'))
for k in io.ENV_KEYS[:5]:io.os.environ[k]='1'
import numpy as np,pandas as pd

def main():
    finished=io.barrier()
    if (io.OUT/'ANALYSIS_COMPLETE.json').exists():io.verify_record(io.OUT/'ANALYSIS_COMPLETE.json');return
    start=io.utc();dest=io.target(io.OUT/'analysis'/('attempt_'+io.stamp()));dest.mkdir(parents=True);outputs=[];protected={}
    b=io.module('_22g_stats_b',io.OLD/'analyze22b_v3.py');s=b.s;table=b.table
    def protect(path,expected=None):
        path=io.Path(path);v=io.sha(path);assert expected is None or v==expected,'Changed source '+str(path);protected[io.rel(path)]=v;return path
    def save(name,frame):
        p=dest/name;io.csv(p,frame);outputs.append(p);return p
    def binary(name,**arrays):
        p=io.target(dest/name)
        with p.open('xb') as f:np.savez_compressed(f,**arrays)
        outputs.append(p);return p
    def rep(name,array):
        p=io.target(dest/'bootstrap'/name)
        with p.open('xb') as f:np.save(f,array,allow_pickle=False)
        outputs.append(p);return io.rel(p)
    panels={};frames={};metrics=[];routes=[];history=[];attempts=[];sources=[];ensembles=[];ensemble_frames={}
    for j in io.read(io.OUT/'GRID_PLAN.json')['jobs']:
        cp,r=io.completion(j);eps=list((io.OUT/'evaluation'/io.job_id(j)).glob('attempt_*/COMPLETE.json'));assert len(eps)==1;er=io.verify_record(eps[0]);folder=eps[0].parent
        mt=pd.read_csv(folder/'metrics.csv',float_precision='round_trip').iloc[0].to_dict();mt.update(arm=j['variant'],evidence='EXPLORATORY',r2os_cs_zero=mt['r2os']);metrics.append(mt)
        frames[j['variant'],j['h'],j['seed']]=s.canon(pd.read_csv(folder/'predictions_losses_weights.csv',float_precision='round_trip'))
        panels[j['variant'],j['h'],j['seed']]=pd.read_csv(folder/'date_metrics.csv',float_precision='round_trip').sort_values('anchor_date').reset_index(drop=True)
        hist=pd.read_csv(cp.parent/'history.csv',float_precision='round_trip');hist['arm']=j['variant'];hist['h']=j['h'];hist['seed']=j['seed'];hist['best_epoch']=r['best_epoch'];history.append(hist)
        if j['model'] in ['static','adaptive']:
            diag=pd.read_csv(folder/'router_diagnostics.csv',float_precision='round_trip');logits=io.read(cp.parent/'static_logits.json') if j['model']=='static' else {}
            for d in diag.to_dict('records'):
                d.update(arm=j['variant'],h=j['h'],seed=j['seed'],evidence='EXPLORATORY')
                ls=[v for k,v in logits.items() if d['branch'] in k and 'logit' in k]
                d['centered_logit_mad']=float(np.abs(np.asarray(ls[0])-np.mean(ls[0])).mean()) if len(ls)==1 else np.nan;routes.append(d)
        sources.append(dict(arm=j['variant'],h=j['h'],seed=j['seed'],predictions=io.rel(folder/'predictions_losses_weights.csv'),complete=io.rel(eps[0]),fit=io.rel(cp)))
        for ac,ar in io.attempts(j):attempts.append(dict(arm=j['variant'],h=j['h'],seed=j['seed'],status=ar['status'],epochs=ar.get('epochs'),best_epoch=ar.get('best_epoch'),started_utc=ar.get('started_utc'),ended_utc=ar.get('ended_utc'),elapsed_seconds=ar.get('elapsed_utc_seconds'),error=ar.get('error'),complete=io.rel(ac)))
    for h in io.HS:
        paths=list((io.OUT/'ridge'/f'h{h}').glob('attempt_*/COMPLETE.json'));assert len(paths)==1;io.verify_record(paths[0]);folder=paths[0].parent
        mt=pd.read_csv(folder/'metrics.csv',float_precision='round_trip').iloc[0].to_dict();mt.update(arm='Ridge-CS',evidence='EXPLORATORY descriptive only',r2os_cs_zero=mt['r2os']);metrics.append(mt)
        frames['Ridge-CS',h,-1]=s.canon(pd.read_csv(folder/'predictions_losses_weights.csv',float_precision='round_trip'));panels['Ridge-CS',h,-1]=pd.read_csv(folder/'date_metrics.csv',float_precision='round_trip').sort_values('anchor_date').reset_index(drop=True)
        sources.append(dict(arm='Ridge-CS',h=h,seed=-1,predictions=io.rel(folder/'predictions_losses_weights.csv'),complete=io.rel(paths[0]),fit=io.rel(folder/'best_model.pkl')))
    M=pd.DataFrame(metrics);assert len(M)==48
    summary=[]
    for (arm,h),g in M.groupby(['arm','h']):
        row=dict(arm=arm,h=h,n_seeds=len(g),evidence='EXPLORATORY descriptive' if arm=='Ridge-CS' else 'EXPLORATORY')
        for col in ['mse','mean_rank_ic','da','r2os_cs_zero']:
            values=g[col].to_numpy(float);row[col+'_mean']=float(np.mean(values)) if np.isfinite(values).all() else np.nan;row[col+'_sd']=float(np.std(values,ddof=1)) if len(g)==5 and np.isfinite(values).all() else np.nan
        summary.append(row)
    for arm in ['Static-CS','Adaptive-CS','LSTM-CS','Ridge-CS']:
        for h in io.HS:
            seeds=[-1] if arm=='Ridge-CS' else io.BOOT_SEEDS;fs=[frames[arm,h,seed] for seed in seeds];f=fs[0][['ticker','anchor_date','split','y_true','y_realized']].copy();f['y_pred']=np.mean([z.y_pred.to_numpy(float) for z in fs],axis=0);ensemble_frames[arm,h]=f
            y=f.y_true.to_numpy(float);raw=f.y_realized.to_numpy(float);pred=f.y_pred.to_numpy(float);valid=y!=0
            ic=s.rank_ic_rows(pred.reshape(391,17),raw.reshape(391,17));iccs=s.rank_ic_rows(pred.reshape(391,17),y.reshape(391,17));assert np.allclose(ic,iccs,equal_nan=True,atol=0,rtol=0)
            mse=float(np.mean((pred-y)**2));ensembles.append(dict(arm=arm,h=h,n_seeds=len(seeds),mse=mse,r2os_cs_zero=1-mse/float(np.mean(y*y)),mean_rank_ic=float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan,da=float(np.mean(np.sign(pred[valid])==np.sign(y[valid]))) if np.isfinite(pred).all() else np.nan,evidence='EXPLORATORY descriptive; equal-weight ensemble'))
            save(f'ensemble_predictions/{arm}_h{h}.csv',f)
    # Existing T0/T3 paths are read only now, after the45-fit barrier.
    complete=io.read(protect(io.B2RUN/'ANALYSIS_COMPLETE.json'));entry=next(a for a in complete['artifacts'] if a['path'].endswith('/prediction_files.csv'))
    registry=pd.read_csv(protect(io.ROOT/entry['path'],entry['sha256']));oldrank=[]
    for r in registry[(registry.setting.isin(['T0','T3']))&registry.model.isin(['static','adaptive','lstm'])].itertuples():
        ec=io.read(protect(io.ROOT/r.complete));expected=next(a['sha256'] for a in ec['artifacts'] if a['path']==r.predictions);protect(io.ROOT/r.predictions,expected)
        date_expected=next(a['sha256'] for a in ec['artifacts'] if a['path']==r.date_metrics)
        d=pd.read_csv(protect(io.ROOT/r.date_metrics,date_expected),float_precision='round_trip').sort_values('anchor_date').reset_index(drop=True);arm={'static':'Static','adaptive':'Adaptive','lstm':'LSTM'}[r.model]+'-'+r.setting;panels[arm,r.h,r.seed]=d
        oldrank.append(dict(arm=arm,h=r.h,seed=r.seed,mean_rank_ic=float(d.rank_ic.mean()),evidence='EXPLORATORY descriptive T0/T3 reference'))
    # Primary/BENCHMARK: fixed five-seed date means; NW primary, MBB robustness.
    ic_tests=[];disagreements=[]
    for h in io.HS:
        wt,di,si=s.draws(h,5,391);dates=s.dates(h,391,0);binary(f'draws/h{h}.npz',joint_seed_weights=wt,joint_date_indices=di,joint_seed_indices=si,date_only_indices=dates,seeds=np.array(io.BOOT_SEEDS))
        for arm in ['Static-CS','Adaptive-CS','LSTM-CS']:
            X=np.stack([panels[arm,h,seed].rank_ic.to_numpy(float) for seed in io.BOOT_SEEDS]);x=np.mean(X,axis=0);defined=bool(np.isfinite(x).all())
            mean=se=t=pnw=pmbb=lo=hi=np.nan;boot=np.full(20000,np.nan)
            if defined:
                mean,se,t,n=s.nw_t(x,h);pnw=float(2*s.stats.norm.sf(abs(t))) if np.isfinite(t) else np.nan;boot=x[dates].mean(axis=1);stats=s.series_stats(boot,mean);lo,hi=stats['ci_lower_95'],stats['ci_upper_95'];pmbb=stats['centered_p_two_sided']
            ic_tests.append(dict(family='BENCHMARK' if arm=='LSTM-CS' else 'PRIMARY',arm=arm,h=h,mean_rank_ic=mean,nw_se=se,nw_t=t,nw_lag=h,nw_ci95_lower=mean-1.959963984540054*se,nw_ci95_upper=mean+1.959963984540054*se,mbb_ci95_lower=lo,mbb_ci95_upper=hi,p_nw=pnw,p_mbb=pmbb,full_support=defined,n_finite_dates=int(np.isfinite(x).sum()),bootstrap=rep(f'IC_{arm}_h{h}.npy',boot),evidence='EXPLORATORY'))
    IC=pd.DataFrame(ic_tests)
    for family,g in IC.groupby('family'):
        assert len(g)==(3 if family=='BENCHMARK' else 6)
        for method in ['nw','mbb']:
            pcol='p_'+method;values={i:float(v) if np.isfinite(v) else 1. for i,v in g[pcol].items()}
            for i,v in s.holm(values).items():IC.loc[i,pcol+'_holm']=v
    IC['flag']=(IC.mean_rank_ic>0)&(IC.p_nw_holm<.05);IC['interpretation']=IC.apply(lambda r:'EXPLORATORY: with a cross-sectional target, the architecture\'s forecasts carry a statistically detectable ranking signal in the original window' if r.flag and r.family=='PRIMARY' else 'EXPLORATORY: benchmark ranking signal detected' if r.flag else 'not detected',axis=1)
    for r in IC.itertuples():
        if (r.p_nw_holm<.05)!=(r.p_mbb_holm<.05):disagreements.append(dict(family=r.family,arm=r.arm,h=r.h,nw_holm_p=r.p_nw_holm,mbb_holm_p=r.p_mbb_holm,evidence='EXPLORATORY'))
    specs=[('SECONDARY-A','Static-CS','Static-T3','rank_ic'),('SECONDARY-A','Adaptive-CS','Adaptive-T3','rank_ic'),('SECONDARY-B','Static-CS','Adaptive-CS','rank_ic'),('SECONDARY-B','Static-CS','Adaptive-CS','mse'),('SECONDARY-C','Adaptive-CS','LSTM-CS','rank_ic'),('SECONDARY-C','Static-CS','LSTM-CS','rank_ic')]
    rows=[];differences=[]
    for family,left,right,metric in specs:
        for h in io.HS:
            a=[panels[left,h,seed] for seed in io.BOOT_SEEDS];z=[panels[right,h,seed] for seed in io.BOOT_SEEDS];assert all(np.array_equal(d.anchor_date,a[0].anchor_date) for d in a+z)
            X=np.ascontiguousarray(np.stack([aa[metric].to_numpy(float)-zz[metric].to_numpy(float) for aa,zz in zip(a,z)]));defined=bool(np.isfinite(X).any(axis=1).all());theta=np.nan;boot=np.full(20000,np.nan)
            if defined:theta,boot,_=s.distributions(X,h);defined=bool(np.isfinite(theta) and np.isfinite(boot).all())
            stats=s.series_stats(boot,theta) if defined else dict(theta_hat=np.nan,ci_lower_95=np.nan,ci_upper_95=np.nan,centered_p_two_sided=np.nan)
            sd=np.nanmean(X,axis=1) if np.isfinite(X).any(axis=1).all() else np.full(5,np.nan);win=int((sd<0 if metric=='mse' else sd>0).sum()) if np.isfinite(sd).all() else np.nan;lose=int((sd>0 if metric=='mse' else sd<0).sum()) if np.isfinite(sd).all() else np.nan
            denominator=float(M[(M.arm=='Adaptive-CS')&(M.h==h)].mse.mean());upper=float(np.quantile(boot if metric=='mse' else -boot,.95,method='linear')) if defined and family=='SECONDARY-B' else np.nan
            rows.append(dict(family=family,contrast=left+'-minus-'+right,first=left,second=right,h=h,metric=metric,**stats,test_defined=defined,holm_input_p=stats['centered_p_two_sided'] if defined else 1.,seed_wins=win,seed_losses=lose,seed_ties=5-win-lose if np.isfinite(win) else np.nan,n_missing_seed_dates=int((~np.isfinite(X)).sum()),denominator_mean_cs_mse=denominator,estimate_pct=100*stats['theta_hat']/denominator if family=='SECONDARY-B' and metric=='mse' else np.nan,ci95_lower_pct=100*stats['ci_lower_95']/denominator if family=='SECONDARY-B' and metric=='mse' else np.nan,ci95_upper_pct=100*stats['ci_upper_95']/denominator if family=='SECONDARY-B' and metric=='mse' else np.nan,adaptive_improvement_upper95=upper,adaptive_mse_improvement_upper95_pct=100*upper/denominator if family=='SECONDARY-B' and metric=='mse' else np.nan,bootstrap=rep(f'{left}_minus_{right}_h{h}_{metric}.npy',boot),evidence='EXPLORATORY'))
            binary(f'inputs/{left}_minus_{right}_h{h}_{metric}.npz',differences=X,seeds=np.array(io.BOOT_SEEDS))
            for i,seed in enumerate(io.BOOT_SEEDS):
                for date,v in zip(a[0].anchor_date,X[i]):differences.append(dict(family=family,first=left,second=right,h=h,metric=metric,seed=seed,anchor_date=date,difference=v,evidence='EXPLORATORY'))
    J=s.corrected(pd.DataFrame(rows),['contrast','metric'],p='holm_input_p');assert len(J)==18
    J['direction_label']=J.apply(lambda r:'not significant' if not r.p_holm_3h<.05 or r.theta_hat==0 else 'significant improvement of the first arm' if (r.metric=='mse' and r.theta_hat<0) or (r.metric=='rank_ic' and r.theta_hat>0) else 'significant deterioration of the first arm',axis=1)
    J['interpretation']=J.apply(lambda r:'EXPLORATORY: the cross-sectional target improves ranking' if r.family=='SECONDARY-A' and r.direction_label=='significant improvement of the first arm' else 'EXPLORATORY: the cross-sectional target deteriorates ranking' if r.family=='SECONDARY-A' and r.direction_label=='significant deterioration of the first arm' else 'EXPLORATORY: '+r.direction_label,axis=1)
    # Descriptive top5 portfolios use only the original execution/cost/metric functions.
    engine=io.module('_22g_portfolio_engine',io.ROOT/'revision_work/scripts/portfolio/engine.py');C,Rdf=engine.load_prices();cal=list(Rdf.index);pos={d:i for i,d in enumerate(cal)};R=Rdf.to_numpy();rf,filled,missing=engine.load_rf(cal);assert missing==0
    for ticker in engine.TICKERS:protect(engine.RAW/f'{ticker}_daily.csv')
    protect(engine.RF);portfolio=[];dailyrows=[];weightrows=[];rebalances=[]
    for (arm,h),f in ensemble_frames.items():
        P=engine.pivot(f);anchors=list(P.index);reb=anchors[::h];execution=[pos[d]+1 for d in reb];end=min(execution[-1]+h,len(cal)-1)
        for mode in ['top5_long_only','top5_minus_bottom5']:
            note='EXPLORATORY descriptive only; long-short excludes borrowing costs and cash interest;10bp per unit full-book L1 turnover'
            if not np.isfinite(P.to_numpy(float)).all():
                portfolio.append(dict(arm=arm,h=h,portfolio=mode,cost_bp=10,gross_target=2 if mode=='top5_minus_bottom5' else 1,status='UNDEFINED',reason='Nonfinite forecast retained; portfolio undefined',note=note,evidence='EXPLORATORY descriptive only'))
                continue
            targets=[]
            for d in reb:
                score=P.loc[d].to_numpy(float)
                assert np.isfinite(score).all(),'Undefined ensemble cannot form descriptive portfolio: '+arm
                order=np.argsort(-score,kind='mergesort');w=np.zeros(17);w[order[:5]]=.2
                if mode=='top5_minus_bottom5':w[order[-5:]]=-.2
                targets.append(w);weightrows.append(dict(arm=arm,h=h,portfolio=mode,decision_date=d,execution_date=cal[pos[d]+1],gross_exposure=float(np.abs(w).sum()),net_exposure=float(w.sum()),**dict(zip(engine.TICKERS,w)),evidence='EXPLORATORY descriptive only'))
            days,gross,turn,period=engine.simulate(R,pos,execution,targets,end);net,pnet=engine.apply_costs(days,gross,execution,turn,period,.001);met=engine.metrics(net,pnet,turn,rf.to_numpy()[days],h)
            note='EXPLORATORY descriptive only; long-short excludes borrowing costs and cash interest;10bp per unit full-book L1 turnover'
            portfolio.append(dict(arm=arm,h=h,portfolio=mode,cost_bp=10,gross_target=2 if mode=='top5_minus_bottom5' else 1,status='COMPLETE',**met,note=note,evidence='EXPLORATORY descriptive only'))
            for d,g,n in zip(days,gross,net):dailyrows.append(dict(arm=arm,h=h,portfolio=mode,date=cal[d],gross_return=g,net_return=n,evidence='EXPLORATORY descriptive only'))
            for d,e,t,pg,pn in zip(reb,execution,turn,period,pnet):rebalances.append(dict(arm=arm,h=h,portfolio=mode,decision_date=d,execution_date=cal[e],turnover=t,cost_fraction=.001*t,gross_period_return=pg,net_period_return=pn,evidence='EXPLORATORY descriptive only'))
    R=pd.DataFrame(routes);RS=R.groupby(['arm','h','branch'],as_index=False).agg(n_seeds=('seed','nunique'),weight_mad=('mean_abs_deviation_from_equal',b.finite_mean),within_model_sd=('mean_sample_sd',b.finite_mean),between_date_share=('between_date_share',b.finite_mean),centered_logit_mad=('centered_logit_mad',b.finite_mean),n_observation_invariant=('constant_to_1e7','sum'));RS['evidence']='EXPLORATORY descriptive'
    ranks=pd.DataFrame(oldrank);ranks=pd.concat([ranks,pd.DataFrame([dict(arm=r['arm'],h=r['h'],seed=r['seed'],mean_rank_ic=r['mean_rank_ic'],evidence='EXPLORATORY CS') for r in metrics])],ignore_index=True)
    rank_summary=ranks.groupby(['arm','h'],as_index=False).agg(n_seeds=('seed','nunique'),rank_ic_mean=('mean_rank_ic','mean'),rank_ic_sd=('mean_rank_ic','std'));rank_summary['evidence']='EXPLORATORY descriptive'
    for name,data in [('test_metrics_per_seed.csv',M),('five_seed_summary.csv',pd.DataFrame(summary)),('ensemble_metrics.csv',pd.DataFrame(ensembles)),('IC_tests.csv',IC),('joint_bootstrap_tests.csv',J),('NW_MBB_disagreements.csv',pd.DataFrame(disagreements,columns=['family','arm','h','nw_holm_p','mbb_holm_p','evidence'])),('paired_seed_date_differences.csv',pd.DataFrame(differences)),('router_per_seed.csv',R),('router_summary.csv',RS),('rank_ic_T0_T3_CS.csv',rank_summary),('all_training_attempts.csv',pd.DataFrame(attempts)),('learning_curves.csv',pd.concat(history,ignore_index=True)),('prediction_sources.csv',pd.DataFrame(sources)),('portfolio_metrics.csv',pd.DataFrame(portfolio)),('portfolio_daily_returns.csv',pd.DataFrame(dailyrows)),('portfolio_weights.csv',pd.DataFrame(weightrows)),('portfolio_rebalances.csv',pd.DataFrame(rebalances))]:save(name,data)
    save('protected_sources.csv',pd.DataFrame([dict(path=p,sha256=h) for p,h in sorted(protected.items())]));assert all(io.sha(io.ROOT/p)==h for p,h in protected.items())
    report='# BLOCKING\n\nNone active. All45 neural cells and3 deterministic Ridge cells retained.\n\nProtocol SHA-256: `'+io.check_protocol()+'`.\n\nThis analysis was motivated by the positive Rank IC observed in earlier analyses of the same test window; all results are EXPLORATORY. W3 T0 remains primary.\n\n'
    report+='## Gates, timing and sigma_cs\n\n'+table(pd.DataFrame([io.read(io.OUT/'DATA_GATE.json')]).drop(columns=['prior_partition_label_proof','trainval_counts'],errors='ignore'))+'\n'+table(pd.read_csv(io.OUT/'sigma_cs.csv'))+'\n'+table(pd.DataFrame([io.read(io.OUT/'TIMING_GATE.json')]).drop(columns=['pilots'],errors='ignore'))+'\nTEST started after the45-fit barrier at '+finished['utc']+'. Queue caffeinate -dimsu and per-process12/12 telemetry are preserved in events/,fits/.\n\n'
    report+='## Five-seed metrics — EXPLORATORY\n\nMSE/R²/DA use float64 y_cs;Rank IC equals raw-realized-return IC. R² benchmark is zero. Ridge is deterministic,n1,SD N/A.\n\n'+table(pd.DataFrame(summary))+'\n\n## Ensembles — EXPLORATORY descriptive\n\n'+table(pd.DataFrame(ensembles))+'\n\n'
    report+='## PRIMARY/BENCHMARK Rank IC — EXPLORATORY\n\nTwo-sidedNW(lagh) primary;MBB B20000,Lh robustness;separate6-/3-test Holm families per method;flag only positive mean andNW Holm<.05.\n\n'+table(IC)+'\n\n## NW/MBB disagreements\n\n'+(table(pd.DataFrame(disagreements)) if disagreements else 'None.\n')+'\n\n## SECONDARY contrasts — EXPLORATORY\n\nJoint paired seed×date bootstrap,Holm3h percontrast×metric;no NW contrasts. SECONDARY-B bounds are signed95th percentiles,negative values retained;MSE percentages use fixed Adaptive-CS meanCS-MSE.\n\n'+table(J)+'\n\n'
    report+='## T0/T3/CS Rank IC and routing — EXPLORATORY descriptive\n\n'+table(rank_summary)+'\n'+table(RS)+'\n\n## Portfolios — EXPLORATORY descriptive only, no tests\n\nTop5 long-only:+.2 each. Long-short:+.2 top5/−.2 bottom5,long100%,short100%,gross200%,net0. Long-short results exclude borrowing costs and cash interest. Original Step16 next-session execution/drift/cost/metrics used;10bp applies to full-book L1 turnover including initial turnover2. Sharpe retains the original engine rf convention.\n\n'+table(pd.DataFrame(portfolio))+'\n\n## Every training attempt and best epoch\n\n'+table(pd.DataFrame(attempts))+'\n\n## Files\n\n'+'\n'.join('- ['+p.name+']('+str(p)+')' for p in outputs if p.suffix=='.csv' and p.parent==dest)+'\n\nNo follow-up variant or selection was run. Checkpoints,predictions,losses,logits/routers,all bootstrap arrays/indices/matrices retained.\n'
    rp=io.OUT/'REPORT_STEP22G_FINAL.md';io.text(rp,report);outputs.append(rp)
    io.js(io.OUT/'ANALYSIS_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),started_utc=start,ended_utc=io.utc(),analysis_directory=io.rel(dest),neural_cells=45,ridge_cells=3,ic_tests=9,joint_tests=18,artifacts=[io.artifact(p) for p in outputs]))
    print('REPORT COMPLETE:',rp,flush=True)
if __name__=='__main__':main()
