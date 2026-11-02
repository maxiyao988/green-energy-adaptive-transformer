"""Post-barrier Step22E endpoints; import Step22D metrics and stratified estimators."""
import sys
sys.dont_write_bytecode=True
import io_e as io
for k in io.ENV_KEYS[:5]:io.os.environ[k]='1'
sys.path.append(str(io.ROOT/'revision_work/outputs/_deps_pyarrow15'))
import numpy as np,pandas as pd
from reuse_d import statistics

def main():
    finished=io.barrier()
    if (io.OUT/'ANALYSIS_COMPLETE.json').exists():io.verify_record(io.OUT/'ANALYSIS_COMPLETE.json');return
    start=io.utc();dest=io.target(io.OUT/'analysis'/('attempt_'+io.stamp()));dest.mkdir(parents=True)
    G=statistics(dest);s=G['s'];save=G['save'];protect=G['protect'];readcsv=G['readcsv'];binary=G['binary'];replicates=G['replicates']
    per_seed=G['per_seed'];strict_mean=G['strict_mean'];series_result=G['series_result'];joint_boot=G['joint_boot'];table=G['table']
    labels={};keys={};reference=[];frames={};files=[];best=[];logits=[];limitations=[]
    for w in ['W1','W2','W3']:
        for h in io.HS:
            p=io.ROOT/f'revision_work/outputs/analysis16/labels/{w}_h{h}.parquet'
            f=pd.read_parquet(protect(p));f.anchor_date=pd.to_datetime(f.anchor_date).dt.strftime('%Y-%m-%d')
            z=s.canon(f[f.split=='test']);assert len(z)==6647 and z.groupby('anchor_date').size().eq(17).all()
            m=float(f.loc[f.split=='train','y_true'].to_numpy(float).mean());y=z.y_true.to_numpy(float)
            labels[w,h]=dict(y=y,m=m,gtm_mse=float(np.mean((y-m)**2)),zero_mse=float(np.mean(y*y)),dates=z.anchor_date.unique())
            keys[w,h]=z[['ticker','anchor_date','y_true']]
            reference.append(dict(window=w,h=h,n_train=int((f.split=='train').sum()),n_test=6647,n_dates=391,train_mean=m,train_mean_test_mse=labels[w,h]['gtm_mse']))
    G['all_labels']=labels
    assert len(set().union(*(set(labels[w,5]['dates']) for w in ['W1','W2','W3'])))==1173

    def find_fit(root,pattern):
        found=list((root/pattern).glob('attempt_*/COMPLETE.json'))
        good=[p for p in found if io.read(protect(p))['status']=='COMPLETE']
        assert len(good)==1,(root,pattern,len(good));return good[0]

    def add(setting,model,w,h,seed,path,kind,expected_sha=None,fit_cp=None,route_source=None):
        path=protect(io.ROOT/path if not isinstance(path,io.Path) else path,expected_sha)
        raw=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path,float_precision='round_trip')
        f=s.canon(raw);expected=keys[w,h]
        assert len(f)==6647 and f[['ticker','anchor_date']].equals(expected[['ticker','anchor_date']]),(setting,model,w,h,seed)
        assert np.array_equal(f.y_true.to_numpy(np.float32),expected.y_true.to_numpy(np.float32))
        key=(setting,model,w,h,int(seed));assert key not in frames
        frames[key]=f
        cp=protect(fit_cp) if fit_cp is not None else None
        record=io.read(cp) if cp is not None else {}
        best.append(dict(setting=setting,model=model,window=w,h=h,seed=int(seed),best_epoch=record.get('best_epoch'),epochs=record.get('epochs',record.get('epochs_trained')),fit_complete=io.rel(cp) if cp else 'NOT STORED',status='STORED' if record.get('best_epoch') is not None else 'NOT STORED'))
        files.append(dict(setting=setting,model=model,window=w,h=h,seed=int(seed),source=kind,predictions=io.rel(path),sha256=io.sha(path),fit_complete=io.rel(cp) if cp else 'NOT STORED'))
        if model=='static':
            lp=cp.parent/'static_logits.json' if cp else None
            blob=io.read(protect(lp)) if lp is not None and lp.exists() else {}
            routed=raw
            if route_source is not None:
                rp=protect(io.ROOT/route_source)
                routed=pd.read_parquet(rp) if rp.suffix=='.parquet' else pd.read_csv(rp,float_precision='round_trip')
                routed=s.canon(routed)
                assert routed[['ticker','anchor_date']].equals(expected[['ticker','anchor_date']])
            for branch,names in [('daily',io.WEIGHTS[:4]),('weekly',io.WEIGHTS[4:])]:
                options=[v for k,v in blob.items() if branch in k and 'logit' in k]
                a=np.asarray(options[0],float) if len(options)==1 else None
                if a is not None:assert a.shape==(4,) and np.isfinite(a).all()
                weights=routed[names].to_numpy(float) if all(n in routed for n in names) else None
                if weights is not None:
                    if not np.isfinite(weights).all():weights=None
                    else:assert np.ptp(weights,axis=0).max()<1e-6
                logits.append(dict(setting=setting,model=model,window=w,h=h,seed=int(seed),branch=branch,
                  centered_logit_mad=float(np.abs(a-a.mean()).mean()) if a is not None else np.nan,
                  softmax_weight_mad_from_quarter=float(np.abs(weights-.25).mean()) if weights is not None else np.nan,
                  logits_source=io.rel(lp) if a is not None else 'NOT STORED',weights_source=route_source or io.rel(path),
                  status='STORED' if a is not None and weights is not None else 'NOT STORED'))
                if a is None or weights is None:limitations.append(f'{setting}/static/{w}/h{h}/s{seed}/{branch}: logit or weight diagnostic NOT STORED')

    # New60 cells: strict record/artifact verification follows the all-training barrier.
    attempts=[]
    for job in io.read(io.OUT/'GRID_PLAN.json')['jobs']:
        cp,record=io.completion(job);assert cp is not None
        ec=list((io.OUT/'evaluation'/io.job_id(job)).glob('attempt_*/COMPLETE.json'));assert len(ec)==1
        er=io.verify_record(ec[0]);ep=ec[0].parent/'predictions_losses_weights.csv'
        add('T3',job['model'],job['window'],job['h'],job['seed'],ep,'Step22E new T3',fit_cp=cp)
        for ap,ar in io.attempts(job):
            attempts.append(dict(**job,status=ar['status'],epochs=ar.get('epochs'),best_epoch=ar.get('best_epoch'),elapsed_seconds=ar.get('elapsed_utc_seconds'),started_utc=ar.get('started_utc'),ended_utc=ar.get('ended_utc'),error=ar.get('error'),complete=io.rel(ap),adapter_complete=(ap.parent/'ADAPTER_COMPLETE.json').exists()))

    registry=readcsv(io.ROOT/'revision_work/outputs/analysis16/registry.csv').fillna('')
    checks=readcsv(io.ROOT/'revision_work/outputs/analysis16/prediction_checks.csv').set_index('id')
    old_sources=io.read(protect(io.OLD/'T0_SOURCES.json'))['sources']
    for r in registry[(registry.input_set=='C')&registry.model.isin(['static','adaptive'])&(registry.config=='daily_weekly')&(((registry['group']=='rolling')&registry.window.isin(['W1','W2']))|((registry['group']=='core')&(registry.window=='W3')))].itertuples():
        c=checks.loc[r.id];assert c.keys_equal and c.labels_float32_equal
        protect(io.ROOT/r.source,r.sha256)
        fit=(io.ROOT/r.source).parent/'COMPLETE.json'
        if not fit.exists():
            old=next(z for z in old_sources if z['model']==r.model and z['h']==r.h and z['seed']==r.seed)
            fit=io.ROOT/old['complete']
        add('T0',r.model,r.window,r.h,r.seed,io.ROOT/r.canonical,'Step16/Step22D T0',c.canonical_sha256,fit,r.source)

    # W3 T3 and all LSTM cells are reused, never retrained or re-inferred.
    b2=io.ROOT/'revision_work/outputs/step22b2';bc=io.read(protect(b2/'ANALYSIS_COMPLETE.json'))
    br=next(a for a in bc['artifacts'] if a['path'].endswith('/prediction_files.csv'))
    bf=readcsv(protect(io.ROOT/br['path'],br['sha256']))
    for r in bf[(bf.setting=='T3')&bf.model.isin(['static','adaptive'])].itertuples():
        ec=io.read(protect(io.ROOT/r.complete));expected=next(a['sha256'] for a in ec['artifacts'] if a['path']==r.predictions)
        fit=find_fit(b2/'fits',f'T3_{r.model}_h{r.h}_s{r.seed}')
        add('T3',r.model,'W3',r.h,r.seed,io.ROOT/r.predictions,'Step22B2 W3 T3',expected,fit)
    dc=io.read(protect(io.D/'ANALYSIS_COMPLETE.json'));dr=next(a for a in dc['artifacts'] if a['path'].endswith('/lstm_prediction_files.csv'))
    lf=readcsv(protect(io.ROOT/dr['path'],dr['sha256']));assert len(lf)==90
    for r in lf.itertuples():
        if r.window in ['W1','W2']:fit=find_fit(io.D/'fits',f'{r.setting}_lstm_{r.window}_h{r.h}_s{r.seed}')
        elif r.setting=='T3':fit=find_fit(b2/'fits',f'T3_lstm_h{r.h}_s{r.seed}')
        else:fit=io.ROOT/next(z['complete'] for z in old_sources if z['model']=='lstm' and z['h']==r.h and z['seed']==r.seed)
        add(r.setting,'lstm',r.window,r.h,r.seed,io.ROOT/r.predictions,'Step22D LSTM (including W3 sources)',r.sha256,fit)
    assert len(frames)==270
    save('prediction_sources.csv',pd.DataFrame(files));save('all_training_attempts.csv',pd.DataFrame(attempts));save('reference_values.csv',pd.DataFrame(reference));save('best_epochs_per_seed.csv',pd.DataFrame(best))

    # Metrics use the unchanged Step22D function definitions, with all five seeds retained.
    rows=[];ensrows=[];components={};ensemble_pred={};daily=[]
    for setting in ['T0','T3']:
        for model in ['static','adaptive','lstm']:
            for w in ['W1','W2','W3']:
                for h in io.HS:
                    ps=np.stack([frames[setting,model,w,h,seed].y_pred.to_numpy(float) for seed in io.SEEDS]);ls=[];ics=[]
                    for seed,p in zip(io.SEEDS,ps):
                        metrics,loss,ic=per_seed(p,w,h);rows.append(dict(setting=setting,model=model,window=w,h=h,seed=seed,**metrics));ls.append(loss);ics.append(ic)
                    components[setting,model,w,h]=dict(mse=np.stack(ls),rank_ic=np.stack(ics))
                    for i,date in enumerate(labels[w,h]['dates']):
                        for j,seed in enumerate(io.SEEDS):daily.append(dict(setting=setting,model=model,window=w,h=h,seed=seed,anchor_date=date,mse=ls[j][i],rank_ic=ics[j][i]))
                    ep=ps.mean(axis=0);ensemble_pred[setting,model,w,h]=ep
                    metric,_,_=per_seed(ep,w,h);ensrows.append(dict(setting=setting,model=model,window=w,h=h,**metric))
                    f=keys[w,h].copy();f['split']='test';f['y_pred']=ep;save(f'ensemble_predictions/{setting}_{model}_{w}_h{h}.csv',f)
    single=pd.DataFrame(rows);en=pd.DataFrame(ensrows)
    for ws in [['W1','W2'],['W1','W2','W3']]:
        for setting in ['T0','T3']:
            for model in ['static','adaptive','lstm']:
                for h in io.HS:
                    for seed in io.SEEDS+['ensemble']:
                        z=en if seed=='ensemble' else single[single.seed==seed]
                        z=z[(z.setting==setting)&(z.model==model)&(z.h==h)&z.window.isin(ws)];assert len(z)==len(ws)
                        mse=strict_mean(z.mse);r=dict(setting=setting,model=model,window='+'.join(ws),h=h,mse=mse,mean_rank_ic=strict_mean(z.mean_rank_ic),n_dates=391*len(ws),n_rows=6647*len(ws),gtm_mse=float(z.gtm_mse.mean()),zero_mse=float(z.zero_mse.mean()),n_valid_da=int(z.n_valid_da.sum()),n_correct_da=float(z.n_correct_da.sum()) if z.n_correct_da.notna().all() else np.nan)
                        r.update(da=r['n_correct_da']/r['n_valid_da'],r2os_train_mean=1-mse/r['gtm_mse'],r2os_zero=1-mse/r['zero_mse'])
                        (ensrows if seed=='ensemble' else rows).append(r if seed=='ensemble' else dict(seed=seed,**r))
    P=pd.DataFrame(rows);E=pd.DataFrame(ensrows);S=G['summary'](P,['setting','model','window','h'])
    assert len(P)==450 and len(E)==len(S)==90
    save('metrics_per_seed.csv',P);save('five_seed_summary.csv',S);save('ensemble_metrics.csv',E);save('date_metrics_per_seed.csv',pd.DataFrame(daily))
    B=pd.DataFrame(best);BS=B.groupby(['setting','model','window','h'],as_index=False).agg(n_seeds=('seed','nunique'),best_epoch_min=('best_epoch','min'),best_epoch_median=('best_epoch','median'),best_epoch_max=('best_epoch','max'),best_epoch_mean=('best_epoch','mean'),best_epoch_sd=('best_epoch','std'))
    save('best_epoch_distribution.csv',BS)
    L=pd.DataFrame(logits)
    pooled=[]
    for windows in [['W1','W2'],['W1','W2','W3']]:
        selected=L[L.window.isin(windows)]
        for (setting,h,seed,branch),g in selected.groupby(['setting','h','seed','branch']):
            assert len(g)==len(windows)
            pooled.append(dict(setting=setting,model='static',window='+'.join(windows),h=h,seed=seed,branch=branch,
                centered_logit_mad=strict_mean(g.centered_logit_mad),softmax_weight_mad_from_quarter=strict_mean(g.softmax_weight_mad_from_quarter),
                logits_source=';'.join(g.logits_source),weights_source=';'.join(g.weights_source),status='STORED' if g.status.eq('STORED').all() else 'NOT STORED'))
    L=pd.concat([L,pd.DataFrame(pooled)],ignore_index=True);assert len(L)==300
    save('static_logits_per_seed.csv',L)
    logsummary=L.groupby(['setting','window','h','branch'],as_index=False).agg(n_seeds=('seed','nunique'),centered_logit_mad_mean=('centered_logit_mad',lambda z:float(z.mean()) if len(z)==5 and z.notna().all() else np.nan),centered_logit_mad_sd=('centered_logit_mad',lambda z:float(z.std(ddof=1)) if len(z)==5 and z.notna().all() else np.nan),weight_mad_mean=('softmax_weight_mad_from_quarter',lambda z:float(z.mean()) if len(z)==5 and z.notna().all() else np.nan),weight_mad_sd=('softmax_weight_mad_from_quarter',lambda z:float(z.std(ddof=1)) if len(z)==5 and z.notna().all() else np.nan))
    save('static_logit_summary.csv',logsummary)

    # Exact Step5 CW formula imported from the Step22D source module's registered source.
    import ast
    tree=ast.parse((io.ROOT/'revision_work/scripts/skill/a2_skill.py').read_text())
    ast=__import__('ast');fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cell_metrics')
    assign=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='f' for t in n.targets))
    cwcode=compile(ast.Module(body=[assign],type_ignores=[]),str(io.ROOT/'revision_work/scripts/skill/a2_skill.py'),'exec')
    e1=[];e2=[];nw3=[]
    for model in ['static','adaptive']:
        for h in io.HS:
            for setting in ['T3','T0']:
                series={w:np.mean(components[setting,model,w,h]['rank_ic'],axis=0) for w in ['W1','W2']}
                binary(f'inputs/E1_{setting}_{model}_h{h}.npz',**series)
                r=series_result(series,['W1','W2'],h,h,tag=f'E1_{setting}_{model}_h{h}')
                e1.append(dict(family='E1 SECONDARY',role='SECONDARY' if setting=='T3' else 'T0 descriptive; no flag',setting=setting,model=model,h=h,window='W1+W2',**r))
            series={}
            for w in ['W1','W2']:
                ep=ensemble_pred['T3',model,w,h];y=labels[w,h]['y'];m=labels[w,h]['m']
                env=dict(y=y,b=np.full(len(y),m),p=ep);exec(cwcode,env)
                v=env['f'].reshape(391,17).mean(axis=1)
                if np.allclose(ep,m,rtol=1e-6,atol=1e-9) or not np.isfinite(ep).all():v[:]=np.nan
                series[w]=v
            binary(f'inputs/E2_{model}_h{h}.npz',**series)
            e2.append(dict(family='E2 CONFIRMATORY',model=model,h=h,window='W1+W2',**series_result(series,['W1','W2'],h,h-1,one_sided=True,tag=f'E2_{model}_h{h}')))
            three={w:np.mean(components['T3',model,w,h]['rank_ic'],axis=0) for w in ['W1','W2','W3']}
            # Reuse the NW estimator; this line reports a normal NW interval only, without a test.
            parts=[s.nw_t(three[w],h) if np.isfinite(three[w]).all() else (np.nan,np.nan,np.nan,0) for w in three]
            mean=float(np.mean([v[0] for v in parts]));se=float(np.sqrt(np.sum([v[1]**2 for v in parts]))/3)
            nw3.append(dict(model=model,setting='T3',h=h,window='W1+W2+W3',n_dates=1173,mean_rank_ic=mean,nw_se=se,nw_ci95_lower=mean-1.959963984540054*se,nw_ci95_upper=mean+1.959963984540054*se,label='includes the selection window W3'))

    def contrast(family,name,model,h,metric,first,second,denominator):
        X={w:components[first[0],first[1],w,h][metric]-components[second[0],second[1],w,h][metric] for w in ['W1','W2']}
        binary(f'inputs/{family}_{name}_{model}_h{h}_{metric}.npz',**X,seeds=np.array(io.SEEDS))
        boot=(joint_boot(X['W1'],'W1',h)+joint_boot(X['W2'],'W2',h))/2
        theta=strict_mean(np.concatenate(list(X.values()),axis=1));defined=np.isfinite(theta) and np.isfinite(boot).all()
        stats=s.series_stats(boot,theta) if defined else dict(ci_lower_95=np.nan,ci_upper_95=np.nan,centered_p_two_sided=np.nan)
        den=float(np.mean([np.mean(components[denominator[0],denominator[1],w,h]['mse']) for w in ['W1','W2']]))
        improve=boot if metric=='mse' else -boot;upper=float(np.quantile(improve,.95,method='linear')) if defined and family=='E4' else np.nan
        return dict(family=family+' CONFIRMATORY',contrast=name,model=model,h=h,metric=metric,window='W1+W2',estimate=theta,
          ci95_lower=stats['ci_lower_95'],ci95_upper=stats['ci_upper_95'],p_mbb=stats['centered_p_two_sided'],p_nw=np.nan,p_nw_holm=np.nan,
          denominator_mean_mse=den,estimate_pct=100*theta/den if metric=='mse' else np.nan,ci95_lower_pct=100*stats['ci_lower_95']/den if metric=='mse' else np.nan,ci95_upper_pct=100*stats['ci_upper_95']/den if metric=='mse' else np.nan,
          adaptive_improvement_upper95=upper,adaptive_mse_improvement_upper95_pct=100*upper/den if metric=='mse' and family=='E4' else np.nan,
          defined=bool(defined),bootstrap=replicates(f'{family}_{name}_{model}_h{h}_{metric}.npy',boot),decision_method='joint seed×date MBB; NW N/A')
    e3=[];e4=[]
    for h in io.HS:
        for model in ['static','adaptive']:e3.append(contrast('E3','T3-minus-T0',model,h,'mse',('T3',model),('T0',model),('T0',model)))
        for metric in ['mse','rank_ic']:e4.append(contrast('E4','static-minus-adaptive','static-vs-adaptive',h,metric,('T3','static'),('T3','adaptive'),('T3','adaptive')))
    E1=pd.DataFrame(e1);E2=pd.DataFrame(e2);E3=pd.DataFrame(e3);E4=pd.DataFrame(e4);families=[];disagreements=[]
    def holm(frame,indices,name,method):
        pcol='p_'+method;assert len(indices) in [3,6]
        values={i:float(frame.loc[i,pcol]) if np.isfinite(frame.loc[i,pcol]) else 1. for i in indices}
        for i,p in s.holm(values).items():
            frame.loc[i,pcol+'_holm']=p
            families.append(dict(family=name,method=method,model=frame.loc[i,'model'],h=int(frame.loc[i,'h']),raw_p=frame.loc[i,pcol],holm_input=values[i],holm_p=p,n_tests=len(indices)))
    for method in ['nw','mbb']:
        holm(E1,E1.index[E1.role=='SECONDARY'].tolist(),'E1 SECONDARY',method)
        holm(E2,E2.index.tolist(),'E2 CONFIRMATORY',method)
    for model in ['static','adaptive']:holm(E3,E3.index[E3.model==model].tolist(),'E3 CONFIRMATORY '+model,'mbb')
    for metric in ['mse','rank_ic']:holm(E4,E4.index[E4.metric==metric].tolist(),'E4 CONFIRMATORY '+metric,'mbb')
    for frame,name in [(E1,'E1 SECONDARY'),(E2,'E2 CONFIRMATORY')]:
        for i,r in frame.iterrows():
            if name.startswith('E1') and r.role!='SECONDARY':continue
            positive=bool(r['mean']>0);nw=bool(r.p_nw_holm<.05);mbb=bool(r.p_mbb_holm<.05)
            frame.loc[i,'flag']=positive and nw
            if nw!=mbb:disagreements.append(dict(family=name,model=r.model,h=int(r.h),mean=r['mean'],p_nw_holm=r.p_nw_holm,p_mbb_holm=r.p_mbb_holm,nw_rejects=nw,mbb_rejects=mbb,positive_NW_flag=positive and nw))
    E2['interpretation']=E2.flag.map(lambda b:'In held-out windows, the T3 ensemble forecasts beat the historical-mean benchmark.' if b else 'not confirmed in held-out windows')
    E3['label']=E3.apply(lambda r:'significant improvement' if r.estimate<0 and r.p_mbb_holm<.05 else 'significant deterioration' if r.estimate>0 and r.p_mbb_holm<.05 else 'not significant',axis=1)
    E3['interpretation']=E3.label.map({'significant improvement':'the MSE improvement from T3 replicates in held-out windows','significant deterioration':'T3 increased MSE in held-out windows; the W3 improvement did not replicate','not significant':'not significant'})
    E4['label']=E4.apply(lambda r:'not significant' if not r.p_mbb_holm<.05 or r.estimate==0 else 'favours Static' if (r.metric=='mse' and r.estimate<0) or (r.metric=='rank_ic' and r.estimate>0) else 'favours Adaptive',axis=1)
    E1['evidence_note']='SECONDARY only; never confirmation; T0 rows descriptive with no flag'
    save('E2_CONFIRMATORY_CW.csv',E2);save('E3_CONFIRMATORY_setting.csv',E3);save('E4_CONFIRMATORY_routing.csv',E4);save('E1_SECONDARY_rank_ic.csv',E1)
    save('holm_families.csv',pd.DataFrame(families));save('three_window_rank_ic_descriptive.csv',pd.DataFrame(nw3))
    Df=pd.DataFrame(disagreements,columns=['family','model','h','mean','p_nw_holm','p_mbb_holm','nw_rejects','mbb_rejects','positive_NW_flag']);save('NW_MBB_disagreements.csv',Df)
    sources=pd.DataFrame([dict(path=p,sha256=v) for p,v in sorted(G['protected'].items())]);save('protected_sources.csv',sources)
    assert all(io.sha(io.ROOT/p)==v for p,v in G['protected'].items())
    timings=pd.DataFrame([dict(phase='training',start=io.read(io.OUT/'TIMING_PILOT_START.json')['utc'],end=finished['utc'],minutes=io.elapsed(io.read(io.OUT/'TIMING_PILOT_START.json')['utc'],finished['utc'])/60)])
    eval_records=[io.read(p) for p in (io.OUT/'evaluation').glob('*/attempt_*/COMPLETE.json')]
    timings=pd.concat([timings,pd.DataFrame([dict(phase='test_evaluation',start=min(r['started_utc'] for r in eval_records),end=max(r['ended_utc'] for r in eval_records),minutes=sum(r['elapsed_utc_seconds'] for r in eval_records)/60),dict(phase='analysis',start=start,end=io.utc(),minutes=io.elapsed(start,io.utc())/60)])],ignore_index=True)
    save('actual_timing.csv',timings)
    report='# BLOCKING\n\n'+('None active. All60 registered cells are retained.\n' if not limitations else '\n'.join('- '+x for x in limitations)+'\n')+'\n'
    report+='Protocol SHA-256: `'+io.check_protocol()+'`.\n\nThis protocol was written after Step22D was completed. Step22D evaluated the same W1/W2 test windows with the LSTM (T0, T3) and with Static/Adaptive (T0). No Static or Adaptive T3 forecasts for W1 or W2 existed when this protocol was written.\n\n'
    report+='## Gates, timing and exact sigma\n\n'+table(pd.DataFrame([io.read(io.OUT/'DATA_GATE.json')]))+'\n'+table(pd.read_csv(io.OUT/'TRAIN_TARGET_SD.csv'))+'\n'+table(timings)+'\n'+table(pd.DataFrame([io.read(io.OUT/'TIMING_GATE.json')]).drop(columns=['pilots'],errors='ignore'))+'\n'
    report+='All TEST reads follow the all60-training barrier at '+finished['utc']+'. The queue uses caffeinate -dimsu (full command/PID in events/). Attempts and interruption/retry records are preserved below. W3 remains primary in the manuscript; these analyses are supplementary.\n\n'
    report+='## Descriptive per-window and pooled metrics\n\nFive-seed mean±sample SD; MSE is the mean of individual-seed MSEs, not ensemble MSE. DA excludes exactly-zero returns. Pooled R² uses total errors against each window’s own TRAIN mean. No descriptive tests or flags. W3 T3 retains Step22B2’s rounded sigma constants; W1/W2 use Step22D’s unrounded values.\n\n'+table(S.drop(columns=[c for c in S if 'zero' in c]))+'\n\n'
    report+='## Equal-weight five-seed ensembles (descriptive)\n\n'+table(E[['setting','model','window','h','mse','mean_rank_ic','da','r2os_train_mean']])+'\n\n'
    report+='## Best epochs and Static logits (descriptive)\n\nNo pooled training epoch is invented. Centered-logit MAD removes the irrelevant common logit offset; weight MAD is mean|softmax(logit)−0.25|. These are different quantities, both reported.\n\n'+table(BS)+'\n\n'+table(logsummary)+'\n\n'
    report+='## E2 CONFIRMATORY — ensemble Clark–West\n\nW1+W2 only. One-sided CW; NW lag h−1 PRIMARY, date-stratified MBB robustness. Separate6-test Holm families; positive mean and NW Holm p<.05 alone sets flag.\n\n'+table(E2[['model','h','mean','ci95_lower','ci95_upper','p_nw','p_nw_holm','p_mbb','p_mbb_holm','flag','interpretation']])+'\n\n'
    report+='## E3 CONFIRMATORY — T3−T0 setting effect\n\nJoint paired-seed×date MBB, two-sided; Holm across3 horizons within model. NW=N/A, per user clarification. Percentages and interval endpoints use T0’s fixed five-seed pooled mean MSE. Significance labels separate improvement and deterioration without altering p-values.\n\n'+table(E3[['model','h','estimate','ci95_lower','ci95_upper','estimate_pct','ci95_lower_pct','ci95_upper_pct','p_nw','p_nw_holm','p_mbb','p_mbb_holm','label','interpretation']])+'\n\n'
    report+='## E4 CONFIRMATORY — Static-T3−Adaptive-T3 routing\n\nJoint MBB, two-sided; separate3-horizon Holm per metric. NW=N/A. MSE<0 favours Static; IC>0 favours Static; opposite signs favour Adaptive only when corrected p<.05. Adaptive-improvement upper bounds are signed95th percentiles (negative retained), pointwise, with MSE percentages divided by Adaptive’s fixed five-seed mean MSE.\n\n'+table(E4[['metric','h','estimate','ci95_lower','ci95_upper','estimate_pct','ci95_lower_pct','ci95_upper_pct','p_nw','p_nw_holm','p_mbb','p_mbb_holm','label','adaptive_improvement_upper95','adaptive_mse_improvement_upper95_pct']])+'\n\n'
    report+='## E1 SECONDARY — pooled Rank IC and descriptive T0 reference\n\nE1 is SECONDARY in all tables and never confirmation. IC is per-seed date IC averaged across5 seeds, not ensemble IC. NW lag h PRIMARY, MBB robustness; separate6-test Holm families. T0 rows show the same method as descriptive references with no adjusted p/flag and do not enlarge E1’s families.\n\n'+table(E1[['family','role','setting','model','h','mean','nw_se','ci95_lower','ci95_upper','p_nw','p_nw_holm','p_mbb','p_mbb_holm','flag']])+'\n\n'
    report+='## NW/MBB disagreements\n\n'+(table(Df) if len(Df) else 'None.\n')+'\n'
    report+='## Three-window T3 Rank IC — includes the selection window W3\n\nMean and normal NW95% interval only, under the specified window-independence assumption. No p, flag or confirmation.\n\n'+table(pd.DataFrame(nw3))+'\n\n'
    report+='## Every training attempt, including failures and restarts\n\n'+table(pd.DataFrame(attempts)[['model','window','h','seed','status','epochs','best_epoch','elapsed_seconds','error','complete']])+'\n\n'
    report+='## Files and provenance\n\n'+'\n'.join('- ['+p.name+']('+str(p)+')' for p in G['outputs'] if p.suffix=='.csv' and p.parent==dest)+'\n\nAll source forecasts are indexed and hashed in prediction_sources.csv; checkpoints, histories, TRAIN/VAL predictions and router weights stay in fits/, TEST predictions/losses/date/stock summaries in evaluation/. draws/, inputs/ and bootstrap/ retain stream indices, matrices and all20000 replicates. Reused module hashes are in RNG_MANIFEST.json and REUSED_MODULES.csv. No source file or prior output was modified.\n'
    path=io.OUT/'REPORT_STEP22E_FINAL.md';io.text(path,report);G['outputs'].append(path)
    io.js(io.OUT/'ANALYSIS_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),started_utc=start,ended_utc=io.utc(),analysis_directory=io.rel(dest),new_training_cells=60,prediction_cells=270,summary_cells=90,limitations=limitations,artifacts=[io.artifact(p) for p in G['outputs']]))
    print('Report:',path,flush=True)

if __name__=='__main__':main()
