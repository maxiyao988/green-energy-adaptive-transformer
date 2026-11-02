"""Post-barrier window-stratified inference; reuse Step5/16 NW/MBB/Holm/IC code."""
import sys
sys.dont_write_bytecode=True
from io_d import *
for k in ENV_KEYS[:5]:os.environ[k]='1'
sys.path.insert(0,str(ROOT/'revision_work/scripts/analysis16'))
sys.path.insert(0,str(OLD))
sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import numpy as np,pandas as pd
from statistics16_v3 import s
from analyze22b_v3 import table
from functools import lru_cache
import ast
B=20000
BASE=20261021
WINDOWS=['W1','W2','W3']

def main():
    finished=barrier()
    if (OUT/'ANALYSIS_COMPLETE.json').exists():verify_record(OUT/'ANALYSIS_COMPLETE.json');return
    import subprocess
    assert not any(str(OUT/'worker_d.py') in line and ' train ' in line for line in subprocess.run(['ps','-Ao','command'],capture_output=True,text=True).stdout.splitlines())
    started=utc();dest=target(OUT/'analysis'/('attempt_'+stamp()));dest.mkdir(parents=True)
    outputs=[];protected={};problems=[]
    def protect(path,expected=None):
        path=Path(path);actual=sha(path)
        if expected is not None:assert actual==expected,'Source changed: '+str(path)
        protected[rel(path)]=actual;return path
    def save(name,frame):
        p=dest/name;csv(p,frame);outputs.append(p);return p
    def binary(name,**arrays):
        p=target(dest/name)
        with p.open('xb') as f:np.savez_compressed(f,**arrays)
        outputs.append(p);return p
    def replicates(name,values):
        p=target(dest/'bootstrap'/name)
        with p.open('xb') as f:np.save(f,values,allow_pickle=False)
        outputs.append(p);return rel(p)
    def readcsv(path):return pd.read_csv(protect(path),float_precision='round_trip')
    def strict_mean(a):
        a=np.asarray(a,float);return float(a.mean()) if a.size and np.isfinite(a).all() else np.nan
    def summary(frame,keys):
        rows=[]
        for key,g in frame.groupby(keys,dropna=False,sort=False):
            key=key if isinstance(key,tuple) else (key,)
            r=dict(zip(keys,key));n=len(g);r['n_seeds']=n
            for col in ['mse','mean_rank_ic','da','r2os_train_mean','r2os_zero']:
                a=g[col].to_numpy(float);r[col+'_mean']=strict_mean(a);r[col+'_sd']=float(a.std(ddof=1)) if n==5 and np.isfinite(a).all() else np.nan
            rows.append(r)
        return pd.DataFrame(rows)

    cw_tree=ast.parse((ROOT/'revision_work/scripts/skill/a2_skill.py').read_text())
    cw_fn=next(n for n in cw_tree.body if isinstance(n,ast.FunctionDef) and n.name=='cell_metrics')
    cw_assign=next(n for n in cw_fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='f' for t in n.targets))
    cw_code=compile(ast.Module(body=[cw_assign],type_ignores=[]),str(ROOT/'revision_work/scripts/skill/a2_skill.py'),'exec')
    checks=readcsv(ROOT/'revision_work/outputs/analysis16/prediction_checks.csv').set_index('id')
    registry=readcsv(ROOT/'revision_work/outputs/analysis16/registry.csv').fillna('')
    all_labels={};records=[];preds={};keyframes={}
    for w in WINDOWS:
        for h in HS:
            source=ROOT/f'revision_work/outputs/analysis16/labels/{w}_h{h}.parquet'
            labels=pd.read_parquet(protect(source));labels.anchor_date=pd.to_datetime(labels.anchor_date).dt.strftime('%Y-%m-%d')
            d=s.canon(labels[labels.split=='test']);assert len(d)==6647 and d.groupby('anchor_date').size().eq(17).all()
            assert len(labels[labels.split=='train'])==dict(W1=16660,W2=23647,W3=30634)[w]
            y=d.y_true.to_numpy(float);m=float(labels.loc[labels.split=='train','y_true'].to_numpy(float).mean())
            all_labels[w,h]=dict(y=y,m=m,gtm_mse=float(np.mean((y-m)**2)),zero_mse=float(np.mean(y**2)),dates=d.anchor_date.unique())
            keyframes[w,h]=d[['ticker','anchor_date','y_true']]
            records.append(dict(window=w,h=h,n_train=int((labels.split=='train').sum()),n_test=len(d),n_test_dates=d.anchor_date.nunique(),first_test_date=d.anchor_date.min(),last_test_date=d.anchor_date.max(),train_mean=m,train_mean_test_mse=np.mean((y-m)**2),zero_test_mse=np.mean(y**2),label_source=rel(source)))
            for inp in ['O','C']:
                for model in ['static','adaptive','ridge']:
                    group='rolling' if w!='W3' else ('ridge' if model=='ridge' else 'core')
                    q=registry[(registry['group']==group)&(registry.input_set==inp)&(registry.model==model)&(registry.config=='daily_weekly')&(registry.window==w)&(registry.h==h)]
                    assert len(q)==(1 if model=='ridge' else 5),(w,h,inp,model,len(q))
                    arrays=[]
                    for r in q.sort_values('seed').itertuples():
                        ck=checks.loc[r.id];assert ck.keys_equal and ck.labels_float32_equal
                        f=s.canon(pd.read_parquet(protect(ROOT/r.canonical,ck.canonical_sha256)))
                        assert f[['ticker','anchor_date']].equals(d[['ticker','anchor_date']])
                        assert np.array_equal(f.y_true.to_numpy(np.float32),d.y_true.to_numpy(np.float32))
                        protect(ROOT/r.source,r.sha256)
                        arrays.append(f.y_pred.to_numpy(float))
                    preds[w,inp,model,h]=np.stack(arrays)
                preds[w,inp,'train_mean',h]=np.full((1,len(y)),m)
                preds[w,inp,'zero',h]=np.zeros((1,len(y)))
    # Every horizon shares its window calendar; all three TEST windows are disjoint.
    date_sets=[set(all_labels[w,5]['dates']) for w in WINDOWS]
    assert len(set.union(*date_sets))==1173 and sum(map(len,date_sets))==1173
    for w in WINDOWS:
        for h in HS:assert np.array_equal(all_labels[w,h]['dates'],all_labels[w,5]['dates'])
    save('window_reference_values.csv',pd.DataFrame(records))

    @lru_cache(None)
    def draws(w,h):
        code=int(w[1:]);di=s.mbb_index(391,h,B,np.random.default_rng(np.random.SeedSequence([BASE,h,code,0])))
        si=np.random.default_rng(np.random.SeedSequence([BASE,h,code,1])).integers(0,5,size=(B,5))
        wt=np.stack([(si==i).sum(axis=1) for i in range(5)],axis=1)/5
        binary(f'draws/{w}_h{h}.npz',date_indices=di,seed_indices=si,seed_weights=wt,seeds=np.array(SEEDS))
        return di,wt

    def date_boot(x,w,h):
        if not np.isfinite(x).all():return np.full(B,np.nan)
        return np.mean(np.asarray(x,float)[draws(w,h)[0]],axis=1)

    def joint_boot(x,w,h):
        x=np.ascontiguousarray(x,dtype=float)
        if not np.isfinite(x).all():return np.full(B,np.nan)
        di,wt=draws(w,h)
        env=dict(np=np,X=x,wt=wt,didx=di,n_missing=0,missing=[],inp='',bk='',setting='',h=h,contrast='',metric='')
        exec(s.MCODE,env)
        return env['boot']

    def series_result(series,ws,h,lag,test=True,one_sided=False,tag=''):
        weights=np.array([len(series[w]) for w in ws],float);weights/=weights.sum()
        complete=all(len(series[w])==391 and np.isfinite(series[w]).all() for w in ws)
        mean=se=t=np.nan;lo=hi=pmbb=pnw=np.nan;boot=np.full(B,np.nan)
        if complete:
            nw=[s.nw_t(series[w],lag) for w in ws]
            mean=float(np.dot(weights,[a[0] for a in nw]));se=float(np.sqrt(np.dot(weights**2,[a[1]**2 for a in nw])))
            if np.isfinite(se) and se>0:
                t=mean/se
                if test:pnw=float(s.stats.norm.sf(t)*(1 if one_sided else 2)) if one_sided else float(2*s.stats.norm.sf(abs(t)))
            boot=sum(a*date_boot(series[w],w,h) for a,w in zip(weights,ws))
            lo,hi=np.quantile(boot,[.025,.975],method='linear')
            if test:
                pmbb=float((1+np.count_nonzero(boot-mean>=mean))/(B+1)) if one_sided else s.series_stats(boot,mean)['centered_p_two_sided']
        path=replicates(tag+'.npy',boot)
        return dict(mean=mean,nw_se=se,nw_t=t,p_nw=pnw,p_mbb=pmbb,ci95_lower=float(lo),ci95_upper=float(hi),nw_lag=lag,
          n_dates=sum(len(series[w]) for w in ws),n_finite_dates=sum(int(np.isfinite(series[w]).sum()) for w in ws),full_support=complete,
          nw_test_defined=bool(test and np.isfinite(pnw)),mbb_test_defined=bool(test and np.isfinite(pmbb)),bootstrap=path)

    def per_seed(pred,w,h):
        label=all_labels[w,h];y=label['y'];p=np.asarray(pred,float)
        good=np.isfinite(p).all();valid=y!=0
        ic=s.rank_ic_rows(p.reshape(391,17),y.reshape(391,17)) if good else np.full(391,np.nan)
        loss=(p-y)**2
        return dict(mse=float(loss.mean()) if good else np.nan,mean_rank_ic=float(np.mean(ic)) if np.isfinite(ic).all() else np.nan,
          da=float(np.mean(np.sign(p[valid])==np.sign(y[valid]))) if good else np.nan,
          r2os_train_mean=1-float(loss.mean())/label['gtm_mse'] if good else np.nan,
          r2os_zero=1-float(loss.mean())/label['zero_mse'] if good else np.nan,
          n_valid_da=int(valid.sum()),n_correct_da=int(np.count_nonzero(np.sign(p[valid])==np.sign(y[valid]))) if good else np.nan,
          gtm_mse=label['gtm_mse'],zero_mse=label['zero_mse'],n_dates=391,n_rows=6647),loss.reshape(391,17).mean(axis=1),ic

    # Part1: seed/date matrices and all per-window / pooled descriptive quantities.
    component={};perrows=[];dailyrows=[]
    for (w,inp,model,h),P in preds.items():
        losses=[];ics=[]
        for i,p in enumerate(P):
            r,l,ic=per_seed(p,w,h);seed=SEEDS[i] if len(P)==5 else -1
            perrows.append(dict(window=w,input_set=inp,model=model,h=h,seed=seed,**r));losses.append(l);ics.append(ic)
        component[w,inp,model,h]=dict(mse=np.stack(losses),rank_ic=np.stack(ics))
        for ix,date in enumerate(all_labels[w,h]['dates']):
            dailyrows.append(dict(window=w,input_set=inp,model=model,h=h,anchor_date=date,
              mse=float(np.mean(np.array(losses)[:,ix])),rank_ic=strict_mean(np.array(ics)[:,ix]),n_seeds=len(P),finite_seed_ic=int(np.isfinite(np.array(ics)[:,ix]).sum())))
    seedframe=pd.DataFrame(perrows)
    for inp in ['O','C']:
        for model in ['static','adaptive','ridge','train_mean','zero']:
            for h in HS:
                seeds=SEEDS if model in ['static','adaptive'] else [-1]
                for seed in seeds:
                    z=seedframe[(seedframe.input_set==inp)&(seedframe.model==model)&(seedframe.h==h)&(seedframe.seed==seed)]
                    assert len(z)==3
                    r=dict(window='W1+W2+W3',input_set=inp,model=model,h=h,seed=seed,n_dates=1173,n_rows=19941)
                    r['mse']=strict_mean(z.mse);r['mean_rank_ic']=strict_mean(z.mean_rank_ic)
                    r['n_valid_da']=int(z.n_valid_da.sum());r['n_correct_da']=float(z.n_correct_da.sum()) if z.n_correct_da.notna().all() else np.nan
                    r['da']=r['n_correct_da']/r['n_valid_da'];r['gtm_mse']=float(z.gtm_mse.mean());r['zero_mse']=float(z.zero_mse.mean())
                    r['r2os_train_mean']=1-r['mse']/r['gtm_mse'];r['r2os_zero']=1-r['mse']/r['zero_mse'];perrows.append(r)
    P1=pd.DataFrame(perrows);P1S=summary(P1,['window','input_set','model','h'])
    save('part1_metrics_per_seed.csv',P1);save('part1_metric_summary.csv',P1S);save('part1_date_series.csv',pd.DataFrame(dailyrows))
    icrows=[];contrasts=[]
    for inp in ['O','C']:
        for h in HS:
            for model in ['static','adaptive','ridge']:
                series={w:np.mean(component[w,inp,model,h]['rank_ic'],axis=0) for w in WINDOWS}
                binary(f'inputs/part1_{inp}_{model}_h{h}_ic.npz',**series)
                for ws in [[w] for w in WINDOWS]+[WINDOWS]:
                    scope='+'.join(ws);pooled=len(ws)==3
                    family=('PRIMARY' if inp=='C' else 'SECONDARY') if pooled and model!='ridge' else 'descriptive' if model=='ridge' else 'component_reference'
                    r=series_result(series,ws,h,h,test=model!='ridge',tag=f'part1_ic_{scope}_{inp}_{model}_h{h}')
                    icrows.append(dict(window=scope,input_set=inp,model=model,h=h,family=family,**r))
            for metric in ['mse','rank_ic']:
                X={w:component[w,inp,'static',h][metric]-component[w,inp,'adaptive',h][metric] for w in WINDOWS}
                boots={w:joint_boot(X[w],w,h) for w in WINDOWS}
                binary(f'inputs/part1_{inp}_static-minus-adaptive_h{h}_{metric}.npz',**X,seeds=np.array(SEEDS))
                for ws in [[w] for w in WINDOWS]+[WINDOWS]:
                    scope='+'.join(ws);D=np.concatenate([X[w] for w in ws],axis=1);theta=strict_mean(D)
                    boot=sum(boots[w]/len(ws) for w in ws);defined=bool(np.isfinite(boot).all() and np.isfinite(theta))
                    lo,hi=np.quantile(boot,[.025,.975],method='linear') if defined else (np.nan,np.nan)
                    p=s.series_stats(boot,theta)['centered_p_two_sided'] if defined else np.nan
                    denom=float(np.mean([np.mean(component[w,inp,'adaptive',h]['mse']) for w in ws]))
                    improvement=boot if metric=='mse' else -boot;upper=float(np.quantile(improvement,.95,method='linear')) if defined else np.nan
                    sd=D.mean(axis=1);path=replicates(f'part1_contrast_{scope}_{inp}_h{h}_{metric}.npy',boot)
                    contrasts.append(dict(window=scope,input_set=inp,h=h,contrast='static-minus-adaptive',metric=metric,estimate=theta,
                      ci95_lower=lo,ci95_upper=hi,p_mbb=p,test_defined=defined,second_arm_mean_test_mse=denom,
                      estimate_pct_second_mse=100*theta/denom if metric=='mse' else np.nan,
                      ci95_lower_pct_second_mse=100*lo/denom if metric=='mse' else np.nan,ci95_upper_pct_second_mse=100*hi/denom if metric=='mse' else np.nan,
                      adaptive_improvement_upper95=upper,adaptive_mse_improvement_upper95_pct=100*upper/denom if metric=='mse' else np.nan,
                      seed_wins_static=int((sd<0 if metric=='mse' else sd>0).sum()),seed_losses_static=int((sd>0 if metric=='mse' else sd<0).sum()),
                      bootstrap=path,n_dates=391*len(ws),n_seeds=5))
            print('Part1',inp,h,'done',flush=True)
    IC=pd.DataFrame(icrows);CT=pd.DataFrame(contrasts)

    # All new TEST evaluations, then reused W3 LSTM sources. No TEST worker precedes the60-fit barrier.
    lstm_pred={};files=[];runrows=[];histories=[]
    for job in read(OUT/'GRID_PLAN.json')['jobs']:
        cp,record=completion(job);runrows.append(dict(**job,status=record['status'],epochs=record.get('epochs'),best_epoch=record.get('best_epoch'),elapsed_utc_seconds=record.get('elapsed_utc_seconds'),error=record.get('error'),complete=rel(cp)))
        if record['status']!='COMPLETE':problems.append(job_id(job)+': '+record['status'])
        history=readcsv(cp.parent/'history.csv')
        for k,v in job.items():history[k]=v
        histories.append(history)
        done=sorted((OUT/'evaluation'/job_id(job)).glob('attempt_*/COMPLETE.json'));assert len(done)==1
        ev=verify_record(done[0]);verify_record(done[0].parent/'ADAPTER_COMPLETE.json')
        path=done[0].parent/'predictions_losses_weights.csv'
        f=s.canon(readcsv(path));expected=keyframes[job['window'],job['h']]
        assert f[['ticker','anchor_date']].equals(expected[['ticker','anchor_date']]) and np.array_equal(f.y_true.to_numpy(np.float32),expected.y_true.to_numpy(np.float32))
        lstm_pred[job['setting'],job['window'],job['h'],job['seed']]=f.y_pred.to_numpy(float)
        files.append(dict(**job,status=ev['status'],predictions=rel(path),sha256=sha(path),complete=rel(done[0]),reused=False))
    priorB2=ROOT/'revision_work/outputs/step22b2';priorComplete=read(protect(priorB2/'ANALYSIS_COMPLETE.json'))
    for item in priorComplete['artifacts']:
        if item['path'].endswith('/prediction_files.csv'):priorfiles=readcsv(protect(ROOT/item['path'],item['sha256']))
    for r in priorfiles[(priorfiles.model=='lstm')&priorfiles.setting.isin(['T0','T3'])].itertuples():
        p=protect(ROOT/r.predictions);c=read(protect(ROOT/r.complete))
        item=next(a for a in c['artifacts'] if a['path']==r.predictions);assert sha(p)==item['sha256']
        f=s.canon(pd.read_csv(p,float_precision='round_trip'));expected=keyframes['W3',r.h]
        assert f[['ticker','anchor_date']].equals(expected[['ticker','anchor_date']]) and np.array_equal(f.y_true.to_numpy(np.float32),expected.y_true.to_numpy(np.float32))
        lstm_pred[r.setting,'W3',r.h,r.seed]=f.y_pred.to_numpy(float)
        files.append(dict(setting=r.setting,window='W3',model='lstm',h=r.h,seed=r.seed,status=r.status,predictions=r.predictions,sha256=sha(p),complete=r.complete,reused=True))
    assert len(lstm_pred)==90
    save('lstm_prediction_files.csv',pd.DataFrame(files));save('all_training_runs.csv',pd.DataFrame(runrows));save('learning_curves.csv',pd.concat(histories,ignore_index=True))
    part2=[];ensembles=[];lstm_ic={};ensemble_ic={};cw_series={};newdate=[]
    for t in ['T0','T3']:
        for w in WINDOWS:
            for h in HS:
                P=np.stack([lstm_pred[t,w,h,seed] for seed in SEEDS]);series=[]
                for seed,p in zip(SEEDS,P):
                    r,l,ic=per_seed(p,w,h);part2.append(dict(setting=t,window=w,h=h,seed=seed,**r));series.append(ic)
                lstm_ic[t,w,h]=np.stack(series)
                ep=P.mean(axis=0);r,l,ic=per_seed(ep,w,h);ensemble_ic[t,w,h]=ic
                ensembles.append(dict(setting=t,window=w,h=h,**r))
                f=keyframes[w,h].copy();f['split']='test';f['y_pred']=ep
                save(f'ensemble_predictions/{t}_{w}_h{h}.csv',f)
                y=all_labels[w,h]['y'];m=all_labels[w,h]['m']
                # Exact Step5 Clark-West formula; only the ensemble, against own-window mean.
                cw_env=dict(y=y,b=np.full(len(y),m),p=ep)
                exec(cw_code,cw_env)
                cw=cw_env['f'].reshape(391,17).mean(axis=1)
                if np.allclose(ep,m,rtol=1e-6,atol=1e-9) or not np.isfinite(ep).all():cw[:]=np.nan
                cw_series[t,w,h]=cw
                for ix,date in enumerate(all_labels[w,h]['dates']):
                    newdate.append(dict(setting=t,window=w,h=h,anchor_date=date,seed_mean_rank_ic=strict_mean(np.array(series)[:,ix]),ensemble_rank_ic=ic[ix],ensemble_mse=l[ix],ensemble_cw=cw[ix] if t=='T3' and w in ['W1','W2'] else np.nan))
    original2=pd.DataFrame(part2);originalE=pd.DataFrame(ensembles)
    for ws in [['W1','W2'],WINDOWS]:
        for t in ['T0','T3']:
            for h in HS:
                for seed in SEEDS+['ensemble']:
                    z=(originalE if seed=='ensemble' else original2[original2.seed==seed])
                    z=z[(z.setting==t)&(z.h==h)&z.window.isin(ws)];assert len(z)==len(ws)
                    mse=strict_mean(z.mse);r=dict(setting=t,window='+'.join(ws),h=h,mse=mse,mean_rank_ic=strict_mean(z.mean_rank_ic),n_dates=391*len(ws),n_rows=6647*len(ws),gtm_mse=float(z.gtm_mse.mean()),zero_mse=float(z.zero_mse.mean()),n_valid_da=int(z.n_valid_da.sum()),n_correct_da=float(z.n_correct_da.sum()) if z.n_correct_da.notna().all() else np.nan)
                    r['da']=r['n_correct_da']/r['n_valid_da'];r['r2os_train_mean']=1-mse/r['gtm_mse'];r['r2os_zero']=1-mse/r['zero_mse']
                    if seed=='ensemble':ensembles.append(r)
                    else:part2.append(dict(seed=seed,**r))
    L=pd.DataFrame(part2);E=pd.DataFrame(ensembles);LS=summary(L,['setting','window','h'])
    save('lstm_metrics_per_seed.csv',L);save('lstm_five_seed_summary.csv',LS);save('lstm_ensemble_metrics.csv',E);save('lstm_date_series.csv',pd.DataFrame(newdate))
    confirm=[]
    for h in HS:
        ic={w:np.mean(lstm_ic['T3',w,h],axis=0) for w in ['W1','W2']}
        binary(f'inputs/confirm_ic_h{h}.npz',**ic)
        cw={w:cw_series['T3',w,h] for w in ['W1','W2']}
        binary(f'inputs/confirm_cw_h{h}.npz',**cw)
        confirm.append(dict(family='CONFIRMATORY_RANK_IC',test='seed_mean_rank_ic',window='W1+W2',input_set='C',model='lstm',setting='T3',h=h,
          **series_result(ic,['W1','W2'],h,h,tag=f'confirm_ic_h{h}')))
        confirm.append(dict(family='CONFIRMATORY_CW',test='ensemble_cw_vs_train_mean',window='W1+W2',input_set='C',model='lstm',setting='T3',h=h,
          **series_result(cw,['W1','W2'],h,h-1,one_sided=True,tag=f'confirm_cw_h{h}')))
    CF=pd.DataFrame(confirm)
    families=[];disagree=[]
    def adjust(frame,indices,name,expected):
        assert len(indices)==expected,(name,len(indices))
        for method in ['nw','mbb']:
            pcol='p_'+method;targetcol=pcol+'_holm'
            adjusted=s.holm({ix:float(frame.loc[ix,pcol]) if np.isfinite(frame.loc[ix,pcol]) else 1.0 for ix in indices})
            for ix,p in adjusted.items():
                frame.loc[ix,targetcol]=p
                families.append(dict(family=name,method=method,h=int(frame.loc[ix,'h']),model=frame.loc[ix,'model'],input_set=frame.loc[ix,'input_set'],raw_p=frame.loc[ix,pcol],holm_input_p=float(frame.loc[ix,pcol]) if np.isfinite(frame.loc[ix,pcol]) else 1.,holm_p=p,family_size=expected))
        for ix in indices:
            pos=frame.loc[ix,'mean']>0;nw=bool(pos and frame.loc[ix,'p_nw_holm']<.05);mbb=bool(pos and frame.loc[ix,'p_mbb_holm']<.05)
            frame.loc[ix,'significantly_positive_NW_primary']=nw;frame.loc[ix,'positive_MBB_robustness']=mbb;frame.loc[ix,'NW_MBB_disagree']=nw!=mbb
            if nw!=mbb:disagree.append(dict(family=name,h=int(frame.loc[ix,'h']),model=frame.loc[ix,'model'],input_set=frame.loc[ix,'input_set'],mean=frame.loc[ix,'mean'],nw_holm_p=frame.loc[ix,'p_nw_holm'],mbb_holm_p=frame.loc[ix,'p_mbb_holm'],NW_positive=nw,MBB_positive=mbb))
    for family in ['PRIMARY','SECONDARY']:adjust(IC,IC.index[IC.family==family].tolist(),family,6)
    for family in ['CONFIRMATORY_RANK_IC','CONFIRMATORY_CW']:adjust(CF,CF.index[CF.family==family].tolist(),family,3)
    CT['p_mbb_holm']=np.nan
    for inp in ['O','C']:
        for metric in ['mse','rank_ic']:
            ix=CT.index[(CT.window=='W1+W2+W3')&(CT.input_set==inp)&(CT.metric==metric)].tolist();assert len(ix)==3
            for i,p in s.holm({i:float(CT.loc[i,'p_mbb']) if np.isfinite(CT.loc[i,'p_mbb']) else 1. for i in ix}).items():CT.loc[i,'p_mbb_holm']=p
    save('part1_rank_ic_inference.csv',IC);save('part1_static_adaptive_contrasts.csv',CT);save('confirmatory_tests.csv',CF)
    F=pd.DataFrame(families);save('rank_ic_and_cw_Holm_families.csv',F)
    D=pd.DataFrame(disagree,columns=['family','h','model','input_set','mean','nw_holm_p','mbb_holm_p','NW_positive','MBB_positive']);save('NW_MBB_disagreements.csv',D)
    # Data/read order, full source protection and every run stay auditable.
    sources=pd.DataFrame([dict(path=p,sha256=v) for p,v in sorted(protected.items())]);save('protected_sources.csv',sources)
    assert all(sha(ROOT/p)==v for p,v in protected.items()),'Existing source changed during analysis'
    datesupport=IC[(IC.model!='ridge')&~IC.full_support]
    if len(datesupport):problems.append('Incomplete date-level IC support; retained as N/A with conservative family placeholders: '+str(len(datesupport)))
    if not CF.full_support.all():problems.append('Undefined confirmatory series retained in family with p=1 placeholder.')
    runframe=pd.DataFrame(runrows)
    columns=['window','input_set','model','h','mse_mean','mse_sd','mean_rank_ic_mean','mean_rank_ic_sd','da_mean','da_sd','r2os_train_mean_mean','r2os_zero_mean']
    report='# BLOCKING\n\n'+('\n'.join('- '+p for p in problems) if problems else 'None. All60 training cells have terminal records; no cell was selected or omitted.')+'\n\n'
    report+='Startup infrastructure deviation: the first three processes in the parent directory stopped before any epoch because the TEST lock rejected an allowed opaque SHA read. Their terminal records remain preserved. This session permits only the original SHA function through that lock; TEST parsing is still blocked. No actual fit was rerun. See STARTUP_RECOVERY.json and the parent STOP_REPORT. Protocol bytes and all training/statistical settings are unchanged.\n\n'
    report+='Protocol SHA-256: `'+check_protocol()+'`. Analysis UTC: '+started+' → '+utc()+'.\n\n'
    report+='## Gates and read order\n\n'+table(pd.DataFrame([read(OUT/'DATA_GATE.json')]))+'\n\n'+table(pd.DataFrame([read(OUT/'TIMING_GATE.json')]).drop(columns=['pilot_jobs'],errors='ignore'))+'\n\n'
    report+='TRAIN SD is sample SD(ddof1), computed in float64 from frozen float32 training labels, with no rounding. W3 T3 retains its original rounded sigma constants.\n\n'+table(pd.read_csv(OUT/'TRAIN_TARGET_SD.csv'))+'\n\n'
    report+='TEST evaluation began after all60 fits terminated at '+finished['utc']+'. Opaque pre-training TEST hashes only; no fallback or TEST deserialization before training. See events/, TIMING_PILOT*, evaluation/ telemetry and DATA_GATE.json.\n\n'
    report+='## Part1 per-window and pooled results\n\nMSE is the mean across five individual-seed forecasts for Static/Adaptive, not ensemble MSE. Ridge and references have one deterministic forecast, so SD is N/A. Pooled R² uses total squared errors against each window’s own TRAIN mean (or zero), not an average of window R². Three disjoint391-date windows give1173 dates. Constant-reference Rank IC is undefined.\n\n'+table(P1S[columns])+'\n\n'
    report+='## Part1 Rank IC: NW primary, stratified MBB robustness\n\nThe IC series is the fixed five-seed mean per date; no ensemble-forecast IC test. NW lag=h; pooled variance is the weighted sum of window mean variances under the specified independence assumption. Both p-values are two-sided and separately Holm adjusted within C PRIMARY6 and O SECONDARY6. Only a positive mean and NW Holm p<.05 sets the positive flag. Ridge has descriptive SE/intervals only. Per-window raw p-values are component references and set no claim.\n\n'
    report+=table(IC[['window','input_set','model','h','family','mean','nw_se','ci95_lower','ci95_upper','p_nw','p_nw_holm','p_mbb','p_mbb_holm','significantly_positive_NW_primary','NW_MBB_disagree']])+'\n\n'
    report+='## Static−Adaptive stratified seed×date contrasts\n\nSeeds are paired within each window and resampled independently across windows. Date blocks never cross windows. B=20000,L=h. First-minus-second MSE/IC differences; Holm across3 horizons within input×metric for pooled rows only. MSE percentages use Adaptive’s fixed five-seed mean MSE. Adaptive improvement is D for MSE and−D for IC; upper bounds are signed95th percentiles, with negatives retained, pointwise rather than simultaneous.\n\n'
    report+=table(CT[['window','input_set','h','metric','estimate','ci95_lower','ci95_upper','p_mbb','p_mbb_holm','ci95_lower_pct_second_mse','ci95_upper_pct_second_mse','adaptive_improvement_upper95','adaptive_mse_improvement_upper95_pct']])+'\n\n'
    report+='## Part2 LSTM descriptive results\n\nT0/T3 five-seed means±sample SD, followed by equal-weight forecast ensembles. W3 is reused from Step22B/22B2, with no retraining or inference. DA excludes exactly-zero realized returns; pooled DA uses all eligible rows. No descriptive metric is selected.\n\n'+table(LS)+'\n\n'+table(E)+'\n\n'
    report+='## W1+W2 confirmation\n\nW3 is excluded from confirmation. CW is one-sided on the T3 five-seed ensemble versus the window-specific TRAIN mean (NW lag h−1). Rank IC uses the five-seed mean of per-date IC (NW lag h), two-sided. NW primary and MBB robustness are separately Holm adjusted across3 horizons in each confirmatory family. Ensemble IC is descriptive only.\n\n'+table(CF[['family','h','mean','nw_se','ci95_lower','ci95_upper','p_nw','p_nw_holm','p_mbb','p_mbb_holm','significantly_positive_NW_primary','NW_MBB_disagree']])+'\n\n'
    report+='## NW/MBB disagreements\n\n'+(table(D) if len(D) else 'None: NW and MBB positive flags agree in all registered pooled IC/CW families.\n')+'\n'
    report+='## Every training run\n\n'+table(runframe[['setting','window','h','seed','status','epochs','best_epoch','elapsed_utc_seconds','error']])+'\n\n'
    report+='## Files and interpretation\n\nW3 remains the primary manuscript evaluation. Pooled results are supplementary. The window-independence variance assumption is specified, not empirically established here. No seed, window or model is selected from TEST outcomes. Undefined results stay in their complete multiplicity family with p=1; raw p remains N/A.\n\n'
    report+='\n'.join('- ['+p.name+']('+str(p)+')' for p in outputs if p.suffix=='.csv' and p.parent==dest)+'\n\nAll indices, seed weights, input matrices and20000-replicate distributions are saved under draws/, inputs/ and bootstrap/. TRAIN/VAL predictions, checkpoint states, epoch losses and telemetry are under fits/. Per-stock/date TEST predictions, losses and summaries are under evaluation/; their paths and hashes are in lstm_prediction_files.csv.\n'
    text(OUT/'REPORT_STEP22D.md',report);outputs.append(OUT/'REPORT_STEP22D.md')
    js(OUT/'ANALYSIS_COMPLETE.json',dict(protocol_sha256=check_protocol(),started_utc=started,ended_utc=utc(),analysis_directory=rel(dest),training_runs=60,reused_w3_lstm_cells=30,part1_summary_cells=len(P1S),confirmatory_rows=len(CF),problems=problems,artifacts=[artifact(p) for p in outputs]))
    print('Report:',OUT/'REPORT_STEP22D.md',flush=True)

if __name__=='__main__':main()
