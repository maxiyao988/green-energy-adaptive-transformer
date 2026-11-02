"""Post-barrier Step22B evaluation and Step16 bootstrap, with fixed Step22F families."""
import sys,types
sys.dont_write_bytecode=True
import io_f as io
import worker_f as worker
np,pd=worker.old.np,worker.old.pd

def main():
    finished=io.barrier()
    if (io.OUT/'ANALYSIS_COMPLETE.json').exists():io.verify_record(io.OUT/'ANALYSIS_COMPLETE.json');return
    start=io.utc();dest=io.target(io.OUT/'analysis'/('attempt_'+io.stamp()));dest.mkdir(parents=True)
    outputs=[];protected={};problems=[]
    def protect(path,expected=None):
        path=io.Path(path);a=io.sha(path);assert expected is None or a==expected,'Changed source: '+str(path)
        protected[io.rel(path)]=a;return path
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
    references=pd.read_csv(protect(io.OUT/'REFERENCE_SOURCES.csv'),keep_default_na=False)
    labels={};dummy={};sources=[];fitrows=[];paths={}
    for h in io.HS:
        label=pd.read_parquet(protect(io.ROOT/f'revision_work/outputs/analysis16/labels/W3_h{h}.parquet'))
        label.anchor_date=pd.to_datetime(label.anchor_date).dt.strftime('%Y-%m-%d');z=label[label.split=='test'].sort_values(['anchor_date','ticker']).reset_index(drop=True)
        assert len(z)==6647 and z.groupby('anchor_date').size().eq(17).all()
        labels[h]=z;dummy[h]=types.SimpleNamespace(ticker=label.ticker.to_numpy(),anchor_date=pd.to_datetime(label.anchor_date).to_numpy(),split=label.split.to_numpy(),y={h:label.y_true.to_numpy(np.float32)})
    # Reference evaluation reuses saved predictions and the original loss/metric code; no model inference.
    old=worker.old;old.install_guard=lambda evaluation=False:None;old.barrier=lambda:finished
    for r in references.itertuples():
        for key in ['source','canonical','complete','history']:
            protect(io.ROOT/getattr(r,key),getattr(r,key+'_sha256'))
        if r.static_logits:protect(io.ROOT/r.static_logits,r.static_logits_sha256)
        folder=io.target(dest/'reference_evaluation'/f'{r.arm}_h{r.h}_s{r.seed}');folder.mkdir(parents=True)
        old.dataset=lambda include_test=False,h=r.h:dummy[h]
        job=dict(setting='T0',model=r.model,h=r.h,seed=r.seed,gate=False,variant=r.arm,window='W3')
        old.evaluate(job,folder)
        sources.append(dict(arm=r.arm,model=r.model,h=r.h,seed=r.seed,provenance='Step16 T0; stored predictions only',evaluation=io.rel(folder/'COMPLETE.json'),predictions=io.rel(folder/'predictions_losses_weights.csv'),fit= r.complete,logits=r.static_logits,history=r.history))
        record=io.read(io.ROOT/r.complete);fitrows.append(dict(arm=r.arm,model=r.model,h=r.h,seed=r.seed,status='REUSED',epochs=record['epochs'],best_epoch=record['best_epoch'],elapsed_seconds=record.get('wall_clock_seconds'),source=r.complete))
        outputs.extend(p for p in folder.iterdir() if p.is_file())
    for j in io.read(io.OUT/'GRID_PLAN.json')['jobs']:
        cp,record=io.completion(j);ec=list((io.OUT/'evaluation'/io.job_id(j)).glob('attempt_*/COMPLETE.json'));assert len(ec)==1;io.verify_record(ec[0])
        sources.append(dict(arm=j['variant'],model=j['model'],h=j['h'],seed=j['seed'],provenance='Step22F new parameter-group run',evaluation=io.rel(ec[0]),predictions=io.rel(ec[0].parent/'predictions_losses_weights.csv'),fit=io.rel(cp),logits=io.rel(cp.parent/'static_logits.json') if j['model']=='static' else '',history=io.rel(cp.parent/'history.csv')))
        fitrows.append(dict(arm=j['variant'],model=j['model'],h=j['h'],seed=j['seed'],status=record['status'],epochs=record['epochs'],best_epoch=record['best_epoch'],elapsed_seconds=record['elapsed_utc_seconds'],source=io.rel(cp)))
    assert len(sources)==90
    # Statistics import follows reference telemetry (which verifies the12/12 worker profile).
    b=io.module('_22f_stats_b',io.OLD/'analyze22b_v3.py');s=b.s;table=b.table
    assert s.BASE==20261019 and s.B==20000
    metrics=[];router=[];frames={};panels={};weights=[];histories=[];ens=[];changes=[]
    for r in sources:
        folder=(io.ROOT/r['evaluation']).parent;mt=pd.read_csv(folder/'metrics.csv',float_precision='round_trip').iloc[0].to_dict();mt.update(arm=r['arm'],evidence='EXPLORATORY supplementary W3 analysis');metrics.append(mt)
        f=s.canon(pd.read_csv(folder/'predictions_losses_weights.csv',float_precision='round_trip'));y=labels[r['h']]
        assert f[['ticker','anchor_date']].equals(y[['ticker','anchor_date']]) and np.array_equal(f.y_true.to_numpy(np.float32),y.y_true.to_numpy(np.float32))
        frames[r['arm'],r['h'],r['seed']]=f;panels[r['arm'],r['h'],r['seed']]=pd.read_csv(folder/'date_metrics.csv',float_precision='round_trip').sort_values('anchor_date').reset_index(drop=True)
        diag=pd.read_csv(folder/'router_diagnostics.csv',float_precision='round_trip')
        logits=io.read(protect(io.ROOT/r['logits'])) if r['logits'] else {}
        if r['model'] in ['static','adaptive']:
            for branch,names in [('daily',io.WEIGHTS[:4]),('weekly',io.WEIGHTS[4:])]:
                d=diag[diag.branch==branch].iloc[0].to_dict();d.update(arm=r['arm'],model=r['model'],h=r['h'],seed=r['seed'],evidence='EXPLORATORY supplementary W3 analysis')
                a=f[names].to_numpy(float);d.update({f'weight_{i+1}':float(np.mean(a[:,i])) for i in range(4)})
                ls=[v for k,v in logits.items() if branch in k and 'logit' in k]
                log=np.array(ls[0],float) if len(ls)==1 else None
                d['centered_logit_mad']=float(np.abs(log-log.mean()).mean()) if log is not None else np.nan
                for i in range(4):d['logit_'+str(i+1)]=float(log[i]) if log is not None else np.nan
                router.append(d)
        hist=pd.read_csv(protect(io.ROOT/r['history']),float_precision='round_trip');best=next(z['best_epoch'] for z in fitrows if (z['arm'],z['h'],z['seed'])==(r['arm'],r['h'],r['seed']))
        for k in ['arm','h','seed']:hist[k]=r[k]
        hist['best_epoch']=best;histories.append(hist)
    M=pd.DataFrame(metrics);R=pd.DataFrame(router)
    S=M.groupby(['arm','h'],as_index=False).agg(n_seeds=('seed','nunique'),n_complete=('status',lambda z:int(z.eq('COMPLETE').sum())),mse_mean=('mse',b.finite_mean),mse_sd=('mse',b.finite_sd),rank_ic_mean=('mean_rank_ic',b.finite_mean),rank_ic_sd=('mean_rank_ic',b.finite_sd),da_mean=('da',b.finite_mean),da_sd=('da',b.finite_sd),r2os_mean=('r2os',b.finite_mean),r2os_sd=('r2os',b.finite_sd))
    S['evidence']='EXPLORATORY supplementary W3 analysis';assert len(S)==18 and S.n_seeds.eq(5).all()
    RS=R.groupby(['arm','h','branch'],as_index=False).agg(n_seeds=('seed','nunique'),weight_mad_mean=('mean_abs_deviation_from_equal',b.finite_mean),weight_mad_sd=('mean_abs_deviation_from_equal',b.finite_sd),within_model_sd_mean=('mean_sample_sd',b.finite_mean),within_model_sd_sd=('mean_sample_sd',b.finite_sd),centered_logit_mad_mean=('centered_logit_mad',b.finite_mean),centered_logit_mad_sd=('centered_logit_mad',b.finite_sd),n_seeds_mad_ge_005=('mean_abs_deviation_from_equal',lambda z:int((z>=.005).sum())),n_defined_mad=('mean_abs_deviation_from_equal','count'),weight_1_mean=('weight_1',b.finite_mean),weight_2_mean=('weight_2',b.finite_mean),weight_3_mean=('weight_3',b.finite_mean),weight_4_mean=('weight_4',b.finite_mean))
    RS['evidence']='EXPLORATORY supplementary W3 analysis';gates=[]
    for arm in ['Static-R10','Static-R100']:
        for h in io.HS:
            a=RS[(RS.arm==arm)&(RS.h==h)].set_index('branch');v=a.loc[['daily','weekly'],'weight_mad_mean'].to_numpy(float)
            status='UNDEFINED' if not np.isfinite(v).all() else 'PASS' if (v>=.005).any() else 'FAIL'
            gates.append(dict(arm=arm,h=h,daily_five_seed_mean_mad=v[0],weekly_five_seed_mean_mad=v[1],status=status,threshold=.005,interpretation='manipulation passed' if status=='PASS' else 'manipulation failed — no conclusion about H' if status=='FAIL' else 'manipulation undefined — no conclusion about H',affects_tests=False,evidence='EXPLORATORY supplementary W3 analysis'))
    gates=pd.DataFrame(gates)
    for arm in list(S.arm.unique()):
        for h in io.HS:
            fs=[frames[arm,h,seed] for seed in io.BOOT_SEEDS];f=fs[0][['ticker','anchor_date','split','y_true']].copy();f['y_pred']=np.mean([f.y_pred.to_numpy(float) for f in fs],axis=0)
            save(f'ensemble_predictions/{arm}_h{h}.csv',f)
            y=f.y_true.to_numpy(float);p=f.y_pred.to_numpy(float);valid=y!=0
            mse=float(np.mean((p-y)**2));m=float(M[(M.arm==arm)&(M.h==h)].global_train_mean.iloc[0]);den=float(np.mean((y-m)**2))
            ic=s.rank_ic_rows(p.reshape(391,17),y.reshape(391,17))
            ens.append(dict(arm=arm,h=h,n_seeds=5,mse=mse,mean_rank_ic=float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan,da=float(np.mean(np.sign(p[valid])==np.sign(y[valid]))) if np.isfinite(p).all() else np.nan,r2os=1-mse/den,evidence='EXPLORATORY descriptive ensemble; not tested'))
    trials=[];diffs=[]
    contrast_specs=[('PRIMARY','Static-R10','Fixed'),('PRIMARY','Static-R100','Fixed'),('SECONDARY','Static-R100','Adaptive-R100'),('SECONDARY','Adaptive-R100','Adaptive'),('SECONDARY','Static-R100','Static')]
    for h in io.HS:
        wt,di,si=s.draws(h,5,391);binary(f'draws/h{h}.npz',seed_weights=wt,date_indices=di,seed_indices=si,seeds=np.array(io.BOOT_SEEDS))
        for family,left,right in contrast_specs:
            for metric in ['mse','rank_ic']:
                a=[panels[left,h,seed] for seed in io.BOOT_SEEDS];z=[panels[right,h,seed] for seed in io.BOOT_SEEDS]
                assert all(np.array_equal(a[0].anchor_date.to_numpy(),v.anchor_date.to_numpy()) for v in a+z)
                X=np.ascontiguousarray(np.stack([v[metric].to_numpy(float)-u[metric].to_numpy(float) for v,u in zip(a,z)]))
                status_ok=all(M[(M.arm==arm)&(M.h==h)].status.eq('COMPLETE').all() for arm in [left,right])
                defined=bool(status_ok and np.isfinite(X).any(axis=1).all());reason='' if defined else 'failed/degenerate arm or seed with no paired-defined dates; retained'
                boot=np.full(20000,np.nan);theta=np.nan
                if defined:theta,boot,_=s.distributions(X,h);defined=bool(np.isfinite(theta) and np.isfinite(boot).all())
                stats=s.series_stats(boot,theta) if defined else dict(theta_hat=np.nan,ci_lower_95=np.nan,ci_upper_95=np.nan,centered_p_two_sided=np.nan)
                delta=np.nanmean(X,axis=1) if np.isfinite(X).any(axis=1).all() else np.full(5,np.nan)
                wins=int(((delta<0) if metric=='mse' else (delta>0)).sum()) if np.isfinite(delta).all() else np.nan
                losses=int(((delta>0) if metric=='mse' else (delta<0)).sum()) if np.isfinite(delta).all() else np.nan
                if defined:
                    direct=np.array([np.nanmean(X[si[i]][:,di[i]],axis=1).mean() for i in range(16)])
                    assert np.max(np.abs(direct-boot[:16]))<1e-12
                denom=float(M[(M.arm==right)&(M.h==h)].mse.mean())
                upper=float(np.quantile(-boot if metric=='mse' else boot,.95,method='linear')) if defined and family=='PRIMARY' else np.nan
                binary(f'inputs/{left}_minus_{right}_h{h}_{metric}.npz',differences=X,seeds=np.array(io.BOOT_SEEDS),dates=a[0].anchor_date.to_numpy(dtype='U10'))
                for i,seed in enumerate(io.BOOT_SEEDS):
                    for date,v in zip(a[0].anchor_date,X[i]):diffs.append(dict(family=family,first=left,second=right,h=h,metric=metric,seed=seed,anchor_date=date,difference=v))
                trials.append(dict(family=family,contrast=left+'-minus-'+right,first=left,second=right,input_set='C',window='W3',h=h,metric=metric,test_defined=defined,undefined_reason=reason if not defined else '',B=20000,L=h,n_seeds=5,n_dates=391,n_missing_seed_dates=int((~np.isfinite(X)).sum()),
                  **stats,holm_input_p=stats['centered_p_two_sided'] if defined else 1.,seed_wins=wins,seed_losses=losses,seed_ties=5-wins-losses if np.isfinite(wins) else np.nan,seed_win_arm=left,denominator_mean_mse=denom,
                  estimate_pct=100*stats['theta_hat']/denom if metric=='mse' and family=='PRIMARY' else np.nan,ci95_lower_pct=100*stats['ci_lower_95']/denom if metric=='mse' and family=='PRIMARY' else np.nan,ci95_upper_pct=100*stats['ci_upper_95']/denom if metric=='mse' and family=='PRIMARY' else np.nan,
                  improvement_upper95=upper,mse_improvement_upper95_pct=100*upper/denom if metric=='mse' and family=='PRIMARY' else np.nan,
                  bootstrap=rep(f'{left}_minus_{right}_h{h}_{metric}.npy',boot),evidence='EXPLORATORY supplementary W3; no out-of-window confirmation'))
    J=s.corrected(pd.DataFrame(trials),['contrast','metric'],p='holm_input_p');assert len(J)==30
    assert J.groupby(['contrast','metric']).size().eq(3).all()
    J['direction_label']=J.apply(lambda r:'not significant' if not r.p_holm_3h<.05 or r.theta_hat==0 else 'significant improvement of the first arm' if (r.metric=='mse' and r.theta_hat<0) or (r.metric=='rank_ic' and r.theta_hat>0) else 'significant deterioration of the first arm',axis=1)
    J['positive_finding']=J.direction_label.eq('significant improvement of the first arm')
    J['manipulation_status']='not applicable'
    J['interpretation']=J.apply(lambda r:'EXPLORATORY: '+r.direction_label if r.positive_finding else r.direction_label,axis=1)
    for i,r in J[J.family=='PRIMARY'].iterrows():
        g=gates[(gates.arm==r['first'])&(gates.h==r.h)].iloc[0];J.loc[i,'manipulation_status']=g.status
        J.loc[i,'interpretation']=g.interpretation if g.status!='PASS' else 'EXPLORATORY: with a larger logit learning rate, learned global scale weights improved on equal weighting end-to-end (W3 only, no out-of-window confirmation)' if r.positive_finding else 'learned global weights did not improve on equal weighting end-to-end even when the weights were allowed to move' if r.test_defined else 'undefined test — no conclusion about H'
    eligible=J[(J.family=='PRIMARY')&(J.manipulation_status=='PASS')]
    overall='No conclusion about H: every manipulation failed or was undefined.' if len(eligible)==0 else 'EXPLORATORY: PRIMARY improvement in passing manipulation cells; W3 only, no out-of-window confirmation.' if eligible.positive_finding.any() else 'learned global weights did not improve on equal weighting end-to-end even when the weights were allowed to move (only passing, defined cells)' if eligible.test_defined.all() else 'No PRIMARY improvement observed in passing cells; undefined tests retained, no complete conclusion about H.'
    save('test_metrics_per_seed.csv',M);save('five_seed_summary.csv',S);save('ensemble_metrics.csv',pd.DataFrame(ens));save('router_diagnostics_per_seed.csv',R);save('manipulation_summary.csv',RS);save('manipulation_gate.csv',gates);save('joint_bootstrap_tests.csv',J);save('paired_seed_date_differences.csv',pd.DataFrame(diffs));save('best_epochs_per_run.csv',pd.DataFrame(fitrows));save('prediction_sources.csv',pd.DataFrame(sources));save('learning_curves_per_epoch.csv',pd.concat(histories,ignore_index=True))
    familyrows=[]
    for (contrast,metric),g in J.groupby(['contrast','metric']):
        for r in g.itertuples():familyrows.append(dict(family=r.family,contrast=contrast,metric=metric,h=r.h,n_tests=3,raw_p=r.centered_p_two_sided,holm_input=r.holm_input_p,holm_p=r.p_holm_3h))
    save('holm_families.csv',pd.DataFrame(familyrows))
    attempts=[]
    for j in io.read(io.OUT/'GRID_PLAN.json')['jobs']:
        for cp,r in io.attempts(j):attempts.append(dict(arm=j['variant'],h=j['h'],seed=j['seed'],status=r['status'],epochs=r.get('epochs'),best_epoch=r.get('best_epoch'),elapsed_seconds=r.get('elapsed_utc_seconds'),started_utc=r.get('started_utc'),ended_utc=r.get('ended_utc'),error=r.get('error'),complete=io.rel(cp),group_record=io.rel(cp.parent/'OPTIMIZER_GROUPS.json') if (cp.parent/'OPTIMIZER_GROUPS.json').exists() else 'NOT STORED'))
    save('all_training_attempts.csv',pd.DataFrame(attempts));save('protected_sources.csv',pd.DataFrame([dict(path=p,sha256=h) for p,h in sorted(protected.items())]))
    assert all(io.sha(io.ROOT/p)==h for p,h in protected.items())
    timing=pd.DataFrame([dict(phase='training',start=io.read(io.OUT/'TIMING_PILOT_START.json')['utc'],end=finished['utc'],minutes=io.elapsed(io.read(io.OUT/'TIMING_PILOT_START.json')['utc'],finished['utc'])/60),dict(phase='analysis',start=start,end=io.utc(),minutes=io.elapsed(start,io.utc())/60)])
    save('timing.csv',timing)
    report='# BLOCKING\n\n'+('None active. All45 new runs and45 reference cells retained.\n' if not problems else '\n'.join(problems)+'\n')+'\nProtocol SHA-256: `'+io.check_protocol()+'`.\n\n'
    report+='The scientific design of this protocol (hypothesis, variants, contrasts and families) was fixed on 2026-10-21 before Step22E was run; this version adds only operational rules. No Static-R10, Static-R100 or Adaptive-R100 runs existed when it was written.\n\nThe preceding statement records user-supplied scientific provenance; this operational file/hash was created after Step22E. W3 T0 remains primary in the manuscript; all new findings are EXPLORATORY supplementary W3 results without out-of-window confirmation.\n\n'
    report+='## Gates and runtime\n\n'+table(pd.DataFrame([io.read(io.OUT/'DATA_GATE.json')]).drop(columns=['prior_manifest','partition_counts'],errors='ignore'))+'\n'+table(pd.DataFrame([io.read(io.OUT/'TIMING_GATE.json')]).drop(columns=['pilots'],errors='ignore'))+'\n'+table(timing)+'\nTEST inference starts only after all45 successful fits at '+finished['utc']+'. Full caffeinate -dimsu invocation/PID is in events/.\n\n'
    report+='## Five-seed descriptive metrics\n\nEXPLORATORY supplementary W3; mean±sample SD of individual-seed metrics, no tests on this table.\n\n'+table(S)+'\n\n## Five-seed ensemble metrics (descriptive)\n\n'+table(pd.DataFrame(ens))+'\n\n'
    report+='## Manipulation check\n\nEXPLORATORY. Gate per variant×h uses five-seed mean MAD per branch: fail only when BOTH branches<.005; pass when at least one≥.005. Undefined statistics are not passes. Per-seed MAD and learned logits/weights are saved in router_diagnostics_per_seed.csv; n_seeds_mad_ge_005 counts all5 seeds. Within-model SD follows Step22B’s constant convention. Gate affects interpretation ONLY: no test/family is removed or changed.\n\n'+table(RS)+'\n\n'+table(gates)+'\n\n'
    cols=['family','first','second','h','metric','theta_hat','ci_lower_95','ci_upper_95','centered_p_two_sided','p_holm_3h','direction_label','seed_wins','seed_losses','seed_ties','manipulation_status','estimate_pct','ci95_lower_pct','ci95_upper_pct','improvement_upper95','mse_improvement_upper95_pct','interpretation','evidence']
    for family in ['PRIMARY','SECONDARY']:report+='## '+family+' contrasts — EXPLORATORY\n\nTwo-sided Step16 joint seed×date bootstrap,B20000,L=h; Holm across3h per contrast×metric; no NW. First-minus-second sign labels retained irrespective of manipulation. PRIMARY bounds are pointwise, signed (negative retained), with Fixed’s fixed five-seed mean MSE denominator.\n\n'+table(J[J.family==family][cols])+'\n\n'
    report+='## H interpretation\n\n'+overall+'\n\nFailed or undefined manipulation cells never support a conclusion about H, even if their reported p-value is small. Direction labels and EXPLORATORY tags preserve any MSE/IC disagreement.\n\n## Best epochs, all arms\n\n'+table(pd.DataFrame(fitrows))+'\n\n## Every new training attempt\n\n'+table(pd.DataFrame(attempts))+'\n\n## Files and provenance\n\n'
    report+='\n'.join('- ['+p.name+']('+str(p)+')' for p in outputs if p.suffix=='.csv' and p.parent==dest)+'\n\nAll trained checkpoints and Adam group/default records are retained under fits/, new TEST forecasts/losses/weights under evaluation/, reference recomputations from saved forecasts under analysis/.../reference_evaluation/. No reference model was trained or re-inferred. inputs/,draws/,bootstrap/ contain exact matrices/indices and20000 replicates. Reused module hashes, fixed RNG streams and numerical profile are in RNG_MANIFEST.json and REUSED_MODULES.csv.\n'
    rp=io.OUT/'REPORT_STEP22F_FINAL.md';io.text(rp,report);outputs.append(rp)
    io.js(io.OUT/'ANALYSIS_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),started_utc=start,ended_utc=io.utc(),analysis_directory=io.rel(dest),training_cells=45,reference_cells=45,prediction_cells=90,inference_rows=30,overall_H_interpretation=overall,problems=problems,artifacts=[io.artifact(p) for p in outputs]))
    print('FINAL REPORT:',rp,flush=True)
if __name__=='__main__':main()
