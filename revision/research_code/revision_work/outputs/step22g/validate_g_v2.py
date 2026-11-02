"""Independent Step22G artifacts/target/RNG/family/portfolio validation; no new tests."""
import sys
sys.dont_write_bytecode=True
import io_g as io
import worker_g as w
np,pd=w.old.np,w.old.pd

def main():
    start=io.utc();finished=io.barrier();a=io.verify_record(io.OUT/'ANALYSIS_COMPLETE.json')
    if (io.OUT/'VALIDATION_COMPLETE.json').exists():io.verify_record(io.OUT/'VALIDATION_COMPLETE.json');return
    dest=io.ROOT/a['analysis_directory'];checks=[]
    def check(name,condition):checks.append(dict(check=name,passed=bool(condition)));assert condition,name
    def csv(name):return pd.read_csv(dest/name,float_precision='round_trip')
    check('Immutable protocol and source hashes',bool(io.check_protocol()))
    ds=w.dataset(True);info=w.INFO[id(ds)];train=ds.split=='train';sigma=io.read(io.OUT/'SIGMAS_CS.json')['values']
    for h in io.HS:check('Exact TRAIN-only float64 sigma '+str(h),float(np.std(info['cs64'][h][train],ddof=1,dtype=np.float64))==sigma[str(h)])
    plan=io.read(io.OUT/'GRID_PLAN.json')['jobs'];check('45 unique neural cells',len(plan)==len({io.job_id(j) for j in plan})==45)
    for j in plan:
        name=io.job_id(j);cp,r=io.completion(j);check('Successful fit/identical setting '+name,r['status']=='COMPLETE' and r['lr']==3e-5 and r['delta']==j['delta'] and j==io.spec(j['model'],j['h'],j['seed']) and io.failed_attempts(j)<2 and not r['test_values_read'])
        check('Training before TEST barrier '+name,io.elapsed(r['ended_utc'],finished['utc'])>=0)
        for phase in ['profile_start','profile_end']:
            profile=r[phase]['profile'];env=profile['environment'];check('12/12 CPU '+name+phase,profile['device']=='cpu' and profile['intra']==profile['inter']==12 and env.get('OMP_NUM_THREADS') is None and env.get('OMP_DYNAMIC')==env.get('MKL_DYNAMIC')=='FALSE')
        hist=pd.read_csv(cp.parent/'history.csv',float_precision='round_trip');check('Finite full history/best epoch '+name,len(hist)==r['epochs'] and np.isfinite(hist[['train_loss','val_loss','val_mse']]).all().all() and 1<=r['best_epoch']<=len(hist) and hist.lr.eq(3e-5).all() and hist.huber_delta.eq(j['delta']).all())
        check('Best validation restoration '+name,float(hist.loc[hist.epoch==r['best_epoch'],'val_loss'].iloc[0])-float(hist.val_loss.min())<=1.0001e-12)
        model=w.old.model_factory(j);model.load_state_dict(w.old.torch.load(cp.parent/'best_state.pt',map_location='cpu',weights_only=True),strict=True)
        ec=list((io.OUT/'evaluation'/name).glob('attempt_*/COMPLETE.json'));check('One TEST evaluation '+name,len(ec)==1);er=io.verify_record(ec[0]);check('TEST after all training '+name,io.elapsed(finished['utc'],er['started_utc'])>=0)
        for sp in ['train','val','test']:
            path=(ec[0].parent/'predictions_losses_weights.csv') if sp=='test' else cp.parent/(sp+'.csv');f=pd.read_csv(path,float_precision='round_trip').sort_values(['anchor_date','ticker']).reset_index(drop=True);mask=ds.split==sp
            expected=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),cs64=info['cs64'][j['h']][mask],cs32=ds.y[j['h']][mask],raw=info['raw'].y[j['h']][mask])).sort_values(['anchor_date','ticker']).reset_index(drop=True)
            check('Counts/keys/targets '+name+sp,len(f)==io.COUNTS[sp] and f[['ticker','anchor_date']].equals(expected[['ticker','anchor_date']]) and (np.array_equal(f.y_true.to_numpy(float),expected.cs64.to_numpy(float)) if sp=='test' else np.array_equal(f.y_true.to_numpy(np.float32),expected.cs32.to_numpy(np.float32))))
            if sp=='test':check('Original realized returns preserved '+name,np.array_equal(f.y_realized.to_numpy(np.float32),expected.raw.to_numpy(np.float32)))
    for file,n in [('test_metrics_per_seed.csv',48),('five_seed_summary.csv',12),('ensemble_metrics.csv',12),('IC_tests.csv',9),('joint_bootstrap_tests.csv',18),('portfolio_metrics.csv',24),('router_per_seed.csv',60),('router_summary.csv',12),('prediction_sources.csv',48)]:check('Full table '+file,len(csv(file))==n)
    b=io.module('_22g_validation_stats',io.OLD/'analyze22b_v3.py');s=b.s
    IC=csv('IC_tests.csv');J=csv('joint_bootstrap_tests.csv')
    for family,g in IC.groupby('family'):
        check('IC family size '+family,len(g)==(6 if family=='PRIMARY' else 3))
        for method in ['nw','mbb']:
            values={i:float(v) if np.isfinite(v) else 1. for i,v in g['p_'+method].items()};adj=s.holm(values)
            check('IC Holm '+family+method,all(np.isclose(adj[i],r['p_'+method+'_holm'],rtol=1e-12,atol=1e-15) for i,r in g.iterrows()))
    check('Positive flag uses NW only',np.array_equal(IC.flag,(IC.mean_rank_ic>0)&(IC.p_nw_holm<.05)))
    check('NW lag h',IC.nw_lag.eq(IC.h).all())
    expected={(r.family,r.arm,r.h) for r in IC.itertuples() if (r.p_nw_holm<.05)!=(r.p_mbb_holm<.05)};d=csv('NW_MBB_disagreements.csv');check('All NW/MBB disagreements retained',expected==set(zip(d.family,d.arm,d.h)))
    check('Six contrast families',len(J.groupby(['contrast','metric']))==6)
    for (contrast,metric),g in J.groupby(['contrast','metric']):
        check('Three-horizon family '+contrast+metric,len(g)==3 and set(g.h)==set(io.HS));adj=s.holm(dict(zip(g.index,g.holm_input_p)));check('Contrast Holm '+contrast+metric,all(np.isclose(adj[i],r.p_holm_3h,rtol=1e-12,atol=1e-15) for i,r in g.iterrows()));check('Undefined placeholder',g.loc[~g.test_defined,'holm_input_p'].eq(1).all())
    for r in J.itertuples():
        boot=np.load(io.ROOT/r.bootstrap,allow_pickle=False);check('20000 replicates',boot.shape==(20000,))
        if r.test_defined:
            lo,hi=np.quantile(boot,[.025,.975],method='linear');check('Saved percentile interval',np.allclose([lo,hi],[r.ci_lower_95,r.ci_upper_95],rtol=1e-12,atol=1e-15));check('All five seed outcomes',r.seed_wins+r.seed_losses+r.seed_ties==5)
            if r.family=='SECONDARY-B':
                bound=float(np.quantile(boot if r.metric=='mse' else -boot,.95,method='linear'));check('Signed Adaptive bound',np.isclose(bound,r.adaptive_improvement_upper95,rtol=1e-12,atol=1e-15))
                if r.metric=='mse':check('Fixed Adaptive-CS denominator',np.allclose(100*np.array([r.theta_hat,lo,hi,bound])/r.denominator_mean_cs_mse,[r.estimate_pct,r.ci95_lower_pct,r.ci95_upper_pct,r.adaptive_mse_improvement_upper95_pct],rtol=1e-12,atol=1e-14))
        label='not significant' if not r.p_holm_3h<.05 or r.theta_hat==0 else 'significant improvement of the first arm' if (r.metric=='mse' and r.theta_hat<0) or (r.metric=='rank_ic' and r.theta_hat>0) else 'significant deterioration of the first arm';check('Direction-specific label',r.direction_label==label)
    for h in io.HS:
        d=np.load(dest/f'draws/h{h}.npz',allow_pickle=False);wt,di,si=s.draws(h,5,391);check('Exact Step16 joint/date RNG '+str(h),np.array_equal(wt,d['joint_seed_weights']) and np.array_equal(di,d['joint_date_indices']) and np.array_equal(si,d['joint_seed_indices']) and np.array_equal(d['date_only_indices'],s.dates(h,391,0)))
        paths=list((io.OUT/'ridge'/f'h{h}').glob('attempt_*/COMPLETE.json'));check('One Ridge-CS cell '+str(h),len(paths)==1);r=io.verify_record(paths[0]);check('Ridge after barrier',io.elapsed(finished['utc'],r['started_utc'])>=0);trials=pd.read_csv(paths[0].parent/'validation_trials.csv',float_precision='round_trip');check('Original four-alpha protocol',trials.alpha.tolist()==[.1,1.,10.,100.] and trials.selected.sum()==1 and trials.loc[trials.selected,'alpha'].iloc[0]==trials.loc[trials.val_mse.idxmin(),'alpha'])
    weights=csv('portfolio_weights.csv');reb=csv('portfolio_rebalances.csv');met=csv('portfolio_metrics.csv')
    for mode,g in weights.groupby('portfolio'):
        gross=2 if mode=='top5_minus_bottom5' else 1;check('Portfolio gross/net '+mode,np.allclose(g.gross_exposure,gross) and np.allclose(g.net_exposure,0 if gross==2 else 1))
    check('10bp full L1 turnover',np.allclose(reb.cost_fraction,.001*reb.turnover))
    check('Exact per-period cost application',np.allclose(reb.net_period_return,(1-reb.cost_fraction)*(1+reb.gross_period_return)-1,rtol=1e-12,atol=1e-15))
    check('Long-short initial turnover2',np.allclose(met[(met.portfolio=='top5_minus_bottom5')&(met.status=='COMPLETE')].initial_turnover,2))
    check('Long-only initial turnover1',np.allclose(met[(met.portfolio=='top5_long_only')&(met.status=='COMPLETE')].initial_turnover,1))
    check('Required descriptive borrowing/cash note',met.note.str.contains('exclude.*borrowing|excludes borrowing',regex=True).all() and met.note.str.contains('cash interest').all())
    for r in csv('protected_sources.csv').itertuples():check('Read-only source '+r.path,io.sha(io.ROOT/r.path)==r.sha256)
    check('Everything EXPLORATORY',all(csv(name).evidence.str.contains('EXPLORATORY').all() for name in ['five_seed_summary.csv','ensemble_metrics.csv','IC_tests.csv','joint_bootstrap_tests.csv','portfolio_metrics.csv']))
    cp=io.OUT/'VALIDATION_CHECKS.csv';io.csv(cp,pd.DataFrame(checks));report=io.OUT/'REPORT_STEP22G_VALIDATION.md';io.text(report,'# BLOCKING\n\nNone.\n\nIndependent validation PASS: '+str(len(checks))+' checks; no new hypothesis tests.\n\nProtocol SHA-256: `'+io.check_protocol()+'`.\n\nVerified45 neural fits,3 deterministic Ridge cells,target arithmetic and TRAIN-only sigma,float32/float64 conventions,counts/keys/original returns,12/12 profile,retries,TEST barrier,9 IC tests/18 paired tests,Holm/flags/labels/negative-retained bounds,exact stored draws,200%gross long-short full-L1 costs and excluded borrowing/cash note,immutable sources. All findings EXPLORATORY;W3 T0 remains primary.\n\nChecks: ['+cp.name+']('+str(cp)+').\n');io.js(io.OUT/'VALIDATION_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),started_utc=start,ended_utc=io.utc(),passed=True,n_checks=len(checks),artifacts=[io.artifact(cp),io.artifact(report)]));print('INDEPENDENT VALIDATION PASS',len(checks),flush=True)
if __name__=='__main__':main()
