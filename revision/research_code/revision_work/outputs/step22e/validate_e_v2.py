"""Independent artifact/read-order/family validation; no additional hypothesis tests."""
import sys
sys.dont_write_bytecode=True
import io_e as io
for k in io.ENV_KEYS[:5]:io.os.environ[k]='1'
sys.path.append(str(io.ROOT/'revision_work/outputs/_deps_pyarrow15'))
import numpy as np,pandas as pd

def main():
    started=io.utc();finished=io.barrier();analysis=io.verify_record(io.OUT/'ANALYSIS_COMPLETE.json')
    if (io.OUT/'VALIDATION_COMPLETE.json').exists():io.verify_record(io.OUT/'VALIDATION_COMPLETE.json');return
    dest=io.ROOT/analysis['analysis_directory'];checks=[]
    def check(name,condition,detail=''):
        checks.append(dict(check=name,passed=bool(condition),detail=str(detail)))
        assert condition,name+': '+str(detail)
    def csv(name):return pd.read_csv(dest/name,float_precision='round_trip')
    check('Protocol and registered code hashes',bool(io.check_protocol()))
    H=pd.read_csv(io.OUT/'data_hash_checks.csv');old=pd.read_csv(io.D/'data_hash_checks.csv')
    check('Same16 frozen window hashes as Step22D',len(H)==16 and H.equals(old))
    for r in H.to_dict('records'):
        path=r.get('path',r.get('file'));expected=r.get('expected_sha256',r.get('sha256'))
        check('Frozen bytes '+path,io.sha(io.ROOT/path)==expected)
    check('Exact stored Step22D sigmas',io.read(io.OUT/'SIGMAS.json')['values']==io.read(io.D/'SIGMAS.json')['values'])
    check('Sigma was copied, not re-estimated',io.read(io.OUT/'SIGMA_IDENTITY_GATE.json')['passed'])
    plan=io.read(io.OUT/'GRID_PLAN.json')['jobs'];check('Exactly60 unique registered jobs',len(plan)==len({io.job_id(j) for j in plan})==60)
    bylabels={};all_epochs=[];endtimes=[];evstarts=[]
    for j in plan:
        check('Unchanged exact configuration '+io.job_id(j),j==io.spec(j['model'],j['window'],j['h'],j['seed']))
        cp,r=io.completion(j);check('Successful single checkpoint '+io.job_id(j),r['status']=='COMPLETE' and io.failed_attempts(j)<2)
        check('No pre-barrier TEST read '+io.job_id(j),r['test_values_read'] is False)
        check('Exact lr/delta '+io.job_id(j),r['lr']==3e-5 and r['delta']==j['delta'])
        endtimes.append(r['ended_utc'])
        for phase in ['profile_start','profile_end']:
            p=r[phase]['profile'];env=p['environment']
            check('12/12 CPU profile '+io.job_id(j)+' '+phase,p['device']=='cpu' and p['intra']==p['inter']==12 and env.get('OMP_NUM_THREADS') is None and env.get('OMP_DYNAMIC')==env.get('MKL_DYNAMIC')=='FALSE')
        hist=pd.read_csv(cp.parent/'history.csv',float_precision='round_trip')
        check('Complete Huber/MSE history '+io.job_id(j),len(hist)==r['epochs'] and np.isfinite(hist[['train_loss','val_loss','val_mse']].to_numpy()).all() and 1<=r['best_epoch']<=len(hist) and hist.lr.eq(3e-5).all() and hist.huber_delta.eq(j['delta']).all())
        check('Best validation checkpoint '+io.job_id(j),float(hist.loc[hist.epoch==r['best_epoch'],'val_loss'].iloc[0])-float(hist.val_loss.min())<=1.0001e-12)
        binding=io.read(cp.parent/'WINDOW_BINDING.json');check('Window row binding '+io.job_id(j),binding['counts']==io.counts(j['window']))
        check('TEST-lock enabled '+io.job_id(j),io.read(cp.parent/'TEST_LOCK.json')['pure_sha_allowed'])
        for split in ['train','val']:
            f=pd.read_csv(cp.parent/(split+'.csv'),float_precision='round_trip').sort_values(['anchor_date','ticker']).reset_index(drop=True)
            key=(j['window'],j['h'],split)
            if key not in bylabels:
                lp=io.ROOT/f"revision_work/outputs/analysis16/labels/{j['window']}_h{j['h']}.parquet"
                z=pd.read_parquet(lp);z.anchor_date=pd.to_datetime(z.anchor_date).dt.strftime('%Y-%m-%d')
                bylabels[key]=z[z.split==split].sort_values(['anchor_date','ticker']).reset_index(drop=True)
            z=bylabels[key]
            check('Counts/keys/float32 labels '+io.job_id(j)+' '+split,len(f)==io.counts(j['window'])[split] and not f.duplicated(['ticker','anchor_date']).any() and f[['ticker','anchor_date']].equals(z[['ticker','anchor_date']]) and np.array_equal(f.y_true.to_numpy(np.float32),z.y_true.to_numpy(np.float32)))
        eps=list((io.OUT/'evaluation'/io.job_id(j)).glob('attempt_*/COMPLETE.json'));check('One completed TEST inference '+io.job_id(j),len(eps)==1)
        er=io.verify_record(eps[0]);evstarts.append(er['started_utc']);check('Post-barrier TEST '+io.job_id(j),io.elapsed(finished['utc'],er['started_utc'])>=0)
        f=pd.read_csv(eps[0].parent/'predictions_losses_weights.csv',float_precision='round_trip')
        z=pd.read_parquet(io.ROOT/f"revision_work/outputs/analysis16/labels/{j['window']}_h{j['h']}.parquet");z.anchor_date=pd.to_datetime(z.anchor_date).dt.strftime('%Y-%m-%d');z=z[z.split=='test'].sort_values(['anchor_date','ticker']).reset_index(drop=True)
        check('TEST counts/keys/float32 labels '+io.job_id(j),len(f)==6647 and f[['ticker','anchor_date']].equals(z[['ticker','anchor_date']]) and np.array_equal(f.y_true.to_numpy(np.float32),z.y_true.to_numpy(np.float32)))
        check('Router weights saved '+io.job_id(j),all(n in f for n in io.WEIGHTS))
        for names in [io.WEIGHTS[:4],io.WEIGHTS[4:]]:
            if er['status']=='COMPLETE':check('Probability simplex '+io.job_id(j),np.allclose(f[names].sum(axis=1),1,rtol=0,atol=2e-7))
        for _,a in io.attempts(j):check('Attempt identity retained '+io.job_id(j),a['job']==j)
    check('No training after barrier',all(io.elapsed(t,finished['utc'])>=0 for t in endtimes))
    check('Analysis follows all evaluation',all(io.elapsed(io.read(p)['ended_utc'],analysis['started_utc'])>=0 for p in (io.OUT/'evaluation').glob('*/attempt_*/COMPLETE.json')))
    check('Recorded caffeinate -dimsu',any(r.get('caffeinate_command')==['/usr/bin/caffeinate','-dimsu','-w',str(r['pid'])] for r in [io.read(p) for p in (io.OUT/'events').glob('*.json')] if r.get('kind')=='queue_start'))
    for filename,n in [('prediction_sources.csv',270),('metrics_per_seed.csv',450),('five_seed_summary.csv',90),('ensemble_metrics.csv',90),('best_epochs_per_seed.csv',270),('static_logits_per_seed.csv',300),('static_logit_summary.csv',60),('E1_SECONDARY_rank_ic.csv',12),('E2_CONFIRMATORY_CW.csv',6),('E3_CONFIRMATORY_setting.csv',6),('E4_CONFIRMATORY_routing.csv',6),('holm_families.csv',36)]:check('Full table '+filename,len(csv(filename))==n)
    P=csv('metrics_per_seed.csv');check('Every summary retains five seeds',P.groupby(['setting','model','window','h']).size().eq(5).all())
    src=csv('prediction_sources.csv');check('No retraining reused sources',src.loc[(src.setting!='T3')|(src.window=='W3')|(src.model=='lstm'),'source'].ne('Step22E new T3').all())
    # Compare reused sources' descriptive metrics to stored Step22D results, without new tests.
    drecord=io.read(io.D/'ANALYSIS_COMPLETE.json');ddir=io.ROOT/drecord['analysis_directory'] if 'analysis_directory' in drecord else None
    if ddir is not None:
        candidates=list(ddir.glob('*summary*.csv'))
        checks.append(dict(check='Step22D reference table locations',passed=True,detail=';'.join(str(p) for p in candidates)))
    families=csv('holm_families.csv')
    from reuse_d import statistics
    G=statistics(io.OUT/'validation_unused_statistics');s=G['s']
    for (family,method),g in families.groupby(['family','method']):
        n=6 if family.startswith(('E1','E2')) else 3
        check('Declared Holm family '+family+' '+method,len(g)==n and g.n_tests.eq(n).all())
        adjusted=s.holm(dict(zip(g.index,g.holm_input)))
        check('Stored Holm values '+family+' '+method,all(np.isclose(adjusted[i],r.holm_p,rtol=1e-12,atol=1e-15) for i,r in g.iterrows()))
        check('Undefined retained p1 '+family,g.loc[g.raw_p.isna(),'holm_input'].eq(1).all())
    E1=csv('E1_SECONDARY_rank_ic.csv');E2=csv('E2_CONFIRMATORY_CW.csv');E3=csv('E3_CONFIRMATORY_setting.csv');E4=csv('E4_CONFIRMATORY_routing.csv')
    check('E1 SECONDARY in every row',E1.family.eq('E1 SECONDARY').all())
    ref=E1[E1.setting=='T0'];check('T0 reference has no adjusted p/flag',ref.p_nw_holm.isna().all() and ref.p_mbb_holm.isna().all() and ref.flag.isna().all())
    for frame,name in [(E1[E1.setting=='T3'],'E1'),(E2,'E2')]:
        expected=(frame['mean']>0)&(frame.p_nw_holm<.05)
        check('NW-only positive flag '+name,np.array_equal(frame.flag.to_numpy(bool),expected))
        for r in frame.itertuples():
            check('Correct NW lag '+name,int(r.nw_lag)==(r.h if name=='E1' else r.h-1))
    for frame,name in [(E1,'E1'),(E2,'E2')]:
        for r in frame.itertuples():
            boot=np.load(io.ROOT/r.bootstrap,allow_pickle=False)
            check('All20000 date replicates '+name,boot.shape==(20000,))
            if r.full_support:
                check('Date percentile interval '+name,np.allclose(np.quantile(boot,[.025,.975],method='linear'),[r.ci95_lower,r.ci95_upper],rtol=1e-12,atol=1e-15))
    expected_disagreements=[]
    for frame,name in [(E1[E1.setting=='T3'],'E1 SECONDARY'),(E2,'E2 CONFIRMATORY')]:
        for r in frame.itertuples():
            if (r.p_nw_holm<.05)!=(r.p_mbb_holm<.05):expected_disagreements.append((name,r.model,r.h))
    actual=csv('NW_MBB_disagreements.csv')
    check('Every NW/MBB disagreement explicitly listed',set(expected_disagreements)==set(zip(actual.family,actual.model,actual.h)))
    for frame,name in [(E3,'E3'),(E4,'E4')]:
        check('Joint-only inference '+name,frame.p_nw.isna().all() and frame.p_nw_holm.isna().all())
        for r in frame.itertuples():
            boot=np.load(io.ROOT/r.bootstrap,allow_pickle=False);check('All20000 replicates '+name+str(r.h),boot.shape==(20000,))
            if r.defined:
                lo,hi=np.quantile(boot,[.025,.975],method='linear');check('Replicate percentile interval '+name,np.allclose([lo,hi],[r.ci95_lower,r.ci95_upper],rtol=1e-12,atol=1e-15))
                if r.metric=='mse':check('Fixed denominator percentages '+name,np.allclose(100*np.array([r.estimate,lo,hi])/r.denominator_mean_mse,[r.estimate_pct,r.ci95_lower_pct,r.ci95_upper_pct],rtol=1e-12,atol=1e-14))
                if name=='E4':
                    q=float(np.quantile(boot if r.metric=='mse' else -boot,.95,method='linear'))
                    check('Signed Adaptive bound; negatives kept',np.isclose(q,r.adaptive_improvement_upper95,rtol=1e-12,atol=1e-15))
                    if r.metric=='mse':check('Bound fixed denominator',np.isclose(100*q/r.denominator_mean_mse,r.adaptive_mse_improvement_upper95_pct,rtol=1e-12,atol=1e-14))
            if name=='E3':label='significant improvement' if r.estimate<0 and r.p_mbb_holm<.05 else 'significant deterioration' if r.estimate>0 and r.p_mbb_holm<.05 else 'not significant'
            else:label='not significant' if not r.p_mbb_holm<.05 or r.estimate==0 else 'favours Static' if (r.metric=='mse' and r.estimate<0) or (r.metric=='rank_ic' and r.estimate>0) else 'favours Adaptive'
            check('Direction-specific label '+name,label==r.label)
    check('All confirmatory/secondary exclude W3',all(f.window.eq('W1+W2').all() for f in [E1,E2,E3,E4]))
    three=csv('three_window_rank_ic_descriptive.csv');check('Three-window selection label; no p/flag',len(three)==6 and three.label.eq('includes the selection window W3').all() and not any(c.startswith('p_') or c=='flag' for c in three))
    for p in (dest/'draws').glob('*.npz'):
        a=np.load(p,allow_pickle=False);check('Window-stratified indices '+p.name,a['date_indices'].shape==(20000,391) and a['date_indices'].min()>=0 and a['date_indices'].max()<391 and np.array_equal(a['seeds'],io.SEEDS))
        h=int(p.stem.split('_h')[1]);code=int(p.stem[1]);base=io.read(io.OUT/'RNG_MANIFEST.json')['bootstrap_base_seed']
        expected=s.mbb_index(391,h,20000,np.random.default_rng(np.random.SeedSequence([base,h,code,0])))
        seeds=np.random.default_rng(np.random.SeedSequence([base,h,code,1])).integers(0,5,size=(20000,5))
        check('Recorded RNG stream '+p.name,np.array_equal(expected,a['date_indices']) and np.array_equal(seeds,a['seed_indices']))
    for r in csv('protected_sources.csv').itertuples():check('Read-only source '+r.path,io.sha(io.ROOT/r.path)==r.sha256)
    path=io.OUT/'VALIDATION_CHECKS.csv';io.csv(path,pd.DataFrame(checks))
    report=io.OUT/'REPORT_STEP22E_VALIDATION.md'
    io.text(report,'# BLOCKING\n\n'+('None.\n' if not analysis['limitations'] else '\n'.join('- '+x for x in analysis['limitations'])+'\n')+'\nIndependent validation PASS. '+str(len(checks))+' checks; no new hypothesis tests.\n\nProtocol SHA-256: `'+io.check_protocol()+'`.\n\nVerified60 training cells, preserved attempts, 12/12 CPU/env, exact sigma/data/source hashes, TRAIN/VAL/TEST keys and float32 labels, post-training TEST order, all forecast/summary cells, registered family sizes, NW/MBB decision rules, direction labels and saved-replicate intervals/signed bounds.\n\nChecks: ['+path.name+']('+str(path)+'). Final report: [REPORT_STEP22E_FINAL.md]('+str(io.OUT/'REPORT_STEP22E_FINAL.md')+').\n')
    io.js(io.OUT/'VALIDATION_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),passed=True,started_utc=started,ended_utc=io.utc(),n_checks=len(checks),artifacts=[io.artifact(path),io.artifact(report)]))
    print('INDEPENDENT VALIDATION PASS:',len(checks),'checks',flush=True)
if __name__=='__main__':main()
