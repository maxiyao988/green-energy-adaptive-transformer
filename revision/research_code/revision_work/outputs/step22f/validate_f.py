"""Independent Step22F validation of artifacts, groups, families and effect bounds."""
import sys
sys.dont_write_bytecode=True
import io_f as io
import worker_f as w
np,pd=w.old.np,w.old.pd

def main():
    start=io.utc();finished=io.barrier();a=io.verify_record(io.OUT/'ANALYSIS_COMPLETE.json')
    if (io.OUT/'VALIDATION_COMPLETE.json').exists():io.verify_record(io.OUT/'VALIDATION_COMPLETE.json');return
    dest=io.ROOT/a['analysis_directory'];checks=[];limitations=[]
    def check(name,value,detail=''):
        checks.append(dict(check=name,passed=bool(value),detail=str(detail)));assert value,name+': '+str(detail)
    def readcsv(name):return pd.read_csv(dest/name,float_precision='round_trip')
    check('Protocol and immutable code',bool(io.check_protocol()))
    H=pd.read_csv(io.OUT/'tensor_hash_checks.csv');check('Eight immutable tensor files',len(H)==8)
    for r in H.itertuples():check('Opaque window bytes '+r.path,io.sha(io.ROOT/r.path)==r.expected_sha256)
    plan=io.read(io.OUT/'GRID_PLAN.json')['jobs'];check('All45 unique jobs',len(plan)==len({io.job_id(j) for j in plan})==45)
    all_labels={h:pd.read_parquet(io.ROOT/f'revision_work/outputs/analysis16/labels/W3_h{h}.parquet') for h in io.HS}
    for h,f in all_labels.items():f.anchor_date=pd.to_datetime(f.anchor_date).dt.strftime('%Y-%m-%d')
    trainingends=[];evalstarts=[];evalends=[]
    for j in plan:
        name=io.job_id(j);check('Exact configuration '+name,j==io.spec(j['variant'],j['h'],j['seed']))
        cp,r=io.completion(j);check('Single successful model '+name,r['status']=='COMPLETE' and io.failed_attempts(j)<2)
        trainingends.append(r['ended_utc']);check('Original base lr and Huber '+name,r['lr']==1e-4 and r['delta']==1 and not r['test_values_read'])
        for phase in ['profile_start','profile_end']:
            profile=r[phase]['profile'];env=profile['environment']
            check('12/12 CPU '+name+' '+phase,profile['device']=='cpu' and profile['intra']==profile['inter']==12 and env.get('OMP_NUM_THREADS') is None and env.get('OMP_DYNAMIC')==env.get('MKL_DYNAMIC')=='FALSE')
        model=w.old.model_factory(j);state=w.old.torch.load(cp.parent/'best_state.pt',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
        check('Parameter count '+name,w.old.formal.count_trainable_parameters(model)==(107785 if j['model']=='static' else 116297))
        g=io.read(cp.parent/'OPTIMIZER_GROUPS.json');named=dict(model.named_parameters());groups=g['groups']
        names=[n for row in groups for n in row['parameter_names']]
        check('Exhaustive disjoint parameter groups '+name,len(names)==len(set(names))==len(named) and set(names)==set(named))
        opt=[row['options'] for row in groups];keys=set(opt[0])-{'lr','group_role'}
        check('Unchanged Adam defaults '+name,keys==set(opt[1])-{'lr','group_role'} and all(opt[0][k]==opt[1][k] for k in keys) and opt[0]['betas']==[.9,.999] and opt[0]['eps']==1e-8 and opt[0]['weight_decay']==0 and opt[0]['amsgrad'] is False)
        check('Exact per-group LR '+name,opt[0]['lr']==1e-4 and opt[1]['lr']==1e-4*j['multiplier'])
        fast=groups[1]['parameter_names'];expected=[n for n in named if n.endswith('.aggregator.scale_logits')] if j['model']=='static' else [n for n in named if '.aggregator.router.' in n]
        check('Both branch accelerated membership '+name,set(fast)==set(expected) and sum(named[n].numel() for n in fast)==(8 if j['model']=='static' else 8520) and groups[0]['parameter_count']==107777)
        check('Optimizer groups retain original order '+name,all(row['parameter_names']==[n for n in named if n in row['parameter_names']] for row in groups))
        end=cp.parent/'OPTIMIZER_GROUPS_END.json'
        if end.exists():check('No group/scheduler drift '+name,io.read(end)['groups']==g['groups'])
        else:limitations.append(name+': optimizer end-sidecar missing; model retained per interruption rule')
        hist=pd.read_csv(cp.parent/'history.csv',float_precision='round_trip');check('Complete finite Huber/val-MSE history '+name,len(hist)==r['epochs'] and np.isfinite(hist[['train_loss','val_loss','val_mse']].to_numpy()).all() and hist.lr.eq(1e-4).all() and hist.huber_delta.eq(1).all())
        check('Best validation checkpoint '+name,1<=r['best_epoch']<=r['epochs'] and float(hist.loc[hist.epoch==r['best_epoch'],'val_loss'].iloc[0])-float(hist.val_loss.min())<=1.0001e-12)
        records=[]
        for split in ['train','val']:records.append((split,pd.read_csv(cp.parent/(split+'.csv'),float_precision='round_trip')))
        eps=list((io.OUT/'evaluation'/name).glob('attempt_*/COMPLETE.json'));check('One complete TEST '+name,len(eps)==1)
        er=io.verify_record(eps[0]);evalstarts.append(er['started_utc']);evalends.append(er['ended_utc'])
        records.append(('test',pd.read_csv(eps[0].parent/'predictions_losses_weights.csv',float_precision='round_trip')))
        for split,f in records:
            f=f.sort_values(['anchor_date','ticker']).reset_index(drop=True);z=all_labels[j['h']];z=z[z.split==split].sort_values(['anchor_date','ticker']).reset_index(drop=True)
            check('Split counts/keys/float32 labels '+name+' '+split,len(f)==io.COUNTS[split] and not f.duplicated(['ticker','anchor_date']).any() and f[['ticker','anchor_date']].equals(z[['ticker','anchor_date']]) and np.array_equal(f.y_true.to_numpy(np.float32),z.y_true.to_numpy(np.float32)))
        for _,attempt in io.attempts(j):check('Identical retry '+name,attempt['job']==j)
    check('All training before TEST barrier',all(io.elapsed(t,finished['utc'])>=0 for t in trainingends))
    check('All TEST after barrier',all(io.elapsed(finished['utc'],t)>=0 for t in evalstarts))
    check('Analysis after all new TEST inference',all(io.elapsed(t,a['started_utc'])>=0 for t in evalends))
    events=[io.read(p) for p in (io.OUT/'events').glob('*.json')]
    check('caffeinate -dimsu recorded',any(r.get('caffeinate_command')==['/usr/bin/caffeinate','-dimsu','-w',str(r.get('pid'))] for r in events))
    for file,n in [('test_metrics_per_seed.csv',90),('five_seed_summary.csv',18),('ensemble_metrics.csv',18),('router_diagnostics_per_seed.csv',150),('manipulation_summary.csv',30),('manipulation_gate.csv',6),('joint_bootstrap_tests.csv',30),('holm_families.csv',30),('prediction_sources.csv',90),('best_epochs_per_run.csv',90)]:check('Every cell '+file,len(readcsv(file))==n)
    b=io.module('_22f_validation_stats_b',io.OLD/'analyze22b_v3.py');s=b.s
    J=readcsv('joint_bootstrap_tests.csv');R=readcsv('router_diagnostics_per_seed.csv');S=readcsv('manipulation_summary.csv');G=readcsv('manipulation_gate.csv');F=readcsv('holm_families.csv')
    for (contrast,metric),g in J.groupby(['contrast','metric']):
        check('Three horizons '+contrast+' '+metric,len(g)==3 and set(g.h)==set(io.HS))
        adjusted=s.holm(dict(zip(g.index,g.holm_input_p)))
        check('Stored Holm '+contrast+' '+metric,all(np.isclose(adjusted[i],r.p_holm_3h,rtol=1e-12,atol=1e-15) for i,r in g.iterrows()))
        check('Undefined conservative p1 '+contrast,g.loc[~g.test_defined,'holm_input_p'].eq(1).all())
    check('Exactly10 prespecified families',len(J.groupby(['contrast','metric']))==10 and len(J[J.family=='PRIMARY'])==12 and len(J[J.family=='SECONDARY'])==18)
    for r in G.itertuples():
        summary=S[(S.arm==r.arm)&(S.h==r.h)].set_index('branch');v=summary.loc[['daily','weekly'],'weight_mad_mean'].to_numpy(float)
        expected='UNDEFINED' if not np.isfinite(v).all() else 'PASS' if (v>=.005).any() else 'FAIL'
        check('Variant/h manipulation '+r.arm+str(r.h),expected==r.status and not r.affects_tests)
    for r in S.itertuples():
        a=R[(R.arm==r.arm)&(R.h==r.h)&(R.branch==r.branch)]
        check('Per-seed manipulation count '+r.arm+str(r.h)+r.branch,len(a)==5 and int((a.mean_abs_deviation_from_equal>=.005).sum())==r.n_seeds_mad_ge_005)
    for r in J.itertuples():
        boot=np.load(io.ROOT/r.bootstrap,allow_pickle=False);check('20k saved replicates '+r.contrast+str(r.h)+r.metric,boot.shape==(20000,))
        if r.test_defined:
            lo,hi=np.quantile(boot,[.025,.975],method='linear');check('Percentile interval',np.allclose([lo,hi],[r.ci_lower_95,r.ci_upper_95],rtol=1e-12,atol=1e-15))
            check('Five seed outcomes',r.seed_wins+r.seed_losses+r.seed_ties==5)
            if r.family=='PRIMARY':
                upper=float(np.quantile(-boot if r.metric=='mse' else boot,.95,method='linear'));check('First-arm signed bound; negatives retained',np.isclose(upper,r.improvement_upper95,rtol=1e-12,atol=1e-15))
                if r.metric=='mse':check('Fixed denominator percent endpoints/bound',np.allclose(100*np.array([r.theta_hat,lo,hi,upper])/r.denominator_mean_mse,[r.estimate_pct,r.ci95_lower_pct,r.ci95_upper_pct,r.mse_improvement_upper95_pct],rtol=1e-12,atol=1e-14))
        label='not significant' if not r.p_holm_3h<.05 or r.theta_hat==0 else 'significant improvement of the first arm' if (r.metric=='mse' and r.theta_hat<0) or (r.metric=='rank_ic' and r.theta_hat>0) else 'significant deterioration of the first arm'
        check('Direction-specific label',label==r.direction_label)
        check('Positive finding labelled EXPLORATORY',not r.positive_finding or 'EXPLORATORY' in r.evidence)
        if r.family=='PRIMARY':
            g=G[(G.arm==r.first)&(G.h==r.h)].iloc[0];check('Interpretation only manipulation gate',r.manipulation_status==g.status)
            if g.status=='FAIL':check('Failed manipulation never supports H','manipulation failed — no conclusion about H'==r.interpretation)
        else:check('SECONDARY no unrequested bounds',pd.isna(r.improvement_upper95))
    check('No NW tests/columns',not any(c.lower().startswith(('p_nw','nw_')) for c in J))
    for h in io.HS:
        d=np.load(dest/f'draws/h{h}.npz',allow_pickle=False);wt,di,si=s.draws(h,5,391)
        check('Step16 exact RNG streams '+str(h),np.array_equal(d['seed_weights'],wt) and np.array_equal(d['date_indices'],di) and np.array_equal(d['seed_indices'],si) and np.array_equal(d['seeds'],io.BOOT_SEEDS))
    for r in readcsv('protected_sources.csv').itertuples():check('Read-only source '+r.path,io.sha(io.ROOT/r.path)==r.sha256)
    paths=readcsv('prediction_sources.csv');check('Only45 new training cells;45 references reused',paths.provenance.eq('Step22F new parameter-group run').sum()==45 and paths.provenance.str.contains('stored predictions only').sum()==45)
    for r in paths[paths.provenance.str.contains('stored predictions only')].itertuples():
        ref=io.verify_record(io.ROOT/r.evaluation);check('Reference metrics after barrier',io.elapsed(finished['utc'],ref['started_utc'])>=0)
    cp=io.OUT/'VALIDATION_CHECKS.csv';io.csv(cp,pd.DataFrame(checks))
    report=io.OUT/'REPORT_STEP22F_VALIDATION.md';io.text(report,'# BLOCKING\n\n'+('None.\n' if not limitations else '\n'.join('- '+x for x in limitations)+'\n')+'\nIndependent validation PASS: '+str(len(checks))+' artifact,optimizer,profile,data,read-order,contrast/family/manipulation and effect-bound checks. No new hypothesis tests.\n\nProtocol SHA-256: `'+io.check_protocol()+'`.\n\nSource files unchanged;45 new fits,45 reused reference cells,90 forecast cells,30 test rows and10 three-horizon Holm families retained. Manipulation status alters interpretation only. PRIMARY bounds use signed first-arm improvement and Fixed’s fixed denominator; negative upper bounds retained.\n\nChecks: ['+cp.name+']('+str(cp)+').\n')
    io.js(io.OUT/'VALIDATION_COMPLETE.json',dict(protocol_sha256=io.check_protocol(),started_utc=start,ended_utc=io.utc(),passed=True,n_checks=len(checks),limitations=limitations,artifacts=[io.artifact(cp),io.artifact(report)]))
    print('INDEPENDENT VALIDATION PASS:',len(checks),'checks',flush=True)
if __name__=='__main__':main()
