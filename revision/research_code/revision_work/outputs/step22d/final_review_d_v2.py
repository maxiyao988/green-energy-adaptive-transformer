"""Independent post-completion artifact/read-order checks; no new hypothesis tests."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'session_v3'))
import io_d as io
import os
for k in io.ENV_KEYS[:5]:os.environ[k]='1'
import numpy as np,pandas as pd

def main():
    complete=io.barrier();record=io.verify_record(io.OUT/'ANALYSIS_COMPLETE.json')
    data_checks=pd.read_csv(io.OUT/'data_hash_checks.csv')
    assert len(data_checks)==16
    assert all(io.sha(io.ROOT/r.path)==r.expected_sha256 for r in data_checks.itertuples())
    assert not record['problems'],record['problems']
    root=io.ROOT/record['analysis_directory'];runs=[]
    readorder=[]
    for job in io.read(io.OUT/'GRID_PLAN.json')['jobs']:
        cp,r=io.completion(job);assert r['status']=='COMPLETE'
        io.verify_record(cp.parent/'ADAPTER_COMPLETE.json')
        assert r['lr']==job['learning_rate'] and r['delta']==job['delta']
        assert r['test_values_read'] is False
        for point in ['profile_start','profile_end']:
            p=r[point]['profile'];assert p['device']=='cpu' and p['intra']==p['inter']==12
            assert p['environment']['OMP_NUM_THREADS'] is None and p['environment']['OMP_DYNAMIC']==p['environment']['MKL_DYNAMIC']=='FALSE'
        hist=pd.read_csv(cp.parent/'history.csv')
        assert len(hist)==r['epochs'] and hist.epoch.eq(r['best_epoch']).sum()==1
        assert hist.huber_delta.eq(job['delta']).all() or np.allclose(hist.huber_delta,job['delta'],rtol=0,atol=1e-16)
        assert hist.lr.eq(job['learning_rate']).all()
        for sp in ['train','val']:
            frame=pd.read_csv(cp.parent/(sp+'.csv'),usecols=['ticker','anchor_date','split'])
            assert len(frame)==io.counts(job['window'])[sp] and frame.split.eq(sp).all() and not frame.duplicated(['ticker','anchor_date']).any()
        ec=sorted((io.OUT/'evaluation'/io.job_id(job)).glob('attempt_*/COMPLETE.json'));assert len(ec)==1
        er=io.verify_record(ec[0]);io.verify_record(ec[0].parent/'ADAPTER_COMPLETE.json');assert er['status']=='COMPLETE'
        frame=pd.read_csv(ec[0].parent/'predictions_losses_weights.csv',usecols=['ticker','anchor_date','split'])
        assert len(frame)==6647 and frame.split.eq('test').all() and frame.anchor_date.nunique()==391 and not frame.duplicated(['ticker','anchor_date']).any()
        for line in (ec[0].parent/'telemetry.jsonl').read_text().splitlines():
            t=io.json.loads(line)['profile'];assert t['intra']==t['inter']==12 and t['device']=='cpu'
            assert t['environment']['OMP_NUM_THREADS'] is None and t['environment']['OMP_DYNAMIC']==t['environment']['MKL_DYNAMIC']=='FALSE'
        assert io.dt.datetime.fromisoformat(r['ended_utc'])<=io.dt.datetime.fromisoformat(complete['utc'])<=io.dt.datetime.fromisoformat(er['started_utc'])
        assert io.dt.datetime.fromisoformat(er['ended_utc'])<=io.dt.datetime.fromisoformat(record['started_utc'])
        runs.append(dict(job_id=io.job_id(job),status='PASS',epochs=r['epochs'],best_epoch=r['best_epoch']))
        readorder.append(dict(job_id=io.job_id(job),train_end=r['ended_utc'],training_barrier=complete['utc'],test_start=er['started_utc'],test_end=er['ended_utc']))
    families=pd.read_csv(root/'rank_ic_and_cw_Holm_families.csv')
    assert len(families)==36
    for (name,method),g in families.groupby(['family','method']):
        expected=6 if name in ['PRIMARY','SECONDARY'] else 3
        assert len(g)==expected and g.family_size.eq(expected).all()
    for file,n in [('part1_metric_summary.csv',120),('part1_rank_ic_inference.csv',72),('part1_static_adaptive_contrasts.csv',48),('confirmatory_tests.csv',6),('lstm_prediction_files.csv',90)]:
        assert len(pd.read_csv(root/file))==n,(file,n)
    disagreements=[]
    for filename in ['part1_rank_ic_inference.csv','confirmatory_tests.csv']:
        d=pd.read_csv(root/filename)
        for r in d[d.p_nw_holm.notna()].itertuples():
            actual=bool(r.mean>0 and r.p_nw_holm<.05)
            assert (str(r.significantly_positive_NW_primary).lower()=='true')==actual
            nw=bool(r.p_nw_holm<.05);mbb=bool(r.p_mbb_holm<.05)
            if nw!=mbb:disagreements.append(dict(table=filename,family=r.family,input_set=r.input_set,model=r.model,h=r.h,mean=r.mean,p_nw_holm=r.p_nw_holm,p_mbb_holm=r.p_mbb_holm,nw_rejects=nw,mbb_rejects=mbb,positive_NW_flag=actual))
        for r in d.itertuples():
            b=np.load(io.ROOT/r.bootstrap,allow_pickle=False)
            assert b.shape==(20000,)
            if np.isfinite(b).all():
                lo,hi=np.quantile(b,[.025,.975],method='linear')
                assert np.allclose([lo,hi],[r.ci95_lower,r.ci95_upper],rtol=1e-13,atol=1e-15)
    contrasts=pd.read_csv(root/'part1_static_adaptive_contrasts.csv')
    assert len(contrasts[contrasts.window=='W1+W2+W3'])==12
    for r in contrasts.itertuples():
        b=np.load(io.ROOT/r.bootstrap,allow_pickle=False);assert b.shape==(20000,)
        if np.isfinite(b).all():
            lo,hi=np.quantile(b,[.025,.975],method='linear');upper=np.quantile(b if r.metric=='mse' else -b,.95,method='linear')
            assert np.allclose([lo,hi,upper],[r.ci95_lower,r.ci95_upper,r.adaptive_improvement_upper95],rtol=1e-13,atol=1e-15)
            if r.metric=='mse':assert np.isclose(r.adaptive_mse_improvement_upper95_pct,100*upper/r.second_arm_mean_test_mse,rtol=1e-13,atol=1e-15)
    protected=pd.read_csv(root/'protected_sources.csv')
    assert all(io.sha(io.ROOT/r.path)==r.sha256 for r in protected.itertuples())
    assert (BASE/'REPORT_STEP22D_FINAL.md').read_bytes()==(io.OUT/'REPORT_STEP22D.md').read_bytes()
    dest=BASE/('validation_'+io.stamp());dest.mkdir()
    for name,frame in [('run_checks.csv',pd.DataFrame(runs)),('read_order.csv',pd.DataFrame(readorder)),('all_NW_MBB_rejection_disagreements.csv',pd.DataFrame(disagreements,columns=['table','family','input_set','model','h','mean','p_nw_holm','p_mbb_holm','nw_rejects','mbb_rejects','positive_NW_flag']))]:
        with (dest/name).open('x',newline='') as f:frame.to_csv(f,index=False,float_format='%.17g')
    with (dest/'FINAL_VALIDATION.json').open('x') as f:
        io.json.dump(dict(utc=io.utc(),passed=True,protocol_sha256=io.check_protocol(),runs=60,evaluations=60,protected_sources=len(protected),window_tensor_hashes_reverified=len(data_checks),nw_mbb_rejection_disagreements=disagreements,report=io.rel(BASE/'REPORT_STEP22D_FINAL.md')),f,indent=2)
    print('FINAL VALIDATION PASS:',dest)
    print('NW/MBB adjusted rejection disagreements, including negative means:',len(disagreements))
    for r in disagreements:print(r)
if __name__=='__main__':main()
