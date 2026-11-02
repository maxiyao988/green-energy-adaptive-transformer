"""Hash-only TEST identity gate; compute hyperparameter SDs from TRAIN rows only."""
from io_d import *
import sys,os
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
for k in ENV_KEYS[:5]:os.environ[k]='1'
import numpy as np,pandas as pd
from window_data import load

def main():
    if (OUT/'DATA_GATE.json').exists():
        assert read(OUT/'DATA_GATE.json')['passed'];return
    started=utc();manifest=read(B2/'tensor_manifest.json');by={x['path']:x for x in manifest['files']}
    design=read(B2/'design.json')
    registry=pd.read_csv(ROOT/'revision_work/outputs/analysis16/registry.csv')
    checks=pd.read_csv(ROOT/'revision_work/outputs/analysis16/prediction_checks.csv')
    protected=pd.read_csv(ROOT/'revision_work/outputs/analysis16/protected_label_sources.csv')
    proof={r.path:r.sha256 for r in protected.itertuples()}
    hashrows,runrows,stats=[],[],[]
    values={}
    for w in ['W1','W2']:
        folder=B2/'tensors'/w
        chosen=[folder/'sample_index.csv']+[folder/f'y_h{h}.npy' for h in HS]+[folder/'C'/f'X_{b}_{kind}.npy' for b in ['daily','weekly'] for kind in ['raw','standardized']]
        for path in chosen:
            item=by[rel(path)];actual=sha(path)
            assert actual==item['sha256'],path
            if path.name.startswith('y_') or path.name=='sample_index.csv':assert proof[rel(path)]==actual
            hashrows.append(dict(path=rel(path),expected_sha256=item['sha256'],actual_sha256=actual,passed=True,read_mode='opaque streaming hash; no TEST deserialization'))
        meta=pd.read_csv(folder/'sample_index.csv')
        assert not meta.duplicated(['ticker','anchor_date']).any()
        for sp,n in counts(w).items():
            z=meta[meta.split==sp];r=design['windows'][w]['ranges'][sp]
            assert len(z)==n and z.anchor_date.nunique()==r['n_anchors'] and z.anchor_date.min()==r['first'] and z.anchor_date.max()==r['last']
        refs=registry[(registry['group']=='rolling')&(registry.input_set=='C')&(registry.window==w)&registry.model.isin(['static','adaptive'])]
        assert len(refs)==30
        for r in refs.itertuples():
            ck=checks[checks.id==r.id];assert len(ck)==1 and ck.keys_equal.iloc[0] and ck.labels_float32_equal.iloc[0]
            cp=(ROOT/r.source).parent/'COMPLETE.json';record=read(cp)
            deps={d['path']:d['sha256'] for d in record['dependencies']}
            for path in [folder/'sample_index.csv',folder/f'y_h{r.h}.npy']:
                assert deps[rel(path)]==by[rel(path)]['sha256']
            runrows.append(dict(id=r.id,complete=rel(cp),complete_sha256=sha(cp),keys_and_float32_labels_previously_verified=True,dependency_tensor_hashes_equal=True))
        ds=load(w,False);train=ds.split=='train';values[w]={}
        for h in HS:
            y=ds.y[h][train].astype(np.float64);sd=float(y.std(ddof=1));assert np.isfinite(sd) and sd>0
            values[w][str(h)]=sd
            stats.append(dict(window=w,h=h,n_train=len(y),mean=float(y.mean()),sd_ddof0=float(y.std(ddof=0)),sd_ddof1=sd,selected_delta=sd))
            # Direct TRAIN/VAL label comparison reads only those columns, never TEST predictions.
            for model in ['static','adaptive']:
                r=refs[(refs.model==model)&(refs.h==h)&(refs.seed==42)].iloc[0]
                for sp in ['train','val']:
                    f=pd.read_parquet((ROOT/r.source).parent/f'{sp}.parquet',columns=['ticker','anchor_date','y_true'])
                    f.anchor_date=pd.to_datetime(f.anchor_date).dt.strftime('%Y-%m-%d');f=f.sort_values(['ticker','anchor_date']).reset_index(drop=True)
                    mask=ds.split==sp
                    mine=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),y_true=ds.y[h][mask])).sort_values(['ticker','anchor_date']).reset_index(drop=True)
                    assert f[['ticker','anchor_date']].equals(mine[['ticker','anchor_date']]) and np.array_equal(f.y_true.to_numpy(np.float32),mine.y_true.to_numpy(np.float32))
    csv(OUT/'TRAIN_TARGET_SD.csv',pd.DataFrame(stats));csv(OUT/'data_hash_checks.csv',pd.DataFrame(hashrows));csv(OUT/'existing_rolling_identity_checks.csv',pd.DataFrame(runrows))
    js(OUT/'SIGMAS.json',dict(utc=utc(),ddof=1,arithmetic='float64 from frozen float32 TRAIN labels',values=values))
    js(OUT/'DATA_GATE.json',dict(started_utc=started,ended_utc=utc(),passed=True,protocol_sha256=sha(OUT/'PROTOCOL_STEP22D.md'),
       hash_checks=len(hashrows),existing_run_identity_checks=len(runrows),fallback_used=False,test_label_contents_deserialized=False,
       test_predictions_read=False,train_val_direct_float32_matches=True,causal_standardization_bitwise_equal=True))
    print('DATA GATE PASS:',len(hashrows),'tensor hashes,',len(runrows),'old-run identity proofs; TEST not deserialized.')
if __name__=='__main__':main()
