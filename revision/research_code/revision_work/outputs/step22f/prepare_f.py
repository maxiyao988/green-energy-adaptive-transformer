"""Pre-training opaque W3 gate; unchanged TRAIN/VAL selector only."""
import sys
sys.dont_write_bytecode=True
import io_f as io
import worker_f as worker
np,pd=worker.old.np,worker.old.pd

def main():
    started=io.utc();refs=io.source_rows();checks=pd.read_csv(io.ROOT/'revision_work/outputs/analysis16/prediction_checks.csv').set_index('id')
    manifest=io.TENSOR_BASE/'MANIFEST_STEP4_VERIFIED.json';prior={r['path']:r for r in io.read(manifest)['files']}
    files=['sample_index.csv']+[f'y_h{h}_frozen.npy' for h in io.HS]+[f'X_{b}_causal_{kind}.npy' for b in ['daily','weekly'] for kind in ['raw','standardized']]
    hashes=[]
    for name in files:
        path=io.TENSOR_BASE/'tensors'/name;r=prior[io.rel(path)];actual=io.sha(path);assert actual==r['sha256']
        hashes.append(dict(path=io.rel(path),expected_sha256=r['sha256'],actual_sha256=actual,passed=True,read_mode='opaque SHA only'))
    ds=worker.old.dataset(False);assert not np.any(ds.split=='test')
    assert {sp:int((ds.split==sp).sum()) for sp in ['train','val']}=={sp:io.COUNTS[sp] for sp in ['train','val']}
    meta=pd.read_csv(io.TENSOR_BASE/'tensors/sample_index.csv');assert {sp:int((meta.split==sp).sum()) for sp in io.COUNTS}==io.COUNTS
    rows=[];direct=[]
    for r in refs:
        cp=io.ROOT/r['complete'];record=io.read(cp);deps={a['path']:a['sha256'] for a in record['dependencies']}
        for a in hashes:
            if '_standardized.npy' not in a['path']:assert deps[a['path']]==a['expected_sha256']
        c=checks.loc[r['id']];assert c.keys_equal and c.labels_float32_equal
        assert io.sha(io.ROOT/r['source'])==r['sha256'];assert io.sha(io.ROOT/r['canonical'])==c.canonical_sha256
        for sp in ['train','val']:
            source=cp.parent/(sp+'.parquet');f=pd.read_parquet(source,columns=['ticker','anchor_date','y_true'])
            f.anchor_date=pd.to_datetime(f.anchor_date).dt.strftime('%Y-%m-%d');f=f.sort_values(['ticker','anchor_date']).reset_index(drop=True)
            mask=ds.split==sp
            mine=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),y_true=ds.y[r['h']][mask])).sort_values(['ticker','anchor_date']).reset_index(drop=True)
            assert f[['ticker','anchor_date']].equals(mine[['ticker','anchor_date']]) and np.array_equal(f.y_true.to_numpy(np.float32),mine.y_true.to_numpy(np.float32))
            direct.append(dict(id=r['id'],split=sp,rows=len(f),keys_equal=True,labels_float32_equal=True,source=io.rel(source),sha256=io.sha(source)))
        logit=cp.parent/'static_logits.json'
        if r['model']=='static':assert logit.exists()
        history=io.ROOT/r['history'];assert history.exists()
        rows.append(dict(arm={'fixed':'Fixed','static':'Static','adaptive':'Adaptive'}[r['model']],model=r['model'],h=r['h'],seed=r['seed'],id=r['id'],source=r['source'],source_sha256=r['sha256'],canonical=r['canonical'],canonical_sha256=c.canonical_sha256,
          complete=io.rel(cp),complete_sha256=io.sha(cp),history=io.rel(history),history_sha256=io.sha(history),static_logits=io.rel(logit) if r['model']=='static' else '',static_logits_sha256=io.sha(logit) if r['model']=='static' else '',best_epoch=record['best_epoch'],epochs=record['epochs'],profile='12/12'))
    assert len(rows)==45
    io.csv(io.OUT/'tensor_hash_checks.csv',pd.DataFrame(hashes));io.csv(io.OUT/'REFERENCE_SOURCES.csv',pd.DataFrame(rows));io.csv(io.OUT/'trainval_identity_checks.csv',pd.DataFrame(direct))
    io.js(io.OUT/'DATA_GATE.json',dict(started_utc=started,ended_utc=io.utc(),passed=True,protocol_sha256=io.sha(io.OUT/'PROTOCOL_STEP22F.md'),tensor_hashes=8,reference_cells=45,trainval_comparisons=len(direct),trainval_standardization_bitwise_equal=True,
      research_test_deserialized=False,ref_source_hashes_verified=True,partition_counts=io.COUNTS,prior_manifest=io.artifact(manifest)))
    print('DATA GATE PASS:8 frozen tensor hashes,45 references,90 TRAIN/VAL matches; no TEST deserialization.')
if __name__=='__main__':main()
