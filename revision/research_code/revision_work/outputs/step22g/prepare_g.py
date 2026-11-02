"""TRAIN/VAL-only cross-sectional target setup and opaque frozen W3 gate."""
import sys
sys.dont_write_bytecode=True
import io_g as io
import worker_g as w
np,pd=w.old.np,w.old.pd

def main():
    start=io.utc();hashes=pd.read_csv(io.F/'tensor_hash_checks.csv');assert len(hashes)==8
    for r in hashes.itertuples():assert io.sha(io.ROOT/r.path)==r.expected_sha256
    assert io.read(io.F/'DATA_GATE.json')['passed']
    ds=w.dataset(False);info=w.INFO[id(ds)];train=ds.split=='train';values={};rows=[];alltargets=[]
    dates=np.unique(ds.anchor_date)
    for h in io.HS:
        y=info['cs64'][h];sd=float(np.std(y[train],ddof=1,dtype=np.float64));assert np.isfinite(sd) and sd>0;values[str(h)]=sd
        cast=ds.y[h].astype(np.float64)
        centered=np.array([y[ds.anchor_date==d].mean() for d in dates]);assert np.max(np.abs(centered))<1e-14
        rows.append(dict(h=h,n_train=int(train.sum()),sigma_cs=sd,ddof=1,train_mean=float(y[train].mean()),max_abs_date_mean=float(np.max(np.abs(centered))),max_float32_cast_error=float(np.max(np.abs(y-cast))),target_precision='float64 demeans/SD;float32 model targets'))
        alltargets.append(pd.DataFrame(dict(ticker=ds.ticker,anchor_date=pd.to_datetime(ds.anchor_date).strftime('%Y-%m-%d'),split=ds.split,h=h,y_realized=info['raw'].y[h],y_cs_float64=y,y_cs_training_float32=ds.y[h])))
    io.csv(io.OUT/'tensor_hash_checks.csv',hashes);io.csv(io.OUT/'sigma_cs.csv',pd.DataFrame(rows));io.csv(io.OUT/'targets_trainval.csv',pd.concat(alltargets,ignore_index=True))
    io.js(io.OUT/'SIGMAS_CS.json',dict(utc=io.utc(),values=values,ddof=1,arithmetic='float64 per-date mean and TRAIN SD, before model float32 cast',source='frozen original V2 realized returns; TRAIN only'))
    io.js(io.OUT/'DATA_GATE.json',dict(protocol_sha256=io.sha(io.OUT/'PROTOCOL_STEP22G.md'),started_utc=start,ended_utc=io.utc(),passed=True,tensor_hashes=8,trainval_counts={sp:int((ds.split==sp).sum()) for sp in ['train','val']},test_deserialized=False,causal_trainval_bitwise_standardization=True,prior_partition_label_proof=io.artifact(io.F/'DATA_GATE.json'),all_dates_have17_stocks=True))
    print('DATA/TARGET GATE PASS:',values,'no TEST deserialization.')
if __name__=='__main__':main()
