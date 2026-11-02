"""Pre-training archived source inventory: metadata and opaque hashes only."""
import sys
sys.dont_write_bytecode=True
import io_e as io
import csv

def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def one_fit(root,pattern):
    good=[p for p in (root/pattern).glob('attempt_*/COMPLETE.json') if io.read(p).get('status')=='COMPLETE']
    assert len(good)==1,(pattern,len(good));return good[0]
def main():
    started=io.utc();records=[];reg=rows(io.ROOT/'revision_work/outputs/analysis16/registry.csv')
    ck={r['id']:r for r in rows(io.ROOT/'revision_work/outputs/analysis16/prediction_checks.csv')}
    old=io.read(io.OLD/'T0_SOURCES.json')['sources']
    def add(setting,model,w,h,seed,source,expected,cp,kind,route_source=None):
        source=io.Path(source);assert source.is_file() and io.sha(source)==expected
        c=io.read(cp);lp=cp.parent/'static_logits.json';history=cp.parent/'history.csv'
        if model=='static':assert lp.exists()
        assert history.exists() and c.get('best_epoch') is not None
        records.append(dict(setting=setting,model=model,window=w,h=int(h),seed=int(seed),predictions=io.rel(source),sha256=expected,
          fit_complete=io.rel(cp),fit_complete_sha256=io.sha(cp),history=io.rel(history),history_sha256=io.sha(history),
          static_logits=io.rel(lp) if model=='static' else '',static_logits_sha256=io.sha(lp) if model=='static' else '',source=kind,route_source=route_source or ''))
    for r in reg:
        if r['input_set']=='C' and r['model'] in ['static','adaptive'] and r['config']=='daily_weekly' and ((r['group']=='rolling' and r['window'] in ['W1','W2']) or (r['group']=='core' and r['window']=='W3')):
            c=ck[r['id']];assert c['keys_equal']=='True' and c['labels_float32_equal']=='True';assert io.sha(io.ROOT/r['source'])==r['sha256']
            cp=(io.ROOT/r['source']).parent/'COMPLETE.json'
            if not cp.exists():cp=io.ROOT/next(z['complete'] for z in old if z['model']==r['model'] and z['h']==int(r['h']) and z['seed']==int(r['seed']))
            add('T0',r['model'],r['window'],r['h'],r['seed'],io.ROOT/r['canonical'],c['canonical_sha256'],cp,'Step16 T0',r['source'])
    root=io.ROOT/'revision_work/outputs/step22b2';ac=io.read(root/'ANALYSIS_COMPLETE.json');a=next(a for a in ac['artifacts'] if a['path'].endswith('/prediction_files.csv'))
    assert io.sha(io.ROOT/a['path'])==a['sha256']
    for r in rows(io.ROOT/a['path']):
        if r['setting']=='T3' and r['model'] in ['static','adaptive']:
            ec=io.read(io.ROOT/r['complete']);expected=next(a['sha256'] for a in ec['artifacts'] if a['path']==r['predictions'])
            cp=one_fit(root/'fits',f"T3_{r['model']}_h{r['h']}_s{r['seed']}")
            add('T3',r['model'],'W3',r['h'],r['seed'],io.ROOT/r['predictions'],expected,cp,'Step22B2')
    ac=io.read(io.D/'ANALYSIS_COMPLETE.json');a=next(a for a in ac['artifacts'] if a['path'].endswith('/lstm_prediction_files.csv'))
    assert io.sha(io.ROOT/a['path'])==a['sha256']
    for r in rows(io.ROOT/a['path']):
        if r['window'] in ['W1','W2']:cp=one_fit(io.D/'fits',f"{r['setting']}_lstm_{r['window']}_h{r['h']}_s{r['seed']}")
        elif r['setting']=='T3':cp=one_fit(root/'fits',f"T3_lstm_h{r['h']}_s{r['seed']}")
        else:cp=io.ROOT/next(z['complete'] for z in old if z['model']=='lstm' and z['h']==int(r['h']) and z['seed']==int(r['seed']))
        add(r['setting'],'lstm',r['window'],r['h'],r['seed'],io.ROOT/r['predictions'],r['sha256'],cp,'Step22D LSTM')
    assert len(records)==210 and len({(r['setting'],r['model'],r['window'],r['h'],r['seed']) for r in records})==210
    # Standard library CSV writer; prediction contents are never opened by a parser.
    p=io.OUT/'ARCHIVED_SOURCE_INVENTORY.csv'
    with p.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    io.js(io.OUT/'ARCHIVED_SOURCE_GATE.json',dict(started_utc=started,ended_utc=io.utc(),passed=True,n_cells=210,
      parsed_test_predictions=False,parsed_test_labels=False,opaque_hashes_verified=True,inventory=io.artifact(p)))
    print('Archived210 cells: metadata/history/logit presence and opaque source hashes PASS; no TEST content parsed.')
if __name__=='__main__':main()
