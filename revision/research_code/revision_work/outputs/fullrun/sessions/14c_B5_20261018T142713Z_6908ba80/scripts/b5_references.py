"""Reuse seed42 and the five-seed four-mechanism summary; no training."""
from b5_worker import *
import day_io_v3 as oldio
ARCHIVES={'lstm':ROOT/'dataset/audit/main_v2/lstm/panel_lstm_test_predictions.csv','vanilla':ROOT/'dataset/audit/main_v2/vanilla/panel_vanilla_test_predictions.csv','swim':ROOT/'dataset/audit/main_v2/swim_cpu/panel_swim_test_predictions.csv'}

def canonical(frame):
 f=frame.copy();f['anchor_date']=pd.to_datetime(f.anchor_date).dt.strftime('%Y-%m-%d');assert not f.duplicated(['ticker','anchor_date']).any()
 return f.sort_values(['ticker','anchor_date']).reset_index(drop=True)

def main():
 ds=day.common.load_frozen_panel();mask=ds.split=='test';rows=[];sources=[]
 index=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d')))
 for inp in ['O','C']:
  for model in ['lstm','vanilla','swim']:
   for h in [5,10,20]:
    if inp=='O':
     source=ARCHIVES[model];f=pd.read_csv(source);f=f[(f.frequency_setting=='daily_weekly')&(f.horizon.astype(str).str.replace('d','',regex=False).astype(int)==h)&(f.seed==42)]
     provenance='archived MAIN V2'+(' CPU SWiM formal' if model=='swim' else '')
     f=f[['ticker','anchor_date','y_true','y_pred']].copy();f['split']='test';f[['y_true','y_pred']]=f[['y_true','y_pred']].astype(np.float32)
    else:
     spec=oldio.spec_for(f'B4_C_{model}_daily_weekly_h{h}_s42');cp,r=oldio.completed({k:spec[k] for k in oldio.SPEC_KEYS});assert r,spec
     source=cp.parent/'test.parquet';f=pd.read_parquet(source);provenance='14c-day B4 seed42 CPU12/12'
    f=canonical(f);expected=canonical(index.assign(y_true=ds.y[h][mask]));assert len(f)==COUNTS['test'] and (f.split=='test').all()
    assert np.array_equal(f[['ticker','anchor_date']].to_numpy(),expected[['ticker','anchor_date']].to_numpy())
    assert np.array_equal(f.y_true.to_numpy(np.float32),expected.y_true.to_numpy(np.float32))
    assert np.isfinite(f[['y_true','y_pred']]).all().all()
    metric=day.common.evaluate_predictions(f.y_true.to_numpy(),f.y_pred.to_numpy(),f.ticker.to_numpy(),f.anchor_date.to_numpy())
    copy=EXP/'seed42'/f'{inp}_{model}_dw_h{h}_s42_test.parquet'
    with target(copy).open('xb') as out:out.write(f.to_parquet(index=False,engine='pyarrow'))
    rows.append(dict(block='B5',input_set=inp,model=model,frequency_setting='daily_weekly',horizon=h,seed=42,test_mse=metric['pooled_mse'],mean_rank_ic=metric['cross_sectional_rank_ic'],provenance=provenance,source=rel(source),source_sha256=sha(source),prediction=rel(copy),prediction_sha256=sha(copy)))
    sources.append(artifact(source))
 source=ROOT/'revision_work/outputs/fullrun/sessions/20261016T142732626448Z_9ba11b23/summary/cell_metrics.csv'
 four=pd.read_csv(source);assert len(four)==120 and set(four.mechanism)=={'single','fixed','static','adaptive'}
 for _,g in four.groupby(['input_set','mechanism','horizon']):assert set(g.seed)==set(SEEDS) and len(g)==5
 csvout(EXP/'seed42_metrics.csv',rows)
 jsonout(EXP/'reference_provenance.json',dict(utc=utc(),seed42_cells=18,retrained_seed42_summary_cells=0,sources=list({s['path']:s for s in sources}.values()),four_mechanism_summary=artifact(source),notes='O seed42 is archived MAIN V2; C seed42 is 14c-day. The mandatory O LSTM h5 pilot is excluded from five-seed summaries.'))
 print('B5_REFERENCES_PASS',18,120,flush=True)
if __name__=='__main__':main()
