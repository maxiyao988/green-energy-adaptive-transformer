"""No-training audit of matching the imported Weekly26 source interval."""
from b3_io import *
import fullrun14a_worker_v2 as base
import numpy as np,pandas as pd,torch,importlib
sys.path.insert(0,str(ROOT/'revision_work/outputs/fullrun/sessions/14c_B2_20261017T101438Z_1a8d5041/scripts'))
import b2_data

def feature_module():
 old=b2_data.EXP;b2_data.EXP=EXP
 try:return b2_data.feature_module()
 finally:b2_data.EXP=old

def main():
 base.check_profile();common,formal,core=base.common,base.formal,base.causal_core
 ds=common.load_frozen_panel();cds=core.load_causal();formal._formal_check_dataset_contract(ds);formal._formal_check_dataset_contract(cds)
 fb=feature_module();rows=[];examples=[];lengths=[]
 for ticker in sorted(common.FROZEN_TICKERS):
  m=ds.ticker==ticker;anchors=pd.to_datetime(ds.anchor_date[m]).to_numpy();split=ds.split[m]
  d=fb.load_prices(ticker);w=core.builder._reconstruct_weekly_from_daily(d)
  wi=np.searchsorted(w.available_date.to_numpy(),anchors,side='right')-1
  data_dates=pd.DatetimeIndex(d.datetime);all_dates=data_dates.to_numpy();earliest=w.week_start.to_numpy()[wi-25];last=w.available_date.to_numpy()[wi]
  lo=np.searchsorted(all_dates,earliest,side='left');hi=np.searchsorted(all_dates,last,side='right');n=hi-lo
  assert (n>0).all() and (last<=anchors).all()
  ow,_,_,ref,di=fb.windows(d,anchors)
  assert np.array_equal(ow[1].astype(np.float32),ds.X_weekly[m]);assert np.array_equal(fb.transform(ow[1],ref).astype(np.float32),cds.X_weekly[m])
  for j,a in enumerate(anchors):
   first=all_dates[lo[j]];end=all_dates[hi[j]-1]
   assert end==last[j] and end<=a and first>=earliest[j]
   window=w.iloc[wi[j]-25:wi[j]+1]
   weekly_first=all_dates[np.searchsorted(all_dates,window.iloc[0].week_start.to_datetime64())]
   assert weekly_first==first and len(window)==26
   current_after=int((di[j]+1)-hi[j]);assert current_after>=0
   row=dict(ticker=ticker,anchor_date=str(pd.Timestamp(a).date()),split=split[j],valid_daily_sessions=int(n[j]),daily_first_session=str(pd.Timestamp(first).date()),daily_last_session=str(pd.Timestamp(end).date()),weekly_first_underlying_session=str(pd.Timestamp(weekly_first).date()),weekly_last_underlying_session=str(pd.Timestamp(last[j]).date()),weekly_first_week_label=str(pd.Timestamp(earliest[j]).date()),weekly_bars=26,coverage_identical=True,no_bar_after_last_completed_week=True,excluded_unfinished_week_sessions=current_after,causal_reference_date=str(pd.Timestamp(a).date()),causal_reference_after_completed_week=bool(a>last[j]))
   rows.append(row)
  lengths.extend(n.tolist());print('TICKER_DESIGN',ticker,int(n.min()),int(n.max()),flush=True)
 allrows=pd.DataFrame(rows);fixed=max(lengths)
 assert len(allrows)==43571 and not allrows.duplicated(['ticker','anchor_date']).any()
 for s,n in COUNTS.items():assert len(allrows[allrows.split==s])==n
 for split in ['train','val','test']:
  q=allrows[(allrows.ticker=='AES')&(allrows.split==split)];examples.append(q.iloc[len(q)//2].to_dict())
 distributions=[]
 for split,g in [('all',allrows)]+[(s,allrows[allrows.split==s]) for s in ['train','val','test']]:
  for n,count in g.valid_daily_sessions.value_counts().sort_index().items():distributions.append(dict(split=split,valid_daily_sessions=int(n),n_windows=int(count),fraction=float(count/len(g))))
 csvout(EXP/'design_windows.csv',rows);csvout(EXP/'design_session_count_distribution.csv',distributions);csvout(EXP/'design_examples.csv',examples)
 source=[artifact(Path(m.__file__)) for m in [common,formal,core,fb,core.builder]]
 # This verifies the existing optional expert mask rather than training/reimplementing any model.
 formal.set_seed(42);expert=formal.DualAttentionScaleExpert(input_dim=32,patch_size=5);expert.eval()
 valid_n=min(lengths);h=torch.randn(2,fixed,32);mask=torch.arange(fixed)[None,:]<torch.tensor([valid_n,fixed])[:,None]
 with torch.no_grad():
  padded=expert(h,valid_mask=mask);trimmed=expert(h[:1,:valid_n])
 finite=bool(torch.isfinite(padded).all());difference=float((padded[:1]-trimmed).abs().max()) if finite else None
 evidence=dict(utc=utc(),no_training=True,fixed_daily_length=fixed,valid_daily_min=min(lengths),valid_daily_max=max(lengths),row_counts=COUNTS,rows=43571,all_coverage_identical=True,weekly_O_and_C_bitwise_unchanged=True,no_daily_or_weekly_bar_after_last_completed_week=True,causal_reference_after_week_windows=int(allrows.causal_reference_after_completed_week.sum()),causal_reference_formula='log(OHLC / anchor-date Close), unchanged Stage3a reference',expert_optional_mask_existing=True,expert_padding_invariance_probe=dict(valid_daily_min=valid_n,fixed_daily_length=fixed,patch_size=5,finite=finite,max_abs_representation_delta_vs_unpadded=difference),dependencies=source,threads=[12,12])
 jsonout(EXP/'design_diagnostics.json',evidence)
 textout(EXP/'PRETRAIN_DESIGN.md','# BLOCKING\n\nNo training started. The causal reference rule requires resolution before claiming strict matched historical coverage: Stage3a uses anchor-date Close, which can be later than the last completed week. The existing expert token mask also needs an end-to-end padding check because the branch does not forward it and inter-patch attention does not mask fully empty patches.\n\n# B3 design before training\n\nFixed right-padded Daily length: '+str(fixed)+'. Valid sessions range: '+str(min(lengths))+'–'+str(max(lengths))+'.\n\n'+table(['split','valid sessions','windows','fraction'],[[r['split'],r['valid_daily_sessions'],r['n_windows'],r['fraction']] for r in distributions])+'\n'+table(['split','anchor','Daily first','Daily last','Weekly first source session','Weekly last source session','valid sessions'],[[r['split'],r['anchor_date'],r['daily_first_session'],r['daily_last_session'],r['weekly_first_underlying_session'],r['weekly_last_underlying_session'],r['valid_daily_sessions']] for r in examples])+'\nCoverage and formal keys/counts match for all43571 samples. Weekly O/C tensors round-trip bitwise to existing bindings. No bar after the last completed week is selected. Anchor-Close references after that week occur in '+str(evidence['causal_reference_after_week_windows'])+' windows. Probe of existing optional expert mask: '+json.dumps(evidence['expert_padding_invariance_probe'])+'.\n')
 print('B3_DESIGN_DIAGNOSTICS_COMPLETE',json.dumps(evidence),flush=True)
if __name__=='__main__':main()
