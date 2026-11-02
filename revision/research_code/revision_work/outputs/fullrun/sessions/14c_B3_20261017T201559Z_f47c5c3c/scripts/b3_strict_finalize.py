"""Verify the authorized strict policy and append gap/volume design evidence."""
from b3_io import *
from b3_data import base,common,formal,core,feature_module,load_panel,contract
import b3_preflight as prepared
import numpy as np,pandas as pd

def main():
 policy=json.loads((EXP/'causal_reference_policy.json').read_text());assert policy['policy']=='completed_week_close' and policy['user_authorized']
 assert json.loads((EXP/'LOADER_CHECKS.json').read_text())['passed']
 d=load_panel('C');o=load_panel('O');contract(d);contract(o);fb=feature_module();windows=pd.read_csv(EXP/'design_windows.csv')
 per_window=[];checks=[]
 for ticker in sorted(common.FROZEN_TICKERS):
  ix=np.flatnonzero(d.ticker==ticker);g=windows.iloc[ix];prices=fb.load_prices(ticker);dates=pd.DatetimeIndex(prices.datetime)
  anchors=pd.to_datetime(d.anchor_date[ix]).to_numpy();ai=dates.get_indexer(anchors);lo=dates.get_indexer(pd.to_datetime(g.daily_first_session));hi=dates.get_indexer(pd.to_datetime(g.daily_last_session))+1
  gap=ai-(hi-1);assert (gap>=0).all() and np.array_equal(gap,g.excluded_unfinished_week_sessions.to_numpy())
  old,_,_,_,_=fb.windows(prices,anchors);end_close=prices.close.to_numpy()[hi-1]
  expected_weekly=fb.transform(old[1],end_close).astype(np.float32);assert np.array_equal(expected_weekly,d.X_weekly[ix])
  source=prices[core.CHANNELS].to_numpy()
  for n in sorted(g.valid_daily_sessions.unique()):
   local=np.flatnonzero(g.valid_daily_sessions.to_numpy()==n);sample_ix=ix[local]
   raw=np.lib.stride_tricks.sliding_window_view(source,int(n),axis=0).transpose(0,2,1)[lo[local]]
   expected=fb.transform(raw,end_close[local]).astype(np.float32)
   assert np.array_equal(expected,d.X_daily[sample_ix,:n])
   assert np.array_equal(raw.astype(np.float32),o.X_daily[sample_ix,:n])
   # Audit the window means independently from padding/reference selection.
   lv=np.log1p(raw[:,:,4]);dv=(lv-lv.mean(axis=1,keepdims=True)).astype(np.float32)
   wv=np.log1p(old[1][local,:,4]);wv=(wv-wv.mean(axis=1,keepdims=True)).astype(np.float32)
   assert np.array_equal(dv,d.X_daily[sample_ix,:n,4]) and np.array_equal(wv,d.X_weekly[sample_ix,:,4])
  for h in [5,10,20]:
   label=np.log(prices.close.to_numpy()[ai+h]/prices.close.to_numpy()[ai]).astype(np.float32);assert np.array_equal(label,d.y[h][ix])
  for j,k in enumerate(ix):per_window.append(dict(ticker=ticker,anchor_date=str(pd.Timestamp(anchors[j]).date()),split=d.split[k],gap_sessions=int(gap[j]),completed_week_final_session=str(dates[hi[j]-1].date()),denominator_date=str(dates[hi[j]-1].date()),anchor_label_reference_date=str(pd.Timestamp(anchors[j]).date()),valid_daily_sessions=int(hi[j]-lo[j]),weekly_bars=26))
  checks.append(dict(ticker=ticker,n_windows=len(ix),both_price_branches_use_completed_week_end_close=True,daily_volume_centering_real_matched_sessions_only=True,weekly_volume_centering_same26_completed_weeks=True,padding_excluded=True,labels_still_use_anchor_close=True,O_matched_input_unchanged=True,max_gap_sessions=int(gap.max())))
 frame=pd.DataFrame(per_window);assert len(frame)==43571
 grouped=frame.groupby(['anchor_date','split']);assert grouped.gap_sessions.nunique().eq(1).all()
 anchor_frame=grouped.gap_sessions.first().reset_index()
 distribution=[]
 for unit,df in [('ticker_anchor_window',frame),('unique_anchor_date',anchor_frame)]:
  for split in ['all','train','val','test']:
   sub=df if split=='all' else df[df.split.eq(split)]
   for gap,count in sub.gap_sessions.value_counts().sort_index().items():distribution.append(dict(unit=unit,split=split,gap_sessions=int(gap),count=int(count),fraction=float(count/len(sub))))
 per_ticker=[]
 for (ticker,split),sub in frame.groupby(['ticker','split']):
  for gap,count in sub.gap_sessions.value_counts().sort_index().items():per_ticker.append(dict(ticker=ticker,split=split,gap_sessions=int(gap),count=int(count),fraction=float(count/len(sub))))
 csvout(EXP/'gap_by_window.csv',per_window);csvout(EXP/'gap_distribution.csv',distribution);csvout(EXP/'gap_distribution_per_ticker.csv',per_ticker);csvout(EXP/'strict_coverage_volume_checks.csv',checks)
 jsonout(EXP/'STRICT_POLICY_CHECKS.json',dict(utc=utc(),passed=True,policy=policy['policy'],both_branches_denominator_verified=True,volume_means_over_same_calendar_interval=True,labels_unchanged_anchor_close=True,O_unchanged=True,windows=43571,unique_anchor_dates=len(anchor_frame),gap_min=int(frame.gap_sessions.min()),gap_max=int(frame.gap_sessions.max()),gap_positive_windows=int((frame.gap_sessions>0).sum()),gap_definition='number of this security daily-source sessions after completed-week final session through anchor inclusive; zero when anchor is completed-week final session',no_training=True))
 addendum='\n\n## Strict-policy additions required by the user\n\nBoth C price branches use the last session Close of the most recent completed week. Daily volume centering is log1p(daily Volume) minus its mean over the122–127 real daily sessions in this completed26-week interval; Weekly volume centering is log1p(weekly summed Volume) minus its mean over these same26 completed weekly bars. Padding and all unfinished-current-week source sessions are excluded from both volume means. O matched tensors and all frozen labels are verified unchanged; labels still use anchor-date Close, not the feature denominator.\n\nB3-C Weekly price values differ from the formal C pipeline because the denominator changes from anchor Close to completed-week end Close. Comparisons of B3 with formal Daily90+Weekly26 combine the historical coverage change with exclusion of the unfinished current week; the C comparison additionally reflects the expressly selected denominator change. These comparisons do not isolate only a Daily-length effect.\n\nGap is measured in source trading sessions from the completed-week final session to the anchor (0 means the same session). All17 securities agree on the gap at each common anchor. Both sample-level and unique-anchor distributions are retained.\n\n'+table(['unit','split','gap sessions','count','fraction'],[[r['unit'],r['split'],r['gap_sessions'],r['count'],r['fraction']] for r in distribution])+f'\n[Full gaps]({EXP}/gap_by_window.csv), [gap distribution]({EXP}/gap_distribution.csv), [per-ticker gap distribution]({EXP}/gap_distribution_per_ticker.csv), [strict source/volume/label checks]({EXP}/strict_coverage_volume_checks.csv).\n'
 original_textout,original_jsonout=prepared.textout,prepared.jsonout
 def text_with_addendum(path,body):
  if Path(path).name=='DESIGN_REPORT_STEP14C_B3.md':body+=addendum
  return original_textout(path,body)
 def json_with_checks(path,value):
  if Path(path).name=='preflight.json':value=dict(value,strict_policy_checks_passed=True,gap_distribution=artifact(EXP/'gap_distribution.csv'),dependencies=value['dependencies']+[artifact(EXP/n) for n in ['STRICT_POLICY_CHECKS.json','gap_distribution.csv','strict_coverage_volume_checks.csv','LOADER_CHECKS.json']])
  return original_jsonout(path,value)
 prepared.textout=text_with_addendum;prepared.jsonout=json_with_checks
 prepared.final();print('STRICT_B3_PREFLIGHT_COMPLETE',utc(),flush=True)
if __name__=='__main__':main()
