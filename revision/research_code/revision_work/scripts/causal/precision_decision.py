"""Separate ticker confounding from event localization; assess adjustment-rounding compatibility."""
from core import *
from build_and_audit import load_prices,adjusted

def main():
 rows=[];levels=[];nearrows=[]
 for t in common.FROZEN_TICKERS:
  o=load_prices(t).set_index('datetime');n=load_prices(t,True).set_index('datetime');a=adjusted(n.reset_index()).set_index('datetime');ix=o.index.intersection(n.index);o=o.loc[ix];n=n.loc[ix];a=a.loc[ix]
  q=o[CHANNELS[:4]].to_numpy()/a[CHANNELS[:4]].to_numpy();relative=np.log(q);spread=np.ptp(relative,axis=1)
  # Piecewise reference only removes the two separately documented genuine vendor steps.
  regimes=np.zeros(len(ix),int)
  stepdates={'CWEN':['2026-06-01'],'NEE':['2026-06-05']}.get(t,[])
  for date in stepdates:regimes+=(ix>=date)
  residual=np.empty(len(ix))
  for regime in np.unique(regimes):
   mask=regimes==regime;residual[mask]=np.abs(relative[mask,3]-np.median(relative[mask,3]))
  # Compatibility envelope for serial float32 multiplication of both snapshots' later dividend factors.
  # This is a conditional rounding bound, not evidence of Yahoo's undocumented internal arithmetic.
  full=pd.read_csv(ROOT/f'data_raw_revision/prices/{t}.csv');dates=pd.to_datetime(full.Date);div=(full.Dividends!=0).to_numpy();later=np.array([int((div&(dates>d)).sum()) for d in ix]);oldlater=np.array([int((div&(dates>d)&(dates<=CUTOFF)).sum()) for d in ix]);ops=oldlater+later+4;u=np.finfo(np.float32).eps/2;bound=ops*u/(1-ops*u)
  retained=(ix>='2015-08-05');ratio=residual[retained]/bound[retained]
  rows.append(dict(ticker=t,max_daily_OHLC_log_scale_spread=float(spread[retained].max()),max_piecewise_close_log_scale_residual=float(residual[retained].max()),max_residual_over_conditional_cumulative_rounding_envelope=float(ratio.max()),fraction_residuals_inside_envelope=float((ratio<=1).mean()),n_dividend_events_new=int(div.sum()),interpretation='Compatibility only: serial float32 vendor factor arithmetic is not authenticated'))
 d=csvout('precision_adjustment_rounding_compatibility.csv',rows)
 w=pd.read_csv(OUT/'precision_window_details.csv');top=w.nlargest(20,'max_standardized_window')
 for t,g in w.groupby('ticker'):
  q=g.original_source_close;hi=g.nlargest(min(20,len(g)),'max_standardized_window');rank=np.searchsorted(np.sort(q),hi.original_source_close)/len(q)
  levels.append(dict(ticker=t,all_source_close_min=float(q.min()),all_source_close_median=float(q.median()),all_source_close_max=float(q.max()),top20_source_close_min=float(hi.original_source_close.min()),top20_source_close_median=float(hi.original_source_close.median()),top20_median_price_percentile=float(np.median(rank))))
 csvout('precision_price_level_check.csv',levels)
 assoc=pd.read_csv(OUT/'precision_action_association.csv');active=assoc[(assoc.ticker!='all')&(assoc.n_large_difference_windows>0)]
 # Ticker-stratified enrichment must be used: dividend-free tickers have exactly zero differences.
 mx=float(active.concentration_ratio.max());compatible=bool((d.fraction_residuals_inside_envelope==1).all());sp=float(d.max_daily_OHLC_log_scale_spread.max());e=json.loads((OUT/'precision_diagnostic_evidence.json').read_text())
 passed=bool(e['below_revised_max'] and mx<1.1 and compatible and sp<1e-12)
 decision=dict(completed_utc=utc(),original_test='FAILED at 1e-6; unchanged',status='PASS at numerical precision (documented tolerance)' if passed else 'STOP: numerical explanation not adequately supported',stage3a_authorized_by_diagnostic=passed,revised_standardized_tolerance=1e-4,observed_max_standardized=e['standardized_distribution']['max'],max_ticker_stratified_action_enrichment=mx,max_daily_OHLC_log_scale_spread=sp,all_price_scale_residuals_inside_conditional_float32_cumulative_envelope=compatible,top20_near_source_actions=int(top.near_action_or_step_source_pm5.sum()),top20_near_anchor_actions=int(top.near_action_or_step_anchor_pm5.sum()),justification='Tiny OHLC changes share a common daily multiplier; volumes and non-dividend tickers agree exactly. No within-ticker event localization; residuals compatible with accumulated float32 dividend-factor rounding. Final CSV serialization alone is insufficient. Vendor internal arithmetic remains unverified.',caveat='Compatibility supports a numerical explanation; it does not authenticate a point-in-time vendor vintage or establish the exact vendor algorithm.')
 jsonout('precision_gate_decision.json',decision);print(json.dumps(decision,indent=2));print(d.to_string(index=False))
if __name__=='__main__':main()
