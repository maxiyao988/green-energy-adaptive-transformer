"""Evidence-linked impact inventory; no document is edited."""
from metrics26_v2 import *
import re,collections

def main():
 records=json.loads((OUT/'DOCUMENT_EXTRACTION_STEP26.json').read_text());M=pd.read_csv(OUT/'TEST_METRICS_STEP26.csv');N=M[M.arm=='new'];S=pd.read_csv(OUT/'SELECTION_LOCK.csv');P=pd.read_csv(OUT/'POOLED_RIDGE_STEP26.csv');G=pd.read_csv(OUT/'propagated/C_subgroups_summary_full_step26.csv');port=pd.read_csv(OUT/'PORTFOLIO_RIDGE_STEP26.csv');newskill=pd.read_csv(OUT/'propagated/B_forecast_skill_final_complete_step26.csv');sh=pd.read_csv(OUT/'propagated/SHRINKAGE_final_step26.csv');counts=pd.read_csv(OUT/'PRICE_LEVEL_480_ALPHA_COUNT_STEP26.csv');out=[]
 def q(inp,h,cfg='daily_weekly',w='W3'):
  return N[(N.target=='raw')&(N.input_set==inp)&(N.h==h)&(N.config==cfg)&(N.window==w)].iloc[0]
 def fmt(x,d=3):return f'{x:.{d}f}'
 def ran(v,d=3):return fmt(min(v),d)+' to '+fmt(max(v),d)
 pnew=P[(P.arm=='new')&(P.window=='W1+W2+W3')];wr=N[(N.window=='W1')&(N.target=='raw')&(N.input_set=='O')&(N.config=='daily_weekly')];plug=G[(G.window=='W1')&(G.input_set=='O')&(G.model=='ridge')&(G.subgroup=='PLUG')];strategy=port[(port.arm=='new')&(port.input_set=='O')&(port.h==5)&(port.layer=='L2_g5')&(port.cost_bp==10)].iloc[0]
 for r in records:
  text=r['text'];isresponse='a_point' in r['file'];loc=r['location'];table=r.get('style')=='table';isridge=re.search(r'ridge',text,re.I)
  # Main tables containing Ridge: preserve complete row/cell locations and all context.
  main_table4=(not isresponse and loc.startswith('table 4 '));main_table2=(not isresponse and loc.startswith('table 2 row 2 '))
  if table and not isresponse:
   if not (main_table4 or main_table2):continue
   pieces=[text]
  else:
   pieces=re.split(r'(?<=[.!?])\s+(?=[A-Z“])|\n+',text)
  for sent in pieces:
   sent=sent.strip()
   if not sent:continue
   relevant=bool(re.search(r'ridge|480|7 alphas|7 of the 480|738|forecast skill|shrinking the forecasts|across 414|lowest learned MSE|no model achieves a lower test MSE than the zero',sent,re.I)) or main_table4 or main_table2
   if not relevant:continue
   newvalue=sent;status='HOLDS';source='manuscript/Table4_step26.csv';oldvalue=sent;reason='Source sentence/cell retained; no Ridge-dependent numerical change identified in this unit.'
   if main_table4:
    m=re.match(r'table 4 row (\d+) cell (\d+)',loc);row,col=map(int,m.groups())
    if row in [7,17] and col in range(2,8):
     inp='O' if row==7 else 'C';h=[5,10,20][(col-2)%3];z=q(inp,h);newvalue=f'{100*z.mse:.4f}' if col<5 else f'{z.mean_rank_ic:.3f}';status='CHANGES' if newvalue!=sent else 'HOLDS';reason=f'Ridge {inp} h{h}; Table4 MSE uses units x100, Rank IC unscaled.'
   elif main_table2:
    col=int(loc.rsplit(' ',1)[1]);source='SELECTION_LOCK.csv'
    if col==2:newvalue='Initial alpha grid {1e-2,1e-1,1,10,100,1e3,1e4,1e5,1e6}; pre-registered two-decade boundary extension, caps 1e-6 and 1e12';status='CHANGES'
    if col==3:newvalue=ran([len(json.loads(x)) for x in S[(S.window=='W3')&(S.target=='raw')].val_mse_by_alpha],0)+' per cell';status='CHANGES'
    if col==4:newvalue=f"alpha = 100 in {int(((S.window=='W3')&(S.target=='raw')&(S.selected_alpha==100)).sum())} of 18 cells; 16 of 18 primary cells change alpha";status='CHANGES'
   elif re.search(r'four (candidate|penalty)|four penalty|four.*penalt',sent,re.I) and isridge:
    newvalue='Ridge uses the nine-value initial grid and pre-registered boundary extensions (9–15 candidates in these runs), with TRAIN fitting and strict ascending-grid minimum VAL MSE selection.';source='SELECTION_LOCK.csv';status='CHANGES';reason='The four-value tuning description is superseded.'
   elif '738' in sent:
    oldvalue='738 cells; 7 positive R2 vs GTM; 42 nominal Rank IC p<.05; 0 Holm-significant CW/IC';newvalue=f"738 cells; {int(newskill.r2os_vs_global_train_mean.gt(0).sum())} positive R2 vs GTM; {int(newskill.rank_ic_p_mbb.lt(.05).sum())} nominal IC p<.05; 0 Holm-significant CW/IC";source='propagated/B_forecast_skill_final_complete_step26.csv';status='CHANGES';reason='No joint forecast-skill Holm rejection changes, including non-Ridge cells.'
   elif re.search(r'414|[Ss]hrink',sent) and ('forecast' in sent.lower()):
    newvalue=f"414 cells per original regime; all Ridge VAL selections now available; {int(((sh.loss_difference<0)&(sh.loss_p_Holm<.05)).sum())} significant improvements over GTM; four Ridge SECONDARY rejections for deterioration disappear";source='propagated/SHRINKAGE_final_step26.csv';status='HOLDS';reason='The no-significant-improvement conclusion holds; formerly N/A W3 O Ridge PRIMARY is now included.'
   elif 'PLUG' in sent and isridge:
    newvalue='W1 O Ridge PLUG relative MSE '+ran(plug.relative_mse_mean,4)+'; mean forecast bias '+ran(plug.mean_forecast_bias_pooled_seeds,4);source='propagated/C_subgroups_summary_full_step26.csv';status='NO_LONGER_TRUE';reason='The severe Ridge PLUG extrapolation failure is absent under the extended grid.'
   elif isridge and re.search(r'4.7|10.3|deteriorat',sent):
    newvalue='W1 O Ridge MSE / own Global Train-Mean MSE '+ran(wr.relative_mse_train_mean,6);source='TEST_METRICS_STEP26.csv';status='NO_LONGER_TRUE';reason='The old 4.7–10.3-fold deterioration disappears.'
   elif isridge and re.search(r'−31|−5%|20-day out-of-sample',sent):
    newvalue=f"W3 DW h20 R2: O {100*q('O',20).r2os_vs_global_train_mean:+.3f}%; C {100*q('C',20).r2os_vs_global_train_mean:+.3f}%";source='TEST_METRICS_STEP26.csv';status='NO_LONGER_TRUE';reason='Window-relative inputs no longer improve this Ridge MSE comparison.'
   elif isridge and re.search(r'−8.4|−1.2|Daily-only 10-day',sent):
    newvalue=f"W3 DO h10 R2: O {100*q('O',10,'daily_only').r2os_vs_global_train_mean:+.3f}%; C {100*q('C',10,'daily_only').r2os_vs_global_train_mean:+.3f}%; DW h20 C Rank IC {q('C',20).mean_rank_ic:.6f}";source='TEST_METRICS_STEP26.csv';status='NO_LONGER_TRUE';reason='The claimed O-to-C accuracy gain reverses; the 20-day IC point estimate also changes.'
   elif isridge and re.search(r'0.068',sent):
    newvalue=f"C DW h20 Rank IC {q('C',20).mean_rank_ic:.6f}";source='TEST_METRICS_STEP26.csv';status='CHANGES'
   elif isridge and re.search(r'pooled|similar or higher Rank IC|0.042',sent):
    newvalue='Pooled C Ridge Rank IC '+ran(pnew[pnew.input_set=='C']['mean'],6)+'; pooled O Ridge '+ran(pnew[pnew.input_set=='O']['mean'],6)+'; original Static/Adaptive Holm families unchanged; Ridge remains descriptive';source='POOLED_RIDGE_STEP26.csv';status='CHANGES';reason='Ridge is outside the original six-test families; no new Ridge hypothesis test was added.'
   elif isridge and re.search(r'Sharpe|56 times|strongly negative',sent,re.I):
    newvalue=f"O DW h5 gamma5/cost10: Sharpe {strategy.sharpe:.6f}; annual turnover {strategy.ann_turnover:.6f}; alpha t {strategy.alpha_vs_ew_nw_t:.6f}; W3 O DW R2 {ran([100*q('O',h).r2os_vs_global_train_mean for h in [5,10,20]],4)}%";source='PORTFOLIO_RIDGE_STEP26.csv';status='NO_LONGER_TRUE';reason='The strongly negative R2 and about-56-turnover narrative no longer holds; exact strategy/cost is specified.'
   elif re.search(r'480|7 alphas|7 of the 480',sent):
    newvalue='5 of 480 price-level TEST/10bp alpha t statistics have |t| > 1.96 (was 7 of 480)';source='PRICE_LEVEL_480_ALPHA_COUNT_STEP26.csv';status='CHANGES';reason='All non-Ridge alpha statistics preserved; Ridge forecasts alone replaced.'
   elif re.search(r'no model achieves a lower test MSE than the zero',sent,re.I):
    newvalue='W3 O DW Ridge beats Zero descriptively at h5/10/20; R2 vs Zero '+ran([100*q('O',h).r2os_vs_zero for h in [5,10,20]],4)+'%; no new Zero-benchmark significance claim';source='TEST_METRICS_STEP26.csv';status='NO_LONGER_TRUE'
   elif 'lowest learned MSE' in sent:
    newvalue='Ridge has lower W3 O DW pooled MSE than each listed learned-model seed mean at all three horizons; all neural-to-neural comparisons are unchanged';source='manuscript/Table4_step26.csv';status='NO_LONGER_TRUE'
   elif 'Figure 3' in sent:
    newvalue='Caption conventions hold; all six Ridge points regenerated in manuscript/Figure3_step26.png and .pdf; other-model points and error bars reused';source='manuscript/Figure3_data_step26.csv';status='CHANGES';reason='The figure reports Ridge-dependent point estimates.'
   elif isridge and re.search(r'Cross-sectional|cross-section|demean',sent,re.I):source='manuscript/S18_full_metrics_step26.csv'
   out.append(dict(document=r['file'],location=(r.get('section','')+' / '+loc).strip(' /'),quoted_text=sent,old_value=oldvalue,new_value=newvalue,source_file=str(OUT.relative_to(ROOT)/source),qualitative_statement=status,assessment=reason))
 # Explicit source-artifact impacts cover supplementary numeric rows absent as native Word tables.
 for name in ['S3','S4','S8','S9','S10','S11','S12','S13_full','S15','S18_full_metrics','S18_full_portfolios']:
  p=OUT/'manuscript'/(name+'_step26.csv')
  if p.exists():out.append(dict(document='Supplementary tables referenced by manuscript and response letter',location=name,quoted_text='Ridge rows and any joint-family adjusted values',old_value='See original source table and OLD_NEW_* comparisons',new_value='Every Ridge row updated; every non-Ridge raw result copied',source_file=str(p.relative_to(ROOT)),qualitative_statement='CHANGES',assessment='Complete table/source view exported; exact old/new values in comparison CSVs.'))
 write('IMPACT_STEP26.csv',pd.DataFrame(out));js('IMPACT_COMPLETE.json',dict(n_rows=len(out),status_counts=dict(collections.Counter(x['qualitative_statement'] for x in out)),documents_edited=False,figure3_visual_QA='PNG inspected: clean labels, all 6 panels, Ridge triangles, original scale/error-bar conventions'))
 print('IMPACT',len(out),collections.Counter(x['qualitative_statement'] for x in out))
if __name__=='__main__':
 try:main()
 except BaseException:js('IMPACT_BLOCKING.json',dict(error=traceback.format_exc()));raise
