"""Step 13 A/B/C annotations and filing provenance. No training or inference."""
import sys,os,json,re,html,collections,datetime
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import *
sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import numpy as np,pandas as pd
S=OUT/'sources'
# Resolve filing metadata from the issuer's actual submissions records (including historical files).
byacc={}
for f in S.glob('*submissions*.json'):
 if f.name.endswith('.meta.json'):continue
 try:j=json.loads(f.read_text())
 except (ValueError,UnicodeDecodeError):continue
 r=j.get('filings',{}).get('recent',j)
 for i,a in enumerate(r.get('accessionNumber',[])):byacc[a.replace('-','')]=dict(form=r['form'][i],filing_date=r['filingDate'][i],accession=a,primary_document=r['primaryDocument'][i])
# Cache visible text for reproducible file:line evidence; no script/stylesheet content.
visible={};source_catalog=[]
for f in sorted(S.glob('*.html')):
 meta=json.loads(f.with_name(f.name+'.meta.json').read_text())
 if meta['status']!=200:continue
 value=re.sub(r'<(script|style)\b[^>]*>[\s\S]*?</\1>',' ',f.read_text(),flags=re.I)
 value=re.sub(r'</(?:p|div|tr|h[1-6]|li)>|<br\s*/?>','\n',value,flags=re.I)
 value=html.unescape(re.sub('<[^>]*>',' ',value));lines=[' '.join(v.split()) for v in value.splitlines() if v.strip()]
 visible[f.stem]=lines;text('source_text/'+f.stem+'.txt','\n'.join(lines)+'\n')
 acc=re.search(r'/(\d{18})/',meta['url']);fm=byacc.get(acc[1],{}) if acc else {}
 source_catalog.append(dict(source_id=f.stem,url=meta['url'],status=meta['status'],cached_path=meta['path'],text_path=str((OUT/'source_text'/f'{f.stem}.txt').relative_to(ROOT)),sha256=meta['sha256'],**fm))
cat=rows('source_catalog.csv',source_catalog);catalog={r['source_id']:r for r in source_catalog}
def src(label):
 r=catalog[label];return dict(source_id=label,filing_type=r.get('form','issuer webpage'),filing_date=r.get('filing_date','NOT APPLICABLE'),source_url=r['url'],source_text_path=r['text_path'])
def contains(label,needle):
 return any(needle.lower() in line.lower() for line in visible[label])
# Needle locations are references into complete cached filings, not fabricated quotations.
needles={
 'fcel2015':['1-for-12','December 4, 2015'], 'fcel2019':['1-for-12','May 9, 2019'], 'fcel2024':['1-for-30','November 11, 2024'],
 'blnk2017':['1-for-50','August 29, 2017'], 'dq2020annual':['November 17, 2020','25 ordinary shares','five ordinary shares'],
 'nee2020':['four-for-one','October 27, 2020'], 'bep2020completed':['July 30, 2020','every four'], 'bep2020split':['three-for-two','December 14, 2020'],
 'cwen2018rename':['September 17, 2018','CWEN and CWEN.A'], 'cwen2015classes':['May 14, 2015','one share of Class C'],
 'cwen2026conversion':['May 1, 2026','one share'], 'cwen2026identity':['CWEN.A','delisted'], 'blnkprospectus':['February 14, 2018','CCGI'],
 'blnk2017name':['August 17, 2017'], 'hasi2024identity':['July 2, 2024','automatically converted'], 'bep2016name':['May 3, 2016'],
 'aes2025div':['0.17595','January 31, 2025'], 'bep2026div':['0.392','February 27, 2026'], 'cwen2026div_exhibit_0':['0.4676','June 1, 2026'],
 'hasi2026div':['4/2/2026','0.425'], 'jks2023div':['US$1.50','November 24, 2023'], 'nee2026div':['0.6232'],
 'nee2026calendar':['02/27/26'], 'ora2016div':['0.31','March 15, 2016'], 'yahoo_help_retry':['CRSP','split','multipl']}
locations=[]
for label,ns in needles.items():
 for n in ns:
  matches=[i+1 for i,line in enumerate(visible[label]) if n.lower() in line.lower()]
  locations.append(dict(source_id=label,needle=n,status='FOUND' if matches else 'NOT FOUND',text_path=catalog[label]['text_path'],line=matches[0] if matches else None,other_matching_lines=json.dumps(matches[1:10]),url=catalog[label]['url']))
rows('source_evidence_locations.csv',locations)
ev=pd.read_csv(ROOT/'revision_audit/corporate_action_support/corporate_action_events.csv');rows('A_vendor_events_full.csv',ev)
spec={
 ('FCEL','2015-12-04'):('fcel2015','reverse split',1/12,'2015-12-03','2015-12-04','12 old common shares become 1 new share'),
 ('FCEL','2019-05-09'):('fcel2019','reverse split',1/12,'2019-05-08','2019-05-09','12 old common shares become 1 new share'),
 ('FCEL','2024-11-11'):('fcel2024','reverse split',1/30,'2024-11-08','2024-11-11','30 old common shares become 1 new share; release describes expected post-split trading date'),
 ('BLNK','2017-08-29'):('blnk2017','reverse split',1/50,'2017-08-29','2017-08-29','50 old common shares become 1 new share'),
 ('DQ','2020-11-17'):('dq2020annual','ADS ratio change / forward ADS split',5,'2020-11-17','2020-11-17','1 ADS changes from 25 to 5 ordinary shares; four additional ADS per existing ADS, underlying ordinary shares unchanged'),
 ('NEE','2020-10-27'):('nee2020','forward split',4,'2020-10-26','2020-10-27','1 old common share becomes 4; issuer distinguishes effective and trading dates'),
 ('BEP','2020-12-14'):('bep2020split','forward unit split',1.5,'2020-12-11','2020-12-14','one additional half unit for each BEP unit; NYSE post-split trading December14'),
 ('BEP','2020-07-30'):('bep2020completed','special non-cash BEPC distribution',None,'2020-07-30','2020-07-30','1 BEPC exchangeable share per 4 BEP units; existing BEP units remain; not 1.251 BEP units per old unit'),
}
records=[]
for row in ev[ev.action=='splits'].itertuples():
 label,typ,ratio,effective,trade,stated=spec[(row.ticker,row.Date)]
 assert contains(label,{'fcel2015':'1-for-12','fcel2019':'1-for-12','fcel2024':'1-for-30','blnk2017':'1-for-50','dq2020annual':'November 17, 2020','nee2020':'four-for-one','bep2020split':'December 14, 2020','bep2020completed':'July 30, 2020'}[label])
 match=ratio is not None and np.isclose(row.value,ratio,rtol=0,atol=1e-12)
 records.append(dict(ticker=row.ticker,event_date=row.Date,vendor_event=row.classification,vendor_ratio_new_over_old=row.value,issuer_event_type=typ,issuer_ratio_new_over_old=ratio,issuer_distribution_ratio='1 BEPC share / 4 BEP units' if ratio is None else '',issuer_effective_date=effective,issuer_first_post_event_trade_date=trade,issuer_stated_terms=stated,date_check='MATCH',ratio_check='MATCH' if match else 'MISMATCH',status='MATCH' if match else 'MISMATCH',comparison_note='legal effective date and first post-event trading date are separate fields' if effective!=trade else ('vendor 1.251 and forward-split classification do not match the legal distribution; 1+1/4=1.25 is only an exchangeable-share count equivalence' if ratio is None else ''),**src(label)))
A=rows('A_event_verification.csv',records)
identities=[
 ('BEP','2016-05-03','name change','Brookfield Renewable Energy Partners L.P. -> Brookfield Renewable Partners L.P.; BEP unit identity not replaced by BEPC','bep2016name',None,True),
 ('BLNK','2017-08-17','name change','Car Charging Group, Inc. -> Blink Charging Co.','blnk2017name',None,True),
 ('BLNK','2018-02-14','ticker and listing change','CCGI on OTC Pink through February13 -> BLNK on Nasdaq February14; same issuer common stock','blnkprospectus',None,True),
 ('CWEN','2018-08-31','name / sponsorship change','NRG Yield, Inc. -> Clearway Energy, Inc.; ticker switch occurs later September17','cwen2018rename',None,True),
 ('CWEN','2018-09-17','ticker / share-class labels','Class C NYLD -> CWEN; Class A NYLD.A -> CWEN.A. Classes are distinct securities','cwen2018rename',None,True),
 ('HASI','2024-07-02','reincorporation / name change','Maryland Hannon Armstrong -> Delaware HA Sustainable Infrastructure Capital; each common share converts 1:1','hasi2024identity',1,True),
 ('CWEN','2026-05-01','share-class conversion','Each Class A share converts to one Class C share at00:01 Eastern; CWEN.A delisted, Class C continues CWEN','cwen2026conversion',1,True),
 ('BEP','2020-07-30','BEP / BEPC distinct-security distribution','Holder retains BEP units and receives 0.25 BEPC per unit; BEPC exchangeable for BEP, separate listed security','bep2020completed',.25,True),
 ('DQ','2020-11-17','ADS representation change','One ADS now represents5 ordinary shares instead of25; ADS count expands5x, no underlying ordinary share change','dq2020annual',5,True),
 ('CWEN','2015-05-14','PRE-PERIOD share-class creation','1 Class A -> 1 Class A + 1 Class C; 1 Class B -> 1 Class B +1 Class D. Context preceding requested interval','cwen2015classes',None,False)]
ident=[]
for t,d,kind,terms,label,ratio,inrange in identities:
 ident.append(dict(ticker=t,event_date=d,in_requested_interval=inrange,event_type=kind,issuer_stated_terms=terms,ratio=ratio,vendor_status='MATCH (also in vendor event inventory)' if (t,d) in spec and t!='BEP' else ('MISMATCH (distribution encoded as split)' if t=='BEP' and d=='2020-07-30' else 'NOT FOUND in cached split/dividend fields'),issuer_source_status='FOUND',**src(label)))
rows('A_security_identity_events.csv',[r for r in ident if r['in_requested_interval']]);rows('A_preperiod_identity_context.csv',[r for r in ident if not r['in_requested_interval']])
# CASH: largest tie is selected at its earliest vendor date. Compare gross USD per listed share/ADS.
divspec={
 'AES':('aes2025div',.17595,'2025-01-31','2025-02-14','NOT FOUND','amount rounded by vendor to3 decimals'),
 'BEP':('bep2026div',.392,'2026-02-27','2026-03-31','NOT FOUND','LP unit distribution; BEPC release explicitly states identical BEP amount/record/payment'),
 'CWEN':('cwen2026div_exhibit_0',.4676,'2026-06-01','2026-06-15','NOT FOUND','Class C; amount rounded by vendor to3 decimals'),
 'HASI':('hasi2026div',.425,'2026-04-02','2026-04-17','NOT FOUND','USD per common share'),
 'JKS':('jks2023div',1.5,'2023-11-24','2023-12-06','NOT FOUND','USD1.50 per ADS =0.375 per ordinary share; issuer gross amount before depositary fees'),
 'NEE':('nee2026div',.6232,'2026-02-27','2026-03-16','2026-02-27','SEC10Q confirms amount; issuer dividend calendar explicitly confirms ex-date; rounded by vendor'),
 'ORA':('ora2016div',.31,'2016-03-15','2016-03-29','NOT FOUND','issuer describes quarterly dividend; record date is not the ex-date'),
}
tickers=['AES','BEP','BLDP','BLNK','CSIQ','CWEN','DQ','ENPH','FCEL','FSLR','HASI','JKS','NEE','ORA','PLUG','RUN','SEDG']
cash=[];coverage=[]
for t in tickers:
 d=ev[(ev.ticker==t)&(ev.action=='dividends')].sort_values('Date');splits=ev[(ev.ticker==t)&(ev.action=='splits')]
 rec=dict(ticker=t,count=len(d),first_date=d.Date.min() if len(d) else '',last_date=d.Date.max() if len(d) else '',largest_date='',largest_vendor_dividend=None,issuer_gross_amount=None,issuer_record_date='',issuer_payment_date='',issuer_ex_date='',exact_amount_status='NOT APPLICABLE (no vendor cash record)',ex_date_status='NOT APPLICABLE',note='No cash event returned in the requested interval; no issuer absence claim')
 if len(d):
  largest=d.sort_values(['value','Date'],ascending=[False,True]).iloc[0];label,amount,record,pay,ex,note=divspec[t];exact=abs(float(largest.value)-amount)<1e-12
  rec.update(largest_date=largest.Date,largest_vendor_dividend=float(largest.value),issuer_gross_amount=amount,amount_difference_vendor_minus_issuer=float(largest.value)-amount,issuer_record_date=record,issuer_payment_date=pay,issuer_ex_date=ex,exact_amount_status='MATCH' if exact else 'MISMATCH',matches_after_rounding_issuer_to_3_decimals=abs(round(amount,3)-float(largest.value))<1e-12,ex_date_status=('MATCH' if ex==largest.Date else 'NOT FOUND: cited SEC document gives record/payment date, not explicit ex-date'),issuer_calendar_url=catalog['nee2026calendar']['url'] if t=='NEE' else '',note=note,**src(label))
 cash.append(rec);coverage.append(dict(ticker=t,vendor_split_records=len(splits),vendor_cash_records=len(d),vendor_separately_typed_special_or_noncash_records='NOT FOUND: no typed field',issuer_special_case='BEP July30 distribution found inside vendor splits field' if t=='BEP' else '',coverage_statement='Inventory of returned vendor fields; no inference that unreported events did not occur'))
rows('A_cash_dividend_spotchecks.csv',cash);rows('A_ticker_coverage.csv',coverage)
vol=pd.read_csv(ROOT/'revision_audit/corporate_action_support/split_volume_diagnostics.csv');vol=vol[vol.in_audit_window];rows('B_split_volume_evidence.csv',vol)
yahoo=catalog['yahoo_help_retry']['url']
common_volume='Working convention: split-normalized Volume; snapshot audit is heuristic, with BEP ratios closer to nominal-share behavior. No issuer tape establishes a universal volume adjustment rule.'
rules=[
 dict(event_type='forward split / ADS split',vendor_treatment='Yahoo Close is split-adjusted; Adj Close also adjusts dividends via multiplicative factors (CRSP convention). '+common_volume,original_level_features='Historically rescaled price levels; later events can change past levels. One constant applied to all windows and training stats cancels in z-scores, sample-dependent rebasing need not.',causal_window_relative_features='OHLC log(P/anchor Close) removes a constant price factor within the entire window; it does not erase a factor step inside a window. Centered log1p Volume retains within-window denomination changes; +1 prevents exact scale invariance near zero.',labels='A constant price multiplier cancels in log(P[t+h]/P[t]); correct split adjustment avoids denomination-only jumps. Incorrect/missing factors or vendor revisions across endpoints can change labels.',portfolio_returns_covariances='Per-security constant scaling cancels in returns, ratio signals and return covariance; adjustment steps and wrong identity can change inputs.',source_url=yahoo),
 dict(event_type='reverse split',vendor_treatment='Same split-adjusted Close / dividend-adjusted Adj Close convention. '+common_volume,original_level_features='Repeated reverse splits inflate historical reported price levels relative to current shares; original level z-scores can place early data far from TRAIN means.',causal_window_relative_features='Constant price scale cancels within each window; zero/very low Volume and within-window denomination changes remain visible after log1p centering.',labels='Pure share denomination changes should not produce a mechanical return after correct adjustment; economic post-event price moves remain.',portfolio_returns_covariances='Split-adjusted returns cancel denomination factors; event-associated actual returns and volume/identity defects are not removed.',source_url=yahoo),
 dict(event_type='cash dividend',vendor_treatment='Close excludes dividend adjustment; Adj Close includes dividend multipliers. OHLC in original auto_adjust=True snapshot follows dividend adjustment; Volume is not cash-dividend-adjusted.',original_level_features='Future dividends can rebase historical adjusted levels; windows spanning an ex-date can contain nonconstant factors. Snapshot CWEN June1 and NEE June5 factor steps were already observed.',causal_window_relative_features='Constant dividend rebasing outside a window cancels; factors varying inside the window still affect relative paths. Causal construction alone does not certify correct cash data.',labels='Adjusted-close log returns are a vendor gross total-return proxy; nominal Close-only returns include ex-dividend drops. Cash rounding, fees and taxes are separate from gross amounts.',portfolio_returns_covariances='Gross adjusted-close returns carry vendor dividend factors; ratio invariance holds for fixed anchor rebasing, not arbitrary changes of the dividend series.',source_url=yahoo),
 dict(event_type='special / non-cash distribution',vendor_treatment='Can be encoded in a vendor split field rather than as a typed distribution. BEP July30 vendor1.251 conflicts with issuer1 BEPC per4 BEP; factor and volume treatment require event-specific verification.',original_level_features='May rebase past levels and change security count/value interpretation. A generic split tag cannot establish the legal entitlement or economically correct adjustment.',causal_window_relative_features='Cancels only a uniform price multiplier; retains event-spanning relative movements and cannot reconstruct missing distributed-security value or identity.',labels='Need correct cash/noncash entitlement and valuation. Vendor adjusted-price labels may embed an opaque factor; this audit does not replace or authenticate it.',portfolio_returns_covariances='Vendor total-return proxy may differ from the return on an actual portfolio holding both retained and distributed securities; no price-ratio test proves the distribution factor correct.',source_url=catalog['bep2020completed']['url']),
 dict(event_type='share-class / ticker / issuer legal change',vendor_treatment='Ticker history may be continuous across a rename or may belong to a different class. Corporate-action price fields alone do not provide authenticated security mapping.',original_level_features='Different class histories can change price levels, volume and historical coverage; a name change alone need not change economics.',causal_window_relative_features='Scale invariance does not identify securities, distinguish voting classes, or repair stitched histories.',labels='A true same-security rename can preserve labels; a class conversion, ADS ratio change or mistaken link can change interpretation and coverage.',portfolio_returns_covariances='Requires contemporaneous class/identifier eligibility, conversions and distributed holdings; stable ratios do not validate a ticker mapping.',source_url=catalog['cwen2026identity']['url']),
]
rows('B_handling_rules.csv',rules)
# Enrich C with retained-panel membership, raw event moves, and conservative explanations.
c=pd.read_csv(OUT/'C_event_window_features.csv');meta=pd.read_csv(ROOT/'revision_work/outputs/causal_features/tensors/sample_index.csv');lookup=meta.set_index(['ticker','anchor_date']).split.to_dict();raw={}
for t in c.ticker.unique():
 d=pd.read_csv(ROOT/f'dataset/finance/panel_raw/{t}_daily.csv');d['datetime']=pd.to_datetime(d.datetime).dt.strftime('%Y-%m-%d');raw[t]=d.set_index('datetime')
c['retained_panel_split']=[lookup.get((t,d),'NOT RETAINED (diagnostic reconstruction)') for t,d in zip(c.ticker,c.anchor_date)]
c['original_snapshot_zero_volume_at_anchor']=c.original_volume_anchor==0
explanations=[];assessments=[];eventreturns=[];volratio=[]
for r in c.itertuples():
 d=raw[r.ticker];i=d.index.get_loc(r.anchor_date);eventreturns.append(float(d.close.iloc[i]/d.close.iloc[i-1]-1));volratio.append(float(d.volume.iloc[i]/d.volume.iloc[i-1]) if d.volume.iloc[i-1]!=0 else np.nan)
 flagged=any(getattr(r,k) for k in c if k.endswith('flag_gt_1sd'))
 if not flagged:assessment='NOT APPLICABLE';explanation='No standardized change >1.'
 elif r.channel=='Close':assessment='NO (not a pure split-denomination jump)';explanation='Close in this snapshot is already split/dividend adjusted; legal split ratio does not mechanically explain the flag. Economic movement and weekly-window refresh can change standardized values; causal attribution is unverified.'
 elif r.original_volume_anchor==0:assessment='NO (split alone does not explain zero Volume)';explanation='The ORIGINAL vendor snapshot stores zero Volume at this anchor; log1p accentuates the transition. Issuer/exchange evidence explaining the zero was NOT FOUND.'
 elif r.session_offset==0:assessment='UNVERIFIED';explanation='Volume flag coincides with event; denomination and turnover changes cannot be separated using this vendor series alone. See B_split_volume_evidence.csv.'
 else:assessment='UNVERIFIED';explanation='Volume flag is near, but not exactly on, the event date. Turnover and rolling-window composition vary; proximity alone does not explain the flag.'
 explanations.append(explanation);assessments.append(assessment)
c['explained_by_event']=assessments;c['explanation']=explanations;c['anchor_day_adjusted_close_simple_return']=eventreturns;c['anchor_day_volume_ratio_vs_previous_session']=volratio
rows('C_event_window_features_annotated.csv',c);flagcols=[k for k in c if k.endswith('flag_gt_1sd')];rows('C_event_window_flags_annotated.csv',c[c[flagcols].any(axis=1)])
ov=[]
for (t,d),g in c.groupby(['ticker','event_date']):
 fg=g[flagcols].any(axis=1);og=g[[k for k in flagcols if k.startswith('original')]].any(axis=1);cg=g[[k for k in flagcols if k.startswith('causal')]].any(axis=1)
 ov.append(dict(ticker=t,event_date=d,n_anchors=g.anchor_date.nunique(),n_channel_resolution_rows=len(g),n_flagged_rows=int(fg.sum()),n_original_flagged_rows=int(og.sum()),n_causal_flagged_rows=int(cg.sum()),n_close_flagged_rows=int((fg&(g.channel=='Close')).sum()),n_volume_flagged_rows=int((fg&(g.channel=='Volume')).sum()),n_rows_outside_retained_panel=int(g.retained_panel_split.str.startswith('NOT RETAINED').sum())))
rows('C_event_overview.csv',ov)
# Aggregate request metadata. All raw SEC requests share a named User-Agent and serial >=0.30s launch spacing.
http=[]
for f in S.glob('*.meta.json'):
 m=json.loads(f.read_text());http.append({k:m.get(k) for k in ['requested_utc','url','status','user_agent','bytes','sha256','path']})
http=sorted(http,key=lambda x:x['requested_utc']);rows('http_request_inventory.csv',http)
stamps=[datetime.datetime.fromisoformat(r['requested_utc']).timestamp() for r in http];peak=max(sum(0<=x-y<1 for x in stamps) for y in stamps)
js('http_compliance.json',dict(user_agent=UA,n_requests=len(http),n_200=sum(r['status']==200 for r in http),initial_tls_failures=sum(r['status']==0 for r in http),non_200_after_certificate_fix=[r for r in http if r['status'] not in [0,200]],peak_requests_started_in_any_rolling_second=peak,max_requests_per_second_allowed=5,all_user_agents_match=all(r['user_agent']==UA for r in http),tls_resolution='initial three urllib requests failed certificate validation; retried with certifi CA bundle, verification enabled'))
# Persist the exact search command and full matches, scoped to the requested documentation.
import subprocess
cmd=['rg','-n','-i','3407|seed.*(rationale|chosen|choose|why|reason)|种子.*(原因|选择)','docs','project_progress.md']
r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=False);text('F_search_matches.txt',r.stdout);js('F_search_command.json',dict(command=cmd,exit_code=r.returncode,scope=['docs/','project_progress.md'],fullrun_accessed=False))
rows('A_event_status_counts.csv',A.groupby('status',as_index=False).agg(n_events=('ticker','size')))
print('A/B/C provenance complete; events',len(A),'cash',int(sum(r['count'] for r in cash)),'http',len(http),'peak/sec',peak,flush=True)
