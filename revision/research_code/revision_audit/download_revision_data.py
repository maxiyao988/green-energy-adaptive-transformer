#!/usr/bin/env python3
"""Download an isolated revision snapshot and audit corporate actions, without training.

Run: .venv/bin/python -B revision_audit/download_revision_data.py
Dependencies may live in data_raw_revision/_packages; original environment is untouched.
All final files use exclusive creation. A completed/partially populated output namespace
is refused, rather than silently overwriting any prior snapshot or audit output.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data_raw_revision'
AUDIT = ROOT / 'revision_audit'
if (RAW / '_packages').is_dir(): sys.path.insert(0, str(RAW / '_packages'))
import csv, hashlib, importlib.metadata as md, io, json, logging, math, os, platform, re, time, zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TICKERS = ['AES','BEP','BLDP','BLNK','CSIQ','CWEN','DQ','ENPH','FCEL','FSLR','HASI','JKS','NEE','ORA','PLUG','RUN','SEDG']
BENCHMARKS = ['SPY','ICLN','PBW','QCLN','TAN']
START, END = '2015-08-05', '2026-07-07'
FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3'
FF_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
COLS = ['Date','Open','High','Low','Close','Adj Close','Volume','Dividends','Stock Splits']
RATIO_RTOL, RATIO_ATOL = 1e-5, 1e-8
ATTEMPTS, BACKOFF = 4, 2.0
ORIGINAL = ROOT / 'dataset/finance/panel_raw'
SUPPORT = AUDIT / 'corporate_action_support'
MANIFEST: dict = {}
LOG = logging.getLogger('revision_download')

def utc(): return datetime.now(timezone.utc).isoformat()
def rel(p): return str(Path(p).relative_to(ROOT))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def write_text(p,text):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8') as f:f.write(text)
def write_json(p,value): write_text(p,json.dumps(value,indent=2,allow_nan=False)+'\n')
def write_csv(p,data,columns=None):
    frame=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data,columns=columns)
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8',newline='') as f:frame.to_csv(f,index=False)
    return frame

def retry(label,fn):
    """Four attempts with 2/4/8 second backoff; failures are explicitly recorded."""
    for attempt in range(1,ATTEMPTS+1):
        before=utc()
        try:
            result=fn()
            MANIFEST['attempts'].append(dict(resource=label,attempt=attempt,start_utc=before,end_utc=utc(),status='OK'))
            return result
        except Exception as exc:
            detail=f'{type(exc).__name__}: {exc}'
            MANIFEST['attempts'].append(dict(resource=label,attempt=attempt,start_utc=before,end_utc=utc(),status='FAILED',error=detail))
            LOG.warning('%s attempt %d/%d failed: %s',label,attempt,ATTEMPTS,detail)
            if attempt<ATTEMPTS:time.sleep(BACKOFF*2**(attempt-1))
    MANIFEST['failures'].append(dict(resource=label,error=detail))
    return None

def dates_local(index):
    dates=pd.DatetimeIndex(index)
    if dates.tz is not None:dates=dates.tz_localize(None)
    return dates.strftime('%Y-%m-%d')

def download_history(symbol,folder):
    record=dict(symbol=symbol,kind=folder,start_utc=utc(),status='FAILED')
    holder={}
    def fetch():
        obj=yf.Ticker(symbol)
        # Exact request specified by the user; do not enable repair or back-adjustment.
        frame=obj.history(period='max',auto_adjust=False,actions=True)
        if frame is None or frame.empty:raise ValueError('empty history returned by yfinance')
        missing=set(COLS[1:])-set(frame.columns)
        if missing:raise ValueError(f'required columns missing: {sorted(missing)}')
        frame=frame.copy();frame.index.name='Date'
        holder['obj']=obj;holder['timezone']=str(frame.index.tz)
        out=frame[COLS[1:]].copy();out.insert(0,'Date',dates_local(frame.index))
        if out.Date.duplicated().any():raise ValueError('duplicate session dates in history')
        return out.reset_index(drop=True).sort_values('Date').reset_index(drop=True)
    frame=retry(symbol+'/history',fetch)
    if frame is None:
        record['end_utc']=utc();MANIFEST['resources'].append(record);return None
    dest=RAW/folder/f'{symbol}.csv'
    write_csv(dest,frame)
    record.update(status='OK',path=rel(dest),rows=len(frame),first_date=frame.Date.min(),last_date=frame.Date.max(),index_timezone=holder['timezone'],sha256=digest(dest))
    if symbol in TICKERS:
        obj=holder['obj']
        for attribute,column in [('dividends','Dividends'),('splits','Stock Splits')]:
            def fetch_action(attribute=attribute,column=column):
                series=getattr(obj,attribute)
                if series is None:raise ValueError('action property returned None')
                action=pd.DataFrame({'Date':dates_local(series.index),column:series.to_numpy()})
                # Empty action files are valid only if history also reports no such events.
                expected=frame.loc[frame[column].fillna(0).ne(0),['Date',column]]
                keys=set(action.Date)
                if not set(expected.Date)<=keys:raise ValueError('property omits events present in history')
                return action.sort_values('Date').reset_index(drop=True)
            action=retry(symbol+'/'+attribute,fetch_action)
            ap=RAW/'actions'/f'{symbol}_{attribute}.csv'
            if action is not None:
                write_csv(ap,action);record[attribute]=dict(status='OK',path=rel(ap),rows=len(action),sha256=digest(ap))
            else:record[attribute]=dict(status='FAILED',path='NOT FOUND')
    record['end_utc']=utc();MANIFEST['resources'].append(record)
    LOG.info('%s: %d sessions %s to %s',symbol,len(frame),frame.Date.min(),frame.Date.max())
    time.sleep(0.4)
    return frame

def get_bytes(url):
    response=requests.get(url,timeout=(15,60),headers={'User-Agent':'Research revision data audit/1.0'})
    response.raise_for_status()
    if not response.content:raise ValueError('empty HTTP response')
    return response.content

def download_fred():
    start=utc()
    def fetch():
        content=get_bytes(FRED_URL)
        frame=pd.read_csv(io.BytesIO(content),na_values=['.'])
        if 'DTB3' not in frame:raise ValueError('FRED response is not a DTB3 CSV')
        datecol=next((c for c in ['observation_date','DATE','Date'] if c in frame),None)
        if datecol is None:raise ValueError('FRED date column NOT FOUND')
        frame=frame.rename(columns={datecol:'Date'})
        frame['Date']=pd.to_datetime(frame.Date,errors='raise').dt.strftime('%Y-%m-%d')
        frame['DTB3']=pd.to_numeric(frame.DTB3,errors='raise')
        frame['daily_simple_rate']=frame.DTB3/100.0/252.0
        return content,frame[['Date','DTB3','daily_simple_rate']]
    result=retry('FRED/DTB3',fetch)
    record=dict(resource='FRED/DTB3',url=FRED_URL,start_utc=start,end_utc=utc(),status='FAILED')
    if result is not None:
        content,frame=result
        with (RAW/'rf/DTB3_source.csv').open('xb') as f:f.write(content)
        write_csv(RAW/'rf/DTB3.csv',frame)
        record.update(status='OK',rows=len(frame),first_date=frame.Date.min(),last_date=frame.Date.max(),missing_rates=int(frame.DTB3.isna().sum()),path=rel(RAW/'rf/DTB3.csv'),sha256=digest(RAW/'rf/DTB3.csv'))
    MANIFEST['resources'].append(record)

def download_ff():
    start=utc()
    def fetch():
        content=get_bytes(FF_URL);rows=[]
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            member=next(n for n in z.namelist() if n.lower().endswith('.csv'))
            source=z.read(member).decode('utf-8-sig')
        for row in csv.reader(io.StringIO(source)):
            if len(row)>=5 and re.fullmatch(r'\d{8}',row[0].strip()):
                values=[float(x.strip()) for x in row[1:5]]
                rows.append([pd.to_datetime(row[0].strip(),format='%Y%m%d').strftime('%Y-%m-%d')]+[None if v in [-99.99,-999.0] else v/100.0 for v in values])
        if not rows:raise ValueError('no daily factor records parsed')
        frame=pd.DataFrame(rows,columns=['Date','Mkt-RF','SMB','HML','RF'])
        if frame.Date.duplicated().any():raise ValueError('duplicate factor dates')
        return content,member,source,frame
    result=retry('FamaFrench/daily_factors',fetch)
    record=dict(resource='FamaFrench/daily_factors',optional=True,url=FF_URL,start_utc=start,end_utc=utc(),status='FAILED')
    if result is not None:
        content,member,source,frame=result
        with (RAW/'ff/F-F_Research_Data_Factors_daily_CSV.zip').open('xb') as f:f.write(content)
        write_text(RAW/'ff/F-F_Research_Data_Factors_daily_source.csv',source)
        write_csv(RAW/'ff/F-F_Research_Data_Factors_daily.csv',frame)
        record.update(status='OK',archive_member=member,rows=len(frame),first_date=frame.Date.min(),last_date=frame.Date.max(),path=rel(RAW/'ff/F-F_Research_Data_Factors_daily.csv'),sha256=digest(RAW/'ff/F-F_Research_Data_Factors_daily.csv'))
    MANIFEST['resources'].append(record)

def audit_data(downloads):
    """Compare vendor snapshots; all heuristics are diagnostic, not PIT validation."""
    events=[];consistency=[];comparisons=[];ratios=[];changes=[];windows=[];volumes=[];jumps=[];quality=[];missing=[];bad=[];original_info=[]
    for symbol in TICKERS:
        original_path=ORIGINAL/f'{symbol}_daily.csv'
        if not original_path.is_file():
            quality.append(dict(ticker=symbol,status='ORIGINAL NOT FOUND'));continue
        old=pd.read_csv(original_path)
        old['Date']=pd.to_datetime(old.datetime).dt.strftime('%Y-%m-%d')
        original_info.append(dict(ticker=symbol,path=rel(original_path),sha256=digest(original_path),rows=len(old),first_date=old.Date.min(),last_date=old.Date.max(),mtime_utc=datetime.fromtimestamp(original_path.stat().st_mtime,timezone.utc).isoformat()))
        new=downloads.get(symbol)
        if new is None:
            quality.append(dict(ticker=symbol,status='NEW HISTORY NOT FOUND'));continue
        new=new.copy().sort_values('Date').reset_index(drop=True)
        # Compare both independently requested action properties with history.
        for kind,column in [('dividends','Dividends'),('splits','Stock Splits')]:
            ap=RAW/'actions'/f'{symbol}_{kind}.csv'
            hist=new.loc[new[column].fillna(0).ne(0),['Date',column]]
            if ap.is_file():
                prop=pd.read_csv(ap)
                joined=hist.merge(prop,on='Date',how='outer',suffixes=('_history','_property'),indicator=True)
                mismatch=(joined['_merge']!='both')|~np.isclose(joined[column+'_history'].to_numpy(dtype=float),joined[column+'_property'].to_numpy(dtype=float),rtol=1e-6,atol=1e-8,equal_nan=False)
                consistency.append(dict(ticker=symbol,action=kind,history_events=len(hist),property_events=len(prop),mismatches=int(mismatch.sum()),status='CHECKED' if not mismatch.any() else 'MISMATCH'))
            else:consistency.append(dict(ticker=symbol,action=kind,status='PROPERTY NOT FOUND'))
            # Event listing uses the history response; independent-property consistency is above.
            for _,r in hist[(hist.Date>=START)&(hist.Date<=END)].iterrows():
                events.append(dict(ticker=symbol,Date=r.Date,action=kind,value=float(r[column]),classification=('reverse split' if r[column]<1 else 'forward split') if kind=='splits' else 'cash dividend',source='history actions=True'))
        joined=old[['Date','close']].rename(columns={'close':'original_Close'}).merge(new[['Date','Close','Adj Close','Volume','Dividends','Stock Splits']],on='Date',how='inner')
        joined=joined.sort_values('Date');joined['ratio']=joined.original_Close/joined['Adj Close'].where(joined['Adj Close']>0)
        prev=joined.ratio.shift(1)
        joined['ratio_change_exact']=joined.ratio.ne(prev)&prev.notna()
        joined['ratio_change_above_tolerance']=~np.isclose(joined.ratio,prev,rtol=RATIO_RTOL,atol=RATIO_ATOL,equal_nan=True)&prev.notna()
        joined.insert(0,'ticker',symbol);ratios.extend(joined.to_dict('records'))
        for pos in range(1,len(joined)):
            r=joined.iloc[pos]
            if r.ratio_change_exact:
                before=joined.iloc[pos-1]
                changes.append(dict(ticker=symbol,previous_Date=before.Date,Date=r.Date,previous_ratio=float(before.ratio),ratio=float(r.ratio),relative_change=float(r.ratio/before.ratio-1) if before.ratio else None,above_tolerance=bool(r.ratio_change_above_tolerance),in_audit_window=START<=r.Date<=END))
        inwindow=joined[(joined.Date>=START)&(joined.Date<=END)]
        q=joined.ratio.dropna()
        comparisons.append(dict(ticker=symbol,overlap_rows=len(joined),overlap_first=joined.Date.min() if len(joined) else 'NOT FOUND',overlap_last=joined.Date.max() if len(joined) else 'NOT FOUND',ratio_min=q.min(),ratio_max=q.max(),ratio_median=q.median(),ratio_exact_change_dates=int(joined.ratio_change_exact.sum()),ratio_significant_change_dates=int(joined.ratio_change_above_tolerance.sum()),window_rows=len(inwindow),window_ratio_min=inwindow.ratio.min(),window_ratio_max=inwindow.ratio.max(),window_ratio_median=inwindow.ratio.median(),window_significant_change_dates=int(inwindow.ratio_change_above_tolerance.sum()),interpretation='date-varying difference above tolerance' if joined.ratio_change_above_tolerance.any() else 'approximately constant rescaling (or equality); cause not identified'))
        olddates=set(old.Date);newdates=set(new.Date)
        missing_old=sorted(olddates-newdates)
        for date in missing_old:missing.append(dict(ticker=symbol,Date=date,reference='original saved daily snapshot'))
        for column in ['Open','High','Low','Close','Adj Close']:
            invalid=~np.isfinite(new[column].to_numpy(dtype=float))|new[column].le(0)
            for _,r in new.loc[invalid,['Date',column]].iterrows():bad.append(dict(ticker=symbol,Date=r.Date,column=column,value=r[column],issue='missing/nonfinite' if not np.isfinite(r[column]) else 'zero/negative'))
        # SPY sessions provide an observed trading-date reference, not an authoritative calendar.
        spy=downloads.get('SPY');missing_spy=[]
        if spy is not None:
            applicable=spy[(spy.Date>=max(START,old.Date.min()))&(spy.Date<=min(END,old.Date.max()))]
            missing_spy=sorted(set(applicable.Date)-newdates)
            for date in missing_spy:missing.append(dict(ticker=symbol,Date=date,reference='SPY observed session within original audit range'))
        panel_path=ROOT/'dataset/multiscale_dataset/panel'/symbol/'meta.csv'
        if panel_path.is_file():
            meta=pd.read_csv(panel_path)
            panel_start=meta.anchor_date.min();panel_end=meta.target_date_20d.max()
        else:panel_start=panel_end='NOT FOUND'
        quality.append(dict(ticker=symbol,status='CHECKED',original_rows=len(old),new_rows=len(new),original_first=old.Date.min(),original_last=old.Date.max(),new_first=new.Date.min(),new_last=new.Date.max(),missing_original_dates=len(missing_old),missing_spy_reference_dates=len(missing_spy) if spy is not None else 'NOT FOUND',duplicate_dates=int(new.Date.duplicated().sum()),nonpositive_close=int(new.Close.le(0).sum()),nonpositive_adj_close=int(new['Adj Close'].le(0).sum()),invalid_price_cells=sum(1 for r in bad if r['ticker']==symbol),zero_volume_rows=int(new.Volume.eq(0).sum()),negative_volume_rows=int(new.Volume.lt(0).sum()),shorter_than_original_range=bool(new.Date.min()>old.Date.min() or new.Date.max()<old.Date.max()),original_panel_feature_anchor_first=panel_start,original_panel_target_last=panel_end,covers_original_audit_range=bool(new.Date.min()<=max(START,old.Date.min()) and new.Date.max()>=min(END,old.Date.max()))))
        # All recorded splits, including those before the requested action-list interval.
        for i in new.index[new['Stock Splits'].fillna(0).ne(0)]:
            event=new.loc[i];s=float(event['Stock Splits'])
            for j in range(max(0,i-5),min(len(new),i+6)):
                r=new.loc[j]
                orig=old.loc[old.Date==r.Date,'close']
                windows.append(dict(ticker=symbol,split_Date=event.Date,split_ratio_new_over_old=s,in_audit_window=START<=event.Date<=END,offset_trading_days=j-i,Date=r.Date,Open=r.Open,High=r.High,Low=r.Low,Close=r.Close,Adj_Close=r['Adj Close'],Volume=r.Volume,Dividends=r.Dividends,Stock_Splits=r['Stock Splits'],original_adjusted_Close=float(orig.iloc[0]) if len(orig) else None))
            pre=new.iloc[max(0,i-5):i];post=new.iloc[i+1:min(len(new),i+6)]
            pre_v=float(pre.Volume.median()) if len(pre) else float('nan');post_v=float(post.Volume.median()) if len(post) else float('nan')
            observed=post_v/pre_v if pre_v>0 else float('nan')
            # Normalized volume ~1; nominal shares traded often ~split ratio. Demand can dominate either.
            if math.isfinite(observed) and observed>0 and s>0 and s!=1:
                dist_one=abs(math.log(observed));dist_split=abs(math.log(observed/s))
                verdict='closer to continuous volume (~1): consistent with split-adjusted volume' if dist_one<dist_split else 'closer to split ratio: consistent with nominal-share volume'
                if abs(math.log(s))<math.log(1.25):verdict='INCONCLUSIVE: split ratio too near 1 for a useful volume heuristic'
            else:verdict='NOT FOUND: insufficient positive volume observations'
            pprev=float(new.loc[i-1,'Close']) if i else float('nan')
            factor=float(event.Close/pprev) if pprev>0 else float('nan')
            volumes.append(dict(ticker=symbol,split_Date=event.Date,split_ratio_new_over_old=s,in_audit_window=START<=event.Date<=END,n_pre=len(pre),n_post=len(post),median_volume_pre5=pre_v,median_volume_post5=post_v,post_pre_volume_ratio=observed,nominal_volume_expected_ratio=s,split_adjusted_volume_expected_ratio=1.0,raw_close_factor=factor,nominal_close_expected_factor=1/s if s>0 else None,split_corrected_close_return=factor*s-1,diagnostic=verdict,conclusion='HEURISTIC ONLY; volume adjustment REMAINS UNVERIFIED'))
        factors=new.Close/new.Close.shift(1)
        for i in new.index[(factors-1).abs()>0.25]:
            r=new.loc[i];s=float(r['Stock Splits']);factor=float(factors.loc[i]);has_split=s>0 and s!=1
            explained=bool(has_split and abs(factor*s-1)<=0.25)
            jumps.append(dict(ticker=symbol,Date=r.Date,in_audit_window=START<=r.Date<=END,previous_Date=new.loc[i-1,'Date'],previous_Close=float(new.loc[i-1,'Close']),Close=float(r.Close),Adj_Close=float(r['Adj Close']),raw_close_simple_return=factor-1,recorded_split_ratio=s,split_corrected_return=factor*s-1 if has_split else None,explained_by_recorded_split=explained,flag='EXPLAINED_BY_RECORDED_SPLIT' if explained else 'UNEXPLAINED_NO_RECORDED_SPLIT' if not has_split else 'RECORDED_SPLIT_BUT_RESIDUAL_GT_25_PERCENT'))
    frames={
        'corporate_action_events.csv':(events,['ticker','Date','action','value','classification','source']),
        'action_source_consistency.csv':(consistency,None),
        'snapshot_ratio_summary.csv':(comparisons,None),
        'snapshot_ratios_by_date.csv':(ratios,None),
        'snapshot_ratio_changes.csv':(changes,['ticker','previous_Date','Date','previous_ratio','ratio','relative_change','above_tolerance','in_audit_window']),
        'split_windows.csv':(windows,['ticker','split_Date','split_ratio_new_over_old','in_audit_window','offset_trading_days','Date','Open','High','Low','Close','Adj_Close','Volume','Dividends','Stock_Splits','original_adjusted_Close']),
        'split_volume_diagnostics.csv':(volumes,None),
        'price_discontinuities.csv':(jumps,None),
        'data_quality_summary.csv':(quality,None),
        'missing_dates.csv':(missing,['ticker','Date','reference']),
        'invalid_prices.csv':(bad,['ticker','Date','column','value','issue']),
        'original_snapshot_inventory.csv':(original_info,None),
    }
    for name,(rows,columns) in frames.items():write_csv(SUPPORT/name,rows,columns)
    write_csv(SUPPORT/'snapshot_ratio_significant_changes.csv',[r for r in changes if r['above_tolerance']],frames['snapshot_ratio_changes.csv'][1])
    MANIFEST['original_snapshot']=dict(directory=rel(ORIGINAL),download_date_user_supplied='2026-07-08',independent_timestamped_download_provenance='NOT FOUND',files=original_info)
    return dict(events=events,comparisons=comparisons,quality=quality,volumes=volumes,jumps=jumps,consistency=consistency)

def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','/').replace('\n',' ') for x in r)+' |' for r in rows])+'\n'

def write_report(evidence):
    """Write a self-contained analysis with qualified, evidence-based risk conclusions."""
    ev=evidence['events'];qs=evidence['quality'];comps=evidence['comparisons'];vs=evidence['volumes'];js=evidence['jumps']
    def ref(name):return f'[corporate_action_support/{name}](corporate_action_support/{name})'
    def number(x):return f'{x:.10g}' if x is not None and pd.notna(x) else 'NOT FOUND'
    summary=[]
    for symbol in TICKERS:
        s=[r for r in ev if r['ticker']==symbol and r['action']=='splits'];d=[r for r in ev if r['ticker']==symbol and r['action']=='dividends']
        available=next((r.get('status')=='CHECKED' for r in qs if r['ticker']==symbol),False)
        summary.append([symbol,len(s) if available else 'NOT FOUND',len(d) if available else 'NOT FOUND','; '.join(f"{r['Date']} {r['value']:g} ({r['classification']})" for r in s) or ('none reported' if available else 'NOT FOUND')])
    failures=MANIFEST['failures']
    text=f'''# Corporate-action and vendor-snapshot audit

Download started (UTC): {MANIFEST['download_started_utc']}. Audit generated (UTC): {utc()}.
Action-list interval: **{START} through {END}, inclusive**. Downloads use maximum available histories.

Original pipeline/data/checkpoints/results were not changed. This is a newly retrieved vendor snapshot, **not a point-in-time dataset**. Prices are saved from the exact requested `Ticker.history(period="max", auto_adjust=False, actions=True)` call, not silently repaired, re-adjusted or spliced. `Close` is labelled vendor/raw-request Close below; `auto_adjust=False` does not prove nominal historical, unsplit-adjusted traded prices. `Adj Close` is saved alongside it.

## Download coverage and provenance

- Successful equity histories: {sum(t in MANIFEST['downloaded_symbols'] for t in TICKERS)}/17.
- Successful benchmark histories: {sum(t in MANIFEST['downloaded_symbols'] for t in BENCHMARKS)}/5.
- Resources failing after four attempts: {len(failures)}; see [download_failures.csv](../data_raw_revision/download_failures.csv) and [current download log](../{MANIFEST['log_path']}).
- Exact installed package versions, per-resource UTC timestamps, SHA-256 hashes, retry history, source URLs and transformations: [MANIFEST.json](../data_raw_revision/MANIFEST.json).
- Source methods: [yfinance history](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html), [FRED DTB3](https://fred.stlouisfed.org/series/DTB3), [French factor definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html).

The original adjusted daily snapshot is `dataset/finance/panel_raw/{{TICKER}}_daily.csv`, with `datetime` and lowercase OHLCV columns. Its `auto_adjust=True` acquisition is specified at `scripts/python/panel_download_daily_weekly.py:63`; daily targets use that Close at `scripts/python/panel_build_multiscale_dataset.py:123`. File ranges/hashes and local mtimes are retained in {ref('original_snapshot_inventory.csv')}. The **2026-07-08 download date was supplied by the user**; a contemporaneously timestamped, cryptographically authenticated download manifest was NOT FOUND. File mtimes and an end date of 2026-07-07 do not independently authenticate the acquisition time.

## a) Splits, reverse splits and dividends

{mdtable(['Ticker','Splits/reverse splits','Dividend events','Split dates and new/old share ratios'],summary)}
Every individual split and cash-dividend date/value in the requested interval is listed in {ref('corporate_action_events.csv')}; dividend values are vendor-reported cash per share, not independent issuer confirmations. The separate `data_raw_revision/actions/*_dividends.csv` and `*_splits.csv` contain full-history `.dividends`/`.splits` property responses. Their union-of-dates/value comparison against the actions embedded in history is in {ref('action_source_consistency.csv')}. An empty action table means no events were returned by this vendor response, not verified absence of all corporate actions.

## b) Original Close / new Adj Close

{mdtable(['Ticker','Overlap rows','Ratio min','Ratio max','Ratio median','Significant change dates (full overlap)','Significant change dates (requested interval)'],[[r['ticker'],r['overlap_rows'],number(r['ratio_min']),number(r['ratio_max']),number(r['ratio_median']),r['ratio_significant_change_dates'],r['window_significant_change_dates']] for r in comps])}

Full-overlap and requested-interval statistics are in {ref('snapshot_ratio_summary.csv')}. All overlapping dates and ratios are in {ref('snapshot_ratios_by_date.csv')}. **Every exact floating-point change** versus the preceding overlapping date is in {ref('snapshot_ratio_changes.csv')}; {ref('snapshot_ratio_significant_changes.csv')} filters changes exceeding `abs(current−previous) > {RATIO_ATOL} + {RATIO_RTOL}*abs(previous)` (10 ppm relative tolerance), to avoid presenting adjustment/serialization noise as economically meaningful vendor changes. Both outputs preserve the exact-change flag and requested-interval membership.

A constant ratio different from one is consistent with a uniform rebasing, including dividend/split adjustments after the old snapshot, but does not identify the cause. A changing ratio can reflect date-specific revisions, changes in adjustment history or other vendor processing. **This comparison measures snapshot differences; it cannot distinguish causally between vendor revisions and corporate actions without independent archived/issuer evidence.** All original dates absent from the new response are listed separately rather than silently dropped from coverage statistics.

## c) Split windows and >25% discontinuities

{ref('split_windows.csv')} contains the split date and up to five observed trading sessions before and after **every recorded split in the retrieved history**, including pre-interval splits flagged `in_audit_window=False`. Columns include vendor Close, Volume, Adj Close and original adjusted Close where available. Truncated edge windows are retained explicitly via offsets and the volume-diagnostic observation counts.

All full-history absolute one-session vendor Close changes above 25% are in {ref('price_discontinuities.csv')}. For a recorded split ratio s=new shares/old shares, the nominal pre/post expectation is Close_t/Close_(t−1)≈1/s. A jump is labelled explained only when a split is recorded on that session and `abs((Close_t/Close_(t−1))*s−1) <= 0.25`. No split means `UNEXPLAINED_NO_RECORDED_SPLIT`; larger residuals despite a split are flagged separately. This is a flagging heuristic, not proof of erroneous prices: genuine market moves can exceed 25%, and Yahoo Close may already be split-normalized.

Total full-history flagged jumps: {len(js)}; not explained by the recorded same-date split: {sum(not r['explained_by_recorded_split'] for r in js)}. Within the requested interval: {sum(r['in_audit_window'] for r in js)} jumps, {sum(r['in_audit_window'] and not r['explained_by_recorded_split'] for r in js)} not explained. No discontinuity is silently fixed.

## d) Does Volume appear split-adjusted?

{mdtable(['Ticker','Split date','Split ratio','Median post/pre Volume (5 sessions each)','Close factor','Diagnostic'],[[r['ticker'],r['split_Date'],number(r['split_ratio_new_over_old']),number(r['post_pre_volume_ratio']),number(r['raw_close_factor']),r['diagnostic']] for r in vs if r['in_audit_window']])}

All splits, including those outside the requested interval, are in {ref('split_volume_diagnostics.csv')}. Medians exclude the split date and use five sessions on each side. If Volume is historically normalized to a common share basis, continuity near 1 is expected under otherwise stable trading activity; nominal shares traded may instead jump by the split ratio. The diagnostic compares absolute log distance to 1 and to s, but actual trading demand, float changes and event activity can dominate. Ratios near 1.0 and insufficient observations are inconclusive. **Volume adjustment remains unverified; this statistical check cannot authenticate the vendor convention.**

## e) Missing dates, invalid prices and shortened histories

{mdtable(['Ticker','Status','Old range','New range','Old dates missing','SPY-reference dates missing','Invalid price cells','Shorter than original range'],[[r['ticker'],r['status'],f"{r.get('original_first','NOT FOUND')} to {r.get('original_last','NOT FOUND')}",f"{r.get('new_first','NOT FOUND')} to {r.get('new_last','NOT FOUND')}",r.get('missing_original_dates','NOT FOUND'),r.get('missing_spy_reference_dates','NOT FOUND'),r.get('invalid_price_cells','NOT FOUND'),r.get('shorter_than_original_range','NOT FOUND')] for r in qs])}

{ref('data_quality_summary.csv')} also records zero/negative Close and Adj Close counts, duplicate dates, zero/negative Volume, the original per-ticker panel anchor/target range and audit-interval coverage. Price validation checks all OHLC and Adj Close for missing/nonfinite/zero/negative values. Row-level issues: {ref('invalid_prices.csv')} and {ref('missing_dates.csv')}. Missing-date tests use (1) the original snapshot's observed dates and (2) observed SPY sessions restricted to the ticker's original audit range. SPY is an empirical reference, not an independently certified exchange calendar; ticker halts/holidays can explain gaps. Missing absolute exchange sessions not represented in either reference remain unverified.

## Risk-free and optional factor conversions

DTB3 is an annualized **discount-basis** quotation in percent. `rf/DTB3_source.csv` preserves the downloaded source CSV, while `rf/DTB3.csv` adds `daily_simple_rate = DTB3 / 100 / 252`, exactly as requested. This is a simple 252-trading-day proxy, not a maturity-corrected T-bill holding return or discount-yield-to-bond-equivalent-yield conversion. Missing rates are retained as missing, with no forward fill, extrapolation or join to equities. Source: [FRED series definition](https://fred.stlouisfed.org/series/DTB3).

When the optional French download succeeds, the original ZIP and member CSV are preserved under `ff/`; the clean CSV has Date, Mkt-RF, SMB, HML, RF, converting source percent units to decimals by dividing by 100. Sentinel −99.99/−999 values are mapped to missing. RF is already a daily factor return and is **not** divided by 252. No equity-calendar alignment or fill is applied. Source: [French factor library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html).

## f) Checked versus unverified risks

| Risk | Status and limit |
| --- | --- |
| Correct request flags and immutable separate snapshot | CHECKED: exact requested history flags; new paths; package versions, retrieval times and hashes recorded. |
| Vendor split/dividend events and adjustment indicators | CHECKED to extent returned: full action properties, history consistency, split windows and Close/Adj Close retained; no issuer-level completeness claim. |
| Changes since original saved data | CHECKED on overlapping dates: ratio summaries plus all exact/significant changes; causal attribution to specific actions/vendor corrections remains unverified. |
| Missing dates, nonpositive prices, short history | CHECKED against saved original dates and observed SPY reference, with failures/missing resources disclosed. |
| Volume split convention | Diagnostic CHECKED; true adjustment policy REMAINS UNVERIFIED. |
| True point-in-time vendor data | REMAINS UNVERIFIED: current and old files do not provide as-of historical vintages or availability timestamps for every historical value. |
| Historical ticker/security identity | REMAINS UNVERIFIED: no independent issuer/security-master mapping or identifier lineage was established. |
| Delisted securities / survivorship | REMAINS UNVERIFIED: the fixed 17-name universe is not a reconstructed historical investable universe including delisted firms. |

This audit does not establish point-in-time data, eliminate survivorship bias, or authorize changing the frozen paper pipeline.
'''
    if failures:text+='\n## Failed resources\n\n'+mdtable(['Resource','Final error'],[[r['resource'],r['error']] for r in failures])
    write_text(AUDIT/'corporate_action_audit.md',text)

def integrity():
    baseline=RAW/'existing_files_before.json'
    if not baseline.is_file():return dict(status='BASELINE NOT FOUND')
    changed=[]
    for item in json.loads(baseline.read_text()):
        p=ROOT/item['path']
        if not p.is_file():changed.append(dict(path=item['path'],issue='missing'));continue
        stat=p.stat()
        if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns']:changed.append(dict(path=item['path'],issue='size/mtime changed'))
        elif item.get('sha256') and digest(p)!=item['sha256']:changed.append(dict(path=item['path'],issue='hash changed'))
    return dict(status='PASS' if not changed else 'FAIL',files_checked=len(json.loads(baseline.read_text())),changes=changed)

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-empty-run',action='store_true',help='Allow existing empty output directories after an interrupted startup; preserve old logs.')
    args=parser.parse_args()
    if args.continue_empty_run and SUPPORT.exists() and any(SUPPORT.iterdir()):raise FileExistsError('Supporting CSVs already exist; refusing overwrite')
    protected=[RAW/'MANIFEST.json',AUDIT/'corporate_action_audit.md']
    if not args.continue_empty_run:protected += [RAW/'download.log',SUPPORT]
    for path in protected:
        if path.exists():raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    for folder in ['prices','actions','benchmarks','rf','ff']:
        p=RAW/folder
        if p.exists() and any(p.iterdir()):raise FileExistsError(f'Refusing nonempty output directory: {p}')
        p.mkdir(parents=True,exist_ok=True)
    SUPPORT.mkdir(parents=True,exist_ok=args.continue_empty_run)
    LOG.setLevel(logging.INFO)
    logpath=RAW/(datetime.now(timezone.utc).strftime('download_%Y%m%dT%H%M%SZ.log') if args.continue_empty_run else 'download.log')
    handler=logging.FileHandler(logpath,mode='x',encoding='utf-8')
    formatter=logging.Formatter('%(asctime)sZ %(levelname)s %(message)s');formatter.converter=time.gmtime
    handler.setFormatter(formatter);LOG.addHandler(handler)
    stream=logging.StreamHandler(sys.stdout);stream.setFormatter(formatter);LOG.addHandler(stream)
    yf.set_tz_cache_location(str(RAW/'_yfinance_cache'))
    packages={d.metadata['Name']:d.version for d in md.distributions(path=[str(RAW/'_packages')])} if (RAW/'_packages').is_dir() else {n:md.version(n) for n in ['yfinance','pandas','numpy','requests']}
    MANIFEST.update(log_path=rel(logpath),prior_startup_logs=[rel(p) for p in sorted(RAW.glob('*.log')) if p!=logpath],startup_correction='Pandas Date index/column ambiguity corrected before any price CSV was written; prior logs retained' if args.continue_empty_run else None,download_started_utc=utc(),python=sys.version,python_executable=sys.executable,platform=platform.platform(),package_versions=packages,actual_import_paths=dict(yfinance=yf.__file__,pandas=pd.__file__,numpy=np.__file__,requests=requests.__file__),script_path=rel(Path(__file__)),script_sha256=digest(Path(__file__)),request=dict(method='yfinance.Ticker(symbol).history',parameters=dict(period='max',auto_adjust=False,actions=True),notes='Other parameters retain installed-version defaults; no repair/back_adjust flag enabled.'),equities=TICKERS,benchmarks=BENCHMARKS,audit_start=START,audit_end=END,retry_policy=dict(max_attempts=ATTEMPTS,backoff_seconds=[2,4,8]),ratio_change_tolerance=dict(rtol=RATIO_RTOL,atol=RATIO_ATOL),risk_free_conversion=dict(source=FRED_URL,series='DTB3',source_units='annualized percent, discount basis',formula='daily_simple_rate = DTB3 / 100 / 252',missing_values='retained; no fill',limitation='simple trading-day proxy, not exact T-bill holding return'),fama_french=dict(url=FF_URL,optional=True,source_units='percent',conversion='divide all four daily factors by 100; no division by 252'),attempts=[],resources=[],failures=[],downloaded_symbols=[],point_in_time=False)
    frames={}
    fatal=None
    try:
        for symbol in TICKERS+BENCHMARKS:
            frame=download_history(symbol,'prices' if symbol in TICKERS else 'benchmarks')
            if frame is not None:frames[symbol]=frame;MANIFEST['downloaded_symbols'].append(symbol)
        download_fred();download_ff()
        MANIFEST['download_completed_utc']=utc()
        evidence=audit_data(frames)
        write_report(evidence)
    except Exception as exc:
        fatal=f'{type(exc).__name__}: {exc}';LOG.exception('Fatal error after preserving completed downloads')
        MANIFEST['fatal_error']=fatal
    finally:
        MANIFEST['completed_utc']=utc();MANIFEST['integrity_check']=integrity()
        write_csv(RAW/'download_failures.csv',MANIFEST['failures'],['resource','error'])
        artifacts=[]
        for folder in [RAW/'prices',RAW/'actions',RAW/'benchmarks',RAW/'rf',RAW/'ff',SUPPORT]:
            for p in sorted(folder.glob('*')):
                if p.is_file():artifacts.append(dict(path=rel(p),bytes=p.stat().st_size,sha256=digest(p)))
        report=AUDIT/'corporate_action_audit.md'
        if report.is_file():artifacts.append(dict(path=rel(report),bytes=report.stat().st_size,sha256=digest(report)))
        MANIFEST['artifacts']=artifacts
        MANIFEST['status']='FAILED' if fatal or MANIFEST['integrity_check']['status']!='PASS' else 'COMPLETE_WITH_DOWNLOAD_FAILURES' if MANIFEST['failures'] else 'COMPLETE'
        write_json(RAW/'MANIFEST.json',MANIFEST)
        LOG.info('SUMMARY: equities %d/17; benchmarks %d/5; failed resources %d; original-file integrity %s; status %s',sum(t in frames for t in TICKERS),sum(t in frames for t in BENCHMARKS),len(MANIFEST['failures']),MANIFEST['integrity_check']['status'],MANIFEST['status'])
        print('Manifest:',rel(RAW/'MANIFEST.json'));print('Audit:',rel(report) if report.is_file() else 'NOT FOUND')
    if fatal or MANIFEST['integrity_check']['status']!='PASS':return 1
    return 2 if MANIFEST['failures'] else 0

if __name__=='__main__':raise SystemExit(main())
