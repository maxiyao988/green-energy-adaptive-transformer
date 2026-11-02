"""Prompt 19: descriptive arithmetic on saved summaries/histories, no model imports."""
from pathlib import Path
import collections
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
assert ROOT.name=='pathformer'
assert not (OUT/'CONTENT_STEP19.json').exists()
os.environ['MPLCONFIGDIR']=str(OUT/('matplotlib_cache_'+Path(__file__).stem))
os.environ['OMP_NUM_THREADS']='1'
BASE='revision_work/outputs/'
A16=BASE+'analysis16/'
A17=BASE+'analysis17/'
M18=BASE+'manuscript18/'
AUDIT=BASE+'data_audit/'
FROZEN=BASE+'frozen/'
CORE=BASE+'fullrun/sessions/20261016T142732626448Z_9ba11b23/'
SEEDS=(42,0,1,21,3407)
MODELS=('single','fixed','same_patch_fixed','static','adaptive','lstm','vanilla','swim','ridge')
sources={}
cache={}
missing=[]
artifacts=[]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def text(s):
    p=ROOT/s
    if s not in sources:sources[s]={'path':s,'sha256_before':digest(p),'bytes':p.stat().st_size}
    return p.read_text()


def rows(s):
    if s not in cache:
        cache[s]=[dict(r,_source=s,_line=i) for i,r in enumerate(csv.DictReader(io.StringIO(text(s))),2)]
    return cache[s]


def one(rr,**key):
    r=[z for z in rr if all(str(z[k])==str(v) for k,v in key.items())]
    assert len(r)==1,(key,len(r))
    return r[0]


def num(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None


def fmt(v):
    if v is None or str(v).lower() in ('','nan','null','none'):return 'N/A'
    x=num(v)
    return format(x,'.6g') if x is not None else str(v)


def clean(r):return {k:v for k,v in r.items() if not k.startswith('_')}


def export_csv(name,rr):
    keys=list(dict.fromkeys(k for r in rr for k in r)) if rr else ['status','reason']
    p=OUT/name
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:fmt(r.get(k)) for k in keys} for r in rr])
    artifacts.append(name)


# A: divide stored seed-summary MSE and SD by a recorded GTM MSE only.
seed_summary=rows(A16+'clarified/B_seed_summary.csv')
gtm=rows(BASE+'skill_analysis/a1/a1_benchmark_mse.csv')
f3=[]
for inp in ('O','C'):
    for h in (5,10,20):
        den=one(gtm,h=h,benchmark='global_train_mean')
        for model in MODELS:
            group='samepatch' if model=='same_patch_fixed' else 'neural' if model in ('lstm','vanilla','swim') else 'ridge' if model=='ridge' else 'core'
            r=one(seed_summary,group=group,input_set=inp,model=model,config='daily_weekly',h=h)
            mean=float(r['mse_mean']);sd=num(r['mse_sd']);denom=float(den['mse'])
            assert denom>0
            assert int(r['n_seeds'])==(1 if model=='ridge' else 5)
            f3.append(dict(input_set=inp,model=model,horizon=h,n_seeds=r['n_seeds'],test_mse_mean=mean,test_mse_seed_sd=sd,global_train_mean_test_mse=denom,
                           mse_ratio=mean/denom,mse_ratio_seed_sd=sd/denom if sd is not None else 'N/A (deterministic Ridge)',
                           summary_source=r['_source'],summary_source_line=r['_line'],reference_source=den['_source'],reference_source_line=den['_line'],
                           numerical_provenance='O neural: Step16 B5FIX/B5 uniform12; C neural: 14c-day/B5; other cells: Step16 registered core/samepatch/Ridge'))
assert len(f3)==54

# B: follow the exact core prediction registry, using the matching saved history.
# No 3/1 C runs, alternate patches, or recovered histories are substituted.
reg=[r for r in rows(A16+'registry.csv') if r['group']=='core']
assert len(reg)==120
best_saved=rows(BASE+'best_epoch_by_run.csv')
history_records=[];run_records=[]
for inp in ('O','C'):
    for model in ('static','adaptive','single','fixed'):
        for h in (5,10,20):
            for seed in SEEDS:
                r=one(reg,input_set=inp,model=model,h=h,seed=seed)
                src=r['source'];selected=[];best=None;best_source=None
                if src.startswith('dataset/audit/'):
                    if model=='adaptive' and seed==42:
                        hist='dataset/audit/main_v2/pathformer/panel_pathformer_training_history.csv'
                        selected=[z for z in rows(hist) if z['frequency_setting']=='daily_weekly' and int(z['horizon'])==h and int(z['seed'])==seed]
                    elif seed==42:
                        hist='dataset/audit/experiment3_pathformer_ablation/experiment3_training_history.csv'
                        selected=[z for z in rows(hist) if z['mechanism']==model and z['horizon']==str(h)+'d' and int(z['seed'])==seed]
                        if model=='single':selected=[z for z in selected if float(z['daily_patch'])==5 and float(z['weekly_patch'])==2]
                    else:
                        hist='dataset/audit/experiment4_multiseed/experiment4_training_history.csv'
                        selected=[z for z in rows(hist) if z['mechanism']==model and z['horizon']==str(h)+'d' and int(z['seed'])==seed]
                    br=one(best_saved,mechanism=model,input_config='daily_weekly',horizon=h,seed=seed,selected_for_formal_comparison='True')
                    best=int(br['best_epoch']);best_source=br['_source']+':'+str(br['_line'])
                else:
                    folder=Path(src).parent;hist=str(folder/'history.csv');complete=str(folder/'COMPLETE.json')
                    if (ROOT/hist).exists() and (ROOT/complete).exists():
                        j=json.loads(text(complete));best=int(j['best_epoch']);best_source=complete+'#/best_epoch'
                        selected=rows(hist)
                ident=f'{inp}_{model}_dw_h{h}_s{seed}'
                if not selected:
                    missing.append(dict(kind='history',cell=ident,expected_source=hist,reason='Matching saved history missing; no substitution'))
                    continue
                selected=sorted(selected,key=lambda z:int(z['epoch']))
                epochs=[int(z['epoch']) for z in selected]
                assert epochs==list(range(1,max(epochs)+1)),(ident,epochs)
                assert best in epochs and best_source is not None
                vals=[]
                for z in selected:
                    tr=float(z.get('train_huber',z.get('train_loss')))
                    va=float(z.get('val_huber',z.get('val_loss')))
                    assert math.isfinite(tr) and math.isfinite(va)
                    vals.append(dict(input_set=inp,mechanism=model,horizon=h,seed=seed,epoch=int(z['epoch']),train_huber=tr,val_huber=va,best_epoch=best,last_epoch=epochs[-1],source_file=hist,source_line=z['_line']))
                b=one(vals,epoch=best);last=vals[-1]
                change=lambda key:100*(last[key]-b[key])/b[key] if b[key]!=0 else None
                run_records.append(dict(input_set=inp,mechanism=model,horizon=h,seed=seed,best_epoch=best,last_epoch=epochs[-1],
                                        train_huber_best=b['train_huber'],train_huber_last=last['train_huber'],train_huber_pct_change=change('train_huber'),
                                        val_huber_best=b['val_huber'],val_huber_last=last['val_huber'],val_huber_pct_change=change('val_huber'),
                                        source_file=hist,best_epoch_source=best_source,primary_prediction_source=src))
                history_records.extend(vals)

curve_groups=collections.defaultdict(list);run_groups=collections.defaultdict(list)
for r in history_records:curve_groups[(r['input_set'],r['mechanism'],r['horizon'],r['epoch'])].append(r)
for r in run_records:run_groups[(r['input_set'],r['mechanism'],r['horizon'])].append(r)
curves=[];learning_summary=[];best_dist=[]
for (inp,m,h,e),rr in sorted(curve_groups.items()):
    assert len({r['seed'] for r in rr})==len(rr)
    curves.append(dict(input_set=inp,mechanism=m,horizon=h,epoch=e,n_seeds_at_epoch=len(rr),contributing_seeds=','.join(str(r['seed']) for r in rr),
                       mean_train_huber=statistics.fmean(r['train_huber'] for r in rr),mean_val_huber=statistics.fmean(r['val_huber'] for r in rr),
                       aggregation='Available histories at this epoch; no padding, last-value carry-forward or replacement'))
for (inp,m,h),rr in sorted(run_groups.items()):
    counts=collections.Counter(r['best_epoch'] for r in rr)
    for ep,n in sorted(counts.items()):best_dist.append(dict(input_set=inp,mechanism=m,horizon=h,best_epoch=ep,n_seeds_at_best_epoch=n,n_histories=len(rr),seeds=','.join(str(r['seed']) for r in rr if r['best_epoch']==ep)))
    learning_summary.append(dict(input_set=inp,mechanism=m,horizon=h,n_histories=len(rr),expected_n_seeds=5,seeds=','.join(str(r['seed']) for r in rr),best_epoch_counts=json.dumps(dict(sorted(counts.items()))),
                                 best_epoch_min=min(counts),best_epoch_median=statistics.median(r['best_epoch'] for r in rr),best_epoch_max=max(counts),
                                 mean_train_huber_pct_change=statistics.fmean(r['train_huber_pct_change'] for r in rr),mean_val_huber_pct_change=statistics.fmean(r['val_huber_pct_change'] for r in rr),
                                 change_definition='100*(last_epoch_loss-best_epoch_loss)/best_epoch_loss, then equal-weight mean over seeds'))
assert len(run_records)+len([m for m in missing if m['kind']=='history'])==120

# C: exact saved code excerpts; no intervention execution.
ivpath='revision_work/scripts/interventions/router_interventions.py'
ivlines=text(ivpath).splitlines()
bindpath='revision_work/scripts/analysis16/interventions16.py'
bindlines=text(bindpath).splitlines()


def excerpt(lines,lo,hi):return '\n'.join(f'{i}: {lines[i-1]}' for i in range(lo,hi+1))


definitions=[
 dict(intervention='I4_date_mean',definition='For each horizon, seed, split and branch separately, replace each sample router vector with the arithmetic mean learned vector across stocks at its own anchor date. Step16 calls this on TEST and VAL separately. Uses same-date split weights, not TRAIN/global weights.',file=ivpath,line_start=123,line_end=135,exact_code=excerpt(ivlines,123,135)),
 dict(intervention='I6_equal_daily_only',definition='Daily branch uses four weights of 0.25; Weekly retains that sample\'s learned vector w. Both branches and the original head remain active.',file=ivpath,line_start=136,line_end=136,exact_code=excerpt(ivlines,136,136)),
 dict(intervention='I6_equal_weekly_only',definition='Weekly branch uses four weights of 0.25; Daily retains that sample\'s learned vector w. Both branches and the original head remain active.',file=ivpath,line_start=137,line_end=137,exact_code=excerpt(ivlines,137,137)),
 dict(intervention='I5_drop_*',definition='Eight separate variants: one of Daily {5,10,20,30} or Weekly {2,4,8,13}. In the affected branch clone the learned weights, set the selected scale to zero, divide every remaining weight by the row sum after zeroing. The other branch returns its learned w unchanged. No epsilon/clamp fallback occurs in this renormalization code.',file=ivpath,line_start=139,line_end=147,exact_code=excerpt(ivlines,139,147)),
 dict(intervention='Step16_binding',definition='Step16 imports the Step9 make_fns unchanged and applies its per-split learned vectors, ticker/date indices and TRAIN means. This excerpt establishes the code path used for stored H results.',file=bindpath,line_start=76,line_end=81,exact_code=excerpt(bindlines,76,81))]
assert 'I6_equal_daily_only' in ivlines[135] and 'I6_equal_weekly_only' in ivlines[136]
assert 'groupby(date_idx).transform("mean")' in '\n'.join(ivlines[122:135])
assert any('iv.make_fns' in x for x in bindlines[75:81]),'Binding excerpt needs correct lines'

# D. Source copies for one S1-S12 worksheet per requested supplementary table.
sheets=[]


def copied(s,rr=None,record_type=None):
    return [dict(clean(r),**({'record_type':record_type} if record_type else {}),source_file=s,source_csv_line=r['_line']) for r in (rows(s) if rr is None else rr)]


def sheet(name,title,note,data,source_files):
    sheets.append(dict(name=name,title=title,notes=note,rows=data,source_files=source_files))


eventfile=AUDIT+'A_event_verification.csv';identityfile=AUDIT+'A_security_identity_events.csv';cashfile=AUDIT+'A_cash_dividend_spotchecks.csv';vendorfile=AUDIT+'A_vendor_events_full.csv';rulefile=AUDIT+'B_handling_rules.csv'
events=rows(eventfile);identities=rows(identityfile);cash=rows(cashfile);rules=rows(rulefile)
ruleids={r['event_type']:'R'+str(i) for i,r in enumerate(rules,1)}
rule_for_type={'forward split':'forward split / ADS split','reverse split':'reverse split','cash dividend':'cash dividend'}
s1=[]
for v in rows(vendorfile):
    ev=[r for r in events if r['ticker']==v['ticker'] and r['event_date']==v['Date']]
    spot=[r for r in cash if r['ticker']==v['ticker'] and r['largest_date']==v['Date'] and v['classification']=='cash dividend']
    assert len(ev)<=1 and len(spot)<=1
    r=ev[0] if ev else spot[0] if spot else None
    rule='special / non-cash distribution' if ev and 'non-cash' in ev[0]['issuer_event_type'] else rule_for_type[v['classification']]
    d=dict(record_type='vendor_action',ticker=v['ticker'],event_date=v['Date'],vendor_event=v['classification'],vendor_value=v['value'],issuer_event=r.get('issuer_event_type','cash dividend spot-check') if r else 'NOT CHECKED (cash checks are largest-dividend spots only)',
           issuer_terms=r.get('issuer_stated_terms',r.get('note','')) if r else '',issuer_filing_form=r.get('filing_type','') if r else 'NOT CHECKED',issuer_filing_date=r.get('filing_date','') if r else 'NOT CHECKED',issuer_filing_url=r.get('source_url','') if r else 'NOT CHECKED',
           verification_status=r.get('status',r.get('exact_amount_status','')) if r else 'NOT CHECKED',handling_rule_id=ruleids[rule],handling_rule_type=rule,source_file=vendorfile,issuer_source_file=r['_source'] if r else 'N/A',source_csv_line=v['_line'])
    s1.append(d)
for r in events:
    rule='special / non-cash distribution' if 'non-cash' in r['issuer_event_type'] else 'reverse split' if 'reverse' in r['issuer_event_type'] else 'forward split / ADS split'
    s1.append(dict(record_type='issuer_verified_event',ticker=r['ticker'],event_date=r['event_date'],vendor_event=r['vendor_event'],vendor_value=r['vendor_ratio_new_over_old'],issuer_event=r['issuer_event_type'],issuer_terms=r['issuer_stated_terms'],issuer_filing_form=r['filing_type'],issuer_filing_date=r['filing_date'],issuer_filing_url=r['source_url'],verification_status=r['status'],date_check=r['date_check'],ratio_check=r['ratio_check'],issuer_effective_date=r['issuer_effective_date'],issuer_first_post_event_trade_date=r['issuer_first_post_event_trade_date'],handling_rule_id=ruleids[rule],handling_rule_type=rule,source_file=eventfile,source_csv_line=r['_line']))
for r in identities:
    rule='special / non-cash distribution' if 'distribution' in r['event_type'] else 'share-class / ticker / issuer legal change'
    s1.append(dict(record_type='security_identity',ticker=r['ticker'],event_date=r['event_date'],issuer_event=r['event_type'],issuer_terms=r['issuer_stated_terms'],issuer_filing_form=r['filing_type'],issuer_filing_date=r['filing_date'],issuer_filing_url=r['source_url'],verification_status=r['issuer_source_status'],in_requested_interval=r['in_requested_interval'],handling_rule_id=ruleids[rule],handling_rule_type=rule,source_file=identityfile,source_csv_line=r['_line']))
for r in cash:
    s1.append(dict(record_type='cash_dividend_spotcheck_summary',ticker=r['ticker'],event_date=r['largest_date'],vendor_event='largest cash dividend spot-check',vendor_value=r['largest_vendor_dividend'],issuer_event='cash dividend',issuer_terms=r['note'],issuer_filing_form=r['filing_type'],issuer_filing_date=r['filing_date'],issuer_filing_url=r['source_url'],verification_status=r['exact_amount_status'],cash_dividend_count=r['count'],cash_first_date=r['first_date'],cash_last_date=r['last_date'],handling_rule_id=ruleids['cash dividend'],handling_rule_type='cash dividend',source_file=cashfile,source_csv_line=r['_line']))
for r in rules:
    s1.append(dict(record_type='handling_rule',handling_rule_id=ruleids[r['event_type']],handling_rule_type=r['event_type'],**{k:r[k] for k in ['vendor_treatment','original_level_features','causal_window_relative_features','labels','portfolio_returns_covariances']},handling_rule_source_url=r['source_url'],source_file=rulefile,source_csv_line=r['_line']))
sheet('S1','Corporate actions, issuer records and handling rules','Vendor actions, issuer verification and security identity are distinct record types. Match rule IDs to handling_rule rows. Cash dividends are issuer-checked only at the recorded largest-dividend spots. NOT CHECKED is not MATCH. BEP 2020-07-30 retains the vendor/issuer mismatch; issuer effective and first post-event trading dates stay separate.',s1,[vendorfile,eventfile,identityfile,cashfile,rulefile])
sheet('S2','Mean shift and Global Train-Mean','W3 pooled TRAIN/VAL/TEST realized log returns and forecast. Values copied from Step18 T1. Its squared-bias/MSE share was NOT STORED and is retained as such; this prompt does not request a new S2 bias-share calculation.',copied(M18+'T1_mean_shift.csv'),[M18+'T1_mean_shift.csv'])
sheet('S3','Financial forecast benchmarks','Step18 T2. Raw ranking signals and calibrated forecasts stay separate. Undefined CW/Rank IC remain N/A; absent results remain NOT STORED. NW conventions and source numerical profiles are preserved.',copied(M18+'T2_financial_benchmarks.csv'),[M18+'T2_financial_benchmarks.csv'])
bfile=A16+'clarified/B_forecast_skill_final.csv'
brows=[r for r in rows(bfile) if r['group']=='single_resolution' and r['config'] in ('daily_only','weekly_only')]
assert len(brows)==60
sheet('S4','Single-resolution configurations','Includes every stored single-resolution row. n=1 means conditional on seed42; training randomness not reflected, SD/seed wins inapplicable. The verified O Adaptive Weekly-only alias legitimately reuses five B3 seeds. Numerical provenance is copied, including profile-not-verified labels.',copied(bfile,brows),[bfile])
afile=A16+'A_paired_tests.csv';arows=[r for r in rows(afile) if r['metric']=='rank_ic']
sheet('S5','Rank IC contrasts between aggregation mechanisms','Step16 A, joint seed x date and conditional date-only inference. Contrast signs and Holm families are copied unchanged. All intervals and p-values are stored results, not calculated in Step19.',copied(afile,arows),[afile])
primary=M18+'T6_frozen_head_primary.csv';secondary=FROZEN+'five_seed_summary.csv'
s6=copied(primary,record_type='primary_experts_and_head_frozen')
for r in rows(secondary):
    if r['setting']!='head_retrained':continue
    s6.append(dict(record_type='secondary_experts_frozen_head_retrained',input_set=r['input_set'],backbone=r['backbone'],setting=r['setting'],variant=r['variant'],horizon=r['horizon'],n_seeds=r['n_seeds'],test_mse_mean=r['mse_mean'],test_mse_sd=r['mse_sd'],rank_ic_mean=r['rank_ic_mean'],rank_ic_sd=r['rank_ic_sd'],trainable_parameters=r['trainable_parameters'],source_file=secondary,source_csv_line=r['_line']))
sheet('S6','Frozen representations: absolute forecast metrics','Primary S1/S2 freeze experts AND head. Secondary is S1 only, experts frozen and head retrained with Static/constant-input/Adaptive aggregation. Fixed rows in the secondary source are the reused frozen-head baseline, not a newly trained Fixed head. Values are five-seed means/SDs; all copied.',s6,[primary,secondary])
hfiles=[A16+'H_intervention_metrics.csv',A16+'H_seed_summary.csv',A16+'H_bootstrap_tests_v2.csv']
s7=[]
for p,typ in zip(hfiles,['per_seed_absolute_and_delta_metrics','five_seed_summary','stored_inference']):s7.extend(copied(p,record_type=typ))
sheet('S7','Router interventions I1-I6','All I1-I6 variants including eight I5 drop-scale cases and both I6 variants; I0 is retained as the baseline. TEST/VAL per-seed metrics, seed summaries and stored loss/Rank-IC tests are distinguished by record_type. No intervention inference is run. Definitions and exact file:line are in REPORT_STEP19.',s7,hfiles)
cfiles=[M18+'F5_stored_plot_data.csv',A16+'C_rolling_CW.csv',A16+'C_rolling_pooled_CW.csv']
s8=[]
for p,typ in zip(cfiles,['five_seed_rolling_metrics','per_seed_stored_Clark_West','pooled_stored_Clark_West']):s8.extend(copied(p,record_type=typ))
sheet('S8','Rolling-origin forecasts and Clark-West','Includes Static, Adaptive, Ridge and Zero relative-MSE summaries and the stored CW results vs Global Train-Mean and Zero. Deterministic models have no seed SD. Per-seed and pooled Newey-West/joint/date-only rows retain their stored definitions and adjustment families.',s8,cfiles)
sub=A16+'clarified/C_subgroups_summary.csv'
sheet('S9','Stock, volatility and subperiod diagnostics','All windows/inputs/models/horizons. Per-stock Rank IC is not applicable because IC is cross-sectional by date. Group IC uses eligible dates with at least 5 stocks; eligible-date counts and pooled forecast bias are retained. These are descriptive stored subgroup results.',copied(sub),[sub])
gfile=A16+'G_PRIMARY_with_existing_O.csv';grows=rows(gfile)
common=['split','input_set','h','strategy','layer','cost_bp']
s10=[];s11=[]
for r in grows:
    d={k:r[k] for k in common}
    for k in ['ann_geo_return','ann_vol','sharpe','max_drawdown','avg_turnover_ex_initial','initial_turnover','ann_turnover','hit_rate_periods','cumulative_return','n_days','n_rebalances']:d[k]=r[k]
    d.update(provenance=r['provenance'],source_file=gfile,source_csv_line=r['_line']);s10.append(d)
    d={k:r[k] for k in common}
    d.update(gamma=r['layer'].split('_g')[1] if r['layer'].startswith('L2_g') else 'N/A (not mean-variance)',ann_geo_return=r['ann_geo_return'],ann_vol=r['ann_vol'],sharpe=r['sharpe'],avg_turnover_ex_initial=r['avg_turnover_ex_initial'],ce_A2=r['ce_A2'],ce_A5=r['ce_A5'],provenance=r['provenance'],source_file=gfile,source_csv_line=r['_line']);s11.append(d)
sheet('S10','Portfolio metrics: all saved costs and allocations','Stored TEST MV gamma1/2/5/10/20, top-5 and allocation benchmarks, every saved cost. Turnover is average per rebalance excluding initial allocation; annual turnover is separate. Existing O Step8 references retain provenance; O neural primary rows use uniform B5FIX/B5. Missing O Single/Fixed ensembles are not fabricated.',s10,[gfile])
sheet('S11','Gamma/cost sensitivity and certainty equivalents','Gamma is copied from the stored MV layer label. CE A2 and CE A5, return, volatility, Sharpe and turnover are copied from the same Step8/16 results at every saved cost. CE is NOT reconstructed from rounded return/volatility. Non-MV gamma remains inapplicable.',s11,[gfile])
test17=A17+'test_results.csv'
sheet('S12','Shrinkage toward the training mean','Both pre-registered regimes: PRIMARY validation-selected lambda and SECONDARY fixed lambda=0.5. Every planned cell is kept, including lambda=0 and blocked original-Ridge PRIMARY cells. All CW/MBB intervals, raw and Holm p are copied unchanged from Step17. This sheet performs no tests or reselection.',copied(test17),[test17])
inherited=rows(A17+'BLOCKING.csv')
for r in inherited:missing.append(dict(kind='inherited_Step17_'+r['kind'],cell=' / '.join(r[k] for k in ('window','input_set','model','h','member')),expected_source=A17+'BLOCKING.csv',reason=r['reason']))

# Validate spreadsheet records before exclusive export. Missing union fields stay inapplicable.
for s in sheets:
    assert s['rows'] and all(r.get('source_file') for r in s['rows']),s['name']
    s['columns']=list(dict.fromkeys(k for r in s['rows'] for k in r))
    s['data']=[[r.get(k) for k in s['columns']] for r in s['rows']]
    assert all(len(a)==len(s['columns']) for a in s['data'])
    s['row_count']=len(s['rows'])
    # Human-readable exports use six significant digits; JSON keeps full source precision until workbook import.
    export_csv(s['name']+'_table.csv',s['rows'])
    del s['rows']
for name,rr in [('F3_complete_data.csv',f3),('learning_history_cells.csv',history_records),('learning_per_seed.csv',run_records),('learning_best_epoch_distribution.csv',best_dist),('learning_summary.csv',learning_summary),('learning_mean_by_epoch.csv',curves),('intervention_definitions.csv',definitions),('MISSING_HISTORY_AND_INHERITED_LIMITS.csv',missing)]:export_csv(name,rr)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator,FuncFormatter
font=font_manager.findfont('Times New Roman',fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.1})
width=16/2.54
colors={'static':'#0072B2','adaptive':'#D55E00','ridge':'#009E73'}
figdata=[]


def save(fig,name,caption):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        assert not ax.get_title()
        ticks=[t for t in ax.get_xticklabels() if min(ax.get_xlim())<=t.get_position()[0]<=max(ax.get_xlim())]+[t for t in ax.get_yticklabels() if min(ax.get_ylim())<=t.get_position()[1]<=max(ax.get_ylim())]
        for t in ax.texts+ticks+[ax.xaxis.label,ax.yaxis.label]:
            if not t.get_text() or not t.get_visible():continue
            b=t.get_window_extent(renderer)
            assert b.x0>=-2 and b.y0>=-2 and b.x1<=fig.bbox.width+2 and b.y1<=fig.bbox.height+2,(name,t.get_text(),b.bounds)
    for ext in ('png','pdf'):
        with (OUT/(name+'.'+ext)).open('xb') as f:fig.savefig(f,format=ext,dpi=300)
        artifacts.append(name+'.'+ext)
    figdata.append(dict(name=name,caption=caption,width_cm=16,font='Times New Roman',font_size_pt=9,dpi=300))
    plt.close(fig)


fig,axs=plt.subplots(3,2,figsize=(width,25/2.54))
fig.subplots_adjust(left=.225,right=.975,bottom=.075,top=.90,wspace=.34,hspace=.45)
fig.text(.383,.94,'O',ha='center');fig.text(.817,.94,'C',ha='center')
names=['Single','Fixed','Same-patch','Static','Adaptive','LSTM','Vanilla','SWiM','Ridge']
for i,ax in enumerate(axs.flat):
    h=(5,10,20)[i//2];inp=('O','C')[i%2]
    ax.text(-.08,1.025,'('+chr(97+i)+')',transform=ax.transAxes,ha='left',va='bottom')
    ax.set_yticks(range(9),names if i%2==0 else ['']*9);ax.set_ylim(8.7,-.7)
    ax.axvline(1,color='#555555',linestyle='--',linewidth=.7)
    for y,m in enumerate(MODELS):
        d=one(f3,input_set=inp,horizon=h,model=m)
        sd=num(d['mse_ratio_seed_sd'])
        ax.errorbar(d['mse_ratio'],y,xerr=sd,fmt='^' if m=='ridge' else 'o',color=colors.get(m,'#333333'),markersize=3,capsize=2)
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:format(x,'.3g')))
    ax.grid(axis='x',color='#dddddd',linewidth=.4);ax.set_axisbelow(True)
    if i%2==0:ax.set_ylabel('h='+str(h))
    if i//2==2:ax.set_xlabel('TEST MSE / Train-Mean MSE')
save(fig,'F3_complete','Stored five-seed TEST MSE mean and seed SD divided by stored Global Train-Mean TEST MSE. Rows h=5,10,20; O/C columns. Error bars are ±1 seed SD, not intervals. Ridge (triangle) is deterministic n=1, with no seed SD. Dashed line=1. No model is evaluated.')

fig,axs=plt.subplots(1,2,figsize=(width,9/2.54))
fig.subplots_adjust(left=.13,right=.97,bottom=.19,top=.77,wspace=.27)
for i,(inp,ax) in enumerate(zip(('O','C'),axs)):
    ax.text(-.07,1.04,'('+chr(97+i)+')',transform=ax.transAxes,ha='left')
    ax.text(.5,1.15,inp,transform=ax.transAxes,ha='center')
    for m in ('static','adaptive'):
        rr=[r for r in curves if r['input_set']==inp and r['mechanism']==m and r['horizon']==20]
        if not rr:ax.text(.5,.5,'HISTORY MISSING',transform=ax.transAxes,ha='center');continue
        x=[r['epoch'] for r in rr]
        ax.plot(x,[r['mean_train_huber'] for r in rr],color=colors[m],linestyle='-',label=m.capitalize()+' TRAIN')
        ax.plot(x,[r['mean_val_huber'] for r in rr],color=colors[m],linestyle='--',label=m.capitalize()+' VAL')
    ax.set_xlabel('Epoch');ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=4));ax.grid(color='#dddddd',linewidth=.4)
    ax.yaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:format(x,'.3g')))
    if i==0:ax.set_ylabel('Huber loss (delta=1)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,1.005),ncol=2,frameon=False,columnspacing=1.5,handlelength=2)
save(fig,'F7_learning_curves','Static/Adaptive h=20, O left/C right; solid TRAIN, dashed VAL Huber(delta=1). Each epoch averages only seeds whose own saved history contains that epoch. No extension after early stopping; late epochs can have fewer than five seeds. Per-epoch contributing seeds/counts are in learning_mean_by_epoch.csv. No bands or intervals are estimated.')

source_records=[]
for s,r in sources.items():
    after=digest(ROOT/s);assert after==r['sha256_before'],s
    source_records.append(dict(r,sha256_after=after,unchanged=True))
export_csv('SOURCE_MANIFEST.csv',source_records)
payload=dict(generated_utc=datetime.now(timezone.utc).isoformat(),repo=str(ROOT),output=str(OUT),sheets=sheets,missing=missing,definitions=definitions,figures=figdata,
             source_files=source_records,figure_data_rows=len(f3),history_cells=len(run_records),history_rows=len(history_records),learning_summary=learning_summary,
             history_missing=[r for r in missing if r['kind']=='history'],font_path=font,artifacts=artifacts,allowed_arithmetic=['MSE and stored SD divided by GTM MSE','per-seed percentage loss change','means over saved histories','best-epoch counts/min/median/max'],
             prohibited_work_performed=dict(training=False,inference=False,new_p_values=False,new_intervals=False,new_model_evaluation=False))
with (OUT/'CONTENT_STEP19.json').open('x') as f:json.dump(payload,f,indent=2,allow_nan=False)
print(json.dumps(dict(figure_data_rows=len(f3),history_cells=len(run_records),history_missing=len(payload['history_missing']),epoch_rows=len(history_records),sheet_rows={s['name']:s['row_count'] for s in sheets},inherited_Step17_limitations=len(inherited)),indent=2))
