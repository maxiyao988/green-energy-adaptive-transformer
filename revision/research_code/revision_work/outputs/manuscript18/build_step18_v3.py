"""Prompt 18: copy saved results, with no estimators, aggregation or tests.

All numeric table/plot values must be direct source fields.  Missing derived
quantities stay missing, including ratios, SDs, and means over epoch histories.
Every output is opened exclusively.  Re-execution refuses an existing report.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import os
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
assert ROOT.name == 'pathformer', ROOT
assert not (OUT / 'REPORT_STEP18.md').exists(), 'Exclusive output already exists'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLCONFIGDIR'] = str(OUT / 'matplotlib_cache')

BASE = 'revision_work/outputs/'
A16 = BASE + 'analysis16/'
SK = BASE + 'skill_analysis/'
FROZEN = BASE + 'frozen/'
SESS = BASE + 'fullrun/sessions/'
CORE = SESS + '20261016T142732626448Z_9ba11b23/'
DAY = SESS + '14c_day_20261017T050034Z_12db13f7/'
B2S = SESS + '14c_B2_supplement_20261017T162556Z_bfb4b1be/'
B3 = SESS + '14c_B3_20261017T201559Z_f47c5c3c/'
B5 = SESS + '14c_B5_20261018T142713Z_6908ba80/'
FIX = SESS + '14c_B5fix_20261019T055730Z_6da4918c/'
OLD_C = BASE + 'input_sensitivity/step12a/part2_causal_a3/'
NS = 'NOT STORED'
NR = 'NOT RECORDED'
NA = 'N/A (undefined in source)'
sources = {}
tables = {}
copies = []
notes = {}
gaps = []


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_text(s):
    p = ROOT / s
    if s not in sources:
        sources[s] = {'path': s, 'sha256_before': sha(p), 'bytes': p.stat().st_size}
    return p.read_text()


def rows(s):
    return [dict(r, _source=s, _line=i) for i, r in
            enumerate(csv.DictReader(io.StringIO(read_text(s))), 2)]


def one(rr, **key):
    found = [r for r in rr if all(str(r[k]) == str(v) for k, v in key.items())]
    assert len(found) == 1, (key, len(found))
    return found[0]


def fmt(v):
    if v is None or str(v).strip().lower() in ('', 'nan', 'none', 'null', 'inf', '-inf'):
        return NA
    if isinstance(v, bool):
        return str(v)
    try:
        x = float(v)
        return format(x, '.6g') if math.isfinite(x) else NA
    except (ValueError, TypeError):
        return str(v)


def cp(d, name, src, col):
    value = src.get(col, NS)
    d[name] = fmt(value)
    copies.append({'destination': d, 'field': name, 'source': src['_source'],
                   'line': src['_line'], 'column': col, 'formatted_value': fmt(value)})


def add(name, rr, caption):
    assert name not in tables
    tables[name] = rr
    notes[name] = caption


def gap(item, finding):
    gaps.append({'item': item, 'status': NS, 'finding': finding})


# T1. Saved pooled means; do not turn bias2 and MSE into a new ratio.
means = rows(BASE + 'mean_shift_table.csv')
skill5 = rows(SK + 'a2_original/a2_skill_by_cell.csv')
t1 = []
for h in (5, 10, 20):
    src = one(means, horizon=h, ticker='POOLED')
    gtm = one(skill5, model='naive', config='global_train_mean', h=h)
    d = {'horizon': h, 'unit': 'raw h-session log return', 'bias_squared_share_of_test_mse': NS}
    for name, col in [('train_mean', 'train_mean_return'), ('val_mean', 'validation_mean_return'),
                      ('test_mean', 'test_mean_return'), ('global_train_mean_forecast', 'global_train_mean_prediction')]:
        cp(d, name, src, col)
    cp(d, 'gtm_test_mse', gtm, 'mse')
    cp(d, 'gtm_squared_bias', gtm, 'bias2')
    d['source'] = src['_source'] + '; ' + gtm['_source']
    t1.append(d)
add('T1_mean_shift', t1, 'W3 pooled across stocks. Means are copied from Step 3; the Global Train-Mean TEST MSE and squared bias are copied from Step 5. The squared-bias/MSE share is not stored, so no division is performed. Step 3 uses float32 label accumulation; Step 5 uses its recorded float64 arithmetic.')
gap('T1', 'Squared bias as a share of Global Train-Mean TEST MSE is not stored. Stored squared bias and MSE are retained separately.')

# T2. Reference signals retain their stored distinction: raw ranks vs calibrated forecasts.
skill16 = rows(A16 + 'clarified/B_forecast_skill_final.csv')
signal = rows(SK + 'a1/a1_signal_ranking.csv')
t2 = []
metric_map = {'test_mse': 'mse', 'r2_os_vs_train_mean': 'r2os_vs_global_train_mean',
              'cw_p_nw_vs_train_mean': 'cw_p_nw_one_sided', 'mean_rank_ic': 'mean_rank_ic',
              'rank_ic_nw_t': 'rank_ic_nw_t'}
for h in (5, 10, 20):
    for model, config in [('naive', 'zero'), ('naive', 'global_train_mean'),
                          ('bench', 'ticker_train_mean'), ('bench', 'expanding_pooled_mean'),
                          ('bench', 'expanding_ticker_mean'), ('bench', 'ols_mom6_1'), ('bench', 'ols_rev1m')]:
        src = one(skill5, model=model, config=config, h=h)
        d = {'input_set': 'O source; common financial reference', 'model': config, 'configuration': 'financial reference', 'horizon': h, 'seed': 'deterministic'}
        for n, c in metric_map.items(): cp(d, n, src, c)
        d['source'] = src['_source']
        d['note'] = 'Step 5 NW convention (IC lag h-1); raw one-sided CW vs Train-Mean'
        t2.append(d)
    for name in ('mom6_1', 'rev1m'):
        src = one(signal, signal=name, h=h)
        d = {'input_set': 'O source; common financial reference', 'model': name + '_raw', 'configuration': 'uncalibrated ranking signal', 'horizon': h, 'seed': 'deterministic',
             'test_mse': NS, 'r2_os_vs_train_mean': NS, 'cw_p_nw_vs_train_mean': NS}
        cp(d, 'mean_rank_ic', src, 'mean_rank_ic')
        cp(d, 'rank_ic_nw_t', src, 'rank_ic_nw_t')
        d['source'] = src['_source']
        d['note'] = 'Raw-signal MSE/CW not stored; do not substitute its TRAIN-OLS-calibrated forecast'
        t2.append(d)
    for inp in ('O', 'C'):
        for config in ('daily_only', 'weekly_only', 'daily_weekly'):
            src = one(skill16, group='ridge', input_set=inp, model='ridge', config=config, h=h, window='W3')
            d = {'input_set': inp, 'model': 'Ridge', 'configuration': config, 'horizon': h, 'seed': 42}
            for n, c in metric_map.items(): cp(d, n, src, c)
            d['source'] = src['_source']
            d['note'] = 'Step 16 recorded NW convention (IC lag h); uniform 12/12 C source; O formal Ridge'
            t2.append(d)
add('T2_financial_benchmarks', t2, 'Step 5 financial references and Step 16 Ridge skill rows. Expanding pooled and ticker versions are both shown. Raw signals have stored ranking diagnostics but no stored MSE/CW. OLS prefixes denote TRAIN-only calibration. Undefined constant-cross-section Rank IC and self-benchmark CW remain N/A. NW conventions are identified per source; p-values are not re-estimated or adjusted here.')
gap('T2', 'Raw MOM(6-1) and REV(1m) have stored Rank IC/NW t but no raw-forecast MSE, R2 or CW p. Their calibrated forecasts are separate rows.')

# T3. Source-code/configuration extraction, no model instantiation or parameter counting.
source_code = {}
for name in ('panel_baseline_ridge.py', 'panel_baseline_lstm.py', 'panel_baseline_vanilla_transformer.py', 'panel_baseline_swim.py', 'panel_baseline_swim_cpu.py', 'panel_pathformer_main_v2.py', 'panel_pathformer_experiment3.py', 'panel_pathformer_experiment4_multiseed.py'):
    s = 'scripts/python/' + name
    source_code[name] = read_text(s).splitlines()


def code_ref(name, phrase):
    hits = [i for i, line in enumerate(source_code[name], 1) if phrase in line]
    assert hits, (name, phrase)
    return 'scripts/python/' + name + ':' + str(hits[0])


day_cells = rows(DAY + 'cell_metrics.csv')
ridge_selected = []
ridge_o = rows('dataset/audit/main_v2/ridge/panel_ridge_summary_metrics.csv')
params = {}
for src in day_cells:
    if src['horizon'] != '5' or src['block'] != 'B4': continue
    s = src['completion']
    j = json.loads(read_text(s))
    params[(src['model'], src['frequency_setting'])] = (j['parameter_count'], s)
for src in ridge_o:
    d = {'input_set': 'O', 'model': 'Ridge', 'configuration': src['frequency_setting'], 'horizon': src['horizon']}
    cp(d, 'selected_alpha', src, 'best_alpha')
    cp(d, 'selection_val_mse', src, 'val_mse_at_best_alpha')
    d['source'] = src['_source']
    ridge_selected.append(d)
for src in day_cells:
    if src['model'] != 'ridge': continue
    summary = str(Path(src['completion']).parent / 'trainer_summary.csv')
    saved = one(rows(summary))
    d = {'input_set': 'C', 'model': 'Ridge', 'configuration': src['frequency_setting'], 'horizon': src['horizon']}
    cp(d, 'selected_alpha', saved, 'best_alpha')
    cp(d, 'selection_val_mse', saved, 'val_mse_at_best_alpha')
    d['source'] = summary
    ridge_selected.append(d)
add('T3_Ridge_selected_configurations', ridge_selected, 'Recorded selected alpha and validation MSE per O/C × resolution × horizon; read from formal O and 14c-day C trainer summaries. No alpha search is run.')

neural_stop = 'Validation Huber(delta=1); strict improvement; patience=10; max_epochs=100; restore best state'
path_stop = 'Validation Huber(delta=1); improvement >1e-12; patience=10; max_epochs=100; restore best state'
t3 = []
for model in ('Ridge', 'LSTM', 'Vanilla', 'SWiM', 'Single', 'Fixed', 'Static', 'Adaptive'):
    d = {'model': model, 'input_sets': 'O and C; identical architecture', 'historical_search_trials_outside_recorded_protocol': NR}
    if model == 'Ridge':
        name = 'panel_baseline_ridge.py'
        d.update(candidate_range='alpha={0.1,1,10,100}', number_of_trials='4 alpha candidates per cell', selection_metric='Minimum pooled VALIDATION MSE; strict comparison in grid order', stopping_rule='No early stopping; sklearn Ridge fit for each alpha', selected_configuration='See T3_Ridge_selected_configurations.csv', optimizer_batch_loss='sklearn Ridge; flattened standardized features')
        d['source_code'] = code_ref(name, 'ALPHA_GRID =') + '; ' + code_ref(name, 'if val_mse <')
    elif model in ('LSTM', 'Vanilla', 'SWiM'):
        name = {'LSTM': 'panel_baseline_lstm.py', 'Vanilla': 'panel_baseline_vanilla_transformer.py', 'SWiM': 'panel_baseline_swim.py'}[model]
        config = {'LSTM': 'hidden_size=64; num_layers=1; unidirectional; dual-frequency fusion head for D+W', 'Vanilla': 'd_model=64; heads=4; encoder_layers=2; FFN=128; dropout=0.1; ReLU; max_len=512', 'SWiM': 'd_model=64; heads=4; blocks=2; FFN=128; dropout=0.1; D window/shift=10/5; W=4/2'}[model]
        d.update(candidate_range='One fixed configuration in the formal entrypoint; no candidate search loop recorded', number_of_trials=NR, selection_metric='Minimum VALIDATION Huber across epochs; no architecture selection recorded', stopping_rule=neural_stop, selected_configuration=config, optimizer_batch_loss='Adam lr=0.0001; batch=32; Huber delta=1; gradient clip norm=1')
        d['source_code'] = code_ref(name, 'if val_loss < best_val_loss:') + '; ' + code_ref(name, {'LSTM':'HIDDEN_SIZE =', 'Vanilla':'D_MODEL =', 'SWiM':'NUM_BLOCKS ='}[model])
    else:
        name = 'panel_pathformer_experiment3.py' if model in ('Single', 'Fixed', 'Static') else 'panel_pathformer_main_v2.py'
        d.update(candidate_range='D patches={5,10,20,30}; W patches={2,4,8,13}' if model == 'Single' else 'Fixed formal configuration; D patches={5,10,20,30}, W patches={2,4,8,13}', number_of_trials='48 Single candidates across 3 horizons (recorded); later seeds/O/C reuse selection without search' if model == 'Single' else NR,
                 selection_metric='Single patch pair: pooled VALIDATION MSE; ties ascending D then W' if model == 'Single' else 'Minimum VALIDATION Huber across epochs; no architecture selection recorded', stopping_rule=path_stop,
                 selected_configuration=('D patch=5; W patch=2; one expert per branch' if model=='Single' else {'Fixed':'4 experts/branch; equal weights', 'Static':'4 experts/branch; 4 zero-initialized softmax logits/branch', 'Adaptive':'4 experts/branch; per-sample Linear(128,32)-ReLU-Linear(32,4)-softmax router'}[model]) + '; d_model=32; heads=4; FFN=64; dropout=0.1', optimizer_batch_loss='Adam lr=0.0001; batch=32; Huber delta=1; gradient clip norm=1')
        d['source_code'] = code_ref('panel_pathformer_main_v2.py', 'D_MODEL =') + '; ' + code_ref(name, 'if val_loss < min_val_loss - 1e-12:')
        if model=='Single':d['source_code'] += '; ' + code_ref(name, 'EXPECTED_SINGLE_CELLS =') + '; dataset/audit/experiment3_pathformer_ablation/experiment3_single_selection.csv'
    if model in ('Ridge', 'LSTM', 'Vanilla', 'SWiM'):
        key = {'Ridge':'ridge','LSTM':'lstm','Vanilla':'vanilla','SWiM':'swim'}[model]
        pp=[];ss=[]
        for cfg in ('daily_only','weekly_only','daily_weekly'):
            v,s = params[(key,cfg)]
            pp.append(cfg+'='+fmt(v));ss.append(s+'#/parameter_count')
        d['parameter_count'] = '; '.join(pp)
        d['parameter_count_source'] = '; '.join(ss)
    else:
        d['parameter_count'] = {'Single':'30401','Fixed':'107777','Static':'107785','Adaptive':'daily_only=56069; weekly_only=56069; daily_weekly=116297'}[model]
        d['parameter_count_source'] = code_ref('panel_pathformer_experiment3.py', 'EXPECTED_PARAMETER_COUNTS =') if model!='Adaptive' else code_ref('panel_pathformer_main_v2.py','def _expected_model_count(')
    t3.append(d)
add('T3_hyperparameter_budget', t3, 'Configuration and parameter counts are copied from code/constants and saved completion records; no model is constructed. A fixed configuration in code does not establish the number of undocumented historical trials: these are NOT RECORDED. Seed replications are not counted as hyperparameter trials. Single selection was frozen at Daily5/Weekly2 in 14a. Parameter counts refer to the stored resolution configurations, not additional B3 or frozen-head variants.')
read_text('dataset/audit/experiment3_pathformer_ablation/experiment3_single_selection.csv')

# T4. Preserve signs and numerical profiles; do not recalculate or reverse CIs.
t4 = []
paired16 = rows(A16 + 'A_paired_tests.csv')
for src in paired16:
    if src['contrast'] != 'static−adaptive': continue
    d = {'input_set': src['input_set'], 'profile': '12/12; Step16 primary', 'contrast': 'Static−Adaptive', 'method': src['method'], 'metric': src['metric'], 'horizon': src['h'], 'L': src['L'], 'n_seeds': src['n_seeds'], 'B': src['B']}
    for n,c in [('estimate','theta_hat'),('ci_lower_95','ci_lower_95'),('ci_upper_95','ci_upper_95'),('raw_p','centered_p_two_sided'),('holm_p','p_holm_3h')]:cp(d,n,src,c)
    d['paired_seed_t_p'] = NS
    if src['input_set']=='O':
        z=one(rows(SK+'a3_original/a3_seed_level_tests.csv'), h=src['h'], metric='loss' if src['metric']=='squared_loss' else 'rank_ic')
        cp(d,'paired_seed_t_p',z,'paired_t_p_two_sided')
    d['source'] = src['_source']
    t4.append(d)
for inp in ('O','C'):
    # Requested uniform-profile sensitivity rows do not exist in Step16.
    for h,L in ((5,10),(10,20),(20,40)):
        for met in ('squared_loss','rank_ic'):
            d = {'input_set':inp,'profile':'12/12; requested sensitivity','contrast':'Static−Adaptive','method':'date_only','metric':met,'horizon':h,'L':L,'n_seeds':5,'B':20000,
                 'estimate':NS,'ci_lower_95':NS,'ci_upper_95':NS,'raw_p':NS,'holm_p':NS,'paired_seed_t_p':NS,'source':'No stored uniform-profile L=2h row with this contrast'}
            t4.append(d)
add('T4_Static_minus_Adaptive_uniform', t4, 'Step 16 original-window Static−Adaptive inference; the existing Holm-over-3-horizons p is copied as stored. Negative squared loss favors Static; positive Rank IC favors Static. The O two-sided seed-paired t p is copied from Step 5 (p is unchanged by contrast orientation). Missing uniform-profile L=2h values and C seed-level p are not generated. Native Step 5/12A results below preserve Adaptive−Static signs and the distinct C 3/1 profile.')
legacy=[]
for inp,prefix,profile in [('O',SK+'a3_original/','O archived formal'),('C',OLD_C,'C Stage3a/3b 3/1; NOT uniform12')]:
    seedrows=rows(prefix+'a3_seed_level_tests.csv')
    for filename,method in [('a3_date_only_mbb.csv','date_only'),('a3_joint_seed_date_bootstrap.csv','joint_seed_date')]:
        for src in rows(prefix+filename):
            d={'input_set':inp,'profile':profile,'contrast':'Adaptive−Static (native stored orientation)','method':method,'metric':src['metric'],'horizon':src['h'],'L':src['block_length'],'specification':src['specification'],'n_seeds':src['n_seeds'],'B':src['B']}
            for n,c in [('estimate','theta_hat'),('ci_lower_95','ci_lower_95'),('ci_upper_95','ci_upper_95'),('raw_p','centered_p_two_sided'),('holm_p','centered_p_holm_primary')]:
                cp(d,n,src,c)
            if d['holm_p']==NA:d['holm_p']=NS
            sr=one(seedrows,h=src['h'],metric=src['metric'])
            for n,c in [('paired_seed_t_p','paired_t_p_two_sided'),('paired_seed_sign_p','sign_test_p_two_sided'),('paired_seed_wilcoxon_p','wilcoxon_p_two_sided')]:cp(d,n,sr,c)
            d['source']=src['_source']+'; '+sr['_source']
            legacy.append(d)
add('T4_native_orientation_references',legacy,'Stored Step 5 O and Step 12A C results, including L=h and L=2h. These are Adaptive−Static, not the sign of the primary table. C uses the earlier 3/1 profile. Sensitivity Holm p was not stored. Two-sided t, sign and Wilcoxon p are copied; no sign conversion, CI transformation or new test is performed.')
gap('T4', 'Uniform 12/12 Static−Adaptive L=2h rows and C seed-level paired-test p are not stored. Stored O and earlier C 3/1 L=h/2h inference is included separately in its native Adaptive−Static orientation; L=2h Holm p is not stored.')

# T5 and T6 are direct five-seed summaries, without groupby/aggregation.
t5=[]
for src in rows(DAY+'B1_side_by_side_summary.csv'):
    d={'input_set':src['input_set'],'model':src['model'],'horizon':src['horizon'],'n_seeds':src['n_seeds']}
    for n,c in [('test_mse_mean','test_mse_mean'),('test_mse_sd','test_mse_sd'),('rank_ic_mean','mean_rank_ic_mean'),('rank_ic_sd','mean_rank_ic_sd')]:cp(d,n,src,c)
    d['source']=src['_source'];t5.append(d)
add('T5_same_patch',t5,'Stored B1 five-seed mean and sample SD (ddof=1), with Daily5/Weekly2 Single and diverse-patch Fixed. Same-patch Fixed uses four Daily10 and four Weekly4 experts. All O/C cells use the recorded 12/12 profile.')
t6=[]
for src in rows(FROZEN+'five_seed_summary.csv'):
    if src['setting']!='head_frozen':continue
    d={'input_set':src['input_set'],'backbone':src['backbone'],'setting':src['setting'],'variant':src['variant'],'horizon':src['horizon'],'n_seeds':src['n_seeds']}
    for n,c in [('test_mse_mean','mse_mean'),('test_mse_sd','mse_sd'),('rank_ic_mean','rank_ic_mean'),('rank_ic_sd','rank_ic_sd')]:cp(d,n,src,c)
    d['source']=src['_source'];t6.append(d)
assert len(t6)==48
add('T6_frozen_head_primary',t6,'Step 15 primary setting, with both experts and prediction head frozen. S1=Fixed-trained backbone; S2=Adaptive-trained backbone. Fixed, Static, constant_gate and Adaptive values are stored five-seed mean ± sample SD; no evaluation or retraining is run.')

# T7: O has a saved aggregate text; C does not have these requested metrics.
router_path='dataset/audit/experiment5_router_diagnostics/experiment5_router_diagnostics_summary.txt'
router_text=read_text(router_path)
router_metrics=[('sample_tv','sample_dependence_index'),('within_ticker_tv','within_ticker_temporal_tv'),('within_date_tv','within_date_crosssectional_tv'),('normalized_entropy','normalized_entropy_mean'),('effective_scales','effective_scales_mean'),('mean_max_weight','max_weight_mean'),('cross_seed_pairwise_tv','cross_seed_mean_pairwise_seed_tv'),('dominant_scale_agreement','dominant_agreement_share'),('sample_spearman_rho','sample_level_rho_mean'),('date_spearman_rho','date_level_rho_mean')]
t7=[];current=None
for line_no,line in enumerate(router_text.splitlines(),1):
    m=re.fullmatch(r'(5|10|20)d / (daily|weekly)',line)
    if m:
        current={'horizon':int(m[1]),'branch':m[2],'n_seeds':5}
        t7.append(current)
    if current is None:continue
    for n,c in router_metrics:
        m=re.match(r'\s+'+c+r': ([+\-\d.eE]+)',line)
        if m:
            current['O_'+n]=fmt(m[1]);current['C_'+n]=NS
            copies.append({'destination':current,'field':'O_'+n,'source':router_path,'line':line_no,'column':c,'formatted_value':fmt(m[1])})
for d in t7:d['source']=router_path;d['C_source']='Step16 H stores other router quantities; requested metrics NOT STORED'
assert len(t7)==6 and all('O_'+n in d for d in t7 for n,c in router_metrics)
add('T7_router_diagnostics',t7,'O values are copied from the saved original Table 7A/7B diagnostic summary, including its across-seed means. C five-seed sample TV, within-ticker/date TV, entropy, effective scales, max weights, cross-seed TV/agreement and TV-to-Static versus joint-loss sample/date Spearman rho are NOT STORED. Step 16 router SD and between-date variance shares measure different quantities and are not substituted. All rho values remain descriptive.')
gap('T7','All ten requested C five-seed router diagnostics are not stored. O values are available; no diagnostics are recomputed from saved weights.')

# T8: copy saved distributions, never summarize epoch histories.
dist=rows(BASE+'best_epoch_distribution.csv')
t8=[]
for inp in ('O','C'):
    for model in ('single','fixed','static','adaptive','lstm','vanilla','swim'):
        for h in (5,10,20):
            d={'input_set':inp,'mechanism':model,'horizon':h,'configuration':'daily_weekly','requested_n_seeds':5,'best_epoch_counts':NS,'best_epoch_min':NS,'best_epoch_median':NS,'best_epoch_max':NS,'mean_train_huber_change_best_to_stop':NS,'mean_val_huber_change_best_to_stop':NS,'source':NS}
            if inp=='O' and model in ('static','adaptive'):
                src=one(dist,model='multiscale_transformer',mechanism=model,input_config='daily_weekly',horizon=h)
                for n,c in [('best_epoch_counts','best_epoch_counts'),('best_epoch_min','best_epoch_min'),('best_epoch_median','best_epoch_median'),('best_epoch_max','best_epoch_max')]:cp(d,n,src,c)
                d['source']=src['_source']
            t8.append(d)
add('T8_learning_curve_summaries',t8,'Only the stored O Static/Adaptive five-seed best-epoch distributions match the requested five-seed population. The selected archived single-seed distributions are retained separately below, with n_runs as saved. No best-to-stop loss changes or mean-over-seed epoch curves are stored. Raw per-epoch history is not aggregated or differenced.')
t8old=[]
for src in dist:
    d={'input_set':'O','scope':'archived formal population; not B5FIX five-seed neural runs'}
    for c in ('model','mechanism','input_config','horizon','n_runs','n_best_epoch_1','best_epoch_min','best_epoch_median','best_epoch_max','best_epoch_counts'):cp(d,c,src,c)
    d['source']=src['_source'];t8old.append(d)
add('T8_archived_distributions',t8old,'All selected archived Step 3 distribution rows are copied, including single-resolution and nominal seed-42 cells. These do not supply missing five-seed C or B5FIX distributions.')
gap('T8 / F7','Mean best-to-stopping train/validation loss changes and mean-over-seed epoch Huber curves are not stored. Most requested five-seed best-epoch distributions are also absent. F7 is an explicit availability placeholder, not a learning-curve result.')

# Figure data: direct selection of stored numeric fields only.
rolling=rows(B2S+'window_summary_5seed.csv')
five=rows(A16+'clarified/B_seed_summary.csv')
f3=[]
for inp in ('O','C'):
    for h in (5,10,20):
        for model in ('single','fixed','same_patch_fixed','static','adaptive','lstm','vanilla','swim','ridge'):
            d={'input_set':inp,'horizon':h,'model':model,'n_seeds':1 if model=='ridge' else 5,'mse_ratio_mean':NS,'mse_ratio_sd':NS,'source':NS}
            if model in ('static','adaptive','ridge'):
                src=one(rolling,window='W3',input_set=inp,model=model,horizon=h)
                cp(d,'mse_ratio_mean',src,'mse_relative_train_mean_mean')
                cp(d,'mse_ratio_sd',src,'mse_relative_train_mean_sd')
                if d['mse_ratio_sd']==NA:d['mse_ratio_sd']=NS
                d['source']=src['_source']
            group='samepatch' if model=='same_patch_fixed' else 'ridge' if model=='ridge' else 'neural' if model in ('lstm','vanilla','swim') else 'core'
            # Retain the available MSE mean/SD alongside the missing ratio, not as a replacement.
            candidates=[z for z in five if z['group']==group and z['input_set']==inp and z['model']==model and z['config']=='daily_weekly' and z['h']==str(h)]
            assert len(candidates)==1,(inp,h,model,len(candidates))
            src=candidates[0]
            for n,c in [('stored_test_mse_mean','mse_mean'),('stored_test_mse_sd','mse_sd')]:cp(d,n,src,c)
            d['mse_summary_source']=src['_source'];f3.append(d)
add('F3_stored_plot_data',f3,'Ratio means and SDs are stored for W3 Static/Adaptive and the deterministic Ridge mean in the B2 supplement. The requested ratios/ratio SDs for other models are NOT STORED, although their MSE mean/SD exists. No division by a reference MSE, conversion of R2 to a ratio, or resummarization is performed. Ridge has no five-seed SD.')
gap('F3','Only Static/Adaptive ratio mean ± SD and the deterministic Ridge ratio mean are stored. Single/Fixed/same-patch/LSTM/Vanilla/SWiM ratio summaries are NOT STORED; missing rows are labelled, not fabricated.')
f4=[]
for src in rows(B3+'summary_5seed.csv'):
    d={'input_set':src['input_set'],'horizon':src['horizon'],'configuration':src['configuration'],'mechanism':src['mechanism'],'n_seeds':src['n_seeds']}
    for n,c in [('test_mse_mean','test_mse_mean'),('test_mse_sd','test_mse_sd_ddof1')]:cp(d,n,src,c)
    d['source']=src['_source'];f4.append(d)
add('F4_stored_plot_data',f4,'B3 five-seed TEST MSE mean ± sample SD. Daily windows cover the same 26 completed weeks as Weekly. B3-C references the last completed week close, so B3-C Weekly differs from the formal C pipeline. Comparison with the formal Daily90/Weekly26 setting combines coverage and exclusion of the unfinished current week.')
f5=[]
for src in rolling:
    if src['model'] not in ('static','adaptive','ridge','zero'):continue
    d={'window':src['window'],'input_set':src['input_set'],'model':src['model'],'horizon':src['horizon'],'n_seeds':src['n_seeds']}
    cp(d,'mse_ratio_mean',src,'mse_relative_train_mean_mean')
    cp(d,'mse_ratio_sd',src,'mse_relative_train_mean_sd')
    if d['mse_ratio_sd']==NA:d['mse_ratio_sd']='N/A (deterministic; no seed SD)'
    d['source']=src['_source'];f5.append(d)
add('F5_stored_plot_data',f5,'Stored five-seed rolling summaries, each relative to that window/horizon Global Train-Mean MSE. Ridge and Zero are deterministic single forecasts, so seed SD is inapplicable. W3 is the saved reference, without retraining.')
g=rows(A16+'G_PRIMARY_with_existing_O.csv')
f6=[]
strategy_names={'single':'Single','fixed':'Fixed','static':'Static','adaptive':'Adaptive','lstm':'LSTM','vanilla':'Vanilla','swim':'SWiM','EW':'1/N','MINVAR':'Minimum variance','RISKPARITY':'Risk parity'}
for inp in ('O','C'):
    for h in (5,10,20):
        for model in strategy_names:
            strategy=model if model in ('EW','MINVAR','RISKPARITY') else model+'_ens'
            layer='allocation benchmark' if model in ('EW','MINVAR','RISKPARITY') else 'L2_g5'
            hits=[z for z in g if z['input_set']==inp and z['h']==str(h) and z['strategy']==strategy and z['layer']==layer and float(z['cost_bp'])==10 and z['split']=='test']
            if len(hits)>1:
                hits=[z for z in hits if z['provenance'].startswith('Step16')]
            assert len(hits)<=1,(inp,h,strategy,len(hits))
            d={'input_set':inp,'horizon':h,'strategy':strategy_names[model],'allocation':layer,'gamma':5,'cost_bp':10,'sharpe':NS,'source':NS,'provenance':NS}
            if hits:
                cp(d,'sharpe',hits[0],'sharpe');d['source']=hits[0]['_source'];d['provenance']=hits[0]['provenance']
            f6.append(d)
add('F6_stored_plot_data',f6,'Stored seed-ensemble mean–variance portfolio Sharpe (gamma=5, cost=10bp) and 1/N/minimum-variance/risk-parity references. O Single/Fixed ensemble portfolios are NOT STORED. O Static/Adaptive use saved Step 8 reference portfolios; O neural ensembles use Step 16 uniform12 sources. No returns, allocations or Sharpe are recomputed.')
gap('F6','O Single and O Fixed seed-ensemble mean–variance portfolio Sharpe is NOT STORED. All other requested F6 rows are available.')
f7=[{'input_set':inp,'mechanism':m,'horizon':20,'epoch':NS,'mean_train_huber':NS,'mean_val_huber':NS,'source':'Raw per-seed histories exist; requested across-seed epoch means NOT STORED'} for inp in ('O','C') for m in ('static','adaptive')]
add('F7_stored_plot_data',f7,'Requested mean-over-seeds epoch curves are NOT STORED. No history files are averaged. F7 PNG/PDF are clearly marked availability placeholders, not manuscript learning-curve figures.')
add('MISSING_VALUES',gaps,'Missing requested values under the no-new-statistic rule. Other undefined source values remain explicitly N/A in the tables.')

# Verify all copied numbers still equal source fields after six-significant-digit formatting.
for record in copies:
    actual=record['destination'][record['field']]
    assert actual==record['formatted_value'] or (record['formatted_value']==NA and actual in (NS,'N/A (deterministic; no seed SD)')),(record,actual)
assert len(t1)==3 and len(t2)==45 and len(t5)==18 and len(f3)==54 and len(f4)==36 and len(f5)==72 and len(f6)==60

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, FuncFormatter
font_path=font_manager.findfont('Times New Roman',fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.titlesize':9,'figure.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'lines.linewidth':1.1,'savefig.dpi':300})
WIDTH=16/2.54
colors={'static':'#0072B2','adaptive':'#D55E00','ridge':'#009E73','zero':'#CC79A7'}
plot_artifacts=[]
figures=[]


def number(x):
    try:return float(x)
    except (ValueError,TypeError):return None


def six_panels(height):
    fig,axs=plt.subplots(3,2,figsize=(WIDTH,height/2.54))
    fig.subplots_adjust(left=.225,right=.975,bottom=.075,top=.90,wspace=.34,hspace=.45)
    fig.text(.383,.94,'O',ha='center',va='top')
    fig.text(.817,.94,'C',ha='center',va='top')
    for i,ax in enumerate(axs.flat):
        ax.text(-.08,1.025,'('+chr(97+i)+')',transform=ax.transAxes,ha='left',va='bottom')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:format(x,'.3g')))
        ax.grid(axis='x',color='#dddddd',linewidth=.4)
        ax.set_axisbelow(True)
    return fig,axs


def save_plot(fig,name,caption):
    # Perform rendering before creating any final outputs, so layout problems can fail safely.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        assert not ax.get_title(), 'No titles inside images'
        for t in ax.texts+ax.get_xticklabels()+ax.get_yticklabels()+[ax.xaxis.label,ax.yaxis.label]:
            if not t.get_text() or not t.get_visible():continue
            b=t.get_window_extent(renderer)
            assert b.x0>=-2 and b.y0>=-2 and b.x1<=fig.bbox.width+2 and b.y1<=fig.bbox.height+2,(name,t.get_text(),b.bounds,fig.bbox.bounds)
    figures.append((name,fig,caption))


fig,axs=six_panels(25)
models=['single','fixed','same_patch_fixed','static','adaptive','lstm','vanilla','swim','ridge']
labels=['Single','Fixed','Same-patch','Static','Adaptive','LSTM','Vanilla','SWiM','Ridge']
for row,h in enumerate((5,10,20)):
    for col,inp in enumerate(('O','C')):
        ax=axs[row,col]
        ax.set_yticks(range(9),labels if col==0 else ['']*9)
        ax.set_ylim(8.7,-.7)
        ax.axvline(1,color='#555555',linestyle='--',linewidth=.7)
        for y,m in enumerate(models):
            d=one(f3,input_set=inp,horizon=h,model=m)
            x,s=number(d['mse_ratio_mean']),number(d['mse_ratio_sd'])
            if x is None:
                ax.text(.97,y,'NOT STORED',transform=ax.get_yaxis_transform(),ha='right',va='center',color='#666666')
            else:
                ax.errorbar(x,y,xerr=s,fmt='o',color=colors.get(m,'#333333'),markersize=3,capsize=2)
        if col==0:ax.set_ylabel('h='+str(h),labelpad=4)
        if row==2:ax.set_xlabel('TEST MSE / Train-Mean MSE')
save_plot(fig,'F3_relative_MSE','Rows: h=5,10,20; columns: O,C. Points and ±1 seed SD use stored ratio summaries only. Missing ratios are labelled NOT STORED. Ridge is deterministic and has no seed SD. Dashed reference=1. This figure is incomplete for the full requested model set.')

fig,axs=six_panels(19)
cfgs=['daily_only','weekly_only','daily_weekly']
for row,h in enumerate((5,10,20)):
    for col,inp in enumerate(('O','C')):
        ax=axs[row,col];ax.grid(axis='y',color='#dddddd',linewidth=.4)
        ax.set_xticks(range(3),['Daily','Weekly','D+W'])
        for m,offset in [('static',-.08),('adaptive',.08)]:
            ds=[one(f4,input_set=inp,horizon=h,configuration=c,mechanism=m) for c in cfgs]
            ax.errorbar([i+offset for i in range(3)],[float(d['test_mse_mean']) for d in ds],yerr=[float(d['test_mse_sd']) for d in ds],fmt='o-',markersize=3,capsize=2,label=m.capitalize(),color=colors[m])
        ax.set_xlim(-.35,2.35)
        if col==0:ax.set_ylabel('h='+str(h)+' TEST MSE')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.61,1.008),ncol=2,frameon=False)
save_plot(fig,'F4_matched_coverage','B3 five-seed TEST MSE mean ± SD, Daily-matched, Weekly-26, and their combination. Rows: h=5,10,20; O,C columns. In C, both branches use last-completed-week close and matched volume interval; these differ from the formal C reference.')

fig,axs=six_panels(19)
for row,h in enumerate((5,10,20)):
    for col,inp in enumerate(('O','C')):
        ax=axs[row,col];ax.set_xticks(range(3),['W1','W2','W3']);ax.axhline(1,color='#555555',linestyle='--',linewidth=.7)
        for m,off,marker in [('static',-.12,'o'),('adaptive',-.04,'s'),('ridge',.04,'^'),('zero',.12,'D')]:
            ds=[one(f5,input_set=inp,horizon=h,window=w,model=m) for w in ('W1','W2','W3')]
            xx=[i+off for i in range(3)];yy=[float(d['mse_ratio_mean']) for d in ds]
            ax.plot(xx,yy,marker=marker,color=colors[m],markersize=3,label=m.capitalize())
            for x,y,d in zip(xx,yy,ds):
                sd=number(d['mse_ratio_sd'])
                if sd is not None:ax.errorbar(x,y,yerr=sd,color=colors[m],capsize=2)
        ax.set_xlim(-.35,2.35);ax.grid(axis='y',color='#dddddd',linewidth=.4)
        if col==0:ax.set_ylabel('h='+str(h)+' MSE / Train-Mean')
        ax.yaxis.set_major_locator(MaxNLocator(4))
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.61,1.008),ncol=4,frameon=False,columnspacing=.9,handlelength=1.4)
save_plot(fig,'F5_rolling_windows','Stored W1/W2/W3 MSE ratios, each relative to its own TRAIN-mean forecast. Five-seed mean ± sample SD for Static/Adaptive; deterministic Ridge/Zero have no seed SD. Rows: h=5,10,20; O,C columns. Dashed reference=1.')

fig,axs=six_panels(27)
names=list(strategy_names.values())
for row,h in enumerate((5,10,20)):
    for col,inp in enumerate(('O','C')):
        ax=axs[row,col];ax.set_yticks(range(10),names if col==0 else ['']*10);ax.set_ylim(9.7,-.7)
        for y,name in enumerate(names):
            d=one(f6,input_set=inp,horizon=h,strategy=name);value=number(d['sharpe'])
            if value is None:ax.text(.97,y,'NOT STORED',transform=ax.get_yaxis_transform(),ha='right',va='center',color='#666666')
            else:
                is_ref=name in ('1/N','Minimum variance','Risk parity')
                ax.plot(value,y,'D' if is_ref else 'o',color='#666666' if is_ref else '#0072B2',markersize=4)
        if col==0:ax.set_ylabel('h='+str(h))
        if row==2:ax.set_xlabel('Sharpe ratio')
save_plot(fig,'F6_portfolio_Sharpe','Stored TEST portfolio Sharpe for five-seed ensemble mean–variance forecasts, gamma=5, cost=10bp. Blue circles=model ensembles; grey diamonds=allocation benchmarks. Rows: h=5,10,20; O,C columns. Missing O Single/Fixed ensemble portfolios are explicitly labelled.')

fig,axs=plt.subplots(1,2,figsize=(WIDTH,7/2.54))
fig.subplots_adjust(left=.13,right=.97,bottom=.22,top=.83,wspace=.25)
for col,(inp,ax) in enumerate(zip(('O','C'),axs)):
    ax.text(-.06,1.03,'('+chr(97+col)+')',transform=ax.transAxes,ha='left')
    ax.text(.5,1.17,inp,transform=ax.transAxes,ha='center')
    ax.text(.5,.55,'NOT STORED',transform=ax.transAxes,ha='center',va='center')
    ax.text(.5,.32,'Mean over seeds by epoch',transform=ax.transAxes,ha='center',va='center')
    ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('Epoch')
    if col==0:ax.set_ylabel('Train / VAL Huber loss')
save_plot(fig,'F7_learning_curves_NOT_STORED','Availability placeholder only, not a learning-curve result. Static/Adaptive h=20 mean-over-seed train/VAL Huber curves were not stored. Under the extraction-only rule, raw epoch histories are not averaged. Do not use this placeholder as a manuscript result figure.')

# All checks above precede exclusive delivery. No overwrite or destructive cleanup.
table_paths=[]
for name,rr in tables.items():
    fields=list(dict.fromkeys(k for d in rr for k in d))
    p=OUT/(name+'.csv')
    with p.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k:fmt(d.get(k,NS)) for k in fields} for d in rr])
    table_paths.append(p)
for name,fig,caption in figures:
    for ext in ('png','pdf'):
        p=OUT/(name+'.'+ext)
        with p.open('xb') as f:fig.savefig(f,format=ext,dpi=300)
        plot_artifacts.append({'name':name,'path':str(p.relative_to(ROOT)),'format':ext,'width_cm':16,'dpi':300,'font':'Times New Roman','font_size_pt':9,'caption':caption})
    plt.close(fig)

# Read-only source hashes before/after, including configuration evidence.
for s,d in sources.items():
    d['sha256_after']=sha(ROOT/s)
    assert d['sha256_before']==d['sha256_after'],s
    d['unchanged']=True
with (OUT/'SOURCE_MANIFEST.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256_before','sha256_after','unchanged']);w.writeheader();w.writerows(sources.values())
trace=[]
row_lookup={id(d):(name,i+2) for name,rr in tables.items() for i,d in enumerate(rr)}
for rec in copies:
    name,line=row_lookup[id(rec['destination'])]
    trace.append({k:v for k,v in rec.items() if k!='destination'}|{'table':name+'.csv','table_line':line})
with (OUT/'VALUE_PROVENANCE.json').open('x') as f:json.dump(trace,f,indent=2)
with (OUT/'FIGURE_MANIFEST.json').open('x') as f:json.dump(plot_artifacts,f,indent=2)


def markdown(rr,cols=None):
    if cols is None:cols=[k for k in rr[0] if k not in ('source','C_source','parameter_count_source','source_code','note')]
    def esc(v):return fmt(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(esc(d.get(c,NS)) for c in cols)+' |' for d in rr])


report=['# REPORT STEP18 — Manuscript tables and figures','', '## BLOCKING','', 'Requested quantities that are absent from stored results prevent a complete manuscript-ready set. They are labelled `NOT STORED`; no derived statistic or test has been computed.','',markdown(gaps),'', '## Scope and conventions','',
        'Read-only extraction and plotting only. Numeric values have six significant digits (source precision is retained where lower). Original and causal inputs are labelled O and C. No training, inference, losses, means, SDs, ratios, bootstraps, p-values or corrections were recalculated. The builder performs selection, copying, sorting/formatting, source hashing and plot rendering only. Step 17 results are excluded. O neural benchmark summaries use Step 16 sources from B5FIX/B5 uniform12; archived nominal-seed neural values appear only where explicitly labelled as the Step 3 archival learning-distribution reference.', '',
        'N/A denotes an undefined value in the source, such as Rank IC for constant cross-sections or SD for a deterministic model. NOT STORED denotes a missing requested result. NOT RECORDED denotes absent historical search-budget evidence. Source paths and code line references are preserved in the CSVs; VALUE_PROVENANCE.json maps copied numeric fields to source rows/columns.', '']
for name,rr in tables.items():
    if name=='MISSING_VALUES':continue
    report.extend(['## '+name,'',notes[name],'',f'[CSV]({name}.csv)','',markdown(rr),''])
report.extend(['## Figures','', 'All multi-panel images use O on the left and C on the right, panel letters (a), (b), …, 16 cm width, Times New Roman 9 pt, and a colour-blind-safe palette. PNGs use 300 dpi; PDFs embed TrueType font subsets. No panel or figure title is drawn inside an image. PNG/PDF export does not change canvas width with tight bounding-box cropping.',''])
for name,fig,caption in figures:
    report.extend(['### '+name,'',caption,'',f'[PNG]({name}.png) · [PDF]({name}.pdf)',''])
report.extend(['## File index',''])
for p in table_paths:report.append('- ['+p.name+']('+p.name+')')
for p in plot_artifacts:report.append('- ['+Path(p['path']).name+']('+Path(p['path']).name+')')
report.extend(['- [SOURCE_MANIFEST.csv](SOURCE_MANIFEST.csv)','- [VALUE_PROVENANCE.json](VALUE_PROVENANCE.json)','- [FIGURE_MANIFEST.json](FIGURE_MANIFEST.json)','- [VALIDATION_STEP18.json](VALIDATION_STEP18.json)','', '## Validation','', f'{len(sources)} source files have matching before/after SHA-256. {len(copies)} copied field values were checked against their source formatting before export. Every output uses exclusive creation. Figure labels were checked against the canvas bounds; Times New Roman was found at `{font_path}`. No model/trainer or statistical-test module is imported. These are extraction/format checks, not new inferential tests.',''])
with (OUT/'REPORT_STEP18.md').open('x') as f:f.write('\n'.join(report))
validation={'generated_utc':datetime.now(timezone.utc).isoformat(),'no_training':True,'no_inference':True,'no_new_statistics_or_tests':True,'sources_unchanged':True,'source_count':len(sources),'copied_field_count':len(copies),'tables':{name:len(rr) for name,rr in tables.items()},'figure_exports':len(plot_artifacts),'font_path':font_path,'exclusive_creation':True,'missing_requests':gaps,'artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in OUT.iterdir() if p.is_file()]}
with (OUT/'VALIDATION_STEP18.json').open('x') as f:json.dump(validation,f,indent=2)
print(json.dumps({'report':str(OUT/'REPORT_STEP18.md'),'csv_tables':len(tables),'figure_exports':len(plot_artifacts),'source_files_unchanged':len(sources),'missing_items':len(gaps)},indent=2))
