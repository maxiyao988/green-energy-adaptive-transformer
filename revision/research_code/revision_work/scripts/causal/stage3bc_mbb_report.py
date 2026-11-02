"""Add the requested paired MBB and explicit seed tables; never train or overwrite.

Run from the repository with .venv/bin/python -B. All model, metric, paired-loss
and bootstrap code is imported from the preserved formal implementations.
"""
from core import *
from report_stage3bc_final import old_predictions, canonical, table, number
from scripts.python import panel_pathformer_experiment4_inference as inference
import re
import time

NAME = 'stage3bc_mbb'
DEST = OUT / NAME
SEEDS = [0, 1, 21, 42, 3407]
HORIZONS = [5, 10, 20]


def save(name, frame):
    return csvout(f'{NAME}/{name}', frame)


def source_path(mechanism, seed):
    if seed != 42:
        return ROOT / 'dataset/audit/experiment4_multiseed/experiment4_test_predictions.csv'
    if mechanism == 'adaptive':
        return ROOT / 'dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv'
    return ROOT / 'dataset/audit/experiment3_pathformer_ablation/experiment3_test_predictions.csv'


def absolute_old_links(text):
    def replace(match):
        address = match.group(2)
        if address.startswith(('/', 'http:', 'https:', '#')):
            return match.group(0)
        return match.group(1) + str((OUT / address).resolve()) + ')'
    return re.sub(r'(\[[^\]]*\]\()([^\)]+)\)', replace, text)


def main():
    if DEST.exists():
        raise FileExistsError(f'Refusing to overwrite existing output namespace: {DEST}')
    start = time.perf_counter()
    started = utc()
    profile = threads(3)
    qdir = OUT / 'stage3bc_queue'
    old_report = OUT / 'REPORT_STEP4_FINAL.md'
    old_manifest = json.loads((qdir / 'MANIFEST_FINAL.json').read_text())
    assert sha(old_report) == old_manifest['report_sha256']
    # Verify all already-completed Stage3b/c artifacts before reusing them.
    protected = {ROOT / row['path']: row['sha256'] for row in old_manifest['files']}
    protected[old_report] = old_manifest['report_sha256']
    for path, digest in protected.items():
        assert path.is_file() and sha(path) == digest, str(path)
    for row in old_manifest.get('new_source', []):
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    old = old_predictions()
    sources = []
    source_checks = []
    metric_rows = []
    canonical_sets = {}
    cache = pd.read_csv(qdir / 'all_seed_metrics.csv', float_precision='round_trip')
    common_keys = None
    for feature in ['original_levels', 'causal_relative']:
        frames = []
        for mechanism in ['static', 'adaptive']:
            for h in HORIZONS:
                for seed in SEEDS:
                    if feature == 'original_levels':
                        path = source_path(mechanism, seed)
                        frame = old[(mechanism, h, seed)]
                    else:
                        prefix = '' if seed == 42 else 'stage3b/'
                        path = OUT / f'{prefix}preds/{mechanism}_dw_h{h}_s{seed}_test.parquet'
                        frame = canonical(pd.read_parquet(path))
                    protected.setdefault(path, sha(path))
                    assert len(frame) == 6647
                    assert not frame.duplicated(['ticker', 'anchor_date']).any()
                    assert frame.groupby('anchor_date').ticker.nunique().eq(17).all()
                    assert frame.anchor_date.nunique() == 391
                    keys = frame[['ticker', 'anchor_date']]
                    if common_keys is None:
                        common_keys = keys
                    assert keys.equals(common_keys)
                    y = frame.y_true.to_numpy(np.float32)
                    pred = frame.y_pred.to_numpy(np.float32)
                    assert np.array_equal(y, old[(mechanism, h, seed)].y_true.to_numpy(np.float32))
                    assert np.isfinite(y).all() and np.isfinite(pred).all()
                    result = metrics(y, pred, frame.ticker.to_numpy(), frame.anchor_date.to_numpy())
                    previous = cache[(cache.feature_set == feature) & (cache.mechanism == mechanism)
                                     & cache.horizon.eq(h) & cache.seed.eq(seed)]
                    assert len(previous) == 1
                    for key, value in result.items():
                        assert np.isclose(value, previous.iloc[0][key], rtol=1e-12, atol=1e-14, equal_nan=True), (feature, mechanism, h, seed, key)
                    metric_rows.append(dict(feature_set=feature, mechanism=mechanism, horizon=h,
                                            seed=seed, **result))
                    sources.append(dict(feature_set=feature, mechanism=mechanism, horizon=h, seed=seed,
                                        path=rel(path), sha256=sha(path), n_rows=len(frame),
                                        source='archived formal predictions' if feature == 'original_levels'
                                        else ('preserved Stage3a' if seed == 42 else 'completed Stage3b')))
                    source_checks.append(dict(feature_set=feature, mechanism=mechanism, horizon=h, seed=seed,
                                              metric_bridge_passed=True, keys_and_labels_passed=True))
                    selected = frame[['ticker', 'anchor_date', 'y_true', 'y_pred']].copy()
                    selected['mechanism'] = mechanism
                    selected['horizon'] = f'{h}d'
                    selected['seed'] = seed
                    frames.append(selected)
        canonical_sets[feature] = pd.concat(frames, ignore_index=True)
        inference._cross_seed_date_consistency_check(canonical_sets[feature])
        print('VERIFIED 30 TEST CELLS', feature, flush=True)

    m = pd.DataFrame(metric_rows)
    chosen = ['mechanism', 'horizon', 'seed', 'pooled_mse', 'cross_sectional_rank_ic']
    per_seed = m[m.feature_set == 'original_levels'][chosen].merge(
        m[m.feature_set == 'causal_relative'][chosen], on=['mechanism', 'horizon', 'seed'],
        suffixes=('_original', '_causal'), validate='one_to_one').sort_values(['horizon', 'mechanism', 'seed'])
    assert len(per_seed) == 30
    summary = []
    for (h, mechanism), group in per_seed.groupby(['horizon', 'mechanism'], sort=True):
        row = dict(horizon=int(h), mechanism=mechanism, n_seeds=5)
        assert set(group.seed) == set(SEEDS)
        for column in ['pooled_mse_original', 'pooled_mse_causal', 'cross_sectional_rank_ic_original', 'cross_sectional_rank_ic_causal']:
            assert np.isfinite(group[column]).all()
            row[column + '_mean'] = float(group[column].mean())
            row[column + '_sd_ddof1'] = float(group[column].std(ddof=1))
        summary.append(row)
    summary = pd.DataFrame(summary)
    wins = []
    for feature in ['original_levels', 'causal_relative']:
        for h in HORIZONS:
            g = m[m.feature_set.eq(feature) & m.horizon.eq(h)]
            loss = g.pivot(index='seed', columns='mechanism', values='pooled_mse')
            ic = g.pivot(index='seed', columns='mechanism', values='cross_sectional_rank_ic')
            wins.append(dict(feature_set=feature, horizon=h,
                             adaptive_mse_wins=int((loss.adaptive < loss.static).sum()),
                             static_mse_wins=int((loss.static < loss.adaptive).sum()),
                             mse_ties=int((loss.static == loss.adaptive).sum()),
                             adaptive_rank_ic_wins=int((ic.adaptive > ic.static).sum()),
                             static_rank_ic_wins=int((ic.static > ic.adaptive).sum()),
                             rank_ic_ties=int((ic.static == ic.adaptive).sum()),
                             mean_delta_mse=float((loss.adaptive - loss.static).mean())))
    wins = pd.DataFrame(wins)
    for _, row in pd.read_csv(qdir / 'seed_wins.csv').iterrows():
        match = wins[wins.feature_set.eq(row.feature_set) & wins.horizon.eq(row.horizon)].iloc[0]
        assert match.adaptive_mse_wins == row.adaptive_seed_wins
        assert match.static_mse_wins == row.static_seed_wins

    protected_inputs = [qdir / p for p in ['MANIFEST_FINAL.json', 'processes.json', 'all_seed_metrics.csv',
                         'paired_loss_by_date.csv', 'seed_wins.csv', 'asof_contrasts.csv', 'runtimes.csv']]
    protected_inputs += [Path(inference.__file__), inference.PROTOCOL_PATH, Path(__file__),
                         Path(common.__file__), ROOT / 'revision_work/scripts/causal/report_stage3bc_final.py',
                         ROOT / 'revision_work/scripts/causal/core.py']
    archived = ROOT / 'dataset/audit/experiment4_inference'
    protected_inputs += [archived / p for p in ['experiment4_loss_diff_by_date.csv', 'experiment4_block_bootstrap_summary.csv']]
    for path in protected_inputs:
        protected.setdefault(path, sha(path))
    assert sha(inference.PROTOCOL_PATH) == inference.PROTOCOL_SHA256
    assert inference.BOOTSTRAP_B == 20000 and inference.BOOTSTRAP_BASE_SEED == 20260826
    assert inference.PRIMARY_BLOCK_LENGTH == {'5d': 5, '10d': 10, '20d': 20}

    # Import the original loss arithmetic as well as its MBB, including its
    # float32 error/difference calculation and float64 hierarchical averaging.
    seed_date = {}
    date = {}
    bootstrap = {}
    replicates = {}
    for feature, frame in canonical_sets.items():
        seed_date[feature] = inference._build_loss_diff_by_seed_date(frame)
        date[feature] = inference._build_loss_diff_by_date(seed_date[feature])
        if feature == 'original_levels':
            inference._run_final_point_estimate_audit(date[feature])
        replicates[feature], bootstrap[feature] = inference._run_bootstrap(date[feature])
        assert len(seed_date[feature]) == 5865 and len(date[feature]) == 1173
        assert len(replicates[feature]) == 120000 and len(bootstrap[feature]) == 6
        print('MBB FINISHED', feature, flush=True)

    # Positive control: the exact imported pipeline must replay the archived
    # original-feature statistics. Tiny CSV parse roundoff is allowed here only.
    archive_checks = []
    for filename, recomputed, keys in [
        ('experiment4_loss_diff_by_date.csv', date['original_levels'], ['horizon', 'anchor_date']),
        ('experiment4_block_bootstrap_summary.csv', bootstrap['original_levels'], ['horizon', 'specification'])]:
        reference = pd.read_csv(archived / filename, float_precision='round_trip')
        left = recomputed.sort_values(keys).reset_index(drop=True)
        right = reference.sort_values(keys).reset_index(drop=True)
        assert left.shape == right.shape and list(left.columns) == list(right.columns)
        for column in left.columns:
            if pd.api.types.is_numeric_dtype(left[column]):
                a = left[column].to_numpy(float)
                b = right[column].to_numpy(float)
                valid = np.isfinite(a) & np.isfinite(b)
                delta = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.
                assert np.allclose(a, b, rtol=1e-12, atol=1e-15, equal_nan=True), (filename, column, delta)
                archive_checks.append(dict(file=filename, column=column, max_abs_difference=delta, passed=True))
            else:
                assert left[column].astype(str).equals(right[column].astype(str)), (filename, column)

    old_dates = pd.read_csv(qdir / 'paired_loss_by_date.csv', float_precision='round_trip')
    precision_rows = []
    for feature in date:
        for h in HORIZONS:
            new = date[feature][date[feature].horizon.eq(f'{h}d')].sort_values('anchor_date')
            prior = old_dates[old_dates.feature_set.eq(feature) & old_dates.horizon.eq(h)].sort_values('anchor_date')
            assert new.anchor_date.tolist() == prior.anchor_date.tolist()
            a = new.mean_loss_diff.to_numpy(float)
            b = prior.seed_mean_stock_mean_loss_difference.to_numpy(float)
            precision_rows.append(dict(feature_set=feature, horizon=h, n_dates=391,
                                       max_date_delta_from_previous_float64_loss=float(np.max(np.abs(a-b))),
                                       formal_float32_theta=float(a.mean()), previous_float64_theta=float(b.mean())))
    precision = pd.DataFrame(precision_rows)

    save('per_seed_mse_rank_ic_side_by_side.csv', per_seed)
    save('five_seed_mean_sd_side_by_side.csv', summary)
    save('seed_wins_mse_and_rank_ic.csv', wins)
    save('prediction_sources.csv', sources)
    save('metric_bridge_checks.csv', source_checks)
    save('archived_mbb_reproduction_checks.csv', archive_checks)
    save('loss_arithmetic_precision_comparison.csv', precision)
    for feature in date:
        save(f'{feature}_loss_diff_by_seed_date.csv', seed_date[feature])
        save(f'{feature}_loss_diff_by_date.csv', date[feature])
        save(f'{feature}_mbb_summary.csv', bootstrap[feature])
        save(f'{feature}_mbb_replicates.csv', replicates[feature])

    view = []
    for _, row in summary.iterrows():
        output = dict(horizon=int(row.horizon), mechanism=row.mechanism)
        for stem, label in [('pooled_mse_original', 'Original MSE'), ('pooled_mse_causal', 'Causal MSE'),
                            ('cross_sectional_rank_ic_original', 'Original mean Rank IC'),
                            ('cross_sectional_rank_ic_causal', 'Causal mean Rank IC')]:
            output[label] = f'{number(row[stem+"_mean"])} ± {number(row[stem+"_sd_ddof1"])}'
        view.append(output)
    mbb_cols = ['horizon', 'block_length', 'n_dates', 'n_bootstrap', 'theta_hat', 'ci_lower_95',
                'ci_upper_95', 'centered_p_value', 'centered_p_value_holm_primary', 'classification']
    b = bootstrap['causal_relative']
    primary = b[b.specification.eq('PRIMARY')]
    sensitivity = b[b.specification.eq('SENSITIVITY')]
    sensitivity_changed = []
    for h in inference.HORIZONS:
        p = primary[primary.horizon.eq(h)].iloc[0]
        s = sensitivity[sensitivity.horizon.eq(h)].iloc[0]
        if p.classification != s.classification or (p.centered_p_value < .05) != (s.centered_p_value < .05):
            sensitivity_changed.append(h)
    findings = []
    for _, row in primary.iterrows():
        adjusted = row.centered_p_value_holm_primary
        conclusion = ('Adaptive lower loss' if row.theta_hat < 0 else 'Static lower loss') if adjusted < .05 else 'no difference detected after Holm correction'
        findings.append(f'{row.horizon}: {conclusion} (Holm p={number(adjusted)}).')
    finding_text = ' '.join(findings)
    added = [
        '## Stage3b — Per-seed and five-seed comparison (requested additions)\n',
        'The completed queue is reused: 24 Stage3b causal cells plus the six preserved Stage3a seed-42 cells form 30 causal cells. The original comparison uses the 30 archived formal test-prediction cells; **Adaptive original seed 42 is MAIN V2 archived formal predictions**, never a recovery rerun. Static original seed 42 is Experiment3; the other four original seeds are Experiment4. All cells have 6,647 identical ticker/date test keys and the original frozen labels. [Prediction source paths/hashes](prediction_sources.csv).\n',
        'MSE and Rank IC below are recomputed using imported V2 metric functions. Mean Rank IC is the mean of within-date cross-sectional Spearman correlations across 17 stocks, excluding undefined correlations, as in [panel_common.py:200]('+str(ROOT/'scripts/python/panel_common.py')+':200). It is not a pooled correlation. Means and sample SD (ddof=1) are across exactly five seeds: 0, 1, 21, 42, 3407; all requested metric cells are finite.\n',
        '### Per-seed test MSE and mean Rank IC, side by side\n', table(per_seed),
        '[Numeric per-seed table](per_seed_mse_rank_ic_side_by_side.csv).\n',
        '### Five-seed means ± sample SD, side by side\n', table(pd.DataFrame(view)),
        '[Numeric means and SDs](five_seed_mean_sd_side_by_side.csv). The preserved [all-metric cell table]('+str(qdir/'all_seed_metrics.csv')+') and [all-metric five-seed table]('+str(qdir/'five_seed_summary.csv')+') include MAE, Pearson, direction accuracy, PredStd/TrueStd and out-of-sample R².\n',
        '### Seed wins\n', table(wins),
        'MSE wins use lower loss; Rank IC wins use higher IC, paired on the same seed/horizon. Ties are explicit. [Numeric wins](seed_wins_mse_and_rank_ic.csv).\n',
        '### Paired date-level loss and original moving-block bootstrap\n',
        'For each ticker/date/seed/horizon form Adaptive squared error minus Static squared error. Average over 17 stocks at each seed/date, then over the five seeds at each date, then over the 391 ordered dates. Negative favors Adaptive. Imported [formal paired-loss arithmetic]('+str(Path(inference.__file__))+':419) and [date aggregation]('+str(Path(inference.__file__))+':471) perform these steps. The loss calculations and elementwise differences use float32, followed by float64 aggregation, exactly as the original inference runner. The earlier report used float64 squared-loss arithmetic; those immutable outputs remain available. [Precision comparison](loss_arithmetic_precision_comparison.csv) quantifies the small change; the following MBB uses the formal arithmetic.\n',
        'Directly imported [formal MBB runner]('+str(Path(inference.__file__))+':552): non-circular overlapping blocks, uniformly sampled starts, ceil(T/L) sampled blocks concatenated and truncated to T=391; B=20,000; primary L=h; original supplementary L=2h. RNG is default_rng(SeedSequence([20260826, h, L])). Two-sided 95% percentile CIs use linear quantiles; bootstrap SE uses ddof=1. Centered two-sided p=(1+count(|theta_boot−theta_hat|≥|theta_hat|))/(B+1). Holm adjustment uses the three primary horizon p-values. The original CI-sign classification and Holm p-values are both shown; CI classifications are unadjusted. No seed resampling is performed: inference is conditional on these five seeds. The test concerns Adaptive versus Static within causal features, not a test of causal versus original features. [Frozen protocol]('+str(inference.PROTOCOL_PATH)+':130).\n',
        '#### Causal family — PRIMARY, L=h\n', table(primary, mbb_cols),
        finding_text+'\n',
        '#### Causal family — Original sensitivity specification, L=2h\n', table(sensitivity, [c for c in mbb_cols if c != 'centered_p_value_holm_primary']),
        ('The inference is sensitive to the pre-specified block-length choice for '+', '.join(sensitivity_changed)+'.' if sensitivity_changed else 'Primary and sensitivity specifications agree in CI-sign classification and raw p<0.05 decisions at all three horizons.')+' Primary conclusions use L=h and the three-horizon Holm correction; sensitivity p-values are raw as in the original protocol.\n',
        '[Causal stock means by seed/date](causal_relative_loss_diff_by_seed_date.csv), [causal seed means by date](causal_relative_loss_diff_by_date.csv), [all causal MBB statistics](causal_relative_mbb_summary.csv), [120,000 causal bootstrap replicates](causal_relative_mbb_replicates.csv).\n',
        '#### Archived original-feature MBB, reproduced as a positive control\n', table(bootstrap['original_levels'][bootstrap['original_levels'].specification.eq('PRIMARY')], mbb_cols),
        'Both loss aggregation and all six original MBB specifications reproduce the archived original-feature outputs within CSV parsing precision (absolute tolerance 1e-15, relative tolerance 1e-12). [Positive-control checks](archived_mbb_reproduction_checks.csv), [original MBB summary](original_levels_mbb_summary.csv), [original seed/date losses](original_levels_loss_diff_by_seed_date.csv), [original date losses](original_levels_loss_diff_by_date.csv).\n',
        '### Verification and preservation\n',
        f'All {len(old_manifest["files"])} previously manifested Stage3b/c artifacts and the previous final-report hash were checked before reuse. All 60 original/causal test cells pass source, row/key/label and cached-metric bridge checks. This report update runs no training or checkpoint inference. [Metric checks](metric_bridge_checks.csv), [input preservation check](protected_inputs_verification.json), [new manifest](MANIFEST.json). Existing checkpoints, outputs, reports and code remain in place.\n',
    ]
    text = absolute_old_links(old_report.read_text())
    a = text.index('## Stage3b — Five-seed comparison')
    z = text.index('## Stage3c — Ex-post levels / asof levels / causal relatives')
    text = text[:a] + '\n'.join(added) + '\n' + text[z:]
    heading = '# Step4 final — Stage3b and Stage3c\n'
    assert text.count(heading) == 1
    text = text.replace(heading, heading + '\nRequested MSE/Rank-IC seed tables and formal MBB added '+utc()+'. '
                        +finding_text+' The 30-cell Stage3b/3c training queue is already complete; its results are reused below. '
                        +'The previous final report is preserved at '+str(old_report)+'.\n', 1)
    text = text.replace('Only after queue completion did the requested conditional12/12 recoveries or Static control run.',
                        'After queue completion the conditional Adaptive h5 and h20 recoveries ran sequentially at12/12 and were EXACT; the Static control was not triggered because h10 was EXACT.')
    with target(DEST / 'REPORT_STEP4_FINAL.md').open('x') as f:
        f.write(text)

    verification = []
    for path, expected in sorted(protected.items()):
        actual = sha(path)
        assert actual == expected, str(path)
        verification.append(dict(path=rel(path), sha256_before=expected, sha256_after=actual, unchanged=True))
    jsonout(f'{NAME}/protected_inputs_verification.json', dict(checked_utc=utc(), passed=True,
             n_files=len(verification), files=verification))
    outputs = [dict(path=rel(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(DEST.iterdir()) if p.is_file()]
    jsonout(f'{NAME}/MANIFEST.json', dict(started_utc=started, completed_utc=utc(),
             wall_clock_seconds=time.perf_counter()-start, analysis_only=True, training_jobs_started=0,
             reused_training_cells=30, n_recomputed_metric_cells=60, profile=profile,
             source=rel(Path(__file__)), source_sha256=sha(__file__),
             formal_inference_source=rel(Path(inference.__file__)), formal_inference_sha256=sha(inference.__file__),
             protocol_sha256=sha(inference.PROTOCOL_PATH), bootstrap_B=inference.BOOTSTRAP_B,
             bootstrap_base_seed=inference.BOOTSTRAP_BASE_SEED, primary_block_lengths=inference.PRIMARY_BLOCK_LENGTH,
             sensitivity_block_lengths=inference.SENSITIVITY_BLOCK_LENGTH,
             sensitivity_changed_horizons=sensitivity_changed, findings=finding_text, files=outputs))
    print('FINAL REPORT', rel(DEST / 'REPORT_STEP4_FINAL.md'), flush=True)
    print(primary[mbb_cols].to_string(index=False), flush=True)
    print('INPUTS UNCHANGED', len(verification), 'ELAPSED', time.perf_counter()-start, flush=True)


if __name__ == '__main__':
    main()
