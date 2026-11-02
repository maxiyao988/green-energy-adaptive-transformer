"""Post-barrier metrics and the pre-registered Step16 joint bootstrap; no training."""
import sys
sys.dont_write_bytecode = True
from io22b_v3 import *
import os
for _key in ENV_KEYS[:5]:
    os.environ[_key] = '1'
sys.path.insert(0, str(ROOT / 'revision_work/scripts/analysis16'))
sys.path.append(str(ROOT / 'revision_work/outputs/_deps_pyarrow15'))
import numpy as np
import pandas as pd
from statistics16_v3 import s

def finite_mean(a):
    x = np.asarray(a, float)
    return float(x.mean()) if len(x) == 5 and np.isfinite(x).all() else np.nan

def finite_sd(a):
    x = np.asarray(a, float)
    return float(x.std(ddof=1)) if len(x) == 5 and np.isfinite(x).all() else np.nan

def table(frame):
    if frame.empty:
        return 'No rows.\n'
    values = frame.copy()
    for col in values:
        if pd.api.types.is_numeric_dtype(values[col]):
            values[col] = values[col].map(lambda v: f'{v:.6g}' if pd.notna(v) else 'N/A')
    values = values.fillna('N/A').astype(str)
    esc = lambda v: str(v).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(esc, values.columns)) + ' |',
                     '| ' + ' | '.join(['---'] * len(values.columns)) + ' |'] +
                    ['| ' + ' | '.join(map(esc, row)) + ' |' for row in values.itertuples(index=False, name=None)]) + '\n'

def main():
    finished = barrier()
    if (OUT / 'ANALYSIS_COMPLETE.json').exists():
        verify_record(OUT / 'ANALYSIS_COMPLETE.json')
        return
    start = utc()
    dest = target(OUT / 'analysis' / ('attempt_' + stamp()))
    dest.mkdir(parents=True)
    outputs = []
    def save(name, data):
        p = dest / name
        csv(p, data)
        outputs.append(p)
        return p
    plan = read(OUT / 'GRID_PLAN.json')['jobs']
    models = sorted({job['model'] for job in plan})
    jobs = [spec('T0', m, h, seed) for m in models for h in HS for seed in SEEDS] + plan
    metrics, routes, panels, eval_files = [], [], {}, []
    for job in jobs:
        paths = sorted((OUT / 'evaluation' / job_id(job)).glob('attempt_*/COMPLETE.json'))
        assert len(paths) == 1, job_id(job)
        record = verify_record(paths[0])
        assert record['job'] == job
        folder = paths[0].parent
        mt = pd.read_csv(folder / 'metrics.csv').iloc[0].to_dict()
        metrics.append(mt)
        panels[(job['setting'], job['model'], job['h'], job['seed'])] = pd.read_csv(folder / 'date_metrics.csv').sort_values('anchor_date').reset_index(drop=True)
        route = pd.read_csv(folder / 'router_diagnostics.csv')
        for key in ['setting', 'model', 'h', 'seed']:
            route[key] = job[key]
        routes.append(route)
        eval_files.append(dict(**job, status=record['status'], complete=rel(paths[0]), predictions=rel(folder / 'predictions_losses_weights.csv'),
                               date_metrics=rel(folder / 'date_metrics.csv'), stock_metrics=rel(folder / 'stock_metrics.csv')))
    M = pd.DataFrame(metrics)
    assert len(M) == (180 if 'fixed' in models else 135)
    save('test_metrics_per_seed.csv', M)
    save('prediction_files.csv', pd.DataFrame(eval_files))
    keys = ['setting', 'model', 'h']
    summary = M.groupby(keys, as_index=False).agg(n_seeds=('seed', 'nunique'),
                 n_complete=('status', lambda x: int(x.eq('COMPLETE').sum())),
                 mse_mean=('mse', finite_mean), mse_sd=('mse', finite_sd),
                 rank_ic_mean=('mean_rank_ic', finite_mean), rank_ic_sd=('mean_rank_ic', finite_sd),
                 da_mean=('da', finite_mean), da_sd=('da', finite_sd), r2os_mean=('r2os', finite_mean), r2os_sd=('r2os', finite_sd))
    assert summary.n_seeds.eq(5).all()
    save('five_seed_summary.csv', summary)
    save('T0_T1_T2_side_by_side.csv', summary.pivot(index=['model', 'h'], columns='setting', values=['mse_mean', 'mse_sd', 'rank_ic_mean', 'rank_ic_sd', 'da_mean', 'da_sd', 'r2os_mean']).rename_axis(columns=['metric', 'setting']).pipe(lambda d: d.set_axis(['_'.join(c) for c in d.columns], axis=1)).reset_index())
    R = pd.concat(routes, ignore_index=True)
    save('router_diagnostics_per_seed.csv', R)
    route_summary = R.groupby(['setting', 'model', 'h', 'branch'], as_index=False).agg(
        n_seeds=('seed', 'nunique'), mean_abs_deviation=('mean_abs_deviation_from_equal', finite_mean),
        mean_sample_sd=('mean_sample_sd', finite_mean), between_date_share=('between_date_share', finite_mean),
        n_defined_between_date_shares=('between_date_share', 'count'), n_constant_seeds=('constant_to_1e7', 'sum'))
    save('router_diagnostics_summary.csv', route_summary)
    hist_rows, behavior, all_runs = [], [], []
    sources = {(r['model'], r['h'], r['seed']): r for r in source_rows()}
    for job in jobs:
        if job['setting'] == 'T0':
            r = sources[(job['model'], job['h'], job['seed'])]
            path, best, status = ROOT / r['history'], r['best_epoch'], 'COMPLETE'
            run = dict(job=job, status=status, best_epoch=best, source=r['source'], history=rel(path))
        else:
            cp, original = completion(job)
            best, status = original.get('best_epoch'), original['status']
            path = cp.parent / 'history.csv'
            run = {k: v for k, v in original.items() if k not in ['artifacts', 'profile_start', 'profile_end']}
            run.update(source=rel(cp), history=rel(path) if path.exists() else None)
        all_runs.append(dict(**{k: job[k] for k in ['setting', 'model', 'h', 'seed']}, status=status,
                        best_epoch=best, epochs=run.get('epochs'), error=run.get('error'), source=run['source'], history=run['history'],
                        started_utc=run.get('started_utc'), ended_utc=run.get('ended_utc'), elapsed_utc_seconds=run.get('elapsed_utc_seconds')))
        loss_row = {k: job[k] for k in ['setting', 'model', 'h', 'seed']}
        loss_row.update(status=status, best_epoch=best, history_source=rel(path) if path.exists() else None,
                        last_epoch=np.nan, train_pct_change=np.nan, val_pct_change=np.nan)
        if path.exists():
            history = pd.read_csv(path)
            if len(history):
                for k in ['setting', 'model', 'h', 'seed']:
                    history[k] = job[k]
                history['huber_delta'] = {5: .098083, 10: .138578, 20: .195560}[job['h']] if job['setting'] == 'T2' else 1.
                history['history_source'] = rel(path)
                hist_rows.append(history)
                loss_row['last_epoch'] = int(history.epoch.iloc[-1])
                chosen = history[history.epoch == best]
                if len(chosen) == 1:
                    chosen = chosen.iloc[0]
                    last = history.iloc[-1]
                    for col, output in [('train_loss', 'train_pct_change'), ('val_loss', 'val_pct_change')]:
                        loss_row[output] = 100 * (last[col] - chosen[col]) / chosen[col] if chosen[col] != 0 else np.nan
        behavior.append(loss_row)
    histories = pd.concat(hist_rows, ignore_index=True)
    behavior = pd.DataFrame(behavior)
    save('all_runs.csv', pd.DataFrame(all_runs))
    save('learning_curves_per_epoch.csv', histories)
    save('learning_changes_per_seed.csv', behavior)
    bests = behavior.groupby(['setting', 'model', 'h', 'best_epoch'], dropna=False, as_index=False).agg(n_seeds=('seed', 'nunique'))
    save('best_epoch_distribution.csv', bests)
    behavior_summary = behavior.groupby(keys, as_index=False).agg(
        n_seeds=('seed', 'nunique'), mean_train_pct_change=('train_pct_change', finite_mean), mean_val_pct_change=('val_pct_change', finite_mean))
    save('learning_changes_summary.csv', behavior_summary)
    save('learning_mean_by_epoch.csv', histories.groupby(keys + ['epoch', 'huber_delta'], as_index=False).agg(n_seeds_observed=('seed', 'nunique'), train_loss_mean=('train_loss', 'mean'), val_loss_mean=('val_loss', 'mean')))
    tests, differences, rng_checks = [], [], []
    for h in HS:
        _, di, si = s.draws(h, 5, 391)
        with np.load(ROOT / f'revision_work/outputs/analysis16/rng/draws_h{h}.npz') as z:
            assert np.array_equal(di, z['joint_date_indices']) and np.array_equal(si, z['seed_indices'])
        rng_checks.append(dict(h=h, B=20000, L=h, base_seed=20261019, seed_indices_equal=True, date_indices_equal=True))
    contrasts = [('static', 'adaptive'), ('adaptive', 'lstm')] + ([('fixed', 'static')] if 'fixed' in models else [])
    for setting in ['T1', 'T2']:
        for left, right in contrasts:
            for metric, col in [('mse', 'mse'), ('rank_ic', 'rank_ic'), ('da', 'n_correct')]:
                for h in HS:
                    A = [panels[(setting, left, h, seed)] for seed in SEEDS]
                    Z = [panels[(setting, right, h, seed)] for seed in SEEDS]
                    dates = A[0].anchor_date.to_numpy()
                    assert all(np.array_equal(dates, d.anchor_date.to_numpy()) for d in A + Z)
                    X = np.ascontiguousarray(np.stack([a[col].to_numpy(float) - z[col].to_numpy(float) for a, z in zip(A, Z)]))
                    status_ok = all(M[(M.setting == setting) & (M.model == model) & (M.h == h)].status.eq('COMPLETE').all() for model in [left, right])
                    defined = bool(status_ok and np.isfinite(X).any(axis=1).all())
                    reason = '' if defined else 'failed/degenerate fit or a seed with no paired-defined dates; no seed dropped'
                    den = np.stack([a.n_valid.to_numpy(float) for a in A])
                    assert np.array_equal(den, np.stack([z.n_valid.to_numpy(float) for z in Z]))
                    if defined:
                        theta, boot, _ = s.distributions(X, h)
                        if metric == 'da':
                            dp, db, _ = s.distributions(den, h)
                            theta, boot = theta / dp, boot / db
                            seed_delta = X.sum(axis=1) / den.sum(axis=1)
                        else:
                            seed_delta = np.nanmean(X, axis=1)
                        defined = bool(np.isfinite(theta) and np.isfinite(boot).all())
                        if not defined:
                            reason = 'non-finite bootstrap distribution; retained with p=1 multiplicity placeholder'
                    if defined:
                        result = s.series_stats(boot, theta)
                        win = seed_delta < 0 if metric == 'mse' else seed_delta > 0
                        lose = seed_delta > 0 if metric == 'mse' else seed_delta < 0
                        wins, losses, ties = int(win.sum()), int(lose.sum()), int((seed_delta == 0).sum())
                        # Independent direct application of 16 of the SAME draws checks memory-layout/ratio bookkeeping.
                        _, di, si = s.draws(h, 5, 391)
                        direct = []
                        for i in range(16):
                            xx = X[si[i]][:, di[i]]
                            direct.append(xx.sum() / den[si[i]][:, di[i]].sum() if metric == 'da' else np.nanmean(xx, axis=1).mean())
                        direct_error = float(np.max(np.abs(np.asarray(direct) - boot[:16])))
                        assert direct_error < 1e-12
                    else:
                        result = dict(theta_hat=np.nan, ci_lower_95=np.nan, ci_upper_95=np.nan, centered_p_two_sided=np.nan)
                        wins = losses = ties = 0
                        direct_error = np.nan
                    meta = dict(setting=setting, input_set='C', h=h, contrast=left + '-minus-' + right, metric=metric,
                                n_seeds=5, n_dates=391, B=20000, L=h, test_defined=defined, undefined_reason=reason,
                                seed_wins=wins, seed_losses=losses, seed_ties=ties, seed_win_arm=left,
                                positive_favors_first=metric != 'mse', n_missing_seed_dates=int((~np.isfinite(X)).sum()), direct_draw_max_error=direct_error)
                    tests.append(dict(**meta, **result, holm_input_p=result['centered_p_two_sided'] if defined else 1.))
                    for i, seed in enumerate(SEEDS):
                        for date, diff, n in zip(dates, X[i], den[i]):
                            differences.append(dict(setting=setting, h=h, contrast=meta['contrast'], metric=metric, seed=seed,
                                                     anchor_date=date, paired_difference=diff, n_valid_da=int(n),
                                                     value_definition='correct-count numerator' if metric == 'da' else 'within-date loss / Rank IC difference'))
    J = s.corrected(pd.DataFrame(tests), ['setting', 'contrast', 'metric'], p='holm_input_p')
    assert len(J) == (54 if 'fixed' in models else 36)
    assert J.groupby(['setting', 'contrast', 'metric']).size().eq(3).all()
    save('joint_bootstrap_tests.csv', J)
    save('paired_seed_date_differences.csv', pd.DataFrame(differences))
    save('bootstrap_draw_verification.csv', pd.DataFrame(rng_checks))
    verdicts = []
    for setting in ['T1', 'T2']:
        for claim, contrast in [('H1', 'static-minus-adaptive'), ('H2', 'adaptive-minus-lstm')]:
            q = J[(J.setting == setting) & (J.contrast == contrast) & J.metric.isin(['mse', 'rank_ic'])].copy()
            assert len(q) == 6
            positive_favors_adaptive = (q.metric == 'mse') if claim == 'H1' else (q.metric == 'rank_ic')
            q['favors_adaptive'] = np.where(positive_favors_adaptive, q.theta_hat > 0, q.theta_hat < 0)
            reject = q.test_defined & q.favors_adaptive & (q.p_holm_3h < .05)
            verdict = 'NOT MET' if reject.any() else 'MET' if q.test_defined.all() else 'INDETERMINATE'
            verdicts.append(dict(setting=setting, claim=claim, criterion=verdict, n_required_tests=6, n_defined=int(q.test_defined.sum()),
                                 n_significant_favoring_adaptive=int(reject.sum()), details='; '.join(f"h{r.h} {r.metric} Holm={r.p_holm_3h:.6g}" for r in q[reject].itertuples())))
    V = pd.DataFrame(verdicts)
    save('H1_H2_verdicts.csv', V)
    # Check frozen source bytes only AFTER the TEST barrier; these hashes do not feed any choice.
    protected, errors = [], []
    for item in read(OUT / 'RNG_MANIFEST.json')['implementation'] + read(OUT / 'RNG_MANIFEST.json')['source_manifests']:
        actual = sha(ROOT / item['path'])
        good = actual == item['sha256']
        protected.append(dict(path=item['path'], sha256=actual, unchanged=good))
        if not good:
            errors.append('Implementation source changed: ' + item['path'])
    visited = set()
    for r in sources.values():
        original = read(ROOT / r['complete'])
        for item in original['artifacts'] + original.get('dependencies', []):
            path = item['path']
            if path in visited:
                continue
            if not (ROOT / path).is_file():
                errors.append('Original source is missing: ' + path)
                protected.append(dict(path=path, sha256=None, unchanged=False))
                continue
            visited.add(path)
            actual = sha(ROOT / path)
            good = actual == item['sha256']
            protected.append(dict(path=path, sha256=actual, unchanged=good))
            if not good:
                errors.append('Original source hash differs: ' + path)
    save('source_protection.csv', pd.DataFrame(protected))
    failures = M[~M.status.eq('COMPLETE')]
    missing_history = behavior[behavior.last_epoch.isna()]
    problems = errors + ([f'{len(failures)} failed/degenerate evaluation cells; retained with N/A outcomes.'] if len(failures) else [])
    if len(missing_history):
        problems.append(f'{len(missing_history)} missing/empty histories; listed in learning_changes_per_seed.csv.')
    timing = read(OUT / 'TIMING_GATE.json')
    gate = read(OUT / 'REPRODUCIBILITY_GATE.json')
    report = '# BLOCKING\n\n' + ('\n'.join('- ' + x for x in problems) if problems else 'None.') + '\n\n'
    report += '# Step22B training-setting robustness\n\nProtocol SHA-256: `' + check_protocol() + '`.\n\n'
    report += 'All settings and cells are reported. No setting was selected; no tests across settings were added. T0 is reused from the Step16 C 12/12 sources. TEST evaluation followed ALL_TRAINING_FINISHED.json.\n\n'
    report += '## Reproducibility gate\n\n' + table(pd.DataFrame([{k: gate.get(k) for k in ['passed', 'threshold', 'matching_epoch_sequence', 'keys_equal', 'labels_equal', 'max_abs_validation_loss_difference', 'max_abs_validation_prediction_difference', 'test_read']}])) + '\n'
    report += '## Timing gate and queue\n\n' + table(pd.DataFrame([{k: timing[k] for k in ['pilot_elapsed_utc_seconds', 'core_projection_hours', 'with_fixed_projection_hours', 'include_fixed', 'proceed']}]))
    training_rows = pd.DataFrame(all_runs)
    report += '\nUTC elapsed seconds include sleep; monotonic times are separately logged. Gate time is excluded from grid projection. Per-run start/end profiles, five-minute load snapshots and epochs are retained in fit records and telemetry.\n\n'
    report += table(training_rows.groupby(['setting', 'model'], as_index=False).agg(n_runs=('seed', 'size'), failed=('status', lambda a: int((a != 'COMPLETE').sum())))) + '\n'
    report += '## H1/H2 verdicts\n\n' + table(V) + '\n'
    report += 'MET means no pre-registered Adaptive-favoring MSE/Rank IC test reached Holm p<.05. This is a statement about the criterion, not evidence of equivalence. DA is outside the verdict criterion.\n\n'
    compact = []
    for (model, h), g in summary.groupby(['model', 'h']):
        row = dict(model=model, h=h)
        for st in ['T0', 'T1', 'T2']:
            z = g[g.setting == st].iloc[0]
            row[st + '_MSE_mean_SD'] = f'{z.mse_mean:.6g} ± {z.mse_sd:.6g}'
            row[st + '_RankIC_mean_SD'] = f'{z.rank_ic_mean:.6g} ± {z.rank_ic_sd:.6g}'
            row[st + '_DA_mean_SD'] = f'{z.da_mean:.6g} ± {z.da_sd:.6g}'
            row[st + '_R2_OS_mean'] = z.r2os_mean
        compact.append(row)
    compact = pd.DataFrame(compact)
    save('T0_T1_T2_compact.csv', compact)
    report += '## Five-seed T0 / T1 / T2 outcomes\n\n' + table(compact) + '\n' + table(summary) + '\n'
    report += '## Joint seed×date tests\n\nFirst-minus-second; seed wins refer to the first arm. Holm across horizons within contrast × metric × setting.\n\n'
    report += table(J[['setting', 'contrast', 'h', 'metric', 'theta_hat', 'ci_lower_95', 'ci_upper_95', 'centered_p_two_sided', 'p_holm_3h', 'seed_wins', 'seed_losses', 'seed_ties', 'test_defined']]) + '\n'
    report += '## Routing movement\n\n' + table(route_summary) + '\n'
    report += '## Training behavior\n\nLoss changes are percentages from the selected epoch to the last observed epoch; each setting uses its own Huber units.\n\n' + table(behavior_summary) + '\n' + table(bests) + '\n'
    report += '## Stored artifacts\n\n'
    report += '\n'.join('- [' + p.name + '](' + str(p) + ')' for p in outputs) + '\n\n'
    report += 'Prediction files include every stock/date and seed, losses, DA validity/correctness and router weights when applicable. Fixed has no training loss selection beyond its formal checkpoint rule. All fit errors and failed cells remain listed.\n'
    text(OUT / 'REPORT_STEP22B.md', report)
    outputs.append(OUT / 'REPORT_STEP22B.md')
    js(OUT / 'ANALYSIS_COMPLETE.json', dict(protocol_sha256=check_protocol(), started_utc=start, ended_utc=utc(),
       cells=len(M), joint_tests=len(J), include_fixed='fixed' in models, problems=problems, analysis_directory=rel(dest),
       artifacts=[artifact(p) for p in outputs]))
    print('Report:', OUT / 'REPORT_STEP22B.md')

if __name__ == '__main__':
    main()
