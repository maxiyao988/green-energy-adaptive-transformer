"""T3 statistics, descriptive four-setting ensembles, and bounds from the same draws."""
import sys
sys.dont_write_bytecode = True
from io_b2 import *
import os
for _key in ENV_KEYS[:5]:
    os.environ[_key] = '1'
sys.path.insert(0, str(OLD))
sys.path.append(str(ROOT / 'revision_work/outputs/_deps_pyarrow15'))
import numpy as np
import pandas as pd
import analyze22b_v3 as prior
from scripts.python import panel_common as common
s = prior.s
finite_mean, finite_sd, table = prior.finite_mean, prior.finite_sd, prior.table

def main():
    barrier()
    assert all(r['unchanged'] for r in check_old_sources())
    if (OUT / 'ANALYSIS_COMPLETE.json').exists():
        verify_record(OUT / 'ANALYSIS_COMPLETE.json')
        return
    started = utc()
    dest = target(OUT / 'analysis' / ('attempt_' + stamp()))
    dest.mkdir(parents=True)
    outputs, protected = [], {}
    def protect(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None:
            assert actual == expected, 'Source hash mismatch: ' + str(path)
        protected[rel(path)] = actual
        return path
    def read_csv(path):
        protect(path)
        return pd.read_csv(path, float_precision='round_trip')
    def save(name, data):
        path = dest / name
        csv(path, data)
        outputs.append(path)
        return path
    def one(df, **where):
        q = df
        for k, v in where.items():
            q = q[q[k] == v]
        assert len(q) == 1, (where, len(q))
        return q.iloc[0]
    previous = read(protect(OLD / 'ANALYSIS_COMPLETE.json'))
    olddir = ROOT / previous['analysis_directory']
    # Verify old artifacts only here, after the all-T3-training barrier.
    for item in previous['artifacts']:
        protect(ROOT / item['path'], item['sha256'])
    oldM = read_csv(olddir / 'test_metrics_per_seed.csv')
    oldM = oldM[oldM.model.isin(['static', 'adaptive', 'lstm']) & oldM.setting.isin(['T0', 'T1', 'T2'])]
    assert len(oldM) == 135
    metrics, routers, panels, files, new_frames = [], [], {}, [], {}
    for job in read(OUT / 'GRID_PLAN.json')['jobs']:
        paths = sorted((OUT / 'evaluation' / job_id(job)).glob('attempt_*/COMPLETE.json'))
        assert len(paths) == 1
        record = verify_record(paths[0])
        folder = paths[0].parent
        mt = pd.read_csv(folder / 'metrics.csv').iloc[0].to_dict()
        metrics.append(mt)
        # Match Step22B's float parsing and contiguous seed/date layout exactly for new tests.
        date = pd.read_csv(folder / 'date_metrics.csv').sort_values('anchor_date').reset_index(drop=True)
        panels[(job['model'], job['h'], job['seed'])] = date
        route = pd.read_csv(folder / 'router_diagnostics.csv')
        for key in ['setting', 'model', 'h', 'seed']:
            route[key] = job[key]
        routers.append(route)
        frames = read_csv(folder / 'predictions_losses_weights.csv')
        new_frames[(job['model'], job['h'], job['seed'])] = frames
        files.append(dict(**job, status=record['status'], reused=False, predictions=rel(folder / 'predictions_losses_weights.csv'),
                          complete=rel(paths[0]), date_metrics=rel(folder / 'date_metrics.csv'), stock_metrics=rel(folder / 'stock_metrics.csv')))
    newM = pd.DataFrame(metrics)
    assert len(newM) == 45
    save('T3_test_metrics_per_seed.csv', newM)
    M = pd.concat([oldM, newM], ignore_index=True)
    assert len(M) == 180 and not M.duplicated(['setting', 'model', 'h', 'seed']).any()
    save('T0_T1_T2_T3_metrics_per_seed.csv', M)
    summary = M.groupby(['setting', 'model', 'h'], as_index=False).agg(n_seeds=('seed', 'nunique'),
       n_complete=('status', lambda a: int(a.eq('COMPLETE').sum())), mse_mean=('mse', finite_mean), mse_sd=('mse', finite_sd),
       rank_ic_mean=('mean_rank_ic', finite_mean), rank_ic_sd=('mean_rank_ic', finite_sd),
       da_mean=('da', finite_mean), da_sd=('da', finite_sd), r2os_mean=('r2os', finite_mean), r2os_sd=('r2os', finite_sd))
    assert len(summary) == 36 and summary.n_seeds.eq(5).all()
    save('five_seed_summary.csv', summary)
    compact = []
    for (model, h), g in summary.groupby(['model', 'h']):
        row = dict(model=model, h=h)
        for st in ['T0', 'T1', 'T2', 'T3']:
            q = one(g, setting=st)
            row[st + '_MSE'] = f'{q.mse_mean:.6g} ± {q.mse_sd:.6g}'
            row[st + '_RankIC'] = f'{q.rank_ic_mean:.6g} ± {q.rank_ic_sd:.6g}'
            row[st + '_DA'] = f'{q.da_mean:.6g} ± {q.da_sd:.6g}'
            row[st + '_R2_OS'] = q.r2os_mean
        compact.append(row)
    compact = pd.DataFrame(compact)
    save('T0_T1_T2_T3_side_by_side.csv', compact)
    # Equal-weight seed ensembles use stored predictions only; no old-model inference.
    old_protocol = read(protect(OLD / 'RNG_MANIFEST.json'))['protocol_sha256']
    ensemble_rows = []
    label_keys = {}
    for setting in ['T0', 'T1', 'T2', 'T3']:
        for model in ['static', 'adaptive', 'lstm']:
            for h in HS:
                frames, statuses = [], []
                for seed in SEEDS:
                    job = spec(setting, model, h, seed)
                    if setting == 'T3':
                        frame = new_frames[(model, h, seed)].copy()
                        status = one(newM, model=model, h=h, seed=seed)['status']
                    else:
                        found = sorted((OLD / 'evaluation' / job_id(job)).glob('attempt_*/COMPLETE.json'))
                        assert len(found) == 1
                        rec = read(protect(found[0]))
                        assert rec['protocol_sha256'] == old_protocol and rec['job'] == job
                        for a in rec['artifacts']:
                            protect(ROOT / a['path'], a['sha256'])
                        frame = read_csv(found[0].parent / 'predictions_losses_weights.csv')
                        status = rec['status']
                        files.append(dict(**job, status=status, reused=True, predictions=rel(found[0].parent / 'predictions_losses_weights.csv'),
                                          complete=rel(found[0]), date_metrics=rel(found[0].parent / 'date_metrics.csv'), stock_metrics=rel(found[0].parent / 'stock_metrics.csv')))
                    frame.anchor_date = pd.to_datetime(frame.anchor_date).dt.strftime('%Y-%m-%d')
                    frame = frame.sort_values(['anchor_date', 'ticker']).reset_index(drop=True)
                    assert len(frame) == 6647 and not frame.duplicated(['ticker', 'anchor_date']).any()
                    key = frame[['ticker', 'anchor_date', 'y_true']].copy()
                    if h not in label_keys:
                        label_keys[h] = key
                    reference = label_keys[h]
                    assert key[['ticker', 'anchor_date']].equals(reference[['ticker', 'anchor_date']])
                    assert np.array_equal(key.y_true.to_numpy(np.float32), reference.y_true.to_numpy(np.float32))
                    frames.append(frame)
                    statuses.append(status)
                base = label_keys[h].copy()
                y = base.y_true.to_numpy(float)
                predictions = np.stack([f.y_pred.to_numpy(float) for f in frames])
                ok = all(x == 'COMPLETE' for x in statuses) and np.isfinite(predictions).all()
                p = predictions.mean(axis=0) if ok else np.full(len(y), np.nan)
                base['split'], base['y_pred'] = 'test', p
                base['squared_loss'] = (p - y) ** 2
                base['da_valid'] = y != 0
                base['da_correct'] = np.where(y != 0, (np.sign(p) == np.sign(y)).astype(float), np.nan) if ok else np.nan
                m = float(one(M, setting=setting, model=model, h=h, seed=0).global_train_mean)
                ref_mse = float(np.mean((y - m) ** 2))
                ic = common.rank_ic_by_date(y, p, base.ticker.to_numpy(), base.anchor_date.to_numpy()) if ok else None
                mse = float(np.mean((p - y) ** 2)) if ok else np.nan
                row = dict(setting=setting, model=model, h=h, n_seeds=5, n_rows=6647,
                           status='COMPLETE' if ok else 'UNDEFINED', mse=mse,
                           mean_rank_ic=float(ic.rank_ic.mean()) if ok and ic.rank_ic.notna().any() else np.nan,
                           da=float(np.sum(np.sign(p[y != 0]) == np.sign(y[y != 0])) / np.sum(y != 0)) if ok else np.nan,
                           r2os=1 - mse / ref_mse if ok else np.nan, global_train_mean=m, global_train_mean_mse=ref_mse,
                           no_ensemble_test=True, seed_aggregation='equal-weight mean of all five predictions')
                ensemble_rows.append(row)
                for k in ['setting', 'model', 'h']:
                    base[k] = row[k]
                save(f'ensemble_predictions/{setting}_{model}_h{h}.csv', base)
    ensembles = pd.DataFrame(ensemble_rows)
    assert len(ensembles) == 36
    save('ensemble_metrics.csv', ensembles)
    save('prediction_files.csv', pd.DataFrame(files))
    R = pd.concat([read_csv(olddir / 'router_diagnostics_per_seed.csv')] + [r for r in routers if not r.empty], ignore_index=True)
    save('router_diagnostics_per_seed.csv', R)
    router_summary = R.groupby(['setting', 'model', 'h', 'branch'], as_index=False).agg(
       n_seeds=('seed', 'nunique'), mean_abs_deviation=('mean_abs_deviation_from_equal', finite_mean),
       mean_sample_sd=('mean_sample_sd', finite_mean), between_date_share=('between_date_share', finite_mean),
       n_defined_between_date_shares=('between_date_share', 'count'), n_constant_seeds=('constant_to_1e7', 'sum'))
    save('router_diagnostics_summary.csv', router_summary)
    histories, behavior, runs = [], [], []
    for job in read(OUT / 'GRID_PLAN.json')['jobs']:
        cp, rec = completion(job)
        if rec['status'] == 'COMPLETE':
            assert rec['lr'] == 3e-5 and rec['delta'] == {5:.098083,10:.138578,20:.195560}[job['h']]
            verify_record(cp.parent / 'ADAPTER_COMPLETE.json')
        history_path = cp.parent / 'history.csv'
        row = dict(**{k: job[k] for k in ['setting', 'model', 'h', 'seed']}, status=rec['status'],
                   best_epoch=rec.get('best_epoch'), last_epoch=np.nan, train_pct_change=np.nan, val_pct_change=np.nan,
                   history_source=rel(history_path) if history_path.exists() else None)
        run = {**job, **{k: rec.get(k) for k in ['status', 'error', 'epochs', 'best_epoch', 'lr', 'delta', 'started_utc', 'ended_utc', 'elapsed_utc_seconds', 'monotonic_seconds']},
               'record': rel(cp)}
        runs.append(run)
        if history_path.exists():
            hd = pd.read_csv(history_path)
            if len(hd):
                for k in ['setting', 'model', 'h', 'seed']:
                    hd[k] = job[k]
                hd['huber_delta'] = {5: .098083, 10: .138578, 20: .195560}[job['h']]
                hd['history_source'] = rel(history_path)
                histories.append(hd)
                row['last_epoch'] = int(hd.epoch.iloc[-1])
                best = hd[hd.epoch == rec['best_epoch']]
                if len(best) == 1:
                    for col, out in [('train_loss', 'train_pct_change'), ('val_loss', 'val_pct_change')]:
                        b, last = float(best[col].iloc[0]), float(hd[col].iloc[-1])
                        row[out] = 100 * (last - b) / b if b != 0 else np.nan
        behavior.append(row)
    save('all_T3_runs.csv', pd.DataFrame(runs))
    newH = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    if not newH.empty:
        save('T3_learning_curves_per_epoch.csv', newH)
        save('T3_learning_mean_by_epoch.csv', newH.groupby(['model', 'h', 'epoch', 'huber_delta'], as_index=False).agg(n_seeds_observed=('seed', 'nunique'), train_loss_mean=('train_loss', 'mean'), val_loss_mean=('val_loss', 'mean')))
    B = pd.concat([read_csv(olddir / 'learning_changes_per_seed.csv'), pd.DataFrame(behavior)], ignore_index=True)
    save('learning_changes_per_seed.csv', B)
    bests = B.groupby(['setting', 'model', 'h', 'best_epoch'], dropna=False, as_index=False).agg(n_seeds=('seed', 'nunique'))
    save('best_epoch_distribution.csv', bests)
    behavior_summary = B.groupby(['setting', 'model', 'h'], as_index=False).agg(n_seeds=('seed', 'nunique'), mean_train_pct_change=('train_pct_change', finite_mean), mean_val_pct_change=('val_pct_change', finite_mean))
    save('learning_changes_summary.csv', behavior_summary)
    tests, differences, bounds, drawchecks, distributions = [], [], [], [], []
    for h in HS:
        _, di, si = s.draws(h, 5, 391)
        with np.load(protect(ROOT / f'revision_work/outputs/analysis16/rng/draws_h{h}.npz')) as z:
            assert np.array_equal(di, z['joint_date_indices']) and np.array_equal(si, z['seed_indices'])
        drawchecks.append(dict(h=h, B=20000, L=h, seed_base=20261019, seed_indices_equal=True, date_indices_equal=True))
    for left, right in [('static', 'adaptive'), ('adaptive', 'lstm')]:
        contrast = left + '-minus-' + right
        for metric, col in [('mse', 'mse'), ('rank_ic', 'rank_ic'), ('da', 'n_correct')]:
            for h in HS:
                A = [panels[(left, h, seed)] for seed in SEEDS]
                Z = [panels[(right, h, seed)] for seed in SEEDS]
                dates = A[0].anchor_date.to_numpy()
                assert all(np.array_equal(dates, d.anchor_date.to_numpy()) for d in A + Z)
                X = np.ascontiguousarray(np.stack([a[col].to_numpy(float) - z[col].to_numpy(float) for a, z in zip(A, Z)]))
                den = np.stack([a.n_valid.to_numpy(float) for a in A])
                assert np.array_equal(den, np.stack([z.n_valid.to_numpy(float) for z in Z]))
                ok = all(one(newM, model=m, h=h, seed=seed)['status'] == 'COMPLETE' for m in [left, right] for seed in SEEDS)
                defined = bool(ok and np.isfinite(X).any(axis=1).all())
                reason = '' if defined else 'failed/degenerate fit or at least one seed with no paired-defined dates; retained'
                if defined:
                    theta, boot, _ = s.distributions(X, h)
                    if metric == 'da':
                        dpoint, dboot, _ = s.distributions(den, h)
                        theta, boot = theta / dpoint, boot / dboot
                        delta_seed = X.sum(axis=1) / den.sum(axis=1)
                    else:
                        delta_seed = np.nanmean(X, axis=1)
                    defined = bool(np.isfinite(theta) and np.isfinite(boot).all())
                    if not defined:
                        reason = 'non-finite bootstrap distribution'
                if defined:
                    stat = s.series_stats(boot, theta)
                    wins = int((delta_seed < 0 if metric == 'mse' else delta_seed > 0).sum())
                    losses = int((delta_seed > 0 if metric == 'mse' else delta_seed < 0).sum())
                    ties = int((delta_seed == 0).sum())
                    _, di, si = s.draws(h, 5, 391)
                    direct = []
                    for i in range(16):
                        x = X[si[i]][:, di[i]]
                        direct.append(x.sum() / den[si[i]][:, di[i]].sum() if metric == 'da' else np.nanmean(x, axis=1).mean())
                    direct_error = float(np.max(np.abs(np.asarray(direct) - boot[:16])))
                    assert direct_error < 1e-12
                else:
                    boot = np.full(20000, np.nan)
                    stat = dict(theta_hat=np.nan, ci_lower_95=np.nan, ci_upper_95=np.nan, centered_p_two_sided=np.nan)
                    wins = losses = ties = direct_error = np.nan
                ident = f'T3_h{h}_{contrast}_{metric}'
                with target(dest / 'bootstrap' / (ident + '.npy')).open('xb') as handle:
                    np.save(handle, boot, allow_pickle=False)
                outputs.append(dest / 'bootstrap' / (ident + '.npy'))
                with target(dest / 'paired_inputs' / (ident + '.npz')).open('xb') as handle:
                    np.savez_compressed(handle, paired_difference=X, n_valid_da=den, seed_order=np.array(SEEDS), anchor_dates=np.asarray(dates, str))
                outputs.append(dest / 'paired_inputs' / (ident + '.npz'))
                distributions.append(dict(contrast=contrast, metric=metric, h=h, test_defined=defined, file=rel(dest / 'bootstrap' / (ident + '.npy'))))
                tests.append(dict(setting='T3', input_set='C', contrast=contrast, h=h, metric=metric, B=20000, L=h,
                    n_seeds=5, n_dates=391, test_defined=defined, undefined_reason=reason, seed_wins=wins, seed_losses=losses,
                    seed_ties=ties, seed_win_arm=left, positive_favors_first=metric != 'mse', direct_draw_max_error=direct_error,
                    **stat, holm_input_p=stat['centered_p_two_sided'] if defined else 1.))
                for i, seed in enumerate(SEEDS):
                    for date, x, n in zip(dates, X[i], den[i]):
                        differences.append(dict(setting='T3', contrast=contrast, metric=metric, h=h, seed=seed, anchor_date=date,
                             paired_difference=x, n_valid_da=int(n), value_definition='correct-count numerator' if metric == 'da' else 'date-level MSE/IC difference'))
                if metric in ['mse', 'rank_ic']:
                    direction = (1 if left == 'static' else -1) if metric == 'mse' else (-1 if left == 'static' else 1)
                    second = one(summary, setting='T3', model=right, h=h)
                    scale = float(second.mse_mean)
                    upper = float(np.quantile(direction * boot, .95, method='linear')) if defined else np.nan
                    bounds.append(dict(setting='T3', contrast=contrast, first_arm=left, second_arm=right, h=h, metric=metric,
                        estimate_raw=stat['theta_hat'], ci95_lower_raw=stat['ci_lower_95'], ci95_upper_raw=stat['ci_upper_95'],
                        second_arm_mean_test_mse=scale, estimate_pct_second_mse=100 * stat['theta_hat'] / scale if metric == 'mse' else np.nan,
                        ci95_lower_pct_second_mse=100 * stat['ci_lower_95'] / scale if metric == 'mse' else np.nan,
                        ci95_upper_pct_second_mse=100 * stat['ci_upper_95'] / scale if metric == 'mse' else np.nan,
                        adaptive_improvement_direction=direction, adaptive_improvement_upper95_raw=upper,
                        adaptive_mse_improvement_upper95_pct_second_mse=100 * upper / scale if metric == 'mse' else np.nan,
                        test_defined=defined, coverage='pointwise; no simultaneous guarantee', replicate_file=rel(dest / 'bootstrap' / (ident + '.npy'))))
    J = s.corrected(pd.DataFrame(tests), ['contrast', 'metric'], p='holm_input_p')
    assert len(J) == 18 and J.groupby(['contrast', 'metric']).size().eq(3).all()
    save('T3_joint_bootstrap_tests.csv', J)
    save('T3_paired_seed_date_differences.csv', pd.DataFrame(differences))
    save('T3_seed_wins.csv', J[['contrast', 'metric', 'h', 'seed_wins', 'seed_losses', 'seed_ties', 'test_defined']])
    save('bootstrap_draw_verification.csv', pd.DataFrame(drawchecks))
    save('bootstrap_distribution_files.csv', pd.DataFrame(distributions))
    E = pd.DataFrame(bounds)
    assert len(E) == 12
    save('T3_effect_size_bounds.csv', E)
    verdicts = []
    for claim, contrast in [('H1', 'static-minus-adaptive'), ('H2', 'adaptive-minus-lstm')]:
        g = J[(J.contrast == contrast) & J.metric.isin(['mse', 'rank_ic'])]
        assert len(g) == 6
        positive = (g.metric == 'mse') if claim == 'H1' else (g.metric == 'rank_ic')
        favors = np.where(positive, g.theta_hat > 0, g.theta_hat < 0)
        significant = g.test_defined & favors & (g.p_holm_3h < .05)
        verdicts.append(dict(setting='T3', claim=claim, criterion='NOT MET' if significant.any() else 'MET' if g.test_defined.all() else 'INDETERMINATE',
           n_required_tests=6, n_defined=int(g.test_defined.sum()), n_significant_favoring_adaptive=int(significant.sum())))
    V = pd.DataFrame(verdicts)
    save('H1_H2_verdicts.csv', V)
    sourcechecks = []
    for p, before in protected.items():
        after = sha(ROOT / p)
        sourcechecks.append(dict(path=p, expected_sha256=before, actual_sha256=after, unchanged=before == after))
    sourcechecks += check_old_sources()
    save('source_checks.csv', pd.DataFrame(sourcechecks))
    errors = ['Source hash differs: ' + r['path'] for r in sourcechecks if not r['unchanged']]
    failures = newM[~newM.status.eq('COMPLETE')]
    if len(failures):
        errors.append(f'{len(failures)} failed/degenerate T3 cells, retained with N/A outcomes.')
    if not J.test_defined.all():
        errors.append(f'{int((~J.test_defined).sum())} undefined joint tests; conservative Holm placeholders retained.')
    if not all(row['status'] == 'COMPLETE' for row in ensemble_rows):
        errors.append('At least one five-seed ensemble is undefined; no seed was omitted.')
    gate = read(OUT / 'SOURCE_HASH_GATE.json')
    timing = read(OUT / 'TIMING_GATE.json')
    report = '# BLOCKING\n\n' + ('\n'.join('- ' + x for x in errors) if errors else 'None.') + '\n\n'
    report += '# Step22B2 — combined setting T3\n\nProtocol SHA-256: `' + check_protocol() + '`.\n\n'
    report += 'T3 combines learning rate3e-5 and the fixed horizon-specific Huber delta. T0/T1/T2 are reused from Step22B; no configuration or seed was selected. All45 T3 fits are reported. TEST was opened after ALL_TRAINING_FINISHED.json.\n\n'
    report += '## Gates and timing\n\n' + table(pd.DataFrame([dict(source_hash_gate_passed=gate['passed'], registered_source_checks=len(gate['checks']),
         pilot_elapsed_seconds=timing['pilot_elapsed_utc_seconds'], projected_hours=timing['projected_hours'], timing_gate_passed=timing['proceed'])])) + '\n\n'
    trainstart = min(r['started_utc'] for r in runs)
    trainend = max(r['ended_utc'] for r in runs)
    report += f'T3 training start UTC: {trainstart}; end UTC: {trainend}; elapsed UTC hours: {elapsed(trainstart, trainend)/3600:.6g}. Monotonic times and 12/12 thread/load telemetry are retained per fit.\n\n'
    report += '## (a) Five-seed outcomes, T0–T3\n\n' + table(compact) + '\n\n' + table(summary) + '\n\n'
    report += '## (b) Equal-weight five-seed ensembles (descriptive)\n\n' + table(ensembles[['setting', 'model', 'h', 'status', 'mse', 'mean_rank_ic', 'da', 'r2os']]) + '\n\n'
    report += '## (c) T3 joint seed×date tests\n\nB=20000, L=h; first-minus-second; seed wins refer to the first arm. Holm across three horizons within contrast × metric only.\n\n'
    report += table(J[['contrast', 'h', 'metric', 'theta_hat', 'ci_lower_95', 'ci_upper_95', 'centered_p_two_sided', 'p_holm_3h', 'seed_wins', 'seed_losses', 'seed_ties', 'test_defined']]) + '\n\n'
    report += '## (d) H1/H2 T3 verdicts\n\n' + table(V) + '\n\nMET means no specified Adaptive-favoring MSE/Rank IC contrast reached Holm p<.05; it does not imply equivalence. DA is outside these verdicts.\n\n'
    report += '## (e) Training behavior and routing\n\nStep19 loss change is100×(last−selected)/selected; T3 uses its sigma_h Huber units.\n\n' + table(behavior_summary) + '\n\n' + table(bests[bests.setting == 'T3']) + '\n\n' + table(router_summary[router_summary.setting == 'T3']) + '\n\n'
    report += '## (f) Effect-size bounds (no additional tests)\n\nMSE percentages use the fixed **second-arm five-seed mean TEST MSE**. Signed Adaptive improvement upper bound is its95th percentile, with negative values retained. These are pointwise bounds with no simultaneous coverage. The same replicate arrays as(c) are used; no extra tests are computed.\n\n'
    report += table(E[['contrast', 'h', 'metric', 'estimate_raw', 'ci95_lower_raw', 'ci95_upper_raw', 'second_arm_mean_test_mse', 'ci95_lower_pct_second_mse', 'ci95_upper_pct_second_mse', 'adaptive_improvement_upper95_raw', 'adaptive_mse_improvement_upper95_pct_second_mse']]) + '\n\n'
    report += '## Every T3 run\n\n' + table(pd.DataFrame(runs)[['model', 'h', 'seed', 'status', 'epochs', 'best_epoch', 'lr', 'delta', 'elapsed_utc_seconds', 'error']]) + '\n\n'
    report += '## Files\n\n' + '\n'.join('- [' + p.name + '](' + str(p) + ')' for p in outputs if p.suffix == '.csv' and p.parent == dest) + '\n\n'
    report += 'Per-seed stock/date predictions, losses, DA validity/correctness and routing weights are listed in prediction_files.csv; checkpoints, TRAIN/VAL predictions, histories, Static logits and telemetry remain under fits/. Ensemble predictions, all20000-replicate arrays and paired seed/date matrices are retained under this analysis directory. Source hashes are checked after the TEST barrier.\n'
    text(OUT / 'REPORT_STEP22B2.md', report)
    outputs.append(OUT / 'REPORT_STEP22B2.md')
    js(OUT / 'ANALYSIS_COMPLETE.json', dict(protocol_sha256=check_protocol(), started_utc=started, ended_utc=utc(),
       analysis_directory=rel(dest), t3_seed_cells=45, reused_seed_cells=135, ensemble_cells=36, joint_tests=18,
       effect_bound_rows=12, problems=errors, artifacts=[artifact(p) for p in outputs]))
    print('Report:', OUT / 'REPORT_STEP22B2.md', flush=True)

if __name__ == '__main__':
    main()
