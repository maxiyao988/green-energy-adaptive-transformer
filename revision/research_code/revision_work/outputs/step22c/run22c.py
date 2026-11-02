"""Descriptive joint-bootstrap bounds only. No training or hypothesis-test calls."""
import argparse
import csv as csvlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SEEDS = [0, 1, 21, 42, 3407]
HS = [5, 10, 20]
BASE = ROOT / 'revision_work/outputs/analysis16'
A22 = ROOT / 'revision_work/outputs/step22a'
B22 = ROOT / 'revision_work/outputs/step22b'
TOL = 1e-12

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]

def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def get(p):
    return json.loads(Path(p).read_text())

def target(p):
    p = Path(p).absolute()
    assert p.is_relative_to(OUT) and p.resolve() == p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def write(p, value):
    with target(p).open('x') as f:
        f.write(value)

def js(p, value):
    write(p, json.dumps(value, indent=2, allow_nan=False, default=lambda a: a.item() if hasattr(a, 'item') else str(a)) + '\n')

def artifact(p):
    return dict(path=rel(p), sha256=sha(p), bytes=Path(p).stat().st_size)

def verify_delivery():
    manifest = get(OUT / 'DELIVERY_MANIFEST.json')
    for item in manifest['registered_files']:
        assert sha(ROOT / item['path']) == item['sha256'], 'Registered input/code changed: ' + item['path']
    return manifest

def status():
    p = OUT / 'COMPLETE_STEP22C.json'
    if p.exists():
        d = get(p)
        print('Step22C COMPLETE:', d['ended_utc'], 'metric rows:', d['metric_rows'])
        print('Report:', OUT / 'REPORT_STEP22C.md')
        return
    for p in sorted(OUT.glob('STOP_REPORT_*.md'))[-1:]:
        print('Latest stop report:', p)
    runs = sorted((OUT / 'runs').glob('run_*'))
    if runs:
        p = runs[-1]
        print('Latest run:', p, 'saved joint distributions:', len(list((p / 'bootstrap').glob('*.npy'))), '/84')
    else:
        print('Step22C NOT RUN')

def main():
    manifest = verify_delivery()
    final = OUT / 'COMPLETE_STEP22C.json'
    if final.exists():
        for item in get(final)['artifacts']:
            assert sha(ROOT / item['path']) == item['sha256'], item['path']
        print('Already complete; artifacts verified:', OUT / 'REPORT_STEP22C.md')
        return
    if (OUT / 'REPORT_STEP22C.md').exists():
        raise RuntimeError('Report exists without completion marker; refusing any overwrite')
    dest = target(OUT / 'runs' / ('run_' + stamp()))
    dest.mkdir(parents=True)
    started = utc()
    try:
        calculate(dest, started, manifest)
    except BaseException:
        failure = traceback.format_exc()
        write(dest / 'ERROR.txt', failure)
        p = OUT / ('STOP_REPORT_' + stamp() + '.md')
        write(p, '# BLOCKING\n\nStep22C did not publish effect-size tables. Existing files were not changed.\n\n```text\n' + failure + '\n```\n')
        print('STOP:', p, flush=True)
        raise

def calculate(dest, started, manifest):
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'revision_work/scripts/analysis16'))
    sys.path.append(str(ROOT / 'revision_work/outputs/_deps_pyarrow15'))
    import numpy as np
    import pandas as pd
    from statistics16_v3 import s
    # Import source definitions only; never call their main/test/adjustment functions.
    assert s.B == 20000 and s.BASE == 20261019
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute().resolve()
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and not path.is_relative_to(OUT):
            raise RuntimeError('Read-only source protection: ' + str(path))
    sys.addaudithook(guard)
    used = {}
    def protect(path, expected=None):
        path = Path(path)
        h = sha(path)
        if expected is not None:
            assert h == expected, 'Stored source hash mismatch: ' + str(path)
        used[rel(path)] = h
        return path
    def read(path, columns=None):
        protect(path)
        return pd.read_csv(path, usecols=columns, float_precision='round_trip')
    def save(name, data):
        with target(dest / name).open('x', newline='') as f:
            data.to_csv(f, index=False, float_format='%.17g', na_rep='N/A')
    def one(data, **where):
        q = data
        for k, v in where.items():
            q = q[q[k] == v]
        assert len(q) == 1, (where, len(q))
        return q.iloc[0]
    bdir = ROOT / get(protect(B22 / 'ANALYSIS_COMPLETE.json'))['analysis_directory']
    rng16 = get(protect(BASE / 'RNG_MANIFEST.json'))
    rnga = get(protect(A22 / 'RNG_MANIFEST.json'))
    rngb = get(protect(B22 / 'RNG_MANIFEST.json'))
    assert rng16['B'] == rnga['bootstrap_B'] == rngb['B'] == 20000
    assert rng16['model_seed_order'] == rnga['bootstrap_seed_order'] == SEEDS
    assert rngb['training_seeds'] == SEEDS
    assert rnga['bootstrap_seed_base'] == rngb['bootstrap_base_seed'] == 20261019
    assert rng16['joint_seed_date']['entropy'] == '[20261019,h,h,1]'
    assert rnga['bootstrap_entropy'] == '[20261019,h,h,1]'
    assert rngb['bootstrap_seed_sequence'] == ['20261019', 'h', 'h', 1]
    intervals = ['theta_hat', 'ci_lower_95', 'ci_upper_95', 'B', 'L', 'n_seeds', 'n_dates']
    a = read(BASE / 'A_paired_tests.csv', ['section', 'input_set', 'window', 'contrast', 'metric', 'h', 'method', 'first_mean', 'second_mean'] + intervals)
    a = a[(a.contrast == 'static−adaptive') & (a.method == 'joint_seed_date') & (a.window == 'W3')]
    assert len(a) == 12
    fa = read(A22 / 'joint_bootstrap_tests.csv', ['input_set', 'backbone', 'h', 'contrast', 'metric', 'test_defined'] + intervals)
    fa = fa[fa.metric.isin(['mse', 'rank_ic']) & fa.contrast.isin(['static-minus-adaptive', 'adaptive-minus-constant_gate'])]
    assert len(fa) == 48 and fa.test_defined.all()
    fb = read(bdir / 'joint_bootstrap_tests.csv', ['setting', 'input_set', 'h', 'contrast', 'metric', 'test_defined'] + intervals)
    fb = fb[fb.metric.isin(['mse', 'rank_ic']) & fb.contrast.isin(['static-minus-adaptive', 'adaptive-minus-lstm'])]
    assert len(fb) == 24 and fb.test_defined.all()
    da = read(A22 / 'paired_seed_date_differences.csv', ['input_set', 'backbone', 'h', 'contrast', 'metric', 'seed', 'anchor_date', 'paired_difference'])
    db = read(bdir / 'paired_seed_date_differences.csv', ['setting', 'h', 'contrast', 'metric', 'seed', 'anchor_date', 'paired_difference'])
    ma = read(A22 / 'five_seed_summary.csv', ['input_set', 'backbone', 'h', 'variant', 'n_seeds', 'n_complete', 'mse_mean'])
    mb = read(bdir / 'five_seed_summary.csv', ['setting', 'model', 'h', 'n_seeds', 'n_complete', 'mse_mean'])
    registry = read(BASE / 'registry.csv', ['group', 'input_set', 'model', 'config', 'h', 'seed', 'window', 'id', 'canonical'])
    checks = read(BASE / 'prediction_checks.csv', ['id', 'canonical_sha256'])
    inventory = []
    for group, folder in [('1_end_to_end', BASE), ('2_frozen_tuned', A22), ('3_training_settings', B22)]:
        numeric = sorted(rel(p) for p in folder.rglob('*') if p.is_file() and p.suffix in ['.npy', '.npz'])
        # The registered originals have RNG indices only, not target contrast outcomes.
        assert numeric == ([rel(BASE / f'rng/draws_h{h}.npz') for h in [10, 20, 5]] if group == '1_end_to_end' else []), ('Source numeric inventory changed; inspect saved distributions before reconstructing', numeric)
        inventory.append(dict(group=group, source=rel(folder), target_replicates_saved=False,
                              numeric_files=json.dumps(numeric), action='reconstruct from original input and registered draws',
                              excluded_archives='legacy Step5/12A and untuned Step15 distributions; different target analysis'))
    save('source_inventory.csv', pd.DataFrame(inventory))
    draws = {}
    for h in HS:
        weights, dates, seeds = s.draws(h, 5, 391)
        path = protect(BASE / f'rng/draws_h{h}.npz')
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(dates, z['joint_date_indices'])
            assert np.array_equal(seeds, z['seed_indices'])
        assert weights.shape == (20000, 5) and dates.shape == (20000, 391) and seeds.shape == (20000, 5)
        draws[h] = (weights, dates, seeds)
    def paired_matrix(df, **where):
        q = df
        for k, v in where.items():
            q = q[q[k] == v]
        assert not q.duplicated(['seed', 'anchor_date']).any()
        wide = q.pivot(index='seed', columns='anchor_date', values='paired_difference').reindex(SEEDS)
        assert wide.shape == (5, 391) and sorted(q.seed.unique().tolist()) == SEEDS
        X = np.ascontiguousarray(wide.to_numpy(float))
        assert np.isfinite(X).any(axis=1).all()
        return X, wide.columns.to_numpy(str)
    def end_to_end(inp, h):
        labels = pd.read_parquet(protect(BASE / f'labels/W3_h{h}.parquet'))
        labels = s.canon(labels[labels.split == 'test'])
        y = labels.y_true.to_numpy(float)
        assert len(labels) == 6647 and labels.groupby('anchor_date').size().eq(17).all()
        arms = {}
        for model in ['static', 'adaptive']:
            r = registry[(registry['group'] == 'core') & (registry.input_set == inp) & (registry.model == model)
                         & (registry.config == 'daily_weekly') & (registry.h == h) & (registry.window == 'W3')]
            assert sorted(r.seed.tolist()) == SEEDS
            arrays = []
            for seed in SEEDS:
                z = one(r, seed=seed)
                path = protect(ROOT / z.canonical, one(checks, id=z.id).canonical_sha256)
                p = pd.read_parquet(path)
                p = s.canon(p)
                assert p[['ticker', 'anchor_date']].equals(labels[['ticker', 'anchor_date']])
                assert np.array_equal(p.y_true.to_numpy(np.float32), y.astype(np.float32))
                arrays.append(p.y_pred.to_numpy(float))
            arms[model] = np.stack(arrays)
        loss, ic = [], []
        for i in range(5):
            weights = np.eye(5)[i]
            l, c, _, _ = s.ens_series(y, (-1, 17), arms['static'], arms['adaptive'], y.reshape(-1, 17), weights)
            loss.append(l)
            ic.append(c)
        dates = labels.anchor_date.drop_duplicates().to_numpy(str)
        return dict(mse=np.ascontiguousarray(np.stack(loss)), rank_ic=np.ascontiguousarray(np.stack(ic))), dates
    rows, replays = [], []
    def process(group, inp, backbone, setting, h, contrast, metric, X, dates, reference, den, den_source):
        assert int(reference.B) == 20000 and int(reference.L) == h and int(reference.n_seeds) == 5 and int(reference.n_dates) == 391
        assert len(dates) == 391 and np.isfinite(den) and den > 0
        X = np.ascontiguousarray(X, dtype=float)
        wt, didx, sidx = draws[h]
        env = dict(np=np, X=X, wt=wt, didx=didx, n_missing=int((~np.isfinite(X)).sum()), missing=[],
                   inp=inp, bk=backbone, setting=setting, h=h, contrast=contrast, metric=metric)
        # Existing Step15 application branch only: no series_stats(), p-values, corrections or date-only calculations.
        exec(s.MCODE, env)
        boot = env['boot']
        assert boot.shape == (20000,) and np.isfinite(boot).all()
        theta = float(np.nanmean(X, axis=1).mean())
        low, high = np.quantile(boot, [.025, .975], method='linear')
        errors = [abs(theta - float(reference.theta_hat)), abs(float(low) - float(reference.ci_lower_95)), abs(float(high) - float(reference.ci_upper_95))]
        cell = '_'.join(str(v) for v in [group, inp, backbone or 'NA', setting or 'NA', h, contrast, metric])
        replays.append(dict(cell=cell, point_abs_error=errors[0], lower_abs_error=errors[1], upper_abs_error=errors[2], tolerance=TOL, passed=max(errors) <= TOL,
                            stored_point=float(reference.theta_hat), stored_lower=float(reference.ci_lower_95), stored_upper=float(reference.ci_upper_95)))
        save('bootstrap_interval_replay.csv', pd.DataFrame(replays)) if len(replays) == 84 else None
        if max(errors) > TOL:
            save('FAILED_bootstrap_interval_replay.csv', pd.DataFrame(replays))
            raise RuntimeError('STOP: reconstructed distribution does not match stored interval: ' + cell + ' errors=' + str(errors))
        with target(dest / 'bootstrap' / (cell + '.npy')).open('xb') as f:
            np.save(f, boot, allow_pickle=False)
        with target(dest / 'paired_inputs' / (cell + '.npz')).open('xb') as f:
            np.savez_compressed(f, paired_difference=X, seed_order=np.array(SEEDS), anchor_dates=np.asarray(dates, dtype=str))
        is_static = contrast == 'static-minus-adaptive'
        direction = (1 if is_static else -1) if metric == 'mse' else (-1 if is_static else 1)
        upper = float(np.quantile(direction * boot, .95, method='linear'))
        first, second = contrast.split('-minus-')
        row = dict(group=group, input_set=inp, backbone=backbone, setting=setting, h=h, contrast=contrast, first_arm=first, second_arm=second, metric=metric,
                   n_seeds=5, n_dates=391, B=20000, L=h, estimate_raw=theta, ci95_lower_raw=float(low), ci95_upper_raw=float(high),
                   second_arm_mean_test_mse=float(den), denominator_source=den_source,
                   estimate_pct_second_mse=100 * theta / den if metric == 'mse' else np.nan,
                   ci95_lower_pct_second_mse=100 * low / den if metric == 'mse' else np.nan,
                   ci95_upper_pct_second_mse=100 * high / den if metric == 'mse' else np.nan,
                   adaptive_improvement_direction=direction, adaptive_improvement_estimate_raw=direction * theta,
                   adaptive_improvement_upper95_raw=upper,
                   adaptive_mse_improvement_upper95_pct_second_mse=100 * upper / den if metric == 'mse' else np.nan,
                   allows_positive_improvement=upper > 0, coverage='pointwise one-sided 95%; no simultaneous coverage',
                   reconstruction_max_abs_error=max(errors), replicate_file=rel(dest / 'bootstrap' / (cell + '.npy')))
        rows.append(row)
        print(f'[{len(rows):02}/84] {cell}: interval replay OK', flush=True)
    for inp in ['O', 'C']:
        for h in HS:
            matrices, dates = end_to_end(inp, h)
            denom_row = one(a, input_set=inp, h=h, metric='squared_loss')
            for metric, old_metric in [('mse', 'squared_loss'), ('rank_ic', 'rank_ic')]:
                r = one(a, input_set=inp, h=h, metric=old_metric)
                process('1_end_to_end', inp, '', 'T0', h, 'static-minus-adaptive', metric, matrices[metric], dates, r,
                        float(denom_row.second_mean), rel(BASE / 'A_paired_tests.csv') + '#second_mean; mse row')
    for inp in ['O', 'C']:
        for bk in ['S1', 'S2']:
            for contrast in ['adaptive-minus-constant_gate', 'static-minus-adaptive']:
                second = contrast.split('-minus-')[1]
                for h in HS:
                    den = one(ma, input_set=inp, backbone=bk, h=h, variant=second)
                    assert den.n_seeds == den.n_complete == 5
                    for metric in ['mse', 'rank_ic']:
                        r = one(fa, input_set=inp, backbone=bk, h=h, contrast=contrast, metric=metric)
                        X, dates = paired_matrix(da, input_set=inp, backbone=bk, h=h, contrast=contrast, metric=metric)
                        process('2_frozen_tuned', inp, bk, 'tuned_head_frozen', h, contrast, metric, X, dates, r,
                                float(den.mse_mean), rel(A22 / 'five_seed_summary.csv') + '#' + second)
    for setting in ['T1', 'T2']:
        for contrast in ['static-minus-adaptive', 'adaptive-minus-lstm']:
            second = contrast.split('-minus-')[1]
            for h in HS:
                den = one(mb, setting=setting, model=second, h=h)
                assert den.n_seeds == den.n_complete == 5
                for metric in ['mse', 'rank_ic']:
                    r = one(fb, setting=setting, h=h, contrast=contrast, metric=metric)
                    X, dates = paired_matrix(db, setting=setting, h=h, contrast=contrast, metric=metric)
                    process('3_training_settings', 'C', '', setting, h, contrast, metric, X, dates, r,
                            float(den.mse_mean), rel(bdir / 'five_seed_summary.csv') + '#' + second)
    assert len(rows) == 84 and len(replays) == 84
    data = pd.DataFrame(rows)
    save('all_effect_bounds.csv', data)
    context = ['group', 'input_set', 'backbone', 'setting', 'contrast', 'h']
    wide_rows = []
    for meta, frame in data.groupby(context, dropna=False, sort=False):
        m, c = one(frame, metric='mse'), one(frame, metric='rank_ic')
        wide_rows.append(dict(zip(context, meta), second_arm=m.second_arm, second_arm_mean_test_mse=m.second_arm_mean_test_mse,
            mse_difference=m.estimate_raw, mse_ci95_lower=m.ci95_lower_raw, mse_ci95_upper=m.ci95_upper_raw,
            mse_difference_pct_second_mse=m.estimate_pct_second_mse, mse_ci95_lower_pct_second_mse=m.ci95_lower_pct_second_mse,
            mse_ci95_upper_pct_second_mse=m.ci95_upper_pct_second_mse, adaptive_mse_improvement_upper95_pct=m.adaptive_mse_improvement_upper95_pct_second_mse,
            rank_ic_difference=c.estimate_raw, rank_ic_ci95_lower=c.ci95_lower_raw, rank_ic_ci95_upper=c.ci95_upper_raw,
            adaptive_rank_ic_improvement_upper95=c.adaptive_improvement_upper95_raw))
    wide = pd.DataFrame(wide_rows)
    assert len(wide) == 42
    digests = []
    for group, q in wide.groupby('group', sort=False):
        save('table_' + group + '.csv', q.drop(columns='group'))
        mi, ri = q.adaptive_mse_improvement_upper95_pct.max(), q.adaptive_rank_ic_improvement_upper95.max()
        textline = (f'Across every reported context/horizon in this group, pointwise one-sided 95% upper bounds on Adaptive improvement '
                    f'are no larger than {mi:.6g}% of the respective second-arm mean MSE and {ri:.6g} Rank IC units; this is not a simultaneous 95% statement.')
        digests.append(dict(group=group, n_cells=len(q), max_pointwise_upper95_mse_pct=mi, max_pointwise_upper95_rank_ic=ri, summary=textline))
    save('group_bound_digest.csv', pd.DataFrame(digests))
    source_checks = []
    for path, before in used.items():
        after = sha(ROOT / path)
        source_checks.append(dict(path=path, before_sha256=before, after_sha256=after, unchanged=before == after))
        assert before == after, 'Original input changed: ' + path
    save('source_checks.csv', pd.DataFrame(source_checks))
    js(dest / 'provenance.json', dict(started_utc=started, created_utc=utc(), B=20000, L='h', seed_order=SEEDS,
       joint_rng='SeedSequence([20261019,h,h,1]); seeds then common dates; exact saved-index comparison',
       interval_replay_tolerance=TOL, source_files=[dict(path=p, sha256=h) for p, h in used.items()],
       hypotheses_computed=False, p_values_computed=False, multiplicity_adjustments_computed=False,
       normalization='fixed second-arm five-seed mean TEST MSE; Rank IC unscaled',
       one_sided_upper='q0.95(Adaptive-oriented improvement); signed, no clipping',
       application_source='statistics16.MCODE: original Step15 missing-IC joint application only'))
    def fmt(value):
        if pd.isna(value):
            return 'N/A'
        return f'{value:.6g}' if isinstance(value, (float, np.floating)) else str(value)
    def md(frame):
        return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'] +
                         ['| ' + ' | '.join(fmt(v).replace('|', '/') for v in row) + ' |' for row in frame.itertuples(index=False, name=None)])
    report = '# BLOCKING\n\nNone. All84 reconstructed point estimates and95% intervals match stored results within1e-12.\n\n'
    report += '# Step22C — Descriptive effect-size bounds\n\nNo training, new hypothesis tests, p-values or multiplicity adjustments. All comparisons retain five seeds and391 dates, B=20000 and L=h.\n\n'
    report += 'D is first−second. Every MSE percentage uses the **second arm’s fixed five-seed mean TEST MSE**. Rank IC uses raw units. The one-sided95% upper bound is the95th percentile of the signed Adaptive-improvement distribution, rather than the97.5th percentile. Negative upper bounds are retained and allow no positive improvement under that pointwise bound. These are pointwise bounds without simultaneous coverage.\n\n'
    titles = dict(zip(['1_end_to_end', '2_frozen_tuned', '3_training_settings'], ['1. End-to-end: Static−Adaptive (O/C, original window)', '2. Frozen primary: Step22A tuned aggregators (O/C, S1/S2)', '3. Training robustness: Step22B T1/T2 (C)']))
    for group, q in wide.groupby('group', sort=False):
        display = q[['input_set', 'backbone', 'setting', 'contrast', 'h', 'second_arm_mean_test_mse']].copy()
        display['MSE difference; 95% interval'] = [f'{r.mse_difference:.6g}; [{r.mse_ci95_lower:.6g}, {r.mse_ci95_upper:.6g}]' for r in q.itertuples()]
        display['MSE difference (% second arm); 95% interval'] = [f'{r.mse_difference_pct_second_mse:.6g}; [{r.mse_ci95_lower_pct_second_mse:.6g}, {r.mse_ci95_upper_pct_second_mse:.6g}]' for r in q.itertuples()]
        display['Adaptive MSE improvement upper95 (%)'] = q.adaptive_mse_improvement_upper95_pct
        display['Rank IC difference; 95% interval'] = [f'{r.rank_ic_difference:.6g}; [{r.rank_ic_ci95_lower:.6g}, {r.rank_ic_ci95_upper:.6g}]' for r in q.itertuples()]
        display['Adaptive Rank IC improvement upper95'] = q.adaptive_rank_ic_improvement_upper95
        digest = next(d['summary'] for d in digests if d['group'] == group)
        report += '## ' + titles[group] + '\n\n' + md(display) + '\n\n' + digest + '\n\n'
    report += '## Verification and provenance\n\nTarget distributions were not saved; original paired inputs/canonical predictions and stored RNG indices were reused. Legacy distributions from other numerical profiles/settings were excluded. The reconstruction introduces no model or specification selection.\n\n'
    for name in ['all_effect_bounds.csv', 'table_1_end_to_end.csv', 'table_2_frozen_tuned.csv', 'table_3_training_settings.csv', 'group_bound_digest.csv', 'bootstrap_interval_replay.csv', 'source_inventory.csv', 'source_checks.csv', 'provenance.json']:
        report += '- [' + name + '](' + str(dest / name) + ')\n'
    report += '\nAll replicate arrays and paired-input matrices are retained in bootstrap/ and paired_inputs/ below the run directory. Exact source paths and hashes appear in provenance.json.\n'
    write(OUT / 'REPORT_STEP22C.md', report)
    files = sorted(p for p in dest.rglob('*') if p.is_file()) + [OUT / 'REPORT_STEP22C.md']
    js(OUT / 'COMPLETE_STEP22C.json', dict(started_utc=started, ended_utc=utc(), run_directory=rel(dest), metric_rows=84,
       contrast_context_horizon_cells=42, bootstrap_replicates_per_row=20000, problems=[], artifacts=[artifact(p) for p in files]))
    print('COMPLETE:', OUT / 'REPORT_STEP22C.md', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'status'], nargs='?', default='run')
    args = parser.parse_args()
    if args.command == 'status':
        status()
    else:
        main()
