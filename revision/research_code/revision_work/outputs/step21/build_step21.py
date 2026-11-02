"""Step21: saved TEST forecasts only; exclusively create every output.

Run from repository root: .venv/bin/python -B revision_work/outputs/step21/build_step21.py
This entrypoint intentionally refuses to overwrite any existing result.
"""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'revision_work/scripts/analysis16'))
from statistics16_v3 import s
import numpy as np
import pandas as pd
import scipy
import pyarrow
import json
import hashlib
import platform
import datetime
import time

SEEDS = [0, 1, 21, 42, 3407]
HS = [5, 10, 20]
A16 = ROOT / 'revision_work/outputs/analysis16'
FR = ROOT / 'revision_work/outputs/frozen'
INPUTS = {}
OUTPUTS = []
COUNTS = {}
LABELS = {}
DATE_ROWS = []
CHECKS = []
REFERENCES = []
START = datetime.datetime.now(datetime.timezone.utc).isoformat()
WALL_START = time.perf_counter()
MODELS = ['single', 'fixed', 'same_patch_fixed', 'static', 'adaptive',
          'lstm', 'vanilla', 'swim', 'ridge']
NAMES = {'single': 'Single', 'fixed': 'Fixed', 'same_patch_fixed': 'Same-patch Fixed',
         'static': 'Static', 'adaptive': 'Adaptive', 'lstm': 'LSTM',
         'vanilla': 'Vanilla', 'swim': 'SWiM', 'ridge': 'Ridge',
         'constant_gate': 'Constant gate'}


def relative(p):
    return str(Path(p).resolve().relative_to(ROOT))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def register(p, expected=None):
    p = Path(p).resolve()
    assert p.is_relative_to(ROOT)
    if p not in INPUTS:
        INPUTS[p] = digest(p)
    if expected is not None:
        assert INPUTS[p] == expected, f'BLOCKING hash mismatch: {p}'
    return p


def read_csv(p):
    return pd.read_csv(register(p))


def write(name, payload):
    p = OUT / name
    assert p.resolve().is_relative_to(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(payload.encode() if isinstance(payload, str) else payload)
        f.flush()
        os.fsync(f.fileno())
    OUTPUTS.append(p)
    return p


def csv(name, d):
    return write(name, d.to_csv(index=False, float_format='%.17g', na_rep='N/A'))


def js(name, obj):
    return write(name, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load_labels():
    bench = read_csv(ROOT / 'revision_work/outputs/skill_analysis/a1/a1_benchmark_mse.csv')
    for h in HS:
        p = register(A16 / f'labels/W3_h{h}.parquet')
        d = pd.read_parquet(p)
        assert d.groupby('split').size().to_dict() == {'test': 6647, 'train': 30634, 'val': 6290}
        t = s.canon(d[d.split.eq('test')])
        assert len(t) == 6647 and not t.duplicated(['ticker', 'anchor_date']).any()
        assert t.anchor_date.nunique() == 391 and t.ticker.nunique() == 17
        assert (t.groupby('anchor_date').size() == 17).all()
        assert t.groupby('ticker').size().eq(391).all()
        y = t.y_true.to_numpy(float)
        assert np.isfinite(y).all()
        Y = y.reshape(391, 17)
        valid = Y != 0
        n = int(valid.sum())
        gtm_row = bench[bench.h.eq(h) & bench.benchmark.eq('global_train_mean')]
        assert len(gtm_row) == 1
        gtm = float(gtm_row.iloc[0].pred_mean)
        assert np.sign(gtm) == np.sign(d[d.split.eq('train')].y_true.to_numpy(float).mean())
        LABELS[h] = dict(table=t, y=y, Y=Y, valid=valid, count=valid.sum(1),
                         dates=t.anchor_date.unique(), n=n, gtm=gtm)
        positive = int((Y > 0).sum())
        ref = dict(h=h, n_test_rows=6647, n_test_dates=391, n_stocks=17,
                   n_zero_realized_excluded=int((Y == 0).sum()), n_valid_pairs=n,
                   n_positive_realized=positive, n_negative_realized=int((Y < 0).sum()),
                   positive_return_share=positive / n, always_up_da=positive / n,
                   global_train_mean_forecast=gtm,
                   global_train_mean_da=float(((np.sign(gtm) == np.sign(Y)) & valid).sum() / n),
                   reference_source='revision_work/outputs/skill_analysis/a1/a1_benchmark_mse.csv',
                   identical_for_input_sets='O,C')
        REFERENCES.append(ref)
        for date, count in zip(LABELS[h]['dates'], LABELS[h]['count']):
            DATE_ROWS.append(dict(scope='reference', input_set='O,C', backbone='', setting='',
                                  model='valid_pair_count', h=h, seed='N/A', anchor_date=date,
                                  n_valid=int(count), n_correct='N/A', da='N/A', id='reference'))


def evaluate(row, p, expected_hash):
    p = register(p, expected_hash)
    d = s.canon(pd.read_parquet(p))
    lab = LABELS[row['h']]
    t = lab['table']
    assert len(d) == 6647, (row['id'], len(d))
    assert not d.duplicated(['ticker', 'anchor_date']).any(), row['id']
    assert np.array_equal(d[['anchor_date', 'ticker']].to_numpy(),
                          t[['anchor_date', 'ticker']].to_numpy()), row['id']
    if 'split' in d:
        assert d.split.eq('test').all(), row['id']
    yy = d.y_true.to_numpy(float)
    pp = d.y_pred.to_numpy(float)
    assert np.isfinite(yy).all() and np.isfinite(pp).all(), row['id']
    assert np.array_equal(yy.astype('float32'), lab['y'].astype('float32')), row['id']
    assert np.array_equal(yy == 0, lab['y'] == 0), row['id']
    assert np.array_equal(np.sign(yy), np.sign(lab['y'])), row['id']
    P = pp.reshape(391, 17)
    mask = lab['valid']
    correct = ((np.sign(P) == np.sign(lab['Y'])) & mask)
    counts = correct.sum(axis=1).astype('int64')
    ident = row['id']
    assert ident not in COUNTS
    COUNTS[ident] = counts
    n_correct = int(counts.sum())
    n = lab['n']
    row.update(n_test_rows=len(d), n_valid_pairs=n,
               n_zero_realized_excluded=6647-n, n_correct=n_correct, da=n_correct/n,
               n_zero_forecasts_on_retained_pairs=int(((P == 0) & mask).sum()),
               global_train_mean_da=REFERENCES[HS.index(row['h'])]['global_train_mean_da'],
               always_up_da=REFERENCES[HS.index(row['h'])]['always_up_da'],
               prediction_file=relative(p), prediction_sha256=INPUTS[p])
    CHECKS.append(dict(id=ident, scope=row['scope'], prediction_file=relative(p),
                       rows=len(d), dates=391, stocks=17, keys_equal=True,
                       labels_float32_equal=True, labels_sign_equal=True,
                       labels_exact_zero_mask_equal=True, finite=True,
                       label_max_abs_difference=float(np.max(np.abs(yy-lab['y']))),
                       input_hash_verified=True, n_valid_pairs=n))
    for i, date in enumerate(lab['dates']):
        DATE_ROWS.append(dict(scope=row['scope'], input_set=row['input_set'],
                              backbone=row['backbone'], setting=row['setting'],
                              model=row['model'], h=row['h'], seed=row['seed'],
                              anchor_date=date, n_valid=int(lab['count'][i]),
                              n_correct=int(counts[i]), da=float(counts[i]/lab['count'][i]),
                              id=ident))
    return row


def cells():
    reg = read_csv(A16 / 'registry.csv').fillna('')
    checks = read_csv(A16 / 'prediction_checks.csv').set_index('id')
    chosen = reg[reg.group.isin(['core', 'samepatch', 'neural', 'ridge']) &
                 reg.config.eq('daily_weekly') & reg.window.eq('W3')]
    assert len(chosen) == 246
    frozen = reg[reg.group.eq('frozen')]
    assert len(frozen) == 240
    rows = []
    for rr in pd.concat([chosen, frozen]).itertuples(index=False):
        c = checks.loc[rr.id]
        assert all(bool(c[k]) for k in ('keys_equal', 'labels_float32_equal', 'finite'))
        assert c.source_sha256 == rr.sha256
        register(ROOT / rr.source, rr.sha256)
        is_frozen = rr.group == 'frozen'
        row = dict(scope='frozen' if is_frozen else 'table4', input_set=rr.input_set,
                   backbone=rr.config if is_frozen else '',
                   setting='head_frozen' if is_frozen else '', model=rr.model,
                   config='daily_weekly', h=int(rr.h), seed=int(rr.seed), window='W3',
                   id=rr.id, registry_group=rr.group, original_source=rr.source,
                   original_source_sha256=rr.sha256, provenance=rr.provenance,
                   fixed_baseline_reused=False)
        rows.append(evaluate(row, ROOT / rr.canonical, c.canonical_sha256))
    files = read_csv(FR / 'prediction_files.csv')
    fm = read_csv(FR / 'cell_metrics.csv')
    fp = files[files.split.eq('test') & files.setting.eq('head_retrained')]
    assert len(fp) == 120 and fp.backbone.eq('S1').all()
    primary = files[files.split.eq('test') & files.setting.eq('head_frozen')]
    for rr in frozen.itertuples(index=False):
        q = primary[primary.input_set.eq(rr.input_set) & primary.backbone.eq(rr.config) &
                    primary.variant.eq(rr.model) & primary.horizon.eq(rr.h) & primary.seed.eq(rr.seed)]
        assert len(q) == 1 and q.iloc[0].path == rr.source and q.iloc[0].sha256 == rr.sha256
    for rr in fp.itertuples(index=False):
        q = fm[fm.input_set.eq(rr.input_set) & fm.backbone.eq(rr.backbone) &
               fm.setting.eq(rr.setting) & fm.variant.eq(rr.variant) &
               fm.horizon.eq(rr.horizon) & fm.seed.eq(rr.seed)]
        assert len(q) == 1
        meta = q.iloc[0]
        row = dict(scope='frozen', input_set=rr.input_set, backbone='S1', setting='head_retrained',
                   model=rr.variant, config='daily_weekly', h=int(rr.horizon), seed=int(rr.seed),
                   window='W3', id=f'frozen_secondary_{rr.input_set}_{rr.variant}_{rr.horizon}_{rr.seed}',
                   registry_group='frozen_secondary', original_source=rr.path,
                   original_source_sha256=rr.sha256, provenance=str(meta.provenance),
                   fixed_baseline_reused=bool(meta.fixed_baseline_reused))
        rows.append(evaluate(row, ROOT / rr.path, rr.sha256))
        if rr.variant == 'fixed':
            prior = primary[primary.input_set.eq(rr.input_set) & primary.backbone.eq('S1') &
                            primary.variant.eq('fixed') & primary.horizon.eq(rr.horizon) &
                            primary.seed.eq(rr.seed)]
            assert len(prior) == 1 and prior.iloc[0].path == rr.path
            assert row['fixed_baseline_reused']
        else:
            assert not row['fixed_baseline_reused']
    result = pd.DataFrame(rows)
    keys = ['scope', 'input_set', 'backbone', 'setting', 'model', 'h', 'seed']
    assert len(result) == 606 and not result.duplicated(keys).any()
    for k, g in result.groupby(keys[:-1], dropna=False):
        if k[4] == 'ridge':
            assert len(g) == 1
        else:
            assert sorted(g.seed.tolist()) == SEEDS, k
    for inp in ['O', 'C']:
        assert set(result.query('scope=="table4" and input_set==@inp').model) == set(MODELS)
    print('INTEGRITY: all 606 cells verified (246 Table4,240 frozen primary,120 sensitivity)', flush=True)
    return result


def aggregate(d):
    keys = ['scope', 'input_set', 'backbone', 'setting', 'model', 'config', 'h', 'window']
    out = d.groupby(keys, dropna=False).agg(
        n_seeds=('seed', 'nunique'), da_mean=('da', 'mean'), da_sd=('da', 'std'),
        da_min=('da', 'min'), da_max=('da', 'max'), n_valid_pairs_per_seed=('n_valid_pairs', 'first'),
        n_zero_realized_excluded=('n_zero_realized_excluded', 'first'),
        global_train_mean_da=('global_train_mean_da', 'first'), always_up_da=('always_up_da', 'first'),
        fixed_baseline_reused=('fixed_baseline_reused', 'any')).reset_index()
    assert len(out) == 126
    out['sd_definition'] = np.where(out.n_seeds.eq(5), 'sample SD (ddof=1)', 'N/A (deterministic n=1)')
    return out


def bootstrap(d):
    rng_checks = []
    rng_metadata = register(A16 / 'RNG_MANIFEST.json')
    rmeta = json.loads(rng_metadata.read_text())
    assert rmeta['B'] == 20000 and rmeta['model_seed_order'] == SEEDS
    assert s.BASE == 20261019 and s.B == 20000
    den = {}
    for h in HS:
        rp = register(A16 / f'rng/draws_h{h}.npz')
        with np.load(rp) as z:
            wt, di, si = s.draws(h, 5, 391)
            assert np.array_equal(si, z['seed_indices'])
            assert np.array_equal(di, z['joint_date_indices'])
            assert np.array_equal(s.dates(h, 391), z['date_only_indices'])
        assert np.allclose(wt.sum(1), 1, atol=1e-15, rtol=0)
        D = np.tile(LABELS[h]['count'], (5, 1))
        point, joint, _ = s.distributions(D, h)
        assert np.isfinite(joint).all() and (joint > 0).all()
        den[h] = (point, joint)
        rng_checks.append(dict(h=h, B=20000, L=h, T=391, n_seeds=5,
                               seed_indices_equal=True, joint_date_indices_equal=True,
                               date_only_indices_equal=True, draw_file=relative(rp),
                               draw_file_sha256=INPUTS[rp], entropy=[20261019, h, h, 1]))
    results, seed_rows, verification = [], [], []
    group_keys = ['scope', 'input_set', 'backbone', 'setting']
    for key, group in d.groupby(group_keys, dropna=False):
        scope, inp, bk, setting = key
        contrasts = [('static', 'adaptive')]
        if scope == 'frozen':
            contrasts.append(('adaptive', 'constant_gate'))
        for first, second in contrasts:
            for h in HS:
                A = group[group.model.eq(first) & group.h.eq(h)].set_index('seed').loc[SEEDS]
                Z = group[group.model.eq(second) & group.h.eq(h)].set_index('seed').loc[SEEDS]
                X = np.ascontiguousarray(np.stack([COUNTS[a] - COUNTS[z]
                                                   for a, z in zip(A.id, Z.id)]), dtype=float)
                theta_num, joint_num, _ = s.distributions(X, h)
                theta_den, joint_den = den[h]
                theta = theta_num / theta_den
                joint = joint_num / joint_den
                delta = (A.da-Z.da).to_numpy(float)
                assert abs(theta-float(delta.mean())) < 1e-15
                stats = s.series_stats(joint, theta)
                family = '|'.join([scope, inp, bk or 'none', setting or 'none', first+'-minus-'+second])
                meta = dict(scope=scope, input_set=inp, backbone=bk, setting=setting,
                            contrast=first+'-minus-'+second, first_model=first, second_model=second,
                            h=h, metric='directional_accuracy', method='joint_seed_date',
                            n_seeds=5, n_dates=391, n_valid_pairs_per_seed=LABELS[h]['n'],
                            B=20000, L=h, holm_family=family)
                row = dict(**meta, first_da_mean=float(A.da.mean()), first_da_sd=float(A.da.std(ddof=1)),
                           second_da_mean=float(Z.da.mean()), second_da_sd=float(Z.da.std(ddof=1)),
                           seed_difference_mean=float(delta.mean()), seed_difference_sd=float(delta.std(ddof=1)),
                           seed_wins=int((delta > 0).sum()), seed_losses=int((delta < 0).sum()),
                           seed_ties=int((delta == 0).sum()), seed_wins_definition='first DA > second DA',
                           estimate_pp=100*theta,
                           ci_lower_95_pp=100*stats['ci_lower_95'], ci_upper_95_pp=100*stats['ci_upper_95'],
                           **stats)
                results.append(row)
                for seed, value in zip(SEEDS, delta):
                    seed_rows.append(dict(**meta, seed=seed, first_da=float(A.loc[seed].da),
                                          second_da=float(Z.loc[seed].da), da_difference=float(value),
                                          difference_pp=100*float(value)))
                # Independent direct paired resampling of integer success/valid counts, first 32 draws.
                _, didx, sidx = s.draws(h, 5, 391)
                direct = []
                for rep in range(32):
                    dates = didx[rep]
                    numerator = X[sidx[rep]][:, dates].sum()
                    denominator = 5*LABELS[h]['count'][dates].sum()
                    direct.append(float(numerator/denominator))
                error = float(np.max(np.abs(np.array(direct)-joint[:32])))
                assert error < 2e-15, (key, h, error)
                verification.append(dict(**meta, direct_replicates_checked=32,
                                         max_abs_bootstrap_difference=error,
                                         direct_point_difference=abs(theta-float(delta.mean()))))
                if len(results) % 6 == 0:
                    print('BOOTSTRAP', len(results), 'joint comparisons complete', flush=True)
    tests = s.corrected(pd.DataFrame(results), ['holm_family'])
    assert len(tests) == 42 and tests.groupby('holm_family').size().eq(3).all()
    assert tests.p_holm_3h.notna().all()
    assert (tests.p_holm_3h+1e-15 >= tests.centered_p_two_sided).all()
    return tests, pd.DataFrame(seed_rows), rng_checks, verification


def fmt(x):
    if pd.isna(x):
        return 'N/A'
    return f'{float(x):.6g}'


def md(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, r))+' |' for r in rows])+'\n'


def compact_summary(summary, setting, backbone=''):
    ss = summary[summary.setting.eq(setting) & summary.backbone.eq(backbone)]
    order = MODELS if not setting else ['fixed', 'static', 'adaptive', 'constant_gate']
    rows = []
    for model in order:
        for h in HS:
            q = ss[ss.model.eq(model) & ss.h.eq(h)]
            if len(q) == 0:
                continue
            assert len(q) == 2
            vals = []
            for inp in ['O', 'C']:
                r = q[q.input_set.eq(inp)].iloc[0]
                val = fmt(r.da_mean)+((' ± '+fmt(r.da_sd)) if r.n_seeds == 5 else ' (n=1; SD N/A)')
                vals.append(val)
            rows.append([NAMES[model], h, *vals])
    return md(['Model / aggregator', 'h', 'O DA mean ± SD', 'C DA mean ± SD'], rows)


def test_table(tests):
    rows = []
    for r in tests.sort_values(['input_set', 'backbone', 'setting', 'contrast', 'h']).itertuples():
        rows.append([r.input_set, r.backbone or 'Core', r.setting or 'Trained models',
                     NAMES[r.first_model]+' − '+NAMES[r.second_model], r.h,
                     fmt(r.estimate_pp), '['+fmt(r.ci_lower_95_pp)+', '+fmt(r.ci_upper_95_pp)+']',
                     fmt(r.centered_p_two_sided), fmt(r.p_holm_3h),
                     f'{r.seed_wins}/{r.seed_losses}/{r.seed_ties}'])
    return md(['Input', 'Backbone', 'Setting', 'DA contrast', 'h', 'Difference (pp)', '95% interval (pp)',
               'Raw p', 'Holm p (3h)', 'Seed wins/losses/ties'], rows)


def report(d, summary, tests, qa):
    refrows = [[r['h'], r['n_zero_realized_excluded'], r['n_valid_pairs'],
                fmt(r['global_train_mean_forecast']), fmt(r['global_train_mean_da']),
                fmt(r['always_up_da'])] for r in REFERENCES]
    lines = ['# REPORT STEP21 — Directional accuracy', '', '## BLOCKING', '',
             'None. All requested prediction cells and all five paired seeds are present; integrity checks passed.', '',
             '## Definition and coverage', '',
             'DA is the share of TEST stock–date pairs for which sign(forecast) equals sign(realized h-day log return), '
             'after excluding realized returns exactly equal to zero. Forecasts exactly zero are incorrect on retained pairs. '
             'All arithmetic uses stored predictions; no training or checkpoint inference was run. DA is a fraction '
             '(0.5 means 50%); paired differences below are percentage points (pp).', '',
             'Each forecast has 6,647 TEST rows (391 dates × 17 stocks). We report per-seed DA and its five-seed '
             'arithmetic mean and sample SD (ddof=1), not DA of a seed-ensemble forecast. Ridge is deterministic '
             '(n=1; SD N/A). No seed or specification was selected or dropped. Every one of the 606 reported cells '
             'is included in the per-seed CSVs; the 30 S1 sensitivity Fixed rows reuse the primary Fixed reference.', '',
             'O means original price-level inputs; C means causal window-relative inputs. Table4 sources are exactly '
             'the Step16 registry: 246 core/same-patch/neural/Ridge Daily+Weekly W3 cells. In particular, O neural '
             'seed42 is the B5FIX 12/12 forecast and seeds0/1/21/3407 are B5, rather than the archived original-paper '
             'seed42. Frozen primary uses the 240 Step16 S1/S2 head-frozen cells. The 120 stored S1 head-retrained '
             'sensitivity cells are shown separately, as confirmed by the user. Backbones were not retrained.', '',
             '## Reference predictors and zero-return exclusions', '',
             md(['h', 'Zero returns excluded', 'Retained pairs', 'Global Train-Mean forecast',
                 'Global Train-Mean DA', 'Always-up DA / positive share'], refrows),
             'The stored Global Train-Mean is positive at all three horizons, so its DA equals the always-up '
             'positive-return share. These references are identical for O/C and all frozen settings. Positive share '
             'uses the same nonzero-return denominator as DA. No test against either reference is added.', '',
             '## Table4 — all models', '', compact_summary(summary, ''),
             'Full per-seed results: [table4_da_per_seed.csv](table4_da_per_seed.csv). '
             'Summary: [table4_da_summary.csv](table4_da_summary.csv).', '',
             '## Frozen representations — primary, experts and head frozen', '',
             '### S1 — Fixed-trained backbone', '', compact_summary(summary, 'head_frozen', 'S1'),
             '### S2 — Adaptive-trained backbone', '', compact_summary(summary, 'head_frozen', 'S2'),
             'All four aggregators use their saved TEST predictions. Constant gate denotes the fixed-input gating '
             'network; it is distinct from Static logits. Primary per-seed results and summaries are in '
             '[frozen_primary_da_per_seed.csv](frozen_primary_da_per_seed.csv) and '
             '[frozen_primary_da_summary.csv](frozen_primary_da_summary.csv).', '',
             '## Frozen representations — S1 sensitivity, head retrained', '',
             compact_summary(summary, 'head_retrained', 'S1'),
             'Static, Adaptive and constant gate retrain the head with their aggregator while experts stay frozen. '
             'Fixed is the explicitly reused primary Fixed reference, not an additional trained specification. '
             'No S2 head-retrained setting exists. See [frozen_sensitivity_da_per_seed.csv](frozen_sensitivity_da_per_seed.csv) '
             'and [frozen_sensitivity_da_summary.csv](frozen_sensitivity_da_summary.csv).', '',
             '## Paired DA comparisons', '',
             'Only the requested contrasts are tested: Static minus Adaptive in trained core models and in each '
             'frozen setting, plus Adaptive minus constant gate in each frozen setting. Positive differences and '
             'seed wins favor the first named arm. These are comparisons of per-seed DA, not forecasts averaged '
             'over seeds before applying the sign function.', '',
             'We reuse the Step16 joint seed×date bootstrap: B=20,000, L=h, five paired seeds in order '
             '[0,1,21,42,3407], one common set of non-circular overlapping moving date blocks for all sampled seeds, '
             'truncated to391 dates. SeedSequence([20261019,h,h,1]) and all generated seed/date indices match '
             'the stored Step16 arrays exactly. Percentile95% intervals and centered two-sided p values with '
             '+1 correction use the imported Step5 series_stats; Holm uses the existing Step16 helper.', '',
             'Exact-zero exclusions leave different numbers of eligible stocks on different dates. Therefore '
             'each bootstrap replicate divides its resampled correct-count difference by its resampled '
             'valid-pair count, preserving the requested pooled stock–date DA. Equal weighting of date-level '
             'DA percentages would estimate a different quantity. The seed/date sampling settings remain '
             'unchanged. All391 dates are retained; none has an empty eligible cross-section. Independent direct '
             'count resampling verified32 replicates of each of42 comparisons.', '',
             'Each Holm family contains only horizons5/10/20 for a fixed scope, input set, backbone, head '
             'setting and contrast:14 families ×3 tests=42. No correction is shared across distinct input '
             'sets, backbones, settings or contrasts. Family IDs appear in the test CSV. There are no extra '
             'date-only, seed-level or reference-predictor tests.', '',
             '### Trained core cells', '', test_table(tests[tests.scope.eq('table4')]),
             '### Frozen primary', '', test_table(tests[tests.scope.eq('frozen') & tests.setting.eq('head_frozen')]),
             '### Frozen sensitivity', '', test_table(tests[tests.setting.eq('head_retrained')]),
             'All42 rows are reported regardless of their p values. '
             f"{int(tests.p_holm_3h.lt(.05).sum())} of42 have Holm p<0.05 under the specified within-family correction. '", '',
             '## Files and verification', '',
             '- [references.csv](references.csv): denominator/exclusion counts, positive share, stored Train-Mean and DA.',
             '- [paired_da_tests.csv](paired_da_tests.csv): all joint tests, intervals, raw/Holm p and seed wins.',
             '- [paired_da_seed_differences.csv](paired_da_seed_differences.csv): every paired seed difference.',
             '- [da_by_seed_date.csv](da_by_seed_date.csv): correct counts and valid counts for all seed/date cells; '
             'daily DA is supporting data, not the reported pooled estimator.',
             '- [prediction_integrity.csv](prediction_integrity.csv): keys, labels, exact-zero masks and source checks.',
             '- [all_da_per_seed.csv](all_da_per_seed.csv) and [all_da_summary.csv](all_da_summary.csv): complete combined extracts.',
             '- [RNG_MANIFEST.json](RNG_MANIFEST.json), [VALIDATION_STEP21.json](VALIDATION_STEP21.json) and '
             '[INPUT_MANIFEST.json](INPUT_MANIFEST.json): RNG, independent checks, packages and unchanged input hashes.',
             '- [PROTOCOL_STEP21.md](PROTOCOL_STEP21.md) and [build_step21.py](build_step21.py): declared rules and reproducible code.', '',
             f"All {qa['input_files_verified_unchanged']} registered input files have unchanged SHA-256 hashes. "
             'Prediction keys match exactly; labels agree in float32 and, separately, in sign and exact-zero '
             'classification. Original-source and canonical hashes match the stored registries. CSVs retain '
             'full numerical precision; displayed values use six significant digits. Every output was exclusively created.', '']
    return '\n'.join(lines)


def main():
    assert OUT == ROOT / 'revision_work/outputs/step21'
    assert not (OUT/'all_da_per_seed.csv').exists(), 'Refusing overwrite; use a new output directory/version.'
    protocol = register(OUT/'PROTOCOL_STEP21.md')
    assert 'B=20,000' in protocol.read_text()
    load_labels()
    d = cells()
    summary = aggregate(d)
    tests, differences, rng_checks, independent = bootstrap(d)
    for scope, setting, stem in [('table4', '', 'table4'),
                                 ('frozen', 'head_frozen', 'frozen_primary'),
                                 ('frozen', 'head_retrained', 'frozen_sensitivity')]:
        mask = d.scope.eq(scope) & d.setting.eq(setting)
        smask = summary.scope.eq(scope) & summary.setting.eq(setting)
        csv(stem+'_da_per_seed.csv', d[mask].sort_values(['input_set','backbone','model','h','seed']))
        csv(stem+'_da_summary.csv', summary[smask].sort_values(['input_set','backbone','model','h']))
    csv('all_da_per_seed.csv', d)
    csv('all_da_summary.csv', summary)
    csv('references.csv', pd.DataFrame(REFERENCES))
    csv('paired_da_tests.csv', tests)
    csv('paired_da_seed_differences.csv', differences)
    csv('da_by_seed_date.csv', pd.DataFrame(DATE_ROWS))
    csv('prediction_integrity.csv', pd.DataFrame(CHECKS))
    js('RNG_MANIFEST.json', dict(B=20000, base_seed=20261019, model_seed_order=SEEDS,
                               joint_entropy='[20261019,h,h,1]',
                               draw_order='paired seeds first, then shared moving date blocks',
                               block='non-circular overlapping L=h; ceil(391/h) blocks, truncate to391 dates',
                               index_checks=rng_checks,
                               aggregation='bootstrap correct-count difference / bootstrap valid-pair count',
                               holm='three horizons within scope/input/backbone/setting/contrast'))
    # Register local imported implementation files, including the AST source used by Step16.
    for m in list(sys.modules.values()):
        p = getattr(m, '__file__', None)
        if p and Path(p).resolve().is_relative_to(ROOT) and Path(p).suffix == '.py':
            p = Path(p).resolve()
            if not p.is_relative_to(ROOT/'.venv') and not p.is_relative_to(ROOT/'revision_work/outputs/_deps_pyarrow15'):
                register(p)
    for p in [FR/'scripts/frozen_report.py', Path(__file__)]:
        register(p)
    for p, h in INPUTS.items():
        assert digest(p) == h, f'BLOCKING input changed during run: {p}'
    # Compare summary means/SDs and references to all raw numerator/denominator counts.
    assert np.allclose(d.da.to_numpy(), d.n_correct.to_numpy()/d.n_valid_pairs.to_numpy(), rtol=0, atol=0)
    qa = dict(status='PASS', cells=606, table4_cells=246, frozen_primary_cells=240,
              frozen_sensitivity_cells=120, summary_rows=126, joint_tests=42, holm_families=14,
              n_zero_realized={str(r['h']):r['n_zero_realized_excluded'] for r in REFERENCES},
              n_valid_pairs={str(r['h']):r['n_valid_pairs'] for r in REFERENCES},
              source_keys_labels_zero_masks_hashes_verified=True,
              rng_indices_equal_to_stored_step16=True, independent_bootstrap_checks=independent,
              max_direct_bootstrap_error=max(r['max_abs_bootstrap_difference'] for r in independent),
              input_files_verified_unchanged=len(INPUTS),
              existing_inputs_modified=False, exclusive_creation=True,
              packages=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                            scipy=scipy.__version__,pyarrow=pyarrow.__version__),
              start_utc=START, end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              wall_seconds=time.perf_counter()-WALL_START)
    js('VALIDATION_STEP21.json', qa)
    js('INPUT_MANIFEST.json', dict(files=[dict(path=relative(p),sha256=h) for p,h in sorted(INPUTS.items())],
                                  unchanged=True, packages=qa['packages']))
    write('REPORT_STEP21.md', report(d, summary, tests, qa))
    js('COMPLETE_STEP21.json', dict(start_utc=START, end_utc=qa['end_utc'],
                                  cells=606, tests=42, files=[dict(path=relative(p),sha256=digest(p))
                                                          for p in OUTPUTS]))
    print('COMPLETE:', relative(OUT/'REPORT_STEP21.md'), 'cells=606 tests=42', flush=True)
    print('Holm p<.05:', int(tests.p_holm_3h.lt(.05).sum()), flush=True)


if __name__ == '__main__':
    main()
