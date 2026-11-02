"""A2 - forecast-skill metrics for every TEST prediction cell (re-runnable on any prediction set).

Per cell (model, config, horizon, seed):
  1. R2_OS = 1 - MSE_model / MSE_ref, ref in {global_train_mean, zero, ticker_train_mean,
     expanding_pooled_mean} (+ expanding_ticker_mean, reported for completeness).
  2. Clark-West vs Global Train-Mean: f_it = (y-b)^2 - [(y-m)^2 - (b-m)^2], averaged over stocks per
     date; one-sided H1: E f > 0 with Newey-West HAC (lag h-1) and moving block bootstrap (L=h,
     B=20,000, centred one-sided p). Block indices are common to all cells of a horizon
     (SeedSequence([20260826, h, L, 2])).
  3. Exact MSE decomposition: MSE = mean_t (ybar_t - pbar_t)^2 + mean_it ((y-ybar_t) - (p-pbar_t))^2,
     and MSE = bias^2 + Var(y) + Var(p) - 2 Cov(y,p) (population moments). Both verified < 1e-10.
  4. Cross-sectional skill: mean Rank IC (per-date Spearman), NW t (lag h-1), hit rate, XS R^2.
  5. Seed summaries (mean / SD, ddof=1) for every (model, config, h) with >1 seed.
All arithmetic in float64 from the stored float32/float64 values.

Usage: python -B a2_skill.py --preds GLOB [GLOB ...] [--label-family GLOB ...] --out-dir DIR
"""
import argparse
from skill_common import *

REFS = ["global_train_mean", "zero", "ticker_train_mean", "expanding_pooled_mean", "expanding_ticker_mean"]


def cell_metrics(y, p, refs, h, idx_boot, Y, P, gtm):
    r = {}
    e = y - p
    mse = float(np.mean(e ** 2))
    r["mse"] = mse
    for k in REFS:
        r[f"r2os_vs_{k}"] = 1 - mse / float(np.mean((y - refs[k]) ** 2))
    # Clark-West vs Global Train-Mean
    b = refs["global_train_mean"]
    f = (y - b) ** 2 - ((y - p) ** 2 - (b - p) ** 2)
    # The archived GTM constant differs from the float64 recomputation by ~1e-10; treat it as the benchmark itself.
    if np.allclose(p, b, rtol=1e-6, atol=1e-9):
        r.update(cw_mean=np.nan, cw_nw_se=np.nan, cw_nw_t=np.nan, cw_p_nw_one_sided=np.nan, cw_p_mbb_one_sided=np.nan)
        f_t = np.full(Y.shape[0], np.nan)
    else:
        F = f.reshape(Y.shape)  # rows already sorted (date, ticker)
        f_t = F.mean(1)
        m, se, t, _ = nw_t(f_t, h - 1)
        boot = f_t[idx_boot].mean(1)
        r.update(cw_mean=m, cw_nw_se=se, cw_nw_t=t, cw_p_nw_one_sided=float(stats.norm.sf(t)),
                 cw_p_mbb_one_sided=float((1 + np.sum(boot - m >= m)) / (len(boot) + 1)))
    # decompositions
    ybar, pbar = Y.mean(1, keepdims=True), P.mean(1, keepdims=True)
    ts = float(np.mean((ybar - pbar) ** 2))
    xs = float(np.mean(((Y - ybar) - (P - pbar)) ** 2))
    bias2 = float((y.mean() - p.mean()) ** 2)
    vy, vp = float(np.var(y)), float(np.var(p))
    cov = float(np.mean((y - y.mean()) * (p - p.mean())))
    r.update(mse_ts_component=ts, mse_xs_component=xs, ts_share=ts / mse, decomp_abs_error=abs(ts + xs - mse),
             bias2=bias2, var_y=vy, var_pred=vp, minus2cov=-2 * cov,
             moment_decomp_abs_error=abs(bias2 + vy + vp - 2 * cov - mse),
             pred_mean=float(p.mean()), pred_sd=float(np.std(p)), corr_y_pred=float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 1e-12 else np.nan)
    # cross-sectional skill
    ic = rank_ic_rows(P, Y)
    m, se, t, n = nw_t(ic, h - 1)
    r.update(mean_rank_ic=m, rank_ic_nw_se=se, rank_ic_nw_t=t, rank_ic_n_dates=n,
             ic_hit_rate=float(np.mean(ic[np.isfinite(ic)] > 0)) if n else np.nan,
             xs_r2=1 - float(np.sum(((Y - ybar) - (P - pbar)) ** 2)) / float(np.sum((Y - ybar) ** 2)))
    return r, f_t, ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--label-family", nargs="+", default=[str(PREDS_DIR / "static_dw_h*_s0_*.parquet")])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--boot-b", type=int, default=BOOT_B)
    a = ap.parse_args()
    out = Path(a.out_dir)
    files = [f for f in expand_globs(a.preds) if parse_name(f)["split"] == "test"]
    fam = label_family(a.label_family)
    meta = load_meta(sorted(fam[5].ticker.unique()))
    seen, rows, per_date = {}, [], []
    refcache, bootcache = {}, {}
    for f in files:
        info = parse_name(f)
        if info["cell"] in seen:
            raise SystemExit(f"BLOCKING: duplicate cell {info['cell']}: {seen[info['cell']]} and {f}")
        seen[info["cell"]] = f
        h = info["h"]
        d = load_pred(f)
        fh = fam[h]
        test = fh[fh.split == "test"].sort_values(["anchor_date", "ticker"], kind="mergesort").reset_index(drop=True)
        if not d[["ticker", "anchor_date"]].equals(test[["ticker", "anchor_date"]]):
            raise SystemExit(f"BLOCKING: {f} keys differ from label family")
        if not np.array_equal(d.y_true.to_numpy(np.float32), test.y_true.to_numpy(np.float32)):
            raise SystemExit(f"BLOCKING: {f} labels differ from label family")
        if h not in refcache:
            refcache[h] = reference_forecasts(fh, meta, test[["ticker", "anchor_date"]], h)
            rng = np.random.default_rng(np.random.SeedSequence([BOOT_BASE_SEED, h, h, 2]))
            bootcache[h] = mbb_index(N_TEST_DATES, h, a.boot_b, rng)
        refs, gtm = refcache[h]
        y, p = d.y_true.to_numpy(np.float64), d.y_pred.to_numpy(np.float64)
        Y, dates, _ = to_panel(d, "y_true")
        P, _, _ = to_panel(d, "y_pred")
        r, f_t, ic = cell_metrics(y, p, refs, h, bootcache[h], Y, P, gtm)
        rows.append(dict(model=info["model"], config=info["config"], h=h, seed=info["seed"], file=rel(f), **r))
        per_date.append(pd.DataFrame(dict(model=info["model"], config=info["config"], h=h, seed=info["seed"],
                                          anchor_date=dates, cw_f=f_t, rank_ic=ic)))
        print(f"{info['cell']:55s} mse={r['mse']:.6f} r2gtm={r['r2os_vs_global_train_mean']:+.4f} ic={r['mean_rank_ic']:+.4f}", flush=True)
    res = pd.DataFrame(rows).sort_values(["h", "model", "config", "seed"], key=lambda s: s.astype(str)).reset_index(drop=True)
    bad = res[(res.decomp_abs_error >= 1e-10) | (res.moment_decomp_abs_error >= 1e-10)]
    metric_cols = [c for c in res.columns if c not in ("model", "config", "h", "seed", "file")]
    grp = res.groupby(["model", "config", "h"])
    summ = []
    for (m, c, h), g in grp:
        if len(g) < 2:
            continue
        rec = dict(model=m, config=c, h=h, n_seeds=len(g), seeds=",".join(sorted(g.seed, key=lambda s: int(s))))
        for col in metric_cols:
            v = g[col].astype(float)
            rec[f"{col}_mean"], rec[f"{col}_sd"] = float(v.mean()), float(v.std(ddof=1))
        summ.append(rec)
    write_csv(out / "a2_skill_by_cell.csv", res)
    write_csv(out / "a2_seed_summary.csv", pd.DataFrame(summ))
    write_csv(out / "a2_per_date_cw_and_ic.csv", pd.concat(per_date, ignore_index=True))
    write_json(out / "a2_run_info.json", dict(created=utc(), preds=a.preds, n_cells=len(res), label_family=a.label_family,
                                              boot_b=a.boot_b, boot_seed="SeedSequence([20260826, h, h, 2])",
                                              decomposition_failures=bad[["model", "config", "h", "seed"]].to_dict("records"),
                                              max_decomp_abs_error=float(res.decomp_abs_error.max()),
                                              max_moment_decomp_abs_error=float(res.moment_decomp_abs_error.max()),
                                              files=[rel(f) for f in files]))
    if len(bad):
        raise SystemExit(f"BLOCKING: decomposition check failed for {len(bad)} cells")


if __name__ == "__main__":
    main()
