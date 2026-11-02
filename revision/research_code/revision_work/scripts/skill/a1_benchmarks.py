"""A1 - financial benchmarks computed (not trained) on the 6,647 TEST stock-dates.

Benchmarks (all no-look-ahead):
  ticker_train_mean        per-ticker mean of retained TRAIN labels
  expanding_ticker_mean    mean of the ticker's retained labels with TARGET date <= anchor
  expanding_pooled_mean    same, pooled over the 17 tickers
  expanding_ticker_fullhist  sensitivity: the ticker's full frozen label history (pre-panel anchors and
                           purged split-boundary anchors included) with TARGET date <= anchor
  Signals from the ORIGINAL price snapshot (closes at sessions <= anchor):
    MOM12_1 = log(P[t-21]/P[t-252]); MOM6_1 = log(P[t-21]/P[t-126])
    REV1w = -log(P[t]/P[t-5]);      REV1m = -log(P[t]/P[t-21])
  As ranking signals: per-date Spearman IC vs realized h-day label, NW t (lag h-1), hit rate,
  top-3 minus bottom-3 equal-weight realized return.
  As calibrated forecasts: pooled OLS label ~ 1 + signal on TRAIN only, applied to TEST.

Usage: python -B a1_benchmarks.py [--label-family GLOB ...] [--out-dir DIR]
"""
import argparse
from skill_common import *

SIGNALS = {
    "mom12_1": lambda P, i: np.log(P[i - 21] / P[i - 252]),
    "mom6_1": lambda P, i: np.log(P[i - 21] / P[i - 126]),
    "rev1w": lambda P, i: -np.log(P[i] / P[i - 5]),
    "rev1m": lambda P, i: -np.log(P[i] / P[i - 21]),
}
MIN_LAG = {"mom12_1": 252, "mom6_1": 126, "rev1w": 5, "rev1m": 21}


def signal_table(keys):
    """Signals + recomputed labels for (ticker, anchor_date) keys from the original daily closes."""
    out = []
    for t in sorted(keys.ticker.unique()):
        px = pd.read_csv(RAW_DIR / f"{t}_daily.csv")
        px["datetime"] = canon_dates(px["datetime"])
        px = px.sort_values("datetime", kind="mergesort").reset_index(drop=True)
        P = px.close.to_numpy(np.float64)
        pos = pd.Series(np.arange(len(px)), index=px.datetime)
        k = keys[keys.ticker == t]
        i = pos.reindex(k.anchor_date).to_numpy()
        if np.isnan(i).any():
            raise ValueError(f"{t}: anchor date missing from price snapshot")
        i = i.astype(int)
        rec = pd.DataFrame({"ticker": t, "anchor_date": k.anchor_date.to_numpy()})
        for name, f in SIGNALS.items():
            ok = i >= MIN_LAG[name]
            v = np.full(len(i), np.nan)
            v[ok] = f(P, i[ok])
            rec[name] = v
        for h in HORIZONS:
            ok = i + h < len(P)
            v = np.full(len(i), np.nan)
            v[ok] = np.log(P[i[ok] + h] / P[i[ok]])
            rec[f"y_recomputed_{h}d"] = v.astype(np.float32)
            td = np.full(len(i), None, dtype=object)
            td[ok] = px.datetime.to_numpy()[i[ok] + h]
            rec[f"target_from_prices_{h}d"] = td
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def ranking_stats(sig, y, h):
    ic = rank_ic_rows(sig, y)
    m, se, t, n = nw_t(ic, h - 1)
    order = np.argsort(-sig, axis=1, kind="mergesort")
    yy = np.take_along_axis(y, order, axis=1)
    spread = yy[:, :3].mean(1) - yy[:, -3:].mean(1)
    sm, sse, st, _ = nw_t(spread, h - 1)
    return dict(mean_rank_ic=m, rank_ic_nw_se=se, rank_ic_nw_t=t, n_dates=n,
                ic_hit_rate=float(np.mean(ic[np.isfinite(ic)] > 0)),
                top3_minus_bottom3_mean=sm, top3_minus_bottom3_nw_t=st), ic, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-family", nargs="+", default=[str(PREDS_DIR / "static_dw_h*_s0_*.parquet")])
    ap.add_argument("--out-dir", default=str(SKILL_OUT / "a1"))
    ap.add_argument("--bench-dir", default=str(SKILL_OUT / "benchmark_preds"))
    a = ap.parse_args()
    out, bdir = Path(a.out_dir), Path(a.bench_dir)
    fam = label_family(a.label_family)
    tickers = sorted(fam[5].ticker.unique())
    meta = load_meta(tickers)
    checks = {"label_family": a.label_family, "tickers": tickers}

    # --- integrity: frozen retained labels == meta labels == labels recomputed from the price snapshot
    keys_all = fam[5][["ticker", "anchor_date", "split"]].copy()
    sig = signal_table(keys_all)
    for h in HORIZONS:
        f = fam[h].merge(meta[["ticker", "anchor_date", f"y_{h}d", f"target_date_{h}d"]], on=["ticker", "anchor_date"],
                         how="left", validate="one_to_one").merge(sig, on=["ticker", "anchor_date"], how="left", validate="one_to_one")
        checks[f"h{h}_retained_eq_meta"] = bool(np.array_equal(f.y_true.to_numpy(np.float32), f[f"y_{h}d"].to_numpy(np.float32)))
        checks[f"h{h}_retained_eq_price_snapshot"] = bool(np.array_equal(f.y_true.to_numpy(np.float32), f[f"y_recomputed_{h}d"].to_numpy(np.float32)))
        checks[f"h{h}_max_abs_label_vs_price_snapshot"] = float(np.max(np.abs(f.y_true.to_numpy(np.float64) - f[f"y_recomputed_{h}d"].to_numpy(np.float64))))
        checks[f"h{h}_target_dates_eq_price_calendar"] = bool((f[f"target_date_{h}d"] == f[f"target_from_prices_{h}d"]).all())
        checks[f"h{h}_target_after_anchor"] = bool((f[f"target_date_{h}d"] > f.anchor_date).all())
    for name in SIGNALS:
        for sp in ("train", "val", "test"):
            s = sig.merge(keys_all, on=["ticker", "anchor_date"])
            checks[f"{name}_nan_{sp}"] = int(s.loc[s.split == sp, name].isna().sum())
    if not all(v for k, v in checks.items() if k.endswith(("_eq_meta", "_eq_price_snapshot", "_eq_price_calendar", "_after_anchor"))):
        write_json(out / "a1_integrity_checks.json", checks)
        raise SystemExit("BLOCKING: label / calendar integrity check failed")

    mse_rows, rank_rows_, fit_rows, ic_rows = [], [], [], []
    for h in HORIZONS:
        fh = fam[h]
        test = fh[fh.split == "test"].sort_values(["anchor_date", "ticker"], kind="mergesort").reset_index(drop=True)
        keys = test[["ticker", "anchor_date"]]
        y = test.y_true.to_numpy(np.float64)
        refs, gtm = reference_forecasts(fh, meta, keys, h)
        # full-history expanding ticker mean (sensitivity)
        mh = meta[["ticker", f"target_date_{h}d", f"y_{h}d"]]
        refs["expanding_ticker_fullhist"] = expanding_mean(mh.ticker.to_numpy(), mh[f"target_date_{h}d"].to_numpy(), mh[f"y_{h}d"].to_numpy(),
                                                           keys.ticker.to_numpy(), keys.anchor_date.to_numpy(), pooled=False)
        checks[f"h{h}_global_train_mean"] = gtm
        preds = {k: v for k, v in refs.items()}

        # --- signals
        s_train = fh[fh.split == "train"].merge(sig, on=["ticker", "anchor_date"], validate="one_to_one")
        s_test = test.merge(sig, on=["ticker", "anchor_date"], validate="one_to_one")
        for name in SIGNALS:
            st = s_test[name].to_numpy(np.float64)
            if not np.isfinite(st).all():
                raise SystemExit(f"BLOCKING: {name} undefined on a TEST row")
            S, dates, tick = to_panel(s_test.assign(v=st), "v")
            Y, _, _ = to_panel(s_test, "y_true")
            r, ic, spread = ranking_stats(S, Y, h)
            rank_rows_.append(dict(h=h, signal=name, **r))
            ic_rows.append(pd.DataFrame({"h": h, "signal": name, "anchor_date": dates, "rank_ic": ic, "top3_minus_bottom3": spread}))
            ok = s_train[name].notna().to_numpy()
            X = np.column_stack([np.ones(ok.sum()), s_train.loc[ok, name].to_numpy(np.float64)])
            yt = s_train.loc[ok, "y_true"].to_numpy(np.float64)
            beta, *_ = np.linalg.lstsq(X, yt, rcond=None)
            resid = yt - X @ beta
            fit_rows.append(dict(h=h, signal=name, n_train_used=int(ok.sum()), n_train_dropped_nan=int((~ok).sum()),
                                 intercept=beta[0], slope=beta[1], train_r2=1 - resid @ resid / np.sum((yt - yt.mean()) ** 2)))
            preds[f"ols_{name}"] = beta[0] + beta[1] * st

        mse_gtm = np.mean((y - refs["global_train_mean"]) ** 2)
        mse_zero = np.mean(y ** 2)
        for name, p in preds.items():
            mse = float(np.mean((y - p) ** 2))
            mse_rows.append(dict(h=h, benchmark=name, mse=mse, r2os_vs_global_train_mean=1 - mse / mse_gtm,
                                 r2os_vs_zero=1 - mse / mse_zero, pred_mean=float(np.mean(p)), pred_sd=float(np.std(p))))
            if name in ("zero", "global_train_mean"):
                continue  # archived naive predictions are used for these two
            cell = test[["ticker", "anchor_date"]].copy()
            cell["y_true"] = test.y_true.to_numpy(np.float32)
            cell["y_pred"] = np.asarray(p, np.float64)
            write_parquet(bdir / f"bench_{name}_h{h}_sna_test.parquet", cell)

    write_json(out / "a1_integrity_checks.json", checks)
    write_csv(out / "a1_benchmark_mse.csv", pd.DataFrame(mse_rows))
    write_csv(out / "a1_signal_ranking.csv", pd.DataFrame(rank_rows_))
    write_csv(out / "a1_signal_ols_fits.csv", pd.DataFrame(fit_rows))
    write_csv(out / "a1_signal_ic_by_date.csv", pd.concat(ic_rows, ignore_index=True))
    print(pd.DataFrame(mse_rows).to_string())
    print(pd.DataFrame(rank_rows_).to_string())
    print(pd.DataFrame(fit_rows).to_string())
    print(json.dumps(checks, indent=1, default=str))


if __name__ == "__main__":
    main()
