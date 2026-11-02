"""
Shared panel loader + chronological split + train-only normalization + metric
calculator for the accepted FINAL V2 17-stock Daily+Weekly panel.

Formal tickers: 17
  AES, BEP, BLDP, BLNK, CSIQ, CWEN, DQ, ENPH, FCEL, FSLR, HASI, JKS,
  NEE, ORA, PLUG, RUN, SEDG

Formal common anchor dates: 2563

Formal split (post-purge, final V2 contract):
  Train: 1802 dates, 30634 samples, 2016-01-29 to 2023-03-27
  Validation: 370 dates, 6290 samples, 2023-04-26 to 2024-10-14
  Test: 391 dates, 6647 samples, 2024-11-12 to 2026-06-05

Final 20 common Train dates and final 20 common Validation dates are purged for
MAX_HORIZON=20 so the 20-day target never leaks across split boundaries.

This module is the authoritative frozen invariant for all downstream models.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_universe_duration_tradeoff import (
    load_all_tickers,
    build_tradeoff_curve,
    compute_split_for_dates,
)

FROZEN_N_TICKERS = 17
FROZEN_TICKERS = [
    "AES","BEP","BLDP","BLNK","CSIQ","CWEN","DQ","ENPH",
    "FCEL","FSLR","HASI","JKS","NEE","ORA","PLUG","RUN","SEDG",
]
FEATURES = ["open", "high", "low", "close", "volume"]
HORIZONS = [5, 10, 20]
MAX_HORIZON = max(HORIZONS)
FREQUENCY_SETTINGS = ["daily_only", "weekly_only", "daily_weekly"]

AUDIT_DIR = ROOT / "dataset" / "audit"


@dataclass
class PanelDataset:
    ticker: np.ndarray
    anchor_date: np.ndarray
    split: np.ndarray
    X_daily: np.ndarray
    X_weekly: np.ndarray
    y: dict  # horizon (int) -> np.ndarray


def load_frozen_panel() -> PanelDataset:
    """Load the frozen 17-stock panel with the frozen common-date-only 70/15/15 split."""
    records = load_all_tickers()
    _, step_detail = build_tradeoff_curve(records)
    if FROZEN_N_TICKERS not in step_detail:
        raise RuntimeError(f"n_tickers={FROZEN_N_TICKERS} not found in the removal curve.")

    retained = sorted(step_detail[FROZEN_N_TICKERS]["retained"])
    if retained != sorted(FROZEN_TICKERS):
        raise RuntimeError(f"Formal frozen universe mismatch: retained={retained}; expected={sorted(FROZEN_TICKERS)}")
    common_dates = step_detail[FROZEN_N_TICKERS]["common_dates"]
    split_info = compute_split_for_dates(common_dates)

    # Final label purging: remove the last MAX_HORIZON anchor dates from train and
    # validation so that the 20-day target for a retained training sample cannot
    # already cross into the next split. The same common-date universe is retained
    # across all horizons (5d / 10d / 20d), keeping the panel directly comparable.
    train_dates_full = list(split_info["train_dates"])
    val_dates_full = list(split_info["val_dates"])
    test_dates_full = list(split_info["test_dates"])

    train_dates = set(train_dates_full[:-MAX_HORIZON] if len(train_dates_full) > MAX_HORIZON else [])
    val_dates = set(val_dates_full[:-MAX_HORIZON] if len(val_dates_full) > MAX_HORIZON else [])
    test_dates = set(test_dates_full)

    tickers_out, dates_out, split_out = [], [], []
    X_daily_out, X_weekly_out = [], []
    y_out = {h: [] for h in HORIZONS}

    for ticker in retained:
        rec = records[ticker]
        dates = pd.to_datetime(rec["anchor_date_norm"])
        for i, d in enumerate(dates):
            if d in train_dates:
                s = "train"
            elif d in val_dates:
                s = "val"
            elif d in test_dates:
                s = "test"
            else:
                continue  # outside the frozen common-date window
            tickers_out.append(ticker)
            dates_out.append(d)
            split_out.append(s)
            X_daily_out.append(rec["X_daily"][i])
            X_weekly_out.append(rec["X_weekly"][i])
            y_out[5].append(rec["y_5d"][i])
            y_out[10].append(rec["y_10d"][i])
            y_out[20].append(rec["y_20d"][i])

    return PanelDataset(
        ticker=np.array(tickers_out),
        anchor_date=np.array(dates_out),
        split=np.array(split_out),
        X_daily=np.asarray(X_daily_out, dtype=np.float32),
        X_weekly=np.asarray(X_weekly_out, dtype=np.float32),
        y={h: np.asarray(v, dtype=np.float32) for h, v in y_out.items()},
    )


def fit_train_only_normalizer(ds: PanelDataset) -> dict:
    """Per-ticker, per-frequency, per-feature train-only mean/std.
    Returns {(ticker, 'daily'|'weekly'): (mean[5], std[5])}."""
    stats = {}
    train_mask = ds.split == "train"
    for ticker in np.unique(ds.ticker):
        t_mask = (ds.ticker == ticker) & train_mask
        for freq_name, X in (("daily", ds.X_daily), ("weekly", ds.X_weekly)):
            X_train = X[t_mask]  # (n, window, n_features)
            flat = X_train.reshape(-1, X_train.shape[-1])
            mean = flat.mean(axis=0)
            std = flat.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)  # avoid divide-by-zero on constant columns
            stats[(ticker, freq_name)] = (mean, std)
    return stats


def apply_normalizer(ds: PanelDataset, stats: dict, frequency: str):
    """Returns normalized X (or a (daily, weekly) tuple for 'daily_weekly')."""

    def normalize_one(X, freq_name):
        out = np.empty_like(X)
        for ticker in np.unique(ds.ticker):
            mask = ds.ticker == ticker
            mean, std = stats[(ticker, freq_name)]
            out[mask] = (X[mask] - mean) / std
        return out

    if frequency == "daily_only":
        return normalize_one(ds.X_daily, "daily")
    if frequency == "weekly_only":
        return normalize_one(ds.X_weekly, "weekly")
    if frequency == "daily_weekly":
        return normalize_one(ds.X_daily, "daily"), normalize_one(ds.X_weekly, "weekly")
    raise ValueError(f"Unknown frequency setting: {frequency}")


def flatten_features(frequency: str, normalized) -> np.ndarray:
    if frequency in ("daily_only", "weekly_only"):
        X = normalized
        return X.reshape(X.shape[0], -1)
    d, w = normalized
    return np.concatenate([d.reshape(d.shape[0], -1), w.reshape(w.shape[0], -1)], axis=1)


def safe_pearson_corr(x, y) -> float:
    """Pearson correlation, explicitly NaN (not a numpy warning/0) when either
    series has ~zero variance (e.g. a constant predictor such as the zero or
    train_mean naive baselines) or fewer than 2 observations."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_ic_by_date(y_true, y_pred, ticker, anchor_date) -> pd.DataFrame:
    """Proper cross-sectional Rank IC: for each anchor_date, the Spearman rank
    correlation between predictions and realized returns across tickers on
    that date. Returns one row per date with n_tickers and rank_ic (NaN if
    undefined for that date — fewer than 3 tickers or zero cross-sectional
    variance in either series). Do not average this into a single pooled
    Spearman correlation elsewhere; average the per-date `rank_ic` column."""
    df = pd.DataFrame({"date": anchor_date, "ticker": ticker, "y_true": y_true, "y_pred": y_pred})
    rows = []
    for date, g in df.groupby("date"):
        n = len(g)
        if n < 3 or g["y_true"].std() < 1e-12 or g["y_pred"].std() < 1e-12:
            rows.append({"anchor_date": date, "n_tickers": n, "rank_ic": float("nan")})
            continue
        ic, _ = spearmanr(g["y_pred"], g["y_true"])
        rows.append({"anchor_date": date, "n_tickers": n, "rank_ic": float(ic) if not np.isnan(ic) else float("nan")})
    return pd.DataFrame(rows).sort_values("anchor_date").reset_index(drop=True)


def cross_sectional_rank_ic(y_true, y_pred, ticker, anchor_date) -> float:
    """Spearman rank correlation computed within each anchor_date across
    tickers, then averaged over dates — NOT a pooled time-series correlation."""
    ic_df = rank_ic_by_date(y_true, y_pred, ticker, anchor_date)
    valid = ic_df["rank_ic"].dropna()
    return float(valid.mean()) if len(valid) else float("nan")


def evaluate_predictions(y_true, y_pred, ticker, anchor_date) -> dict:
    """Pooled + ticker-averaged metrics, plus cross-sectional Rank IC."""
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    corr = safe_pearson_corr(y_pred, y_true)
    da = float((np.sign(y_pred) == np.sign(y_true)).mean())
    pred_std = float(np.std(y_pred))
    true_std = float(np.std(y_true))
    ratio = pred_std / true_std if true_std > 1e-12 else float("nan")
    rank_ic = cross_sectional_rank_ic(y_true, y_pred, ticker, anchor_date)

    df = pd.DataFrame({"ticker": ticker, "y_true": y_true, "y_pred": y_pred})
    per_ticker_rows = []
    for _, g in df.groupby("ticker"):
        if len(g) < 2:
            continue
        d = g["y_pred"].values - g["y_true"].values
        per_ticker_rows.append({
            "mse": np.mean(d ** 2),
            "mae": np.mean(np.abs(d)),
            "corr": safe_pearson_corr(g["y_pred"].values, g["y_true"].values),
            "da": (np.sign(g["y_pred"]) == np.sign(g["y_true"])).mean(),
            "pred_std": g["y_pred"].std(),
            "true_std": g["y_true"].std(),
        })
    pt = pd.DataFrame(per_ticker_rows)

    return {
        "pooled_mse": mse,
        "pooled_mae": mae,
        "pooled_corr": corr,
        "pooled_da": da,
        "pooled_pred_std": pred_std,
        "pooled_true_std": true_std,
        "pooled_pred_std_over_true_std": ratio,
        "cross_sectional_rank_ic": rank_ic,
        "ticker_avg_mse": float(pt["mse"].mean()),
        "ticker_avg_mae": float(pt["mae"].mean()),
        "ticker_avg_corr": float(pt["corr"].mean()),
        "ticker_avg_da": float(pt["da"].mean()),
        "ticker_avg_pred_std": float(pt["pred_std"].mean()),
        "ticker_avg_true_std": float(pt["true_std"].mean()),
    }
