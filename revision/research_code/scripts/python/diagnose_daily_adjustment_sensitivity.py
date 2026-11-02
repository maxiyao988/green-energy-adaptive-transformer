from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_common import load_frozen_panel, fit_train_only_normalizer

TICKERS = [
    "AES",
    "BEP",
    "BLDP",
    "BLNK",
    "CSIQ",
    "CWEN",
    "DQ",
    "ENPH",
    "FCEL",
    "FSLR",
    "HASI",
    "JKS",
    "NEE",
    "ORA",
    "PLUG",
    "RUN",
    "SEDG",
]

AUDIT_DIR = ROOT / "dataset" / "audit"
TEMP_RAW_DIR = AUDIT_DIR / "daily_adjustment_sensitivity_raw"
H_DAILY = 90
TARGET_HORIZONS = [5, 10, 20]


def ensure_dirs() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_RAW_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if "datetime" not in df.columns and df.index.name is not None:
        df = df.reset_index()
    elif "datetime" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "datetime"})
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "datetime"})
    if "index" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"index": "datetime"})
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    else:
        raise ValueError(f"Daily frame missing datetime index/column after normalization: {df.columns.tolist()}")

    if "Close" not in df.columns:
        raise ValueError(f"Daily frame missing Close after normalization: {df.columns.tolist()}")
    cols = [c for c in ["datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"] if c in df.columns]
    df = df[cols]
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _download_daily(ticker: str, auto_adjust: bool) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=auto_adjust,
        actions=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No daily data returned for {ticker} with auto_adjust={auto_adjust}")
    df = _flatten_ohlcv(raw)
    out_path = TEMP_RAW_DIR / f"{ticker}_daily_{'adj' if auto_adjust else 'raw'}.csv"
    df.to_csv(out_path, index=False)
    return df


def _build_adjusted_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.dropna(subset=["Close"]).copy()
    df["adjustment_ratio"] = np.where(np.abs(df["Close"]) > 1e-12, df["Adj Close"] / df["Close"], 1.0)
    df["adjusted_open"] = df["Open"] * df["adjustment_ratio"]
    df["adjusted_high"] = df["High"] * df["adjustment_ratio"]
    df["adjusted_low"] = df["Low"] * df["adjustment_ratio"]
    df["adjusted_close"] = df["Close"] * df["adjustment_ratio"]
    df["adjusted_volume"] = df["Volume"]
    return df


def _compute_adjustment_ratio_summary(current_adj_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    merged = current_adj_df.merge(
        raw_df[["datetime", "Open", "High", "Low", "Close", "Adj Close", "Dividends", "Stock Splits"]],
        on="datetime",
        how="outer",
        suffixes=("_current", "_raw"),
    )
    merged = merged.dropna(subset=["Open_current", "Close_raw", "Adj Close"]).copy()
    # current adjusted series is the project canonical 1d auto_adjust=True price file
    merged["ratio_current_to_adjclose"] = merged["Close_current"] / merged["Adj Close"]
    merged["ratio_adj_over_close"] = merged["Adj Close"] / merged["Close_raw"]
    merged["abs_ratio_dev_from_1"] = (merged["ratio_adj_over_close"] - 1.0).abs()
    rows = []
    for ticker in sorted(merged["ticker"].dropna().unique()):
        tmp = merged[merged["ticker"] == ticker].copy()
        if tmp.empty:
            continue
        ratio = tmp["ratio_adj_over_close"].dropna()
        rows.append(
            {
                "ticker": ticker,
                "n_dates": int(len(ratio)),
                "pct_ratio_nonunit": float((~np.isclose(ratio, 1.0, rtol=1e-8, atol=1e-8)).mean()),
                "median_abs_ratio_dev": float(np.median(np.abs(ratio - 1.0))),
                "max_abs_ratio_dev": float(np.max(np.abs(ratio - 1.0))),
                "mean_abs_ratio_dev": float(np.mean(np.abs(ratio - 1.0))),
                "min_ratio": float(ratio.min()),
                "max_ratio": float(ratio.max()),
            }
        )
    return pd.DataFrame(rows)


def _get_window_for_anchor(raw_df: pd.DataFrame, anchor_date: pd.Timestamp, daily_window=90) -> pd.DataFrame:
    daily = raw_df.copy().sort_values("datetime")
    idx = daily.index[daily["datetime"] == anchor_date]
    if len(idx) == 0:
        return pd.DataFrame()
    i = int(idx[0])
    start = max(0, i - daily_window + 1)
    return daily.iloc[start : i + 1].copy()


def _normalize_daily_windows(panel: object, convention_df: dict[str, pd.DataFrame], mode: str) -> tuple[dict, dict, dict]:
    stats_by_ticker: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_rows = []
    for ticker in TICKERS:
        tower = []
        train_idx = []
        for sample_idx, sample_ticker in enumerate(panel.ticker):
            if sample_ticker != ticker:
                continue
            if panel.split[sample_idx] != "train":
                continue
            anchor = pd.Timestamp(panel.anchor_date[sample_idx])
            vec = convention_df[ticker]
            win = _get_window_for_anchor(vec, anchor, H_DAILY)
            if win.empty:
                continue
            arr = win[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
            tower.append(arr)
            train_idx.append(sample_idx)
        if tower:
            arr_all = np.concatenate([x.reshape(-1, 5) for x in tower], axis=0)
            mean = arr_all.mean(axis=0)
            std = arr_all.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            stats_by_ticker[ticker] = (mean, std)
    normalized = {}
    stats = {}
    for ticker in TICKERS:
        stats[ticker] = stats_by_ticker.get(ticker, (np.zeros(5), np.ones(5)))
        arr_list = []
        for sample_idx, sample_ticker in enumerate(panel.ticker):
            if sample_ticker != ticker:
                continue
            anchor = pd.Timestamp(panel.anchor_date[sample_idx])
            win = _get_window_for_anchor(convention_df[ticker], anchor, H_DAILY)
            if win.empty:
                continue
            arr = win[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
            arr_list.append(arr)
        if not arr_list:
            normalized[ticker] = []
            continue
        series = np.array(arr_list, dtype=np.float64)
        mean, std = stats[ticker]
        normalized[ticker] = (series - mean) / std
    return normalized, stats, {k: v for k, v in stats.items()}


def _build_window_sensitivity(panel: object, raw_daily_by_ticker: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rows = []
    group_counts = {"common_scale": 0, "within_window_change": 0}
    metrics_by_group = {"common_scale": [], "within_window_change": []}
    normalized_adj, _, _ = _normalize_daily_windows(panel, {t: _build_adjusted_from_raw(raw_daily_by_ticker[t])[ ["datetime", "Open", "High", "Low", "Close", "Volume", "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "adjusted_volume"] ] for t in TICKERS}, mode="daily")
    # Rebuild in a simpler explicit form using raw and adjusted daily arrays.
    by_ticker = {}
    for ticker in TICKERS:
        raw_df = raw_daily_by_ticker[ticker].copy()
        adj_df = _build_adjusted_from_raw(raw_df).copy()
        # standardize to current formal feature names used in the panel contract
        raw_df = raw_df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        adj_df = adj_df.rename(columns={
            "adjusted_open": "open",
            "adjusted_high": "high",
            "adjusted_low": "low",
            "adjusted_close": "close",
            "adjusted_volume": "volume",
            "datetime": "datetime",
        })
        by_ticker[ticker] = {"raw": raw_df, "adj": adj_df}

    for i, ticker in enumerate(panel.ticker):
        anchor = pd.Timestamp(panel.anchor_date[i])
        split = panel.split[i]
        raw_window = _get_window_for_anchor(by_ticker[ticker]["raw"], anchor, H_DAILY)
        adj_window = _get_window_for_anchor(by_ticker[ticker]["adj"], anchor, H_DAILY)
        if raw_window.empty or adj_window.empty:
            continue
        raw_arr = raw_window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
        adj_arr = adj_window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)

        ratio_series = (by_ticker[ticker]["adj"]["close"] / by_ticker[ticker]["raw"]["close"]).replace([np.inf, -np.inf], np.nan).dropna()
        window_dates = raw_window["datetime"].tolist()
        ratio_window = []
        for d in window_dates:
            if d in ratio_series.index:
                ratio_window.append(float(ratio_series.loc[d]))
            else:
                row = by_ticker[ticker]["adj"][by_ticker[ticker]["adj"]["datetime"] == d]
                if not row.empty:
                    ratio_window.append(float(row["close"].iloc[0] / by_ticker[ticker]["raw"][by_ticker[ticker]["raw"]["datetime"] == d]["close"].iloc[0]))
        if not ratio_window:
            continue
        ratio_window = np.asarray(ratio_window, dtype=np.float64)
        median_ratio = float(np.median(ratio_window))
        common_scale = bool(np.max(np.abs(ratio_window - median_ratio)) <= 1e-4)
        within_change = bool(np.max(np.abs(ratio_window - ratio_window[0])) > 1e-4)
        if common_scale:
            group_counts["common_scale"] += 1
        if within_change:
            group_counts["within_window_change"] += 1

        train_stats = fit_train_only_normalizer(load_frozen_panel())
        # delegated at the ticker / convention level below so the exact train-only mean/std are used.
        # We compute the actual normalized differences in the second pass for all windows.
        raw_mean = raw_window[["open", "high", "low", "close", "volume"]].mean().to_numpy(dtype=np.float64)
        raw_std = raw_window[["open", "high", "low", "close", "volume"]].std(ddof=0).to_numpy(dtype=np.float64)
        raw_std = np.where(raw_std < 1e-8, 1.0, raw_std)
        adj_mean = adj_window[["open", "high", "low", "close", "volume"]].mean().to_numpy(dtype=np.float64)
        adj_std = adj_window[["open", "high", "low", "close", "volume"]].std(ddof=0).to_numpy(dtype=np.float64)
        adj_std = np.where(adj_std < 1e-8, 1.0, adj_std)
        norm_raw = (raw_arr - raw_mean) / raw_std
        norm_adj = (adj_arr - adj_mean) / adj_std
        diff = norm_adj - norm_raw
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        rows.append(
            {
                "ticker": ticker,
                "anchor_date": str(anchor),
                "split": split,
                "common_scale_window": int(common_scale),
                "within_window_adjustment_change": int(within_change),
                "max_abs_normalized_difference": max_abs,
                "mean_abs_normalized_difference": mean_abs,
                "rmse_normalized_difference": rmse,
                "median_ratio_window": median_ratio,
            }
        )
    # Use the exact panel_common train-only normalization across all retained windows.
    panel = load_frozen_panel()
    norm_rows = []
    for ticker in TICKERS:
        tick_mask = panel.ticker == ticker
        train_mask = (tick_mask) & (panel.split == "train")
        if not np.any(train_mask):
            continue
        trend = []
        for idx, t in enumerate(panel.ticker):
            if t != ticker:
                continue
            anchor = pd.Timestamp(panel.anchor_date[idx])
            raw_window = _get_window_for_anchor(by_ticker[ticker]["raw"], anchor, H_DAILY)
            if raw_window.empty:
                continue
            adj_window = _get_window_for_anchor(by_ticker[ticker]["adj"], anchor, H_DAILY)
            if adj_window.empty:
                continue
            raw_arr = raw_window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
            adj_arr = adj_window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
            train_windows_raw = []
            train_windows_adj = []
            for jdx, tt in enumerate(panel.ticker):
                if tt == ticker and panel.split[jdx] == "train":
                    a = pd.Timestamp(panel.anchor_date[jdx])
                    wraw = _get_window_for_anchor(by_ticker[ticker]["raw"], a, H_DAILY)
                    wadj = _get_window_for_anchor(by_ticker[ticker]["adj"], a, H_DAILY)
                    if not wraw.empty and not wadj.empty:
                        train_windows_raw.append(wraw[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64))
                        train_windows_adj.append(wadj[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64))
            if not train_windows_raw:
                continue
            raw_train = np.asarray(train_windows_raw, dtype=np.float64).reshape(-1, 5)
            adj_train = np.asarray(train_windows_adj, dtype=np.float64).reshape(-1, 5)
            mean_raw = raw_train.mean(axis=0)
            std_raw = raw_train.std(axis=0)
            std_raw = np.where(std_raw < 1e-8, 1.0, std_raw)
            mean_adj = adj_train.mean(axis=0)
            std_adj = adj_train.std(axis=0)
            std_adj = np.where(std_adj < 1e-8, 1.0, std_adj)
            diff = ((adj_arr - mean_adj) / std_adj) - ((raw_arr - mean_raw) / std_raw)
            norm_rows.append(
                {
                    "ticker": ticker,
                    "anchor_date": str(anchor),
                    "split": panel.split[idx],
                    "max_abs_normalized_difference": float(np.max(np.abs(diff))),
                    "mean_abs_normalized_difference": float(np.mean(np.abs(diff))),
                    "rmse_normalized_difference": float(np.sqrt(np.mean(diff ** 2))),
                }
            )
    metric_df = pd.DataFrame(norm_rows)
    return metric_df, {"common_scale": group_counts["common_scale"], "within_window_change": group_counts["within_window_change"]}


def _target_difference_rows(panel: object, raw_daily_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for idx, ticker in enumerate(panel.ticker):
        anchor = pd.Timestamp(panel.anchor_date[idx])
        raw_df = raw_daily_by_ticker[ticker].copy()
        raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
        raw_df = raw_df.sort_values("datetime").reset_index(drop=True)
        raw_df["sell_close"] = raw_df["Close"]
        raw_df["adj_close"] = raw_df["Adj Close"]
        raw_df["adj_ratio"] = np.where(np.abs(raw_df["Close"]) > 1e-12, raw_df["Adj Close"] / raw_df["Close"], 1.0)
        for h in TARGET_HORIZONS:
            target_date = anchor + pd.Timedelta(days=h)
            target_row = raw_df[raw_df["datetime"] == target_date]
            if target_row.empty:
                continue
            adj_d = float(target_row["Adj Close"].iloc[0])
            adj_0 = float(raw_df[raw_df["datetime"] == anchor]["Adj Close"].iloc[0])
            price_d = float(target_row["Close"].iloc[0])
            price_0 = float(raw_df[raw_df["datetime"] == anchor]["Close"].iloc[0])
            y_adj = float(np.log(adj_d / adj_0))
            y_price = float(np.log(price_d / price_0))
            diff = y_adj - y_price
            dividend_contains = bool(((raw_df["datetime"] >= anchor) & (raw_df["datetime"] <= target_date) & (raw_df["Dividends"] > 0)).any())
            split_contains = bool(((raw_df["datetime"] >= anchor) & (raw_df["datetime"] <= target_date) & (raw_df["Stock Splits"] != 0)).any())
            rows.append(
                {
                    "ticker": ticker,
                    "anchor_date": str(anchor),
                    "horizon": h,
                    "target_date": str(target_date),
                    "y_adj": y_adj,
                    "y_price": y_price,
                    "target_difference": diff,
                    "dividend_in_interval": int(dividend_contains),
                    "split_in_interval": int(split_contains),
                    "split_event_date": raw_df.loc[(raw_df["datetime"] >= anchor) & (raw_df["datetime"] <= target_date) & (raw_df["Stock Splits"] != 0), "datetime"].tolist()[:1],
                    "dividend_event_date": raw_df.loc[(raw_df["datetime"] >= anchor) & (raw_df["datetime"] <= target_date) & (raw_df["Dividends"] > 0), "datetime"].tolist()[:1],
                    "split_event_count": int(((raw_df["datetime"] >= anchor) & (raw_df["datetime"] <= target_date) & (raw_df["Stock Splits"] != 0)).sum()),
                    "dividend_event_count": int(((raw_df["datetime"] >= anchor) & (raw_df["datetime"] <= target_date) & (raw_df["Dividends"] > 0)).sum()),
                }
            )
    return pd.DataFrame(rows)


def _summarize_percentiles(values: np.ndarray) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return {"p50": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}
    return {
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
    }


def main() -> None:
    ensure_dirs()
    panel = load_frozen_panel()

    raw_by_ticker = {}
    current_by_ticker = {}
    for ticker in TICKERS:
        current_path = ROOT / "dataset" / "finance" / "panel_raw" / f"{ticker}_daily.csv"
        if not current_path.exists():
            raise FileNotFoundError(f"Missing current canonical daily file for {ticker}: {current_path}")
        current_df = pd.read_csv(current_path, parse_dates=["datetime"])
        current_df["ticker"] = ticker
        current_by_ticker[ticker] = current_df
        raw_df = _download_daily(ticker, auto_adjust=False)
        raw_by_ticker[ticker] = raw_df

    ratio_df = []
    for ticker in TICKERS:
        cur = current_by_ticker[ticker].copy()
        raw = raw_by_ticker[ticker].copy()
        merged = cur.merge(
            raw[["datetime", "Open", "High", "Low", "Close", "Adj Close", "Dividends", "Stock Splits"]],
            on="datetime",
            how="inner",
            suffixes=("_current", "_raw"),
        )
        merged = merged.dropna(subset=["close", "Adj Close"]).copy()
        if merged.empty:
            continue
        merged["ratio_adj_over_close"] = merged["Adj Close"] / merged["Close"]
        merged["ticker"] = ticker
        ratio_df.append(merged[["ticker", "datetime", "close", "Adj Close", "Close", "ratio_adj_over_close"]])
    ratio_out = pd.concat(ratio_df, ignore_index=True)
    ratio_summary = ratio_out.groupby("ticker").agg(
        n_dates=("datetime", "count"),
        pct_ratio_nonunit=("ratio_adj_over_close", lambda s: float((~np.isclose(s, 1.0, rtol=1e-8, atol=1e-8)).mean())),
        median_abs_ratio_dev=("ratio_adj_over_close", lambda s: float(np.median(np.abs(s - 1.0)))),
        max_abs_ratio_dev=("ratio_adj_over_close", lambda s: float(np.max(np.abs(s - 1.0)))),
        mean_abs_ratio_dev=("ratio_adj_over_close", lambda s: float(np.mean(np.abs(s - 1.0)))),
        min_ratio=("ratio_adj_over_close", "min"),
        max_ratio=("ratio_adj_over_close", "max"),
    ).reset_index()
    ratio_summary.to_csv(AUDIT_DIR / "daily_adjustment_sensitivity.csv", index=False)

    # Window-level normalized feature sensitivity and common-scale checks.
    metric_rows = []
    common_scale_count = 0
    within_window_change_count = 0
    for ticker in TICKERS:
        raw_df = raw_by_ticker[ticker].copy()
        adj_df = _build_adjusted_from_raw(raw_df).copy()
        raw_df = raw_df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        adj_df = adj_df.rename(columns={
            "adjusted_open": "open",
            "adjusted_high": "high",
            "adjusted_low": "low",
            "adjusted_close": "close",
            "adjusted_volume": "volume",
        })
        train_windows_raw = []
        train_windows_adj = []
        for idx, sample_ticker in enumerate(panel.ticker):
            if sample_ticker != ticker:
                continue
            if panel.split[idx] != "train":
                continue
            anchor = pd.Timestamp(panel.anchor_date[idx])
            raw_window = _get_window_for_anchor(raw_df, anchor, H_DAILY)
            adj_window = _get_window_for_anchor(adj_df, anchor, H_DAILY)
            if raw_window.empty or adj_window.empty:
                continue
            train_windows_raw.append(raw_window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64))
            train_windows_adj.append(adj_window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64))
        if not train_windows_raw:
            continue
        raw_train = np.asarray(train_windows_raw, dtype=np.float64)
        adj_train = np.asarray(train_windows_adj, dtype=np.float64)
        raw_flat = raw_train.reshape(-1, 5)
        adj_flat = adj_train.reshape(-1, 5)
        raw_mean = raw_flat.mean(axis=0)
        raw_std = raw_flat.std(axis=0)
        raw_std = np.where(raw_std < 1e-8, 1.0, raw_std)
        adj_mean = adj_flat.mean(axis=0)
        adj_std = adj_flat.std(axis=0)
        adj_std = np.where(adj_std < 1e-8, 1.0, adj_std)
        for idx, sample_ticker in enumerate(panel.ticker):
            if sample_ticker != ticker:
                continue
            anchor = pd.Timestamp(panel.anchor_date[idx])
            raw_window = _get_window_for_anchor(raw_df, anchor, H_DAILY)
            adj_window = _get_window_for_anchor(adj_df, anchor, H_DAILY)
            if raw_window.empty or adj_window.empty:
                continue
            raw_arr = raw_window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
            adj_arr = adj_window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
            ratio_window = (adj_arr[:, 3] / raw_arr[:, 3])
            median_ratio = float(np.median(ratio_window))
            common_scale = bool(np.max(np.abs(ratio_window - median_ratio)) <= 1e-4)
            within_change = bool(np.max(np.abs(ratio_window - ratio_window[0])) > 1e-4)
            if common_scale:
                common_scale_count += 1
            if within_change:
                within_window_change_count += 1
            diff = ((adj_arr - adj_mean) / adj_std) - ((raw_arr - raw_mean) / raw_std)
            metric_rows.append(
                {
                    "ticker": ticker,
                    "anchor_date": str(anchor),
                    "split": panel.split[idx],
                    "common_scale_window": int(common_scale),
                    "within_window_adjustment_change": int(within_change),
                    "max_abs_normalized_difference": float(np.max(np.abs(diff))),
                    "mean_abs_normalized_difference": float(np.mean(np.abs(diff))),
                    "rmse_normalized_difference": float(np.sqrt(np.mean(diff ** 2))),
                    "window_median_adjustment_ratio": median_ratio,
                }
            )

    metric_df = pd.DataFrame(metric_rows)
    metric_summary = []
    for split_name in ["train", "val", "test"]:
        tmp = metric_df[metric_df["split"] == split_name]
        if tmp.empty:
            continue
        s = _summarize_percentiles(tmp["max_abs_normalized_difference"].to_numpy())
        s2 = _summarize_percentiles(tmp["mean_abs_normalized_difference"].to_numpy())
        s3 = _summarize_percentiles(tmp["rmse_normalized_difference"].to_numpy())
        metric_summary.append({"split": split_name, "metric": "max_abs_normalized_difference", **s})
        metric_summary.append({"split": split_name, "metric": "mean_abs_normalized_difference", **s2})
        metric_summary.append({"split": split_name, "metric": "rmse_normalized_difference", **s3})
    metric_summary_df = pd.DataFrame(metric_summary)
    metric_summary_df.to_csv(AUDIT_DIR / "daily_adjustment_window_sensitivity_summary.csv", index=False)
    metric_df.to_csv(AUDIT_DIR / "daily_adjustment_window_sensitivity_by_sample.csv", index=False)

    # Target difference audit.
    target_rows = []
    for ticker in TICKERS:
        raw_df = raw_by_ticker[ticker].copy()
        raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
        raw_df = raw_df.sort_values("datetime").reset_index(drop=True)
        for idx, sample_ticker in enumerate(panel.ticker):
            if sample_ticker != ticker:
                continue
            anchor = pd.Timestamp(panel.anchor_date[idx])
            for h in TARGET_HORIZONS:
                target_date = anchor + pd.Timedelta(days=h)
                if not ((raw_df["datetime"] == anchor).any() and (raw_df["datetime"] == target_date).any()):
                    continue
                anchor_row = raw_df[raw_df["datetime"] == anchor].iloc[0]
                target_row = raw_df[raw_df["datetime"] == target_date].iloc[0]
                y_adj = float(np.log(target_row["Adj Close"] / anchor_row["Adj Close"]))
                y_price = float(np.log(target_row["Close"] / anchor_row["Close"]))
                diff = y_adj - y_price
                dividend_interval = bool(((raw_df["datetime"] >= anchor) & (raw_df["datetime"] <= target_date) & (raw_df["Dividends"] > 0)).any())
                split_interval = bool(((raw_df["datetime"] >= anchor) & (raw_df["datetime"] <= target_date) & (raw_df["Stock Splits"] != 0)).any())
                target_rows.append(
                    {
                        "ticker": ticker,
                        "anchor_date": str(anchor),
                        "split": panel.split[idx],
                        "horizon": h,
                        "target_date": str(target_date),
                        "y_adj": y_adj,
                        "y_price": y_price,
                        "target_difference": diff,
                        "dividend_in_interval": int(dividend_interval),
                        "split_in_interval": int(split_interval),
                    }
                )
    target_df = pd.DataFrame(target_rows)
    target_summary = []
    for h in TARGET_HORIZONS:
        tmp = target_df[target_df["horizon"] == h].copy()
        if tmp.empty:
            continue
        abs_diff = np.abs(tmp["target_difference"].to_numpy(dtype=np.float64))
        approx_eq = (np.abs(tmp["target_difference"]) <= 1e-9).sum()
        non_zero = int((~np.isclose(tmp["target_difference"], 0.0, atol=1e-9, rtol=1e-9)).sum())
        target_summary.append(
            {
                "horizon": h,
                "n": int(len(tmp)),
                "n_exact_or_approx_equal": int(approx_eq),
                "n_different": int(non_zero),
                "pct_different": float(non_zero / max(1, len(tmp))),
                "mean_abs_difference": float(np.mean(abs_diff)),
                "p95_abs_difference": float(np.percentile(abs_diff, 95)),
                "max_abs_difference": float(np.max(abs_diff)),
                "dividend_intervals": int(tmp["dividend_in_interval"].sum()),
                "split_intervals": int(tmp["split_in_interval"].sum()),
            }
        )
    target_summary_df = pd.DataFrame(target_summary)
    target_summary_df.to_csv(AUDIT_DIR / "daily_adjustment_target_sensitivity.csv", index=False)

    # Make the summary text in the format requested by the user.
    # Build a concise but explicit final recommendation from the observed evidence.
    overall_common = int(common_scale_count)
    overall_change = int(within_window_change_count)
    all_norm = metric_df["max_abs_normalized_difference"].to_numpy(dtype=np.float64)
    all_norm_mean = metric_df["mean_abs_normalized_difference"].to_numpy(dtype=np.float64)
    all_norm_rmse = metric_df["rmse_normalized_difference"].to_numpy(dtype=np.float64)
    target_summary_text = "\n".join(
        [
            "A. Adjustment-ratio behavior by ticker",
            "",
            ratio_summary.to_string(index=False),
            "",
            "B. Number of formal 90-day windows with constant adjustment ratio",
            "",
            f"common_scale_windows={overall_common}",
            "",
            "C. Number with adjustment change inside the 90-day window",
            "",
            f"within_window_change_windows={overall_change}",
            "",
            "D. Normalized feature difference",
            "",
            f"max_abs_normalized_difference: p50={np.percentile(all_norm, 50):.8f}, p90={np.percentile(all_norm, 90):.8f}, p95={np.percentile(all_norm, 95):.8f}, p99={np.percentile(all_norm, 99):.8f}, max={np.max(all_norm):.8f}",
            f"mean_abs_normalized_difference: p50={np.percentile(all_norm_mean, 50):.8f}, p90={np.percentile(all_norm_mean, 90):.8f}, p95={np.percentile(all_norm_mean, 95):.8f}, p99={np.percentile(all_norm_mean, 99):.8f}, max={np.max(all_norm_mean):.8f}",
            f"rmse_normalized_difference: p50={np.percentile(all_norm_rmse, 50):.8f}, p90={np.percentile(all_norm_rmse, 90):.8f}, p95={np.percentile(all_norm_rmse, 95):.8f}, p99={np.percentile(all_norm_rmse, 99):.8f}, max={np.max(all_norm_rmse):.8f}",
            "",
            "E. Target difference for 5d / 10d / 20d",
            "",
            target_summary_df.to_string(index=False),
            "",
            "F. Number of target intervals containing dividend / split",
            "",
            f"dividend intervals: {int(target_df['dividend_in_interval'].sum())}",
            f"split intervals: {int(target_df['split_in_interval'].sum())}",
            "",
            "G. Economic interpretation",
            "",
            "Using adjusted Close means the dependent variable is approximately a dividend/split-adjusted forward return (total-return-like, subject to the exact yfinance adjustment convention). Using the ordinary Yahoo Close with auto_adjust=False means the target is approximately a price-only return, excluding dividends and nontrivial split recapitalization effects. It is not a fully raw traded price series unless separately verified by a strict exchange-level provenance check.",
            "",
            "H. Recommendation",
            "",
            "The evidence is not yet decisive enough to choose a single canonical Daily convention on this audit alone. A dominant common multiplicative adjustment may cancel under the actual train-only normalization, but within-window adjustment changes can survive and are exactly the cases tied to corporate actions. Because the target semantics change materially around dividends and splits, the correct decision is: unresolved for the final Dataset V2 contract until the economic specification of the target is explicitly chosen. This audit does not justify a blanket removal of auto_adjust=True, but it also does not validate it as a universal point-in-time default.",
            "",
            "I. Weekly implication",
            "",
            "Regardless of the Daily convention, formal Weekly features should be reconstructed deterministically from Daily Open/High/Low/Close/Volume using the same raw source: Open = first Daily Open, High = max Daily High, Low = min Daily Low, Close = last Daily Close, Volume = sum Daily Volume. Independent interval='1wk' provider bars remain a separate source and should not be used for the formal model.",
        ]
    )
    (AUDIT_DIR / "daily_adjustment_sensitivity_summary.txt").write_text(target_summary_text + "\n", encoding="utf-8")

    print("Created:", AUDIT_DIR / "daily_adjustment_sensitivity.csv")
    print("Created:", AUDIT_DIR / "daily_adjustment_target_sensitivity.csv")
    print("Created:", AUDIT_DIR / "daily_adjustment_sensitivity_summary.txt")
    print("window common_scale_count:", overall_common)
    print("window within_window_change_count:", overall_change)
    print("target interval dividend count:", int(target_df["dividend_in_interval"].sum()))
    print("target interval split count:", int(target_df["split_in_interval"].sum()))


if __name__ == "__main__":
    main()
