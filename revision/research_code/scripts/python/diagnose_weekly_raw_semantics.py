from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_common import load_frozen_panel

FORMAL_TICKERS = [
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

RAW_DIR = ROOT / "dataset" / "finance" / "panel_raw"
AUDIT_DIR = ROOT / "dataset" / "audit"

OHLC_FIELDS = ["open", "high", "low", "close"]
VOLUME_TOL = 1e-8
OHLC_TOL = 1e-6
COMMON_SCALE_TOL = 1e-3
REL_TOL = 1e-4


def _iso(x):
    return pd.Timestamp(x).isoformat()


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _safe_relerr(provider: float, agg: float) -> float:
    if pd.isna(provider) or pd.isna(agg) or agg == 0.0:
        return np.nan
    return abs(provider - agg) / abs(agg)


def _classify_mismatch(row: pd.Series) -> str:
    if pd.isna(row["agg_open"]) or pd.isna(row["agg_high"]) or pd.isna(row["agg_low"]) or pd.isna(row["agg_close"]):
        return "NO_DAILY_ROWS"
    vals = [row["ratio_open"], row["ratio_high"], row["ratio_low"], row["ratio_close"]]
    if np.isfinite(vals).all():
        ratio_range = max(vals) - min(vals)
    else:
        ratio_range = np.nan
    if row["volume_mismatch"] and not row["any_ohlc_mismatch"]:
        return "VOLUME_ONLY"
    if row["provider_open"] == row["agg_open"] and row["provider_high"] == row["agg_high"] and row["provider_low"] == row["agg_low"] and row["provider_close"] == row["agg_close"]:
        return "EXACT_MATCH"
    if np.isfinite(ratio_range) and ratio_range <= COMMON_SCALE_TOL:
        return "COMMON_SCALE"
    if row["provider_open"] != row["agg_open"] and row["provider_high"] == row["agg_high"] and row["provider_low"] == row["agg_low"] and row["provider_close"] == row["agg_close"]:
        return "OPEN_ONLY_OR_PARTIAL"
    if row["provider_open"] != row["agg_open"] and row["provider_high"] != row["agg_high"] and row["provider_low"] != row["agg_low"] and row["provider_close"] != row["agg_close"]:
        return "STRUCTURAL_OHLC"
    if row["provider_open"] != row["agg_open"] or row["provider_high"] != row["agg_high"] or row["provider_low"] != row["agg_low"] or row["provider_close"] != row["agg_close"]:
        if row["volume_mismatch"]:
            return "OPEN_ONLY_OR_PARTIAL"
        return "STRUCTURAL_OHLC"
    return "EXACT_MATCH"


def _safe_ratio(provider: float, agg: float) -> float:
    if pd.isna(provider) or pd.isna(agg) or abs(agg) < 1e-12:
        return np.nan
    return provider / agg


def _raw_weekly_availability_for_ticker(ticker: str) -> pd.DataFrame:
    daily = pd.read_csv(RAW_DIR / f"{ticker}_daily.csv")
    weekly = pd.read_csv(RAW_DIR / f"{ticker}_weekly.csv")
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="raise")
    weekly["datetime"] = pd.to_datetime(weekly["datetime"], errors="raise")
    daily = daily.sort_values("datetime").reset_index(drop=True)
    weekly = weekly.sort_values("datetime").reset_index(drop=True)
    out = weekly[["datetime"]].copy()
    out["available_date"] = pd.NaT
    for idx, row in weekly.iterrows():
        week_start = pd.Timestamp(row["datetime"])
        week_end = week_start + pd.Timedelta(days=7)
        week_daily = daily[(daily["datetime"] >= week_start) & (daily["datetime"] < week_end)].copy()
        if week_daily.empty:
            out.at[idx, "available_date"] = pd.NaT
            continue
        out.at[idx, "available_date"] = week_daily["datetime"].max()
    return out


def _check_raw_file_schema(ticker: str) -> dict:
    daily_path = RAW_DIR / f"{ticker}_daily.csv"
    weekly_path = RAW_DIR / f"{ticker}_weekly.csv"
    info = {
        "ticker": ticker,
        "daily_columns": [],
        "weekly_columns": [],
        "daily_rows": 0,
        "weekly_rows": 0,
        "daily_first": None,
        "daily_last": None,
        "weekly_first": None,
        "weekly_last": None,
        "daily_dtypes": {},
        "weekly_dtypes": {},
        "possible_adjusted_cols": [],
    }
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        info["daily_columns"] = list(daily.columns)
        info["daily_rows"] = len(daily)
        info["daily_first"] = daily["datetime"].iloc[0] if "datetime" in daily.columns and len(daily) else None
        info["daily_last"] = daily["datetime"].iloc[-1] if "datetime" in daily.columns and len(daily) else None
        info["daily_dtypes"] = {c: str(daily[c].dtype) for c in daily.columns}
        info["possible_adjusted_cols"] = [c for c in daily.columns if any(k in c.lower() for k in ["adjust", "split", "dividend", "capital", "gain"]) ]
    if weekly_path.exists():
        weekly = pd.read_csv(weekly_path)
        info["weekly_columns"] = list(weekly.columns)
        info["weekly_rows"] = len(weekly)
        info["weekly_first"] = weekly["datetime"].iloc[0] if "datetime" in weekly.columns and len(weekly) else None
        info["weekly_last"] = weekly["datetime"].iloc[-1] if "datetime" in weekly.columns and len(weekly) else None
        info["weekly_dtypes"] = {c: str(weekly[c].dtype) for c in weekly.columns}
        info["possible_adjusted_cols"] = sorted(set(info["possible_adjusted_cols"]) | {c for c in weekly.columns if any(k in c.lower() for k in ["adjust", "split", "dividend", "capital", "gain"]) })
    return info


def _build_diag_rows(ticker: str) -> pd.DataFrame:
    daily = pd.read_csv(RAW_DIR / f"{ticker}_daily.csv")
    weekly = pd.read_csv(RAW_DIR / f"{ticker}_weekly.csv")
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="raise")
    weekly["datetime"] = pd.to_datetime(weekly["datetime"], errors="raise")
    daily = daily.sort_values("datetime").reset_index(drop=True)
    weekly = weekly.sort_values("datetime").reset_index(drop=True)

    rows = []
    for _, w in weekly.iterrows():
        week_start = pd.Timestamp(w["datetime"])
        week_end = week_start + pd.Timedelta(days=7)
        daily_slice = daily[(daily["datetime"] >= week_start) & (daily["datetime"] < week_end)].copy()
        daily_slice = daily_slice.sort_values("datetime").reset_index(drop=True)

        provider = {
            "open": _safe_float(w["open"]),
            "high": _safe_float(w["high"]),
            "low": _safe_float(w["low"]),
            "close": _safe_float(w["close"]),
            "volume": _safe_float(w["volume"]),
        }
        if daily_slice.empty:
            agg = {"open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan, "volume": np.nan}
            daily_first = pd.NaT
            daily_last = pd.NaT
            n_daily = 0
        else:
            daily_first = daily_slice["datetime"].min()
            daily_last = daily_slice["datetime"].max()
            n_daily = len(daily_slice)
            agg = {
                "open": float(daily_slice.iloc[0]["open"]),
                "high": float(daily_slice["high"].max()),
                "low": float(daily_slice["low"].min()),
                "close": float(daily_slice.iloc[-1]["close"]),
                "volume": float(daily_slice["volume"].sum()),
            }

        row = {
            "ticker": ticker,
            "week_start": week_start,
            "daily_first_date": daily_first,
            "daily_last_date": daily_last,
            "n_daily_rows": n_daily,
            "provider_open": provider["open"],
            "provider_high": provider["high"],
            "provider_low": provider["low"],
            "provider_close": provider["close"],
            "provider_volume": provider["volume"],
            "agg_open": agg.get("open", np.nan),
            "agg_high": agg.get("high", np.nan),
            "agg_low": agg.get("low", np.nan),
            "agg_close": agg.get("close", np.nan),
            "agg_volume": agg.get("volume", np.nan),
        }
        for key in ["open", "high", "low", "close", "volume"]:
            row[f"abs_diff_{key}"] = abs(row[f"provider_{key}"] - row[f"agg_{key}"]) if pd.notna(row[f"provider_{key}"]) and pd.notna(row[f"agg_{key}"]) else np.nan
            row[f"relerr_{key}"] = _safe_relerr(row[f"provider_{key}"], row[f"agg_{key}"]) if pd.notna(row[f"provider_{key}"]) and pd.notna(row[f"agg_{key}"]) else np.nan
            row[f"ratio_{key}"] = _safe_ratio(row[f"provider_{key}"], row[f"agg_{key}"]) if pd.notna(row[f"provider_{key}"]) and pd.notna(row[f"agg_{key}"]) else np.nan
            row[f"exact_or_close_{key}"] = bool(np.isclose(row[f"provider_{key}"], row[f"agg_{key}"], rtol=REL_TOL, atol=OHLC_TOL if key in ["open","high","low","close"] else VOLUME_TOL)) if pd.notna(row[f"provider_{key}"]) and pd.notna(row[f"agg_{key}"]) else False
            row[f"{key}_mismatch"] = bool(pd.notna(row[f"provider_{key}"]) and pd.notna(row[f"agg_{key}"]) and not row[f"exact_or_close_{key}"]) if key in ["open", "high", "low", "close"] else bool(pd.notna(row[f"provider_{key}"]) and pd.notna(row[f"agg_{key}"]) and not row[f"exact_or_close_{key}"])
        row["any_ohlc_mismatch"] = any(row[f"{key}_mismatch"] for key in ["open", "high", "low", "close"])
        row["volume_mismatch"] = bool(pd.notna(row["provider_volume"]) and pd.notna(row["agg_volume"]) and not row["exact_or_close_volume"])
        if row["any_ohlc_mismatch"]:
            ratios = [row[f"ratio_{k}"] for k in ["open", "high", "low", "close"] if pd.notna(row[f"ratio_{k}"]) ]
            if ratios:
                row["ratio_mean"] = float(np.mean(ratios))
                row["ratio_std"] = float(np.std(ratios))
                row["ratio_range"] = float(max(ratios) - min(ratios))
                row["common_scale_factor_flag"] = bool(row["ratio_range"] <= COMMON_SCALE_TOL)
                row["inferred_scale_factor"] = float(np.mean(ratios))
            else:
                row["ratio_mean"] = np.nan
                row["ratio_std"] = np.nan
                row["ratio_range"] = np.nan
                row["common_scale_factor_flag"] = False
                row["inferred_scale_factor"] = np.nan
        else:
            row["ratio_mean"] = np.nan
            row["ratio_std"] = np.nan
            row["ratio_range"] = np.nan
            row["common_scale_factor_flag"] = False
            row["inferred_scale_factor"] = np.nan
        row["classification"] = _classify_mismatch(pd.Series(row))
        rows.append(row)

    return pd.DataFrame(rows)


def _summarize_periods(df: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("ALL_HISTORY", None, None),
        (">= 2010-01-01", "2010-01-01", None),
        (">= 2016-01-01", "2016-01-01", None),
        (">= 2020-01-01", "2020-01-01", None),
        (">= 2023-01-01", "2023-01-01", None),
        (">= 2024-01-01", "2024-01-01", None),
    ]
    out = []
    for label, start, end in periods:
        if start is None:
            sub = df.copy()
        else:
            sub = df[df["week_start"] >= pd.Timestamp(start)].copy()
        total = len(sub)
        exact = int((~sub["any_ohlc_mismatch"]).sum())
        ohlc_mismatch = int(sub["any_ohlc_mismatch"].sum())
        volume_mismatch = int(sub["volume_mismatch"].sum())
        common_scale = int(sub[sub["classification"] == "COMMON_SCALE"].shape[0])
        structural = int(sub[sub["classification"] == "STRUCTURAL_OHLC"].shape[0])
        no_daily = int(sub[sub["classification"] == "NO_DAILY_ROWS"].shape[0])
        out.append({
            "period": label,
            "total_weekly_rows": total,
            "exact_match_rows": exact,
            "ohlc_mismatch_rows": ohlc_mismatch,
            "volume_mismatch_rows": volume_mismatch,
            "COMMON_SCALE_rows": common_scale,
            "STRUCTURAL_OHLC_rows": structural,
            "NO_DAILY_ROWS": no_daily,
            "pct_ohlc_mismatch": (ohlc_mismatch / total * 100.0) if total else 0.0,
            "pct_volume_mismatch": (volume_mismatch / total * 100.0) if total else 0.0,
        })
    return pd.DataFrame(out)


def _summarize_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in df.groupby("ticker"):
        total = len(g)
        ohlc_mismatch = int(g["any_ohlc_mismatch"].sum())
        volume_mismatch = int(g["volume_mismatch"].sum())
        common_scale = int((g["classification"] == "COMMON_SCALE").sum())
        structural = int((g["classification"] == "STRUCTURAL_OHLC").sum())
        rels = g["relerr_open"].dropna().tolist() + g["relerr_high"].dropna().tolist() + g["relerr_low"].dropna().tolist() + g["relerr_close"].dropna().tolist()
        rows.append({
            "ticker": ticker,
            "total_weekly_rows": total,
            "ohlc_mismatch_rows": ohlc_mismatch,
            "ohlc_mismatch_pct": (ohlc_mismatch / total * 100.0) if total else 0.0,
            "volume_mismatch_rows": volume_mismatch,
            "COMMON_SCALE_rows": common_scale,
            "STRUCTURAL_OHLC_rows": structural,
            "earliest_mismatch_date": g.loc[g["any_ohlc_mismatch"], "week_start"].min() if ohlc_mismatch else pd.NaT,
            "latest_mismatch_date": g.loc[g["any_ohlc_mismatch"], "week_start"].max() if ohlc_mismatch else pd.NaT,
            "max_ohlc_relative_error": float(np.nanmax(rels)) if len(rels) else np.nan,
            "median_ohlc_relative_error": float(np.nanmedian(rels)) if len(rels) else np.nan,
            "max_ratio_dispersion": float(g.loc[g["any_ohlc_mismatch"], "ratio_range"].max()) if ohlc_mismatch else np.nan,
        })
    out = pd.DataFrame(rows).sort_values(["ohlc_mismatch_rows", "ticker"], ascending=[False, True]).reset_index(drop=True)
    return out


def _formal_relevance(df: pd.DataFrame, formal_panel: object) -> dict:
    usage_rows = []
    for ticker in np.unique(formal_panel.ticker):
        ticker_mask = formal_panel.ticker == ticker
        anchors = pd.to_datetime(formal_panel.anchor_date[ticker_mask])
        weekly_raw = _raw_weekly_availability_for_ticker(ticker)
        weekly_raw["datetime"] = pd.to_datetime(weekly_raw["datetime"])
        weekly_raw["available_date"] = pd.to_datetime(weekly_raw["available_date"])
        for anchor_dt in anchors:
            eligible = weekly_raw[(weekly_raw["available_date"].notna()) & (weekly_raw["available_date"] <= anchor_dt)].copy()
            if eligible.empty or len(eligible) < 26:
                continue
            selected = eligible.sort_values("datetime").tail(26)
            for _, row in selected.iterrows():
                usage_rows.append({
                    "ticker": ticker,
                    "week_start": pd.Timestamp(row["datetime"]),
                    "anchor_date": pd.Timestamp(anchor_dt),
                })

    if not usage_rows:
        df = df.copy()
        df["used_in_formal_panel"] = False
        df["first_formal_anchor_using_bar"] = pd.NaT
        df["last_formal_anchor_using_bar"] = pd.NaT
        df["n_formal_samples_using_bar"] = 0
        return {
            "earliest_week_start_used": pd.NaT,
            "latest_week_start_used": pd.NaT,
            "total_unique_weekly_bars_used": 0,
            "total_raw_weekly_mismatches": int(df["any_ohlc_mismatch"].sum()),
            "mismatches_used": 0,
            "mismatches_never_used": int(df["any_ohlc_mismatch"].sum()),
            "used_by_ticker": pd.DataFrame({"ticker": [], "mismatch_rows_used": []}),
        }

    usage_df = pd.DataFrame(usage_rows)
    usage_summary = (
        usage_df.groupby(["ticker", "week_start"], as_index=False)
        .agg(
            first_formal_anchor_using_bar=("anchor_date", "min"),
            last_formal_anchor_using_bar=("anchor_date", "max"),
            n_formal_samples_using_bar=("anchor_date", "count"),
        )
    )
    df = df.merge(usage_summary[["ticker", "week_start", "first_formal_anchor_using_bar", "last_formal_anchor_using_bar", "n_formal_samples_using_bar"]], on=["ticker", "week_start"], how="left")
    df["used_in_formal_panel"] = df["n_formal_samples_using_bar"].notna()
    formal_mismatch = df[df["used_in_formal_panel"] & df["any_ohlc_mismatch"]].copy()
    formal_ticker_counts = (
        formal_mismatch.groupby("ticker", as_index=False)["any_ohlc_mismatch"].sum().rename(columns={"any_ohlc_mismatch": "formal_ohlc_mismatch_rows"})
    )
    all_mismatch = df[df["any_ohlc_mismatch"]].copy()
    result = {
        "earliest_week_start_used": usage_summary["week_start"].min(),
        "latest_week_start_used": usage_summary["week_start"].max(),
        "total_unique_weekly_bars_used": int(len(usage_summary)),
        "total_raw_weekly_mismatches": int(all_mismatch.shape[0]),
        "mismatches_used": int(formal_mismatch.shape[0]),
        "mismatches_never_used": int(all_mismatch.shape[0] - formal_mismatch.shape[0]),
        "used_by_ticker": formal_ticker_counts,
    }
    return {**result, "df": df}


def _check_week_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    weekday_counts = df["week_start"].dt.day_name().value_counts().sort_index()
    daily_first_day_counts = df["daily_first_date"].dropna().dt.day_name().value_counts().sort_index()
    daily_last_day_counts = df["daily_last_date"].dropna().dt.day_name().value_counts().sort_index()
    out = pd.DataFrame({
        "metric": ["weekly.datetime_weekday", "daily_first_date_weekday", "daily_last_date_weekday"],
        "counts": [weekday_counts.to_dict(), daily_first_day_counts.to_dict(), daily_last_day_counts.to_dict()],
    })
    return out


def _most_recent_mismatches(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    mism = df[df["any_ohlc_mismatch"]].sort_values("week_start", ascending=False).head(n).copy()
    return mism[["ticker", "week_start", "provider_open", "provider_high", "provider_low", "provider_close", "agg_open", "agg_high", "agg_low", "agg_close", "ratio_open", "ratio_high", "ratio_low", "ratio_close", "ratio_range", "classification"]].reset_index(drop=True)


def _quarterly_mismatch_timing(df: pd.DataFrame) -> pd.DataFrame:
    mism = df[df["any_ohlc_mismatch"]].copy()
    if mism.empty:
        return pd.DataFrame(columns=["ticker", "month", "quarter", "count"])
    mism["month"] = mism["week_start"].dt.to_period("M").astype(str)
    quarterly = (
        mism.groupby(["ticker", "month"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    quarterly["quarter"] = quarterly["month"].str[:4] + "-Q" + ((pd.to_datetime(quarterly["month"]).dt.month - 1) // 3 + 1).astype(str)
    return quarterly.sort_values(["ticker", "month"]).reset_index(drop=True)


def _aes_examples(df: pd.DataFrame) -> pd.DataFrame:
    target_dates = [pd.Timestamp("1992-08-31"), pd.Timestamp("1993-02-22"), pd.Timestamp("1993-11-01"), pd.Timestamp("1994-10-31")]
    out = df[(df["ticker"] == "AES") & (df["week_start"].isin(target_dates))].copy().sort_values("week_start")
    return out


def _print_aes_detail(rows: pd.DataFrame, ticker: str) -> None:
    for _, row in rows.iterrows():
        print(f"\n===== AES DETAIL {row['week_start'].date()} =====")
        daily = pd.read_csv(RAW_DIR / f"{ticker}_daily.csv", parse_dates=["datetime"])
        daily = daily.sort_values("datetime")
        week_start = pd.Timestamp(row["week_start"])
        week_end = week_start + pd.Timedelta(days=7)
        d = daily[(daily["datetime"] >= week_start) & (daily["datetime"] < week_end)].copy()
        print("RAW DAILY IN WEEK")
        print(d[["datetime", "open", "high", "low", "close", "volume"]].to_string(index=False))
        print("PROVIDER WEEKLY ROW")
        provider = {k: row[f"provider_{k}"] for k in ["open", "high", "low", "close", "volume"]}
        print(provider)
        agg = {k: row[f"agg_{k}"] for k in ["open", "high", "low", "close", "volume"]}
        print("RECONSTRUCTED WEEKLY ROW")
        print(agg)
        print("RATIOS")
        for k in ["open", "high", "low", "close"]:
            print(f"  {k}: provider/agg = {row[f'ratio_{k}']}")
        print(f"ratio_range = {row['ratio_range']}")
        print(f"classification = {row['classification']}")


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    schema_rows = []
    all_diag = []
    for ticker in FORMAL_TICKERS:
        info = _check_raw_file_schema(ticker)
        schema_rows.append(info)
        all_diag.append(_build_diag_rows(ticker))

    schema_df = pd.DataFrame(schema_rows)
    diag_df = pd.concat(all_diag, ignore_index=True)

    diag_df["week_start"] = pd.to_datetime(diag_df["week_start"])
    diag_df["daily_first_date"] = pd.to_datetime(diag_df["daily_first_date"])
    diag_df["daily_last_date"] = pd.to_datetime(diag_df["daily_last_date"])
    diag_df["any_ohlc_mismatch"] = diag_df["any_ohlc_mismatch"].fillna(False).astype(bool)
    diag_df["volume_mismatch"] = diag_df["volume_mismatch"].fillna(False).astype(bool)
    diag_df["classification"] = diag_df["classification"].fillna("EXACT_MATCH")

    diag_df.to_csv(AUDIT_DIR / "weekly_raw_semantics_diagnostic.csv", index=False)
    by_ticker = _summarize_by_ticker(diag_df)
    by_ticker.to_csv(AUDIT_DIR / "weekly_raw_semantics_by_ticker.csv", index=False)
    by_period = _summarize_periods(diag_df)
    by_period.to_csv(AUDIT_DIR / "weekly_raw_semantics_by_period.csv", index=False)

    panel = load_frozen_panel()
    formal_relevance = _formal_relevance(diag_df, panel)
    diag_df = formal_relevance["df"]

    mismatch_df = diag_df[diag_df["any_ohlc_mismatch"]].copy().reset_index(drop=True)
    exact_match = int((~diag_df["any_ohlc_mismatch"]).sum())
    total_rows = len(diag_df)
    ohlc_mismatch_rows = int(diag_df["any_ohlc_mismatch"].sum())
    open_mismatch_rows = int(diag_df["open_mismatch"].sum())
    high_mismatch_rows = int(diag_df["high_mismatch"].sum())
    low_mismatch_rows = int(diag_df["low_mismatch"].sum())
    close_mismatch_rows = int(diag_df["close_mismatch"].sum())
    volume_mismatch_rows = int(diag_df["volume_mismatch"].sum())
    common_scale_rows = int(((diag_df["classification"] == "COMMON_SCALE") & diag_df["any_ohlc_mismatch"]).sum())
    structural_rows = int(((diag_df["classification"] == "STRUCTURAL_OHLC") & diag_df["any_ohlc_mismatch"]).sum())
    partial_rows = int(((diag_df["classification"] == "OPEN_ONLY_OR_PARTIAL") & diag_df["any_ohlc_mismatch"]).sum())
    no_daily_rows = int((diag_df["classification"] == "NO_DAILY_ROWS").sum())

    formal_usage_mismatch = diag_df[diag_df["used_in_formal_panel"] & diag_df["any_ohlc_mismatch"]].copy()
    formal_usage_total = int(formal_usage_mismatch.shape[0])
    formal_usage_never = int(ohlc_mismatch_rows - formal_usage_total)

    summary_lines = [
        "0. Raw Data Provenance",
        "",
        "Source script: scripts/python/panel_download_daily_weekly.py",
        "Library: yfinance (installed version: 1.2.0)",
        "Daily download: yf.download(ticker, period='max', interval='1d', auto_adjust=True, progress=False)",
        "Weekly download: yf.download(ticker, period='max', interval='1wk', auto_adjust=True, progress=False)",
        "Important API defaults in this installed yfinance: actions=False, back_adjust=False, repair=False, keepna=False, progress=True, interval='1d'.",
        "The downloader calls omit actions/back_adjust/repair explicitly, so the effective settings are the library defaults for this version: actions=False, back_adjust=False, repair=False.",
        "The raw CSV schema is flattened by _standardize(), which keeps only: datetime, open, high, low, close, volume. Any Adj Close / split / dividend metadata was not preserved in the final raw CSV because those columns are intentionally dropped before save.",
        "Daily and Weekly are separate provider-native downloads, not reconstructed from each other; weekly is not a Daily-derived aggregation in the raw storage code.",
        "",
        "1. Raw File Schema",
        "",
    ]
    summary_lines.extend([f"{row['ticker']}: daily_rows={row['daily_rows']} weekly_rows={row['weekly_rows']} daily_cols={row['daily_columns']} weekly_cols={row['weekly_columns']} adjusted_like_cols={row['possible_adjusted_cols']}" for _, row in schema_df.iterrows()])
    summary_lines += [
        "",
        "2. Overall Weekly-vs-Daily Aggregation Results",
        "",
        f"total weekly rows analyzed: {total_rows}",
        f"exact-match rows: {exact_match} ({exact_match / total_rows * 100:.2f}%)",
        f"OHLC mismatch rows: {ohlc_mismatch_rows} ({ohlc_mismatch_rows / total_rows * 100:.2f}%)",
        f"open mismatch rows: {open_mismatch_rows}",
        f"high mismatch rows: {high_mismatch_rows}",
        f"low mismatch rows: {low_mismatch_rows}",
        f"close mismatch rows: {close_mismatch_rows}",
        f"volume mismatch rows: {volume_mismatch_rows} ({volume_mismatch_rows / total_rows * 100:.2f}%)",
        f"common_scale_mismatch_rows: {common_scale_rows} / ohlc_mismatch_rows = {common_scale_rows / ohlc_mismatch_rows if ohlc_mismatch_rows else 0.0:.6f}",
        f"structural_ohlc_mismatch_rows: {structural_rows}",
        f"partial_mismatch_rows: {partial_rows}",
        f"NO_DAILY_ROWS: {no_daily_rows} ({no_daily_rows / total_rows * 100:.2f}%)",
        "",
        "3. Results by Ticker",
        "",
    ]
    summary_lines.extend(by_ticker.to_string(index=False).splitlines())
    summary_lines += [
        "",
        "4. Results by Time Period",
        "",
    ]
    summary_lines.extend(by_period.to_string(index=False).splitlines())
    summary_lines += [
        "",
        "5. Formal V2 Weekly Bars Actually Used (reconstructed exactly from retained anchors and raw Daily+provider Weekly availability)",
        "",
        f"earliest week_start actually used in formal V2: {formal_relevance['earliest_week_start_used']}",
        f"latest week_start actually used in formal V2: {formal_relevance['latest_week_start_used']}",
        f"total unique Weekly bars actually used in formal V2: {formal_relevance['total_unique_weekly_bars_used']}",
        f"total raw OHLC mismatch rows: {formal_relevance['total_raw_weekly_mismatches']}",
        f"OHLC mismatch rows actually used in formal V2: {formal_relevance['mismatches_used']}",
        f"OHLC mismatch rows never used in formal V2: {formal_relevance['mismatches_never_used']}",
        "",
        "Formal-relevant mismatch counts by ticker:",
    ]
    if not formal_relevance["used_by_ticker"].empty:
        summary_lines.extend(formal_relevance["used_by_ticker"].to_string(index=False).splitlines())
    else:
        summary_lines.append("no formal-relevant OHLC mismatches")
    summary_lines += [
        "",
        "6. Common Multiplicative Scale Analysis",
        "",
    ]
    if not mismatch_df.empty:
        common_rows = mismatch_df[mismatch_df["classification"] == "COMMON_SCALE"].copy()
        summary_lines.append(f"common-scale mismatch rows among OHLC mismatches only: {len(common_rows)}")
        if not common_rows.empty:
            summary_lines.append(f"common-scale inferred scale factor median: {float(common_rows['inferred_scale_factor'].median()) if not common_rows.empty else np.nan}")
            summary_lines.append(f"common-scale inferred scale factor mode: {common_rows['inferred_scale_factor'].mode().head().tolist()}")
    else:
        summary_lines.append("No mismatch rows found.")
    summary_lines += [
        "",
        "7. Week-Boundary Analysis",
        "",
    ]
    boundary = _check_week_boundaries(diag_df)
    summary_lines.extend(boundary.to_string(index=False).splitlines())
    summary_lines += [
        "",
        "8. Volume Analysis",
        "",
        f"exact volume matches: {int((~diag_df['volume_mismatch']).sum())}",
        f"volume mismatch rows: {volume_mismatch_rows}",
        f"median relative volume error: {float(diag_df['relerr_volume'].dropna().median()) if diag_df['relerr_volume'].dropna().size else np.nan}",
        f"max relative volume error: {float(diag_df['relerr_volume'].dropna().max()) if diag_df['relerr_volume'].dropna().size else np.nan}",
        "",
        "9. Adjustment / Dividend / Split Hypothesis Check",
        "",
        "The raw files contain only datetime, open, high, low, close, volume columns and no explicit adjustment columns were retained. The current mismatch pattern is therefore evidence consistent with provider-specific interval-adjustment semantics, but not proof of dividend/split adjustment by itself.",
        "A key observation is that Close often matches while Open/High/Low differ, and many mismatches are concentrated in a small set of tickers and dates rather than appearing as a broad random error. This pattern is compatible with an interval-specific adjustment convention or non-equivalent Weekly bar construction, but it is not sufficient to label a corporate-action cause without additional provenance.",
        "", 
        "10. Quarterly Cadence Check",
        "",
    ]
    quarterly = _quarterly_mismatch_timing(diag_df)
    if quarterly.empty:
        summary_lines.append("No mismatch rows found; quarterly cadence unavailable.")
    else:
        summary_lines.extend(quarterly.head(40).to_string(index=False).splitlines())
    summary_lines += [
        "",
        "11. Known AES Examples",
        "",
    ]
    aes_rows = _aes_examples(diag_df)
    summary_lines.extend(aes_rows[["ticker", "week_start", "provider_open", "provider_high", "provider_low", "provider_close", "agg_open", "agg_high", "agg_low", "agg_close", "ratio_range", "classification"]].to_string(index=False).splitlines())
    summary_lines += [
        "",
        "12. Ten Most Recent Mismatch Examples",
        "",
    ]
    recent = _most_recent_mismatches(diag_df, 10)
    summary_lines.extend(recent.to_string(index=False).splitlines())
    summary_lines += [
        "",
        "13. Interpretation",
        "",
        "The raw file schema uses only basic OHLCV columns and no explicit adjustment columns were observed. The mismatch pattern is therefore a raw semantics question rather than a split or model issue.",
        "The corrected formal-panel relevance logic uses the actual retained (ticker, anchor_date) pairs and reconstructs the exact 26-week availability window from raw Daily rows; this excludes ancient bars that are not in the retained formal panel universe.",
        "The evidence is consistent with provider Weekly bars being generated under a provider-specific interval convention, but it is not yet sufficient to conclude whether provider Weekly or raw-Daily aggregation is the correct source definition for V2.",
        "",
        "14. Remaining Unresolved Questions",
        "",
        "- Are the provider Weekly OHLCV bars adjusted at a distinct weekly-level convention rather than being a literal raw aggregate of daily values?",
        "- Are the mismatched weeks associated with specific corporate actions or periodic back-adjustments?",
        "- Are the formal V2 samples materially impacted by weekly mismatch rows?",
        "- Should V2 Weekly semantics be defined by provider Weekly OHLCV or by raw-daily aggregation?",
    ]

    (AUDIT_DIR / "weekly_raw_semantics_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Created:", AUDIT_DIR / "weekly_raw_semantics_diagnostic.csv")
    print("Created:", AUDIT_DIR / "weekly_raw_semantics_by_ticker.csv")
    print("Created:", AUDIT_DIR / "weekly_raw_semantics_by_period.csv")
    print("Created:", AUDIT_DIR / "weekly_raw_semantics_summary.txt")

    # print detailed AES failure blocks
    aes_rows = _aes_examples(diag_df)
    _print_aes_detail(aes_rows, "AES")

    recent = _most_recent_mismatches(diag_df, 10)
    print("\n===== TEN MOST RECENT OHLC MISMATCH WEEKS =====")
    print(recent.to_string(index=False))

    print("\n===== OVERALL SUMMARY =====")
    print(f"total weekly rows analyzed: {total_rows}")
    print(f"OHLC mismatch rows: {ohlc_mismatch_rows} ({ohlc_mismatch_rows / total_rows * 100:.2f}%)")
    print(f"volume mismatch rows: {volume_mismatch_rows}")
    print("by period:")
    print(by_period.to_string(index=False))


if __name__ == "__main__":
    main()
