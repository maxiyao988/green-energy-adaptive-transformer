from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_common import MAX_HORIZON, load_frozen_panel
from scripts.python.panel_universe_duration_tradeoff import (
    build_tradeoff_curve,
    compute_split_for_dates,
    load_all_tickers,
)

RAW_DIR = ROOT / "dataset" / "finance" / "panel_raw"
PANEL_DIR = ROOT / "dataset" / "multiscale_dataset" / "panel"
AUDIT_DIR = ROOT / "dataset" / "audit"
REPORT_PATH = AUDIT_DIR / "panel_temporal_integrity_report.txt"

FEATURES = ["open", "high", "low", "close", "volume"]
H_DAILY = 90
H_WEEKLY = 26
FORMAL_TICKERS = [
    "AES", "BEP", "BLDP", "BLNK", "CSIQ",
    "CWEN", "DQ", "ENPH", "FCEL", "FSLR",
    "HASI", "JKS", "NEE", "ORA", "PLUG",
    "RUN", "SEDG",
]
REQUIRED_META_COLUMNS = [
    "anchor_date",
    "daily_feature_end",
    "weekly_bar_start",
    "weekly_available_date",
    "target_date_5d",
    "target_date_10d",
    "target_date_20d",
]


def _reconstruct_weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Independently reconstruct weekly bars from canonical Daily OHLCV."""
    daily = daily.copy().copy()
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="raise")
    daily = daily.sort_values("datetime").reset_index(drop=True)
    daily["week_start"] = daily["datetime"].apply(lambda dt: dt - pd.Timedelta(days=dt.weekday()))

    weekly = (
        daily.groupby("week_start", as_index=False)
        .agg(
            week_start=("week_start", "first"),
            available_date=("datetime", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly["available_date"] = pd.to_datetime(weekly["available_date"])
    return weekly[["week_start", "available_date", "open", "high", "low", "close", "volume"]]


def _load_daily_raw(ticker: str) -> pd.DataFrame:
    daily_path = RAW_DIR / f"{ticker}_daily.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing raw Daily file for {ticker}: {daily_path}")
    daily = pd.read_csv(daily_path)
    if "datetime" not in daily.columns:
        raise KeyError(f"Daily raw file missing datetime column for {ticker}: {daily.columns.tolist()}")
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="raise")
    return daily.sort_values("datetime").reset_index(drop=True)


def _check_required_meta_columns(meta: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_META_COLUMNS if col not in meta.columns]
    if missing:
        raise KeyError(
            "Missing required panel metadata columns: "
            + ", ".join(missing)
            + ". Required: "
            + ", ".join(REQUIRED_META_COLUMNS)
        )


def _load_panel_arrays(ticker: str) -> dict:
    ticker_dir = PANEL_DIR / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"Built panel directory missing for {ticker}: {ticker_dir}")

    meta = pd.read_csv(
        ticker_dir / "meta.csv",
        parse_dates=[
            "anchor_date",
            "daily_feature_end",
            "weekly_bar_start",
            "weekly_available_date",
            "target_date_5d",
            "target_date_10d",
            "target_date_20d",
        ],
    )
    _check_required_meta_columns(meta)

    arrays = {
        "X_daily": np.load(ticker_dir / "X_daily.npy"),
        "X_weekly": np.load(ticker_dir / "X_weekly.npy"),
        "y_5d": np.load(ticker_dir / "y_5d.npy"),
        "y_10d": np.load(ticker_dir / "y_10d.npy"),
        "y_20d": np.load(ticker_dir / "y_20d.npy"),
        "meta": meta,
    }
    return arrays


def _compare_float_arrays(actual: np.ndarray, expected: np.ndarray, label: str, issues: list[str], atol: float = 1e-6, rtol: float = 1e-5) -> None:
    if actual.shape != expected.shape:
        issues.append(f"{label}: shape mismatch {actual.shape} vs {expected.shape}")
        return
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        issues.append(
            f"{label}: reconstructed values differ at index {idx} "
            f"(actual={actual[idx]}, expected={expected[idx]}, abs_diff={diff[idx]})"
        )


def _date_to_index_map(daily: pd.DataFrame) -> dict[pd.Timestamp, int]:
    return {pd.Timestamp(dt): idx for idx, dt in enumerate(daily["datetime"].tolist())}


def _compute_original_split_dates() -> dict[str, list[pd.Timestamp]]:
    records = load_all_tickers()
    _, step_detail = build_tradeoff_curve(records)
    common_dates = step_detail[17]["common_dates"]
    split_info = compute_split_for_dates(common_dates)
    return {
        "train": pd.to_datetime(split_info["train_dates"]).tolist(),
        "val": pd.to_datetime(split_info["val_dates"]).tolist(),
        "test": pd.to_datetime(split_info["test_dates"]).tolist(),
    }


def _expected_weekly_window_for_anchor(weekly_expected: pd.DataFrame, anchor_date: pd.Timestamp) -> np.ndarray | None:
    valid = weekly_expected[
        weekly_expected["available_date"].notna() & (weekly_expected["available_date"] <= anchor_date)
    ].copy()
    if valid.empty or len(valid) < H_WEEKLY:
        return None
    window = valid.sort_values("week_start").tail(H_WEEKLY)
    return window[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)


def _fatal_message(label: str, details: str) -> str:
    return f"{label}: {details}"


def _audit_ticker(ticker: str) -> list[str]:
    issues: list[str] = []
    try:
        daily = _load_daily_raw(ticker)
        arrays = _load_panel_arrays(ticker)
    except Exception as exc:
        return [f"{ticker}: {exc}"]

    meta = arrays["meta"]
    X_daily = arrays["X_daily"]
    X_weekly = arrays["X_weekly"]
    y_map = {5: arrays["y_5d"], 10: arrays["y_10d"], 20: arrays["y_20d"]}

    # 1. metadata integrity
    if len(meta) != len(X_daily) or len(meta) != len(X_weekly):
        issues.append(f"{ticker}: meta length mismatch: {len(meta)} vs X_daily={len(X_daily)} vs X_weekly={len(X_weekly)}")
    for col in ["anchor_date", "daily_feature_end", "weekly_bar_start", "weekly_available_date"]:
        if meta[col].isna().any():
            issues.append(f"{ticker}: {col} contains missing values")
    if (meta["daily_feature_end"] != meta["anchor_date"]).any():
        mismatches = int((meta["daily_feature_end"] != meta["anchor_date"]).sum())
        issues.append(f"{ticker}: daily_feature_end != anchor_date violations={mismatches}")

    # 2. daily integrity: 90-bar window with anchor D
    date_to_index = _date_to_index_map(daily)
    for idx, row in meta.iterrows():
        anchor_date = pd.Timestamp(row["anchor_date"])
        anchor_idx = date_to_index.get(anchor_date)
        if anchor_idx is None:
            issues.append(f"{ticker}: anchor_date {anchor_date} not found in raw Daily CSV")
            continue
        if anchor_idx < H_DAILY - 1:
            issues.append(f"{ticker}: anchor_date {anchor_date} too early to construct a 90-day window")
            continue
        expected_daily = daily.iloc[anchor_idx - H_DAILY + 1 : anchor_idx + 1][FEATURES].to_numpy(dtype=np.float32)
        if X_daily[idx].shape != (H_DAILY, len(FEATURES)):
            issues.append(f"{ticker}: X_daily sample {idx} shape mismatch: {X_daily[idx].shape}")
        else:
            _compare_float_arrays(X_daily[idx], expected_daily, f"{ticker} X_daily[{idx}] anchor={anchor_date}", issues, atol=1e-6, rtol=1e-5)

    # 3. weekly integrity: independent reconstruction from canonical Daily only
    weekly_expected = _reconstruct_weekly_from_daily(daily)
    for idx, row in meta.iterrows():
        anchor_date = pd.Timestamp(row["anchor_date"])
        available_date = pd.Timestamp(row["weekly_available_date"])
        if available_date > anchor_date:
            issues.append(
                f"{ticker}: weekly_available_date exceeds anchor_date for sample {idx}: "
                f"available={available_date}, anchor={anchor_date}"
            )

        expected_window = _expected_weekly_window_for_anchor(weekly_expected, anchor_date)
        if expected_window is None:
            issues.append(f"{ticker}: insufficient usable weekly bars for anchor {anchor_date} to form 26-bar window")
            continue
        if X_weekly[idx].shape != (H_WEEKLY, len(FEATURES)):
            issues.append(f"{ticker}: X_weekly sample {idx} shape mismatch: {X_weekly[idx].shape}")
            continue
        _compare_float_arrays(
            X_weekly[idx],
            expected_window,
            f"{ticker} X_weekly[{idx}] anchor={anchor_date}",
            issues,
            atol=1e-6,
            rtol=1e-5,
        )

        expected_bar_start = pd.Timestamp(weekly_expected[
            weekly_expected["available_date"] <= anchor_date
        ].sort_values("week_start").tail(1)["week_start"].iloc[0])
        if pd.Timestamp(row["weekly_bar_start"]) != expected_bar_start:
            issues.append(
                f"{ticker}: weekly_bar_start mismatch for sample {idx}: "
                f"stored={row['weekly_bar_start']}, expected={expected_bar_start}"
            )
        if pd.Timestamp(row["weekly_available_date"]) != pd.Timestamp(
            weekly_expected[
                weekly_expected["available_date"] <= anchor_date
            ].sort_values("week_start").tail(1)["available_date"].iloc[0]
        ):
            issues.append(
                f"{ticker}: weekly_available_date mismatch for sample {idx}: "
                f"stored={row['weekly_available_date']}, expected={weekly_expected[weekly_expected['available_date'] <= anchor_date].sort_values('week_start').tail(1)['available_date'].iloc[0]}"
            )

    # 4. target integrity
    for h in (5, 10, 20):
        target_col = f"target_date_{h}d"
        for idx, row in meta.iterrows():
            anchor_date = pd.Timestamp(row["anchor_date"])
            anchor_idx = date_to_index.get(anchor_date)
            if anchor_idx is None:
                issues.append(f"{ticker}: unable to map anchor date {anchor_date} to raw daily index for h={h}")
                continue
            if anchor_idx + h >= len(daily):
                issues.append(f"{ticker}: insufficient future data for anchor {anchor_date} at h={h}")
                continue
            expected_target_date = pd.Timestamp(daily.iloc[anchor_idx + h]["datetime"])
            expected_y = float(np.log(daily.iloc[anchor_idx + h]["close"] / daily.iloc[anchor_idx]["close"]))
            stored_target_date = pd.Timestamp(row[target_col])
            if stored_target_date != expected_target_date:
                issues.append(
                    f"{ticker}: target_date mismatch for h={h}: stored={stored_target_date}, expected={expected_target_date}"
                )
            actual_y = float(y_map[h][idx])
            if not np.isclose(actual_y, expected_y, rtol=1e-5, atol=1e-6):
                issues.append(
                    f"{ticker}: target value mismatch for h={h}: stored={actual_y}, expected={expected_y}, anchor={anchor_date}"
                )

    return issues


def _panel_consistency_checks(panel: object, issues: list[str]) -> None:
    unique_tickers = set(np.asarray(panel.ticker).tolist())
    target_set = set(FORMAL_TICKERS)
    if len(unique_tickers) != 17:
        issues.append(f"Formal panel has {len(unique_tickers)} unique tickers; expected 17")
    if unique_tickers != target_set:
        missing = sorted(target_set - unique_tickers)
        extra = sorted(unique_tickers - target_set)
        if missing:
            issues.append(f"Formal panel missing tickers: {missing}")
        if extra:
            issues.append(f"Formal panel contains unexpected tickers: {extra}")

    lengths = {
        "ticker": len(panel.ticker),
        "anchor_date": len(panel.anchor_date),
        "split": len(panel.split),
        "X_daily": panel.X_daily.shape[0],
        "X_weekly": panel.X_weekly.shape[0],
        "y_5d": len(panel.y[5]),
        "y_10d": len(panel.y[10]),
        "y_20d": len(panel.y[20]),
    }
    if len(set(lengths.values())) != 1:
        issues.append(f"Formal panel length mismatch: {lengths}")

    if not np.isfinite(panel.X_daily).all():
        issues.append("Formal panel contains non-finite values in X_daily")
    if not np.isfinite(panel.X_weekly).all():
        issues.append("Formal panel contains non-finite values in X_weekly")
    for h in (5, 10, 20):
        if not np.isfinite(panel.y[h]).all():
            issues.append(f"Formal panel contains non-finite values in y_{h}d")

    date_counts = pd.DataFrame({"ticker": panel.ticker, "anchor_date": panel.anchor_date}).groupby("anchor_date")["ticker"].count()
    bad_dates = date_counts[(date_counts != 17)].index.tolist()
    if bad_dates:
        issues.append(f"Formal panel has retained dates with a ticker count != 17: first examples={bad_dates[:5]}")

    train_dates = pd.to_datetime(panel.anchor_date[panel.split == "train"])
    val_dates = pd.to_datetime(panel.anchor_date[panel.split == "val"])
    test_dates = pd.to_datetime(panel.anchor_date[panel.split == "test"])
    if len(train_dates) and len(val_dates) and len(test_dates):
        if train_dates.max() >= val_dates.min():
            issues.append("Formal panel violates max(train) < min(val)")
        if val_dates.max() >= test_dates.min():
            issues.append("Formal panel violates max(val) < min(test)")


def _check_split_purge(panel: object, issues: list[str]) -> dict[str, int | dict[str, int]]:
    split_dates = _compute_original_split_dates()
    original_train = pd.DatetimeIndex(pd.to_datetime(split_dates["train"]))
    original_val = pd.DatetimeIndex(pd.to_datetime(split_dates["val"]))
    original_test = pd.DatetimeIndex(pd.to_datetime(split_dates["test"]))

    actual_train_dates = pd.Index(pd.to_datetime(pd.Series(panel.anchor_date[panel.split == "train"]).unique())).sort_values()
    actual_val_dates = pd.Index(pd.to_datetime(pd.Series(panel.anchor_date[panel.split == "val"]).unique())).sort_values()
    actual_test_dates = pd.Index(pd.to_datetime(pd.Series(panel.anchor_date[panel.split == "test"]).unique())).sort_values()

    expected_train = original_train[:-MAX_HORIZON] if len(original_train) > MAX_HORIZON else pd.DatetimeIndex([])
    expected_val = original_val[:-MAX_HORIZON] if len(original_val) > MAX_HORIZON else pd.DatetimeIndex([])
    expected_test = original_test

    if not expected_train.equals(actual_train_dates):
        issues.append(
            f"Retained train dates do not match expected purge: "
            f"original={len(original_train)}, retained={len(actual_train_dates)}, expected={len(expected_train)}"
        )
    if not expected_val.equals(actual_val_dates):
        issues.append(
            f"Retained val dates do not match expected purge: "
            f"original={len(original_val)}, retained={len(actual_val_dates)}, expected={len(expected_val)}"
        )

    validation_start = original_val.min()
    test_start = original_test.min()
    crossing = {"train": {5: 0, 10: 0, 20: 0}, "val": {5: 0, 10: 0, 20: 0}}

    for ticker in FORMAL_TICKERS:
        raw_daily = _load_daily_raw(ticker)
        date_to_index = _date_to_index_map(raw_daily)
        ticker_rows = pd.DataFrame({
            "ticker": panel.ticker,
            "anchor_date": pd.to_datetime(panel.anchor_date),
            "split": panel.split,
        })
        for row in ticker_rows[ticker_rows["ticker"] == ticker].itertuples(index=False):
            split = row.split
            if split not in {"train", "val"}:
                continue
            anchor_date = pd.Timestamp(row.anchor_date)
            anchor_idx = date_to_index.get(anchor_date)
            if anchor_idx is None or anchor_idx + MAX_HORIZON >= len(raw_daily):
                continue
            for h in (5, 10, 20):
                target_date = pd.Timestamp(raw_daily.iloc[anchor_idx + h]["datetime"])
                if split == "train" and target_date >= validation_start:
                    crossing["train"][h] += 1
                elif split == "val" and target_date >= test_start:
                    crossing["val"][h] += 1

    return {
        "train_crossing_5d": crossing["train"][5],
        "train_crossing_10d": crossing["train"][10],
        "train_crossing_20d": crossing["train"][20],
        "val_crossing_5d": crossing["val"][5],
        "val_crossing_10d": crossing["val"][10],
        "val_crossing_20d": crossing["val"][20],
    }


def _collect_report_lines() -> list[str]:
    panel = load_frozen_panel()
    lines: list[str] = [
        "=" * 80,
        "FORMAL PANEL TEMPORAL INTEGRITY REPORT",
        "=" * 80,
        "",
        "1. Formal Universe / Dataset Contract",
        "",
        f"Formal universe: {FORMAL_TICKERS}",
        f"Retained sample count: {len(panel.ticker)}",
        "Dataset contract: canonical Daily from yfinance 1d with auto_adjust=True; Weekly reconstructed from canonical Daily via Monday-start calendar weeks; anchor date includes D; forward log-return targets at 5d/10d/20d; provider-native Weekly is historical forensic evidence only.",
        "",
    ]

    ticker_issues: list[str] = []
    for ticker in FORMAL_TICKERS:
        ticker_issues.extend(_audit_ticker(ticker))

    lines.append("2. Daily Window Integrity")
    daily_violations = [s for s in ticker_issues if "X_daily" in s or "daily_feature_end" in s or "anchor_date" in s]
    lines.append(f"{'PASS' if not daily_violations else 'FAIL'}; violations={len(daily_violations)}")
    for issue in daily_violations[:5]:
        lines.append(f"  - {issue}")
    lines.append("")

    lines.append("3. Weekly-from-Daily Construction Integrity")
    weekly_value_issues = [s for s in ticker_issues if "X_weekly" in s or "weekly_available_date" in s or "weekly_bar_start" in s]
    lines.append(f"{'PASS' if not weekly_value_issues else 'FAIL'}; violations={len(weekly_value_issues)}")
    for issue in weekly_value_issues[:5]:
        lines.append(f"  - {issue}")
    lines.append("")

    lines.append("4. Weekly Availability / Look-Ahead Integrity")
    weekly_availability_issues = [s for s in ticker_issues if "weekly_available_date" in s or "insufficient usable weekly bars" in s]
    lines.append(f"{'PASS' if not weekly_availability_issues else 'FAIL'}; violations={len(weekly_availability_issues)}")
    for issue in weekly_availability_issues[:5]:
        lines.append(f"  - {issue}")
    lines.append("")

    lines.append("5. Target Date and Value Integrity")
    target_issues = [s for s in ticker_issues if "target" in s]
    lines.append(f"{'PASS' if not target_issues else 'FAIL'}; violations={len(target_issues)}")
    for issue in target_issues[:5]:
        lines.append(f"  - {issue}")
    lines.append("")

    lines.append("6. Split Purging / Boundary Crossing")
    issue_store: list[str] = []
    purge_stats = _check_split_purge(panel, issue_store)
    lines.append(f"Train crossing 5d: {purge_stats['train_crossing_5d']} | Train crossing 10d: {purge_stats['train_crossing_10d']} | Train crossing 20d: {purge_stats['train_crossing_20d']}")
    lines.append(f"Val crossing 5d: {purge_stats['val_crossing_5d']} | Val crossing 10d: {purge_stats['val_crossing_10d']} | Val crossing 20d: {purge_stats['val_crossing_20d']}")
    lines.append(f"{'PASS' if not issue_store else 'FAIL'}; violations={len(issue_store)}")
    for issue in issue_store[:5]:
        lines.append(f"  - {issue}")
    lines.append("")

    lines.append("7. Panel Consistency")
    panel_issues: list[str] = []
    _panel_consistency_checks(panel, panel_issues)
    lines.append(f"{'PASS' if not panel_issues else 'FAIL'}; violations={len(panel_issues)}")
    for issue in panel_issues[:5]:
        lines.append(f"  - {issue}")
    lines.append("")

    lines.append("8. Overall Result")
    critical_failures = ticker_issues + panel_issues + issue_store
    lines.append("OVERALL: PASS" if not critical_failures else "OVERALL: FAIL")
    lines.append("=" * 80)
    return lines, critical_failures


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines, critical_failures = _collect_report_lines()
    report = "\n".join(report_lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved temporal-integrity report: {REPORT_PATH}")
    raise SystemExit(1 if critical_failures else 0)


if __name__ == "__main__":
    main()
