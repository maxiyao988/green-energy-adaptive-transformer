"""
One-off verification: for the frozen 17-stock candidate panel, confirm that
Daily-only, Weekly-only, and Daily+Weekly settings all use the exact same
(ticker, anchor_date) sample index for the train split (expected: 31,008
pooled training observations, per panel_candidate_universe_summary.csv).

This does not train anything and does not change any dataset or split
decision — it only checks that slicing different feature columns (Daily vs
Weekly vs both) from the same underlying per-ticker rows can never silently
diverge the sample set used by different frequency settings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.python.panel_universe_duration_tradeoff import (
    load_all_tickers,
    build_tradeoff_curve,
    compute_split_for_dates,
)

TARGET_N_TICKERS = 17
EXPECTED_TRAIN_SAMPLES = 31008


def sample_set_for_setting(retained: list[str], records: dict, dates: list, feature_key: str) -> set:
    """(ticker, anchor_date) pairs where `feature_key` ('X_daily' or 'X_weekly') is
    non-NaN/non-Inf for that ticker on that date."""
    date_arr = np.array(dates, dtype="datetime64[ns]")
    out = set()
    for ticker in retained:
        rec = records[ticker]
        idx = np.isin(rec["anchor_date_norm"], date_arr)
        if not idx.any():
            continue
        arr = rec[feature_key][idx]
        valid = ~(np.isnan(arr).any(axis=(1, 2)) | np.isinf(arr).any(axis=(1, 2)))
        valid_dates = pd.to_datetime(rec["anchor_date_norm"][idx][valid])
        out.update((ticker, d) for d in valid_dates)
    return out


def main() -> None:
    records = load_all_tickers()
    _, step_detail = build_tradeoff_curve(records)

    if TARGET_N_TICKERS not in step_detail:
        raise RuntimeError(f"n_tickers={TARGET_N_TICKERS} not found in the removal curve.")

    retained = step_detail[TARGET_N_TICKERS]["retained"]
    common_dates = step_detail[TARGET_N_TICKERS]["common_dates"]
    split = compute_split_for_dates(common_dates)

    print(f"17-stock candidate: {sorted(retained)}")
    print(f"n_common_dates={len(common_dates)}, n_train_dates={len(split['train_dates'])}")

    daily_set = sample_set_for_setting(retained, records, split["train_dates"], "X_daily")
    weekly_set = sample_set_for_setting(retained, records, split["train_dates"], "X_weekly")
    both_set = daily_set & weekly_set  # "Daily + Weekly" requires both present

    print(f"daily_only train sample count:      {len(daily_set)}")
    print(f"weekly_only train sample count:     {len(weekly_set)}")
    print(f"daily_and_weekly train sample count: {len(both_set)}")

    identical = (daily_set == weekly_set == both_set)
    matches_expected = (len(daily_set) == EXPECTED_TRAIN_SAMPLES)

    print()
    if identical:
        print("PASS: daily_only, weekly_only, and daily+weekly all use the IDENTICAL (ticker, anchor_date) sample set.")
    else:
        print("FAIL: sample sets diverge across frequency settings.")
        print(f"  daily_only - weekly_only (extra in daily): {len(daily_set - weekly_set)}")
        print(f"  weekly_only - daily_only (extra in weekly): {len(weekly_set - daily_set)}")

    if matches_expected:
        print(f"PASS: train sample count matches the expected 31,008 pooled training observations.")
    else:
        print(f"FAIL: train sample count {len(daily_set)} does not match expected {EXPECTED_TRAIN_SAMPLES}.")

    if not (identical and matches_expected):
        sys.exit(1)


if __name__ == "__main__":
    main()
