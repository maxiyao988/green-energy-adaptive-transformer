"""Step 5.0 - copy archived formal TEST predictions into canonical parquet cells.

Sources (read only; formal 6,647-row-per-cell MAIN_V2 files, never the 6,664-row root files):
  Adaptive seed 42 daily_weekly  dataset/audit/main_v2/pathformer/panel_pathformer_test_predictions.csv
  Ridge / LSTM / Vanilla / SWiM  dataset/audit/main_v2/{ridge,lstm,vanilla,swim_cpu}/panel_*_test_predictions.csv
  Zero / Global Train-Mean       dataset/audit/main_v2/naive/main_v2_naive_test_predictions.csv
Every cell must have 6,647 rows, 17 tickers x 391 dates, and labels equal (float32) to the
frozen replayed TEST labels.

Usage: python -B s0_canonicalize_archived.py [--out-dir DIR] [--label-ref GLOB]
"""
import argparse
from skill_common import *

MV2 = ROOT / "dataset" / "audit" / "main_v2"
SOURCES = [
    # (path, model label, filters, config column or fixed config, seed)
    (MV2 / "pathformer" / "panel_pathformer_test_predictions.csv", "adaptive", {"frequency_setting": ["daily_weekly"], "seed": [42]}, "dw", "42"),
    (MV2 / "ridge" / "panel_ridge_test_predictions.csv", "ridge", {}, "frequency_setting", "42"),
    (MV2 / "lstm" / "panel_lstm_test_predictions.csv", "lstm", {"seed": [42]}, "frequency_setting", "42"),
    (MV2 / "vanilla" / "panel_vanilla_test_predictions.csv", "vanilla", {"seed": [42]}, "frequency_setting", "42"),
    (MV2 / "swim_cpu" / "panel_swim_test_predictions.csv", "swim", {"seed": [42]}, "frequency_setting", "42"),
    (MV2 / "naive" / "main_v2_naive_test_predictions.csv", "naive", {}, "model", "na"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(SKILL_OUT / "canonical_preds"))
    ap.add_argument("--label-ref", default=str(PREDS_DIR / "static_dw_h*_s0_test.parquet"),
                    help="frozen TEST label cells used to verify archived y_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    ref = {parse_name(f)["h"]: load_pred(f) for f in expand_globs([a.label_ref])}
    rows = []
    for path, model, filt, cfg, seed in SOURCES:
        d = pd.read_csv(path)
        if len(d) != 9 * N_TEST_ROWS + 0 and model not in ("adaptive", "naive"):
            raise ValueError(f"{path}: unexpected row count {len(d)} (expected 9 x 6647)")
        for c, allowed in filt.items():
            d = d[d[c].isin(allowed)]
        d["h"] = d["horizon"].astype(str).str.removesuffix("d").astype(int)
        d["config"] = d[cfg] if cfg in d.columns else cfg
        for (config, h), g in d.groupby(["config", "h"], sort=True):
            g = g[["ticker", "anchor_date", "y_true", "y_pred"]].copy()
            g["anchor_date"] = canon_dates(g["anchor_date"])
            g = g.sort_values(["anchor_date", "ticker"], kind="mergesort").reset_index(drop=True)
            if len(g) != N_TEST_ROWS or g.ticker.nunique() != N_TICKERS or g.anchor_date.nunique() != N_TEST_DATES:
                raise ValueError(f"{model}/{config}/h{h}: shape {len(g)}")
            r = ref[h]
            if not (g[["ticker", "anchor_date"]].equals(r[["ticker", "anchor_date"]])):
                raise ValueError(f"{model}/{config}/h{h}: key mismatch with frozen labels")
            y32 = g.y_true.to_numpy(np.float64).astype(np.float32)
            if not np.array_equal(y32, r.y_true.to_numpy(np.float32)):
                raise ValueError(f"{model}/{config}/h{h}: labels differ from frozen labels")
            g["y_true"] = y32
            g["y_pred"] = g.y_pred.to_numpy(np.float64)
            name = f"{model}_{config}_h{h}_s{seed}_test.parquet"
            write_parquet(out / name, g)
            rows.append(dict(file=rel(out / name), model=model, config=config, h=h, seed=seed, rows=len(g),
                             source=rel(path), source_sha256=sha256(path),
                             y_pred_mean=float(g.y_pred.mean()), labels_equal_frozen=True))
    man = pd.DataFrame(rows)
    write_csv(out / "canonical_manifest.csv", man)
    print(man.to_string())


if __name__ == "__main__":
    main()
