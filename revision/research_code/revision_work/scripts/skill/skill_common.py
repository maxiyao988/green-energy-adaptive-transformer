"""Shared helpers for Step 5 forecast-skill diagnostics (no training, no torch).

Every existing repository file is read only. New files are created exclusively
under revision_work/outputs/skill_analysis/ (open mode 'x'; never overwritten).

Canonical prediction file name (TEST split is what the analysis consumes):
    {model}_{config}_h{h}_s{seed}_{split}.parquet
model has no underscore; config may contain underscores ('dw' == daily_weekly);
seed is an integer or 'na' for deterministic predictors.
Required columns: ticker, anchor_date, y_true, y_pred.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
sys.dont_write_bytecode = True
import glob as _glob
import hashlib
import json
import math
import re
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "revision_work" / "outputs"
# SKILL_OUT_DIR may redirect outputs (used only for dry runs outside the repository).
SKILL_OUT = Path(os.environ.get("SKILL_OUT_DIR", OUT_ROOT / "skill_analysis"))
PREDS_DIR = OUT_ROOT / "preds"
FORBIDDEN_OUT = OUT_ROOT / "causal_features"
# pyarrow 15 (NumPy-1.26 compatible) was installed in Step 3 under outputs/_deps_pyarrow15.
sys.path.append(str(OUT_ROOT / "_deps_pyarrow15"))

import warnings
warnings.filterwarnings("ignore", message=".*Pyarrow will become a required dependency.*")
import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = (5, 10, 20)
SEEDS5 = (0, 1, 21, 42, 3407)
N_TICKERS = 17
N_TEST_DATES = 391
N_TEST_ROWS = 6647
SPLIT_COUNTS = {"train": 30634, "val": 6290, "test": 6647}
BOOT_B = 20000
BOOT_BASE_SEED = 20260826
META_DIR = ROOT / "dataset" / "multiscale_dataset" / "panel"
RAW_DIR = ROOT / "dataset" / "finance" / "panel_raw"
CONFIG_ALIASES = {"dw": "daily_weekly", "do": "daily_only", "wo": "weekly_only"}
NAME_RE = re.compile(r"^(?P<model>[a-z0-9]+)_(?P<config>.+)_h(?P<h>\d+)_s(?P<seed>\d+|na)_(?P<split>train|val|test)\.parquet$")


def utc():
    return datetime.now(timezone.utc).isoformat()


def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- exclusive output

def target(p):
    p = Path(p).resolve()
    if not p.is_relative_to(SKILL_OUT.resolve()):
        raise ValueError(f"Output outside permitted directory: {p}")
    if p.is_relative_to(FORBIDDEN_OUT.resolve()):
        raise ValueError(f"Refusing to write under causal_features: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"Refusing overwrite: {p}")
    return p


def write_bytes(p, payload: bytes):
    with target(p).open("xb") as f:
        f.write(payload)


def write_text(p, text: str):
    write_bytes(p, text.encode("utf-8"))


def write_csv(p, df: pd.DataFrame):
    write_bytes(p, df.to_csv(index=False, float_format="%.12g").encode("utf-8"))


def write_json(p, obj):
    write_text(p, json.dumps(obj, indent=2, allow_nan=True, default=str))


def write_parquet(p, df: pd.DataFrame):
    payload = df.to_parquet(index=False, engine="pyarrow")
    write_bytes(p, payload)
    back = pd.read_parquet(p)
    if not back.equals(df.reset_index(drop=True)):
        raise RuntimeError(f"Parquet read-back mismatch: {p}")


# ---------------------------------------------------------------- prediction files

def expand_globs(patterns):
    files = []
    for pat in patterns:
        hits = sorted(_glob.glob(str(pat))) if any(c in str(pat) for c in "*?[") else [str(pat)]
        if not hits:
            raise FileNotFoundError(f"No files match {pat}")
        files.extend(hits)
    out = []
    for f in files:
        f = str(Path(f).resolve())
        if f not in out:
            out.append(f)
    return out


def parse_name(path):
    m = NAME_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"File name does not follow {{model}}_{{config}}_h{{h}}_s{{seed}}_{{split}}.parquet: {path}")
    d = m.groupdict()
    d["h"] = int(d["h"])
    d["config"] = CONFIG_ALIASES.get(d["config"], d["config"])
    d["cell"] = f"{d['model']}|{d['config']}|h{d['h']}|s{d['seed']}"
    return d


def canon_dates(s):
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d")


def load_pred(path, expect_rows=N_TEST_ROWS):
    d = pd.read_parquet(path, columns=["ticker", "anchor_date", "y_true", "y_pred"])
    d["anchor_date"] = canon_dates(d["anchor_date"])
    d = d.sort_values(["anchor_date", "ticker"], kind="mergesort").reset_index(drop=True)
    if expect_rows is not None and len(d) != expect_rows:
        raise ValueError(f"{path}: {len(d)} rows != {expect_rows}")
    if d.duplicated(["ticker", "anchor_date"]).any():
        raise ValueError(f"{path}: duplicate ticker/date keys")
    for c in ("y_true", "y_pred"):
        if not np.isfinite(d[c].to_numpy(np.float64)).all():
            raise ValueError(f"{path}: non-finite {c}")
    return d


def to_panel(d, col):
    """Balanced (date x ticker) float64 matrix; rows sorted by date, columns by ticker."""
    w = d.pivot(index="anchor_date", columns="ticker", values=col).sort_index()
    w = w[sorted(w.columns)]
    if w.isna().any().any():
        raise ValueError("Unbalanced panel")
    return w.to_numpy(np.float64), list(w.index), list(w.columns)


# ---------------------------------------------------------------- statistics

def nw_lrv(x, lag):
    """Bartlett/Newey-West long-run variance with 1/T autocovariances (as in the archived HAC code)."""
    x = np.asarray(x, np.float64)
    T = len(x)
    c = x - x.mean()
    lrv = float(np.dot(c, c) / T)
    for l in range(1, lag + 1):
        lrv += 2.0 * (1.0 - l / (lag + 1)) * float(np.dot(c[l:], c[:-l]) / T)
    return lrv


def nw_t(x, lag):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 2:
        return float("nan"), float("nan"), float("nan"), T
    lrv = nw_lrv(x, lag)
    se = math.sqrt(lrv / T) if lrv > 0 else float("nan")
    return float(x.mean()), se, float(x.mean() / se) if se and se > 0 else float("nan"), T


def rank_rows(a):
    """Average-tie ranks along axis 1."""
    return stats.rankdata(a, axis=1, method="average")


def rank_ic_rows(pred, y):
    """Per-row Spearman correlation (identical to scipy.spearmanr per date); NaN if constant."""
    rp, ry = rank_rows(pred), rank_rows(y)
    rp = rp - rp.mean(1, keepdims=True)
    ry = ry - ry.mean(1, keepdims=True)
    den = np.sqrt((rp ** 2).sum(1) * (ry ** 2).sum(1))
    sp = np.std(pred, axis=1)
    out = np.full(pred.shape[0], np.nan)
    ok = (sp >= 1e-12) & (np.std(y, axis=1) >= 1e-12) & (den > 0)
    out[ok] = (rp[ok] * ry[ok]).sum(1) / den[ok]
    return out


def mbb_index(T, L, B, rng):
    """Moving (overlapping, non-circular) block bootstrap indices, exactly as the archived
    Experiment 4 inference: ceil(T/L) block starts uniform on 0..T-L, concatenate, truncate."""
    K = math.ceil(T / L)
    starts = rng.integers(0, T - L + 1, size=(B, K))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, K * L)[:, :T]
    return idx


def holm(pvals):
    keys = list(pvals)
    order = sorted(keys, key=lambda k: pvals[k])
    m = len(order)
    run, out = 0.0, {}
    for r, k in enumerate(order, start=1):
        run = max(run, (m - r + 1) * pvals[k])
        out[k] = min(run, 1.0)
    return out


def stable_int(s):
    return zlib.crc32(str(s).encode("utf-8"))


# ---------------------------------------------------------------- labels and references

def label_family(patterns):
    """All-split labels (ticker, anchor_date, split, y_true) per horizon from a family of
    {model}_{config}_h{h}_s{seed}_{split}.parquet files (one cell per horizon)."""
    fam = {}
    for f in expand_globs(patterns):
        info = parse_name(f)
        d = pd.read_parquet(f, columns=["ticker", "anchor_date", "y_true"])
        d["anchor_date"] = canon_dates(d["anchor_date"])
        d["split"] = info["split"]
        if len(d) != SPLIT_COUNTS[info["split"]]:
            raise ValueError(f"{f}: {len(d)} rows != {SPLIT_COUNTS[info['split']]}")
        fam.setdefault(info["h"], []).append(d)
    out = {}
    for h, parts in fam.items():
        d = pd.concat(parts, ignore_index=True)
        if sorted(d.split.unique()) != ["test", "train", "val"]:
            raise ValueError(f"label family h={h} lacks a split")
        if d.duplicated(["ticker", "anchor_date"]).any():
            raise ValueError(f"label family h={h} has duplicate keys")
        out[h] = d.sort_values(["ticker", "anchor_date"], kind="mergesort").reset_index(drop=True)
    return out


def load_meta(tickers):
    """Per-ticker frozen meta (anchor date -> target dates) and frozen full-history labels."""
    rows = []
    for t in tickers:
        m = pd.read_csv(META_DIR / t / "meta.csv")
        m["anchor_date"] = canon_dates(m["anchor_date"])
        for h in HORIZONS:
            m[f"target_date_{h}d"] = canon_dates(m[f"target_date_{h}d"])
            y = np.load(META_DIR / t / f"y_{h}d.npy")
            if len(y) != len(m):
                raise ValueError(f"{t}: meta/label length mismatch")
            m[f"y_{h}d"] = y.astype(np.float32)
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def expanding_mean(lab_ticker, lab_target, lab_y, q_ticker, q_date, pooled):
    """Mean of labels whose target date <= query anchor date (no look-ahead).
    pooled=False: same ticker only; pooled=True: all tickers."""
    out = np.full(len(q_ticker), np.nan)
    lab = pd.DataFrame({"ticker": lab_ticker, "target": lab_target, "y": np.asarray(lab_y, np.float64)})
    q = pd.DataFrame({"ticker": q_ticker, "date": q_date, "pos": np.arange(len(q_ticker))})
    groups = [(None, lab, q)] if pooled else [(t, lab[lab.ticker == t], q[q.ticker == t]) for t in sorted(q.ticker.unique())]
    for _, l, qq in groups:
        l = l.sort_values("target", kind="mergesort")
        cs = np.cumsum(l.y.to_numpy())
        n = np.searchsorted(l.target.to_numpy(), qq.date.to_numpy(), side="right")
        vals = np.where(n > 0, cs[np.maximum(n - 1, 0)] / np.maximum(n, 1), np.nan)
        out[qq.pos.to_numpy()] = vals
    return out


def reference_forecasts(fam_h, meta, test_keys, h):
    """Zero, Global Train-Mean, Ticker train mean, expanding (ticker / pooled) forecasts
    aligned to test_keys (DataFrame with ticker, anchor_date)."""
    tr = fam_h[fam_h.split == "train"]
    gtm = float(np.mean(tr.y_true.to_numpy(np.float64)))
    tmean = tr.groupby("ticker").y_true.apply(lambda s: float(np.mean(s.to_numpy(np.float64))))
    tgt = meta[["ticker", "anchor_date", f"target_date_{h}d"]].rename(columns={f"target_date_{h}d": "target"})
    lab = fam_h.merge(tgt, on=["ticker", "anchor_date"], how="left", validate="one_to_one")
    if lab.target.isna().any():
        raise ValueError("Missing target dates for retained labels")
    qt, qd = test_keys.ticker.to_numpy(), test_keys.anchor_date.to_numpy()
    refs = {
        "zero": np.zeros(len(test_keys)),
        "global_train_mean": np.full(len(test_keys), gtm),
        "ticker_train_mean": test_keys.ticker.map(tmean).to_numpy(np.float64),
        "expanding_ticker_mean": expanding_mean(lab.ticker.to_numpy(), lab.target.to_numpy(), lab.y_true, qt, qd, pooled=False),
        "expanding_pooled_mean": expanding_mean(lab.ticker.to_numpy(), lab.target.to_numpy(), lab.y_true, qt, qd, pooled=True),
    }
    for k, v in refs.items():
        if not np.isfinite(v).all():
            raise ValueError(f"reference {k} not finite for h={h}")
    return refs, gtm
