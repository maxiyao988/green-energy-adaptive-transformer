"""Step 8 - portfolio allocation and transaction-cost evaluation (no training, single thread).

Timing (no look-ahead): a forecast made at the close of anchor t is executed at the close of t+1 and held
for h trading days (to the close of t+1+h). The holding interval is therefore shifted by one day relative to
the label interval (close t -> close t+h). Rebalancing uses non-overlapping anchors (first anchor, then every
h-th anchor); because anchors are consecutive trading days, holding periods tile without gaps. Between
rebalances, weights drift with realised daily simple returns from adjusted closes; value is compounded daily.
Covariance at decision date t: Ledoit-Wolf on the 252 daily returns up to and including t, scaled by h.

Strategies
  L1_top3 / L1_top5   long-only equal weight in the k highest scores
  LS_top3_bottom3     top-3 minus bottom-3 spread, diagnostic only (gross, per holding period)
  L2_g{gamma}         long-only mean-variance: max w'mu - gamma/2 w'Sigma w, sum w = 1, 0 <= w <= 0.25,
                      mu = exp(yhat) - 1; gamma = 5 primary, grid {1, 2, 5, 10, 20}; SLSQP, failures fall back
                      to the previous weights and are logged
  EW, MINVAR          1/N and minimum variance (same Sigma and constraints)
  RISKPARITY          long-only equal risk contribution with the same Sigma: minimise 1/2 y'Sigma y - (1/N) sum log y_i
                      (strictly convex; damped Newton), w = y / sum y; risk contributions w_i (Sigma w)_i / w'Sigma w are
                      checked equal to 1/N within 1e-6 at every decision date (no 0.25 cap is imposed)
Costs c in {0, 5, 10, 20} bp per unit of one-way traded value, turnover = sum |w_target - w_drifted|
(the initial build counts as turnover 1), deducted at each rebalance.

Usage:
  python -B engine.py --run-label original --preds GLOB [...] [--baseline-preds GLOB ...]
                      [--adaptive-formal-csv CSV|none] [--signals | --no-signals]
"""
import argparse
import os
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "skill"))
sys.path.insert(0, str(HERE.parent / "ensemble"))
from skill_common import (ROOT, HORIZONS, expand_globs, parse_name, load_pred, canon_dates, nw_lrv, holm, sha256, utc, rel,  # noqa: E402
                          SPLIT_COUNTS, np, pd, stats)
from ensemble_analysis import load_formal_csv, FORMAL_CSV  # noqa: E402
from a1_benchmarks import signal_table, SIGNALS  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

TICKERS = ["AES", "BEP", "BLDP", "BLNK", "CSIQ", "CWEN", "DQ", "ENPH", "FCEL", "FSLR", "HASI", "JKS", "NEE", "ORA", "PLUG", "RUN", "SEDG"]
RAW = ROOT / "dataset" / "finance" / "panel_raw"
RF = ROOT / "data_raw_revision" / "rf" / "DTB3.csv"
OUT_BASE = ROOT / "revision_work" / "outputs" / "portfolio"
GAMMAS = [1, 2, 5, 10, 20]
GAMMA0 = 5
COSTS_BP = [0, 5, 10, 20]
COST0 = 10
WMAX = 0.25
LOOKBACK = 252
BOOT_B = 10000
BOOT_SEED = 20261018
ANN = 252
A1_FITS = ROOT / "revision_work" / "outputs" / "skill_analysis" / "a1" / "a1_signal_ols_fits.csv"


# ---------------------------------------------------------------- exclusive output
class Writer:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_relative_to(OUT_BASE.resolve()) and not os.environ.get("PORTFOLIO_ALLOW_OUT"):
            raise ValueError(f"output root must be under {OUT_BASE}")
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Refusing to write into non-empty run directory {self.root}")

    def path(self, name):
        p = (self.root / name).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(f"Refusing overwrite: {p}")
        return p

    def bytes(self, name, b):
        with self.path(name).open("xb") as f:
            f.write(b)

    def csv(self, name, df):
        self.bytes(name, df.to_csv(index=False, float_format="%.10g").encode())

    def text(self, name, s):
        self.bytes(name, s.encode())


# ---------------------------------------------------------------- data
def load_prices():
    closes = {}
    for t in TICKERS:
        p = pd.read_csv(RAW / f"{t}_daily.csv")
        p["date"] = canon_dates(p["datetime"])
        closes[t] = p.drop_duplicates("date").set_index("date").close.astype(float)
    C = pd.DataFrame(closes).dropna().sort_index()  # common trading calendar
    R = C.pct_change().iloc[1:]
    return C.iloc[1:], R


def load_rf(dates):
    d = pd.read_csv(RF)
    d["date"] = canon_dates(d.Date)
    s = d.set_index("date").daily_simple_rate.astype(float)
    s = s.reindex(sorted(set(s.index) | set(dates))).sort_index()
    raw_missing = s.reindex(dates).isna()
    f = s.ffill(limit=5).reindex(dates)
    return f, int((raw_missing & f.notna()).sum()), int(f.isna().sum())


def pivot(df):
    w = df.pivot(index="anchor_date", columns="ticker", values="y_pred").sort_index()
    return w[TICKERS]


# ---------------------------------------------------------------- optimisation
def polish_active_set(mu, Q, w, ub=WMAX, tol=1e-10, max_iter=200):
    """Exact solution of min 1/2 w'Qw - mu'w s.t. sum w = 1, 0 <= w <= ub by a primal active-set method
    warm-started at w. Returns (w, max KKT violation) or (None, reason)."""
    n = len(mu)
    lo = set(np.where(w <= 1e-7)[0])
    up = set(np.where(w >= ub - 1e-7)[0]) - lo
    for _ in range(max_iter):
        F = [i for i in range(n) if i not in lo and i not in up]
        x = np.zeros(n)
        for i in up:
            x[i] = ub
        if not F:
            # vertex (every weight at a bound, e.g. four names at the 0.25 cap): optimal iff the budget
            # multiplier lam has a feasible interval: lam >= -g_i on lower bounds, lam <= -g_i on upper bounds
            if abs(x.sum() - 1) > 1e-9:
                return None, "vertex with sum != 1"
            g0 = Q @ x - mu
            lam_lo = max((-g0[i] for i in lo), default=-np.inf)
            lam_hi = min((-g0[i] for i in up), default=np.inf)
            if lam_lo <= lam_hi + tol:
                return x, 0.0
            i = max(lo, key=lambda j: -g0[j])  # most violated lower bound -> free it
            lo.discard(i)
            continue
        QF = Q[np.ix_(F, F)]
        rhs_F = mu[F] - Q[np.ix_(F, sorted(up))] @ x[sorted(up)] if up else mu[F].copy()
        K = np.block([[QF, np.ones((len(F), 1))], [np.ones((1, len(F))), np.zeros((1, 1))]])
        sol = np.linalg.solve(K, np.concatenate([rhs_F, [1.0 - x.sum()]]))
        xF, lam = sol[:-1], sol[-1]
        bad_lo = [(xF[k], F[k]) for k in range(len(F)) if xF[k] < -tol]
        bad_up = [(xF[k] - ub, F[k]) for k in range(len(F)) if xF[k] > ub + tol]
        if bad_lo or bad_up:
            if bad_lo and (not bad_up or min(bad_lo)[0] <= -max(bad_up)[0]):
                lo.add(min(bad_lo)[1])
            else:
                up.add(max(bad_up)[1])
            continue
        x[F] = xF
        g = Q @ x - mu + lam  # stationarity residual; = nu_lo - nu_up on bounds
        viol_lo = [(g[i], i) for i in lo if g[i] < -tol]   # need g_i >= 0 at lower bound
        viol_up = [(-g[i], i) for i in up if g[i] > tol]   # need g_i <= 0 at upper bound
        if viol_lo or viol_up:
            cand = (viol_lo + viol_up)
            i = min(cand)[1]
            lo.discard(i)
            up.discard(i)
            continue
        kkt = max(float(np.abs(g[F]).max()) if F else 0.0, abs(x.sum() - 1), float(max(0, -x.min())), float(max(0, x.max() - ub)))
        return x, kkt
    return None, "active set did not converge"


def risk_parity(S, max_iter=200):
    """Equal-risk-contribution weights via the convex formulation min 1/2 y'Sy - (1/N) sum log y (damped Newton
    with positivity-preserving backtracking), normalised to sum 1. Returns (w, max |RC_i - 1/N|, iterations)."""
    n = S.shape[0]
    f = lambda z: 0.5 * z @ S @ z - np.log(z).sum() / n
    y = np.full(n, 1.0 / np.sqrt(n * np.ones(n) @ S @ np.ones(n) / n))
    it = 0
    for it in range(1, max_iter + 1):
        g = S @ y - (1.0 / n) / y
        H = S + np.diag((1.0 / n) / y ** 2)
        step = np.linalg.solve(H, g)
        dec = float(g @ step)  # Newton decrement squared
        if dec < 1e-24:
            break
        t = 1.0
        while np.any(y - t * step <= 0):
            t *= 0.5
        while f(y - t * step) > f(y) - 0.25 * t * dec and t > 1e-16:
            t *= 0.5
        y = y - t * step
        wn = y / y.sum()
        if np.abs(wn * (S @ wn) / (wn @ S @ wn) - 1.0 / n).max() < 1e-12 or t * np.abs(step).max() < 1e-15 * y.max():
            break  # converged (or the step is below machine precision)
    w = y / y.sum()
    rc = w * (S @ w) / (w @ S @ w)
    return w, float(np.abs(rc - 1.0 / n).max()), it


def solve_mv(mu, S, gamma, x0):
    n = len(mu)
    Q = gamma * S
    f = lambda w: -w @ mu + 0.5 * w @ Q @ w
    g = lambda w: -mu + Q @ w
    res = minimize(f, x0, jac=g, method="SLSQP", bounds=[(0.0, WMAX)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
                   options={"ftol": 1e-12, "maxiter": 1000})
    w, kkt = polish_active_set(mu, Q, np.clip(res.x, 0, WMAX))
    if w is None or kkt > 1e-8:
        return None, f"SLSQP: {res.message}; active-set polish: {kkt}"
    return w, "ok"


# ---------------------------------------------------------------- simulation
def simulate(R, cal_pos, exec_idx, targets, end_idx):
    """Gross daily returns from exec_idx[0] to end_idx (first day = 0 gross), turnover per rebalance,
    and per-period gross returns. targets: list of weight vectors (one per rebalance)."""
    days = np.arange(exec_idx[0], end_idx + 1)
    gross = np.zeros(len(days))
    to, per = [], []
    w_drift = np.zeros(R.shape[1])
    for k, (e, wt) in enumerate(zip(exec_idx, targets)):
        to.append(float(np.abs(wt - w_drift).sum()))
        nxt = exec_idx[k + 1] if k + 1 < len(exec_idx) else end_idx
        w = wt.copy()
        v = 1.0
        for d in range(e + 1, nxt + 1):
            r = R[d]
            rp = float(w @ r)
            gross[d - exec_idx[0]] = rp
            v *= 1 + rp
            w = w * (1 + r) / (1 + rp)
        per.append(v - 1)
        w_drift = w
    return days, gross, np.array(to), np.array(per)


def apply_costs(days, gross, exec_idx, to, per, c):
    net = gross.copy()
    rel_idx = np.array(exec_idx) - exec_idx[0]
    net[rel_idx] = (1 + gross[rel_idx]) * (1 - c * to) - 1
    per_net = (1 - c * to) * (1 + per) - 1
    return net, per_net


def ols_nw(y, x, lag=5):
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lag + 1):
        G = Xu[l:].T @ Xu[:-l]
        S += (1 - l / (lag + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.diag(V)), u


def metrics(net, per_net, to, rf, h, bench_ex=None):
    n = len(net)
    ex = net - rf
    v = np.cumprod(1 + net)
    m = dict(n_days=n, n_rebalances=len(to), ann_geo_return=float(v[-1] ** (ANN / n) - 1), ann_vol=float(net.std(ddof=1) * np.sqrt(ANN)),
             sharpe=float(ex.mean() / ex.std(ddof=1) * np.sqrt(ANN)) if ex.std() > 0 else np.nan,
             max_drawdown=float((v / np.maximum.accumulate(np.concatenate([[1.0], v]))[1:] - 1).min()),
             avg_turnover_ex_initial=float(to[1:].mean()) if len(to) > 1 else np.nan, initial_turnover=float(to[0]),
             ann_turnover=float(to[1:].mean() * ANN / h) if len(to) > 1 else np.nan,
             hit_rate_periods=float((per_net > 0).mean()), cumulative_return=float(v[-1] - 1),
             ce_A2=float((net.mean() - 1.0 * net.var(ddof=1)) * ANN), ce_A5=float((net.mean() - 2.5 * net.var(ddof=1)) * ANN))
    if bench_ex is not None:
        b, se, _ = ols_nw(ex, bench_ex, lag=5)
        m.update(alpha_vs_ew_ann=float(b[0] * ANN), alpha_vs_ew_nw_t=float(b[0] / se[0]), beta_vs_ew=float(b[1]))
    return m


def grs(Rex, f):
    T, N = Rex.shape
    X = np.column_stack([np.ones(T), f])
    B = np.linalg.lstsq(X, Rex, rcond=None)[0]
    E = Rex - X @ B
    Sig = E.T @ E / (T - 2)
    a = B[0]
    mu_f, var_f = f.mean(), f.var(ddof=1)
    stat = (T / N) * ((T - N - 1) / (T - 2)) * float(a @ np.linalg.solve(Sig, a)) / (1 + mu_f ** 2 / var_f)
    return stat, float(stats.f.sf(stat, N, T - N - 1)), T, N


def cbb_diff(ra, rs, rf, h, B, tag):
    """Circular block bootstrap (block = h days) of the paired daily returns; CIs for the annualised mean
    return difference and the Sharpe difference."""
    T = len(ra)
    rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, h, h, zlib.crc32(tag.encode()) % (2 ** 31)]))
    K = int(np.ceil(T / h))
    starts = rng.integers(0, T, size=(B, K))
    idx = ((starts[:, :, None] + np.arange(h)[None, None, :]) % T).reshape(B, K * h)[:, :T]
    d = ra - rs
    md = d[idx].mean(1) * ANN
    ea, es = (ra - rf)[idx], (rs - rf)[idx]
    sd = ea.mean(1) / ea.std(1, ddof=1) * np.sqrt(ANN) - es.mean(1) / es.std(1, ddof=1) * np.sqrt(ANN)
    out = {}
    for k, b, t0 in (("mean_ret_diff_ann", md, d.mean() * ANN),
                     ("sharpe_diff", sd, (ra - rf).mean() / (ra - rf).std(ddof=1) * np.sqrt(ANN) - (rs - rf).mean() / (rs - rf).std(ddof=1) * np.sqrt(ANN))):
        lo, hi = np.quantile(b, [0.025, 0.975])
        out.update({f"{k}": float(t0), f"{k}_ci_lo": float(lo), f"{k}_ci_hi": float(hi), f"{k}_ci_excludes_0": bool(lo > 0 or hi < 0),
                    f"{k}_p_centered": float((1 + np.sum(np.abs(b - t0) >= abs(t0))) / (B + 1))})
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="model val/test parquet cells (daily_weekly only are used)")
    ap.add_argument("--baseline-preds", nargs="*", default=[], help="TEST-only baseline cells (e.g. canonical ridge/lstm/vanilla/swim)")
    ap.add_argument("--adaptive-formal-csv", default=str(FORMAL_CSV))
    ap.add_argument("--train-labels", default=str(ROOT / "revision_work/outputs/preds/static_dw_h{h}_s0_train.parquet"))
    ap.add_argument("--no-signals", action="store_true")
    ap.add_argument("--models", default=None, help="regex: keep only prediction sets whose name matches (gtm is always kept as a check)")
    ap.add_argument("--report-subset", default=None, help="comma list: report only these strategies (+ static/adaptive seed means)")
    ap.add_argument("--boot-b", type=int, default=BOOT_B)
    ap.add_argument("--out-root", default=str(OUT_BASE))
    a = ap.parse_args()
    W = Writer(Path(a.out_root) / a.run_label)
    blocking, qp_log, notes, rp_rows = [], [], [], []
    C, Rdf = load_prices()
    cal = list(Rdf.index)
    pos = {d: i for i, d in enumerate(cal)}
    R = Rdf.to_numpy()
    rf_s, n_ffill, n_rf_missing = load_rf(cal)
    rf = rf_s.to_numpy()
    # ---- predictions
    sets = {}  # (name, h, split) -> pivot DataFrame
    meta = {}
    for f in expand_globs(a.preds) + (expand_globs(a.baseline_preds) if a.baseline_preds else []):
        info = parse_name(f)
        if info["config"] != "daily_weekly" or info["split"] not in ("val", "test"):
            continue
        d = load_pred(f, expect_rows=SPLIT_COUNTS[info["split"]])
        name = f"{info['model']}_s{info['seed']}"
        sets[(name, info["h"], info["split"])] = pivot(d)
        meta[(name, info["h"], info["split"])] = rel(f)
    if a.adaptive_formal_csv.lower() != "none":
        for (mech, h, seed, sp), (d, src) in load_formal_csv(a.adaptive_formal_csv).items():
            key = (f"adaptive_s{seed}", h, sp)
            if key in sets:
                blocking.append(f"formal CSV duplicates {meta[key]}")
            sets[key], meta[key] = pivot(d), src
    anchors = {}
    for (name, h, sp), P in sets.items():
        ref = anchors.setdefault((h, sp), list(P.index))
        if list(P.index) != ref or P.isna().any().any():
            blocking.append(f"{name} h={h} {sp}: anchor dates / tickers differ from the reference set")
    for (h, sp), ad in anchors.items():
        if any(d not in pos for d in ad):
            blocking.append(f"h={h} {sp}: anchor date not in the common price calendar")
    # ensembles (average predictions first)
    for mech in ("static", "adaptive"):
        for h in HORIZONS:
            for sp in ("val", "test"):
                ks = sorted([k for k in sets if k[0].startswith(mech + "_s") and k[1] == h and k[2] == sp], key=lambda k: int(k[0].split("_s")[1]))
                if len(ks) >= 2:
                    sets[(f"{mech}_ens", h, sp)] = sum(sets[k] for k in ks) / len(ks)
                    meta[(f"{mech}_ens", h, sp)] = "mean of " + ",".join(k[0] for k in ks)
    # signals (raw for ranking; TRAIN-only OLS-calibrated for mu) and GTM
    if not a.no_signals:
        fits = pd.read_csv(A1_FITS)
        keys = pd.DataFrame([(t, d) for (h, sp), ad in anchors.items() if h == 5 for d in ad for t in TICKERS], columns=["ticker", "anchor_date"]).drop_duplicates()
        sig = signal_table(keys)
        for s in SIGNALS:
            raw = sig.pivot(index="anchor_date", columns="ticker", values=s)[TICKERS]
            for h in HORIZONS:
                fr = fits[(fits.h == h) & (fits.signal == s)].iloc[0]
                for sp in ("val", "test"):
                    if (h, sp) not in anchors:
                        continue
                    rr = raw.loc[anchors[(h, sp)]]
                    if rr.isna().any().any():
                        blocking.append(f"signal {s} undefined at some {sp} anchors")
                    sets[(f"sig_{s}", h, sp)] = fr.intercept + fr.slope * rr  # calibrated yhat (Layer 2)
                    sets[(f"sigraw_{s}", h, sp)] = rr  # raw signal (Layer 1 ranking)
                    meta[(f"sig_{s}", h, sp)] = f"OLS a={fr.intercept:.5f} b={fr.slope:.5f} (TRAIN only, Step 5 A1)"
    gtm = {}
    for h in HORIZONS:
        tr = pd.read_parquet(a.train_labels.format(h=h), columns=["y_true"])
        gtm[h] = float(np.mean(tr.y_true.to_numpy(np.float64)))
        for sp in ("val", "test"):
            if (h, sp) in anchors:
                sets[("gtm", h, sp)] = pd.DataFrame(gtm[h], index=anchors[(h, sp)], columns=TICKERS)
    if blocking:
        W.text("REPORT_STEP8.md", "# REPORT STEP 8\n\n## BLOCKING\n\n" + "\n".join(f"- {b}" for b in blocking) + "\n")
        raise SystemExit("BLOCKING: " + "; ".join(blocking))

    # ---------------------------------------------------------------- run all strategies
    daily_rows, reb_rows, met_rows, ls_rows, w_rows, cum_rows = [], [], [], [], [], []
    sims = {}  # (split, h, strategy) -> (days, gross, exec_idx, to, per)
    for (h, sp), ad in sorted(anchors.items()):
        reb = ad[0::h]
        exec_idx = [pos[d] + 1 for d in reb]
        end_idx = exec_idx[-1] + h
        if end_idx > len(cal) - 1:
            notes.append(f"{sp} h={h}: last holding period truncated at {cal[-1]}")
            end_idx = len(cal) - 1
        Sig = {}
        for d in reb:
            i = pos[d]
            X = R[i - LOOKBACK + 1:i + 1]
            if len(X) < LOOKBACK:
                blocking.append(f"{sp} h={h} {d}: fewer than {LOOKBACK} return days")
            Sig[d] = LedoitWolf().fit(X).covariance_ * h
        strategies = {}
        n = len(TICKERS)
        strategies["EW"] = [np.full(n, 1 / n) for _ in reb]

        def mv_path(mu_by_date, gamma, label):
            prev, out = np.full(n, 1 / n), []
            for d in reb:
                w, msg = solve_mv(mu_by_date[d], Sig[d], gamma, prev)
                if w is None:
                    qp_log.append(dict(split=sp, h=h, strategy=label, date=d, message=str(msg), action="fallback to previous weights"))
                    w = prev
                out.append(w)
                prev = w
            return out
        strategies["MINVAR"] = mv_path({d: np.zeros(n) for d in reb}, 1.0, "MINVAR")
        rp = []
        for d in reb:
            w, dev, iters = risk_parity(Sig[d])
            rp_rows.append(dict(split=sp, h=h, decision_date=d, max_abs_rc_deviation=dev, max_weight=float(w.max()), min_weight=float(w.min()), newton_iterations=iters))
            if dev > 1e-6:
                blocking.append(f"{sp} h={h} {d}: risk-parity risk contributions not equal (max |RC - 1/N| = {dev:.2e})")
            rp.append(w)
        strategies["RISKPARITY"] = rp
        names = sorted({k[0] for k in sets if k[1] == h and k[2] == sp and not k[0].startswith("sigraw_")
                        and (a.models is None or k[0] == "gtm" or __import__("re").fullmatch(a.models, k[0]))})
        for name in names:
            P = sets[(name, h, sp)]
            mu = {d: np.expm1(P.loc[d].to_numpy(np.float64)) for d in reb}
            for g in GAMMAS:
                strategies[f"{name}|L2_g{g}"] = mv_path(mu, float(g), f"{name}|L2_g{g}")
            if name == "gtm":
                continue  # constant scores: Layer 1 undefined
            rank_src = sets[(name.replace("sig_", "sigraw_"), h, sp)] if name.startswith("sig_") else P
            for k in (3, 5):
                ws = []
                for d in reb:
                    order = np.argsort(-rank_src.loc[d].to_numpy(np.float64), kind="mergesort")
                    w = np.zeros(n)
                    w[order[:k]] = 1 / k
                    ws.append(w)
                strategies[f"{name}|L1_top{k}"] = ws
            # long-short diagnostic (gross, per holding period)
            spread = []
            for kk, d in enumerate(reb):
                order = np.argsort(-rank_src.loc[d].to_numpy(np.float64), kind="mergesort")
                e, nx = exec_idx[kk], (exec_idx[kk + 1] if kk + 1 < len(exec_idx) else end_idx)
                cum = np.prod(1 + R[e + 1:nx + 1], axis=0) - 1
                spread.append(cum[order[:3]].mean() - cum[order[-3:]].mean())
            spread = np.array(spread)
            ls_rows.append(dict(split=sp, h=h, strategy=name, n_periods=len(spread), mean_period_spread=float(spread.mean()),
                                ann_spread=float(spread.mean() * ANN / h), t_stat=float(spread.mean() / spread.std(ddof=1) * np.sqrt(len(spread))),
                                hit_rate=float((spread > 0).mean())))
        # GTM Layer 2 must equal minimum variance (constant mu)
        if ("gtm", h, sp) in sets:
            dmax = max(float(np.abs(a_ - b_).max()) for a_, b_ in zip(strategies[f"gtm|L2_g{GAMMA0}"], strategies["MINVAR"]))
            notes.append(f"{sp} h={h}: max |w(GTM, L2 gamma=5) - w(MINVAR)| = {dmax:.2e}")
            if dmax > 1e-4:
                blocking.append(f"{sp} h={h}: GTM Layer 2 does not reproduce minimum variance (max weight diff {dmax:.2e})")
        for sname, targets in strategies.items():
            days, gross, to, per = simulate(R, pos, exec_idx, targets, end_idx)[0:4]
            sims[(sp, h, sname)] = (days, gross, exec_idx, to, per)
            for d, wt in zip(reb, targets):
                w_rows.append(dict(split=sp, h=h, strategy=sname, decision_date=d, execution_date=cal[pos[d] + 1], **dict(zip(TICKERS, wt))))
        # metrics
        ew_cache = {}
        for c_bp in COSTS_BP:
            c = c_bp / 1e4
            dE, gE, eE, tE, pE = sims[(sp, h, "EW")]
            ew_net, _ = apply_costs(dE, gE, eE, tE, pE, c)
            ew_ex = ew_net - rf[dE]
            for sname in strategies:
                days, gross, ex_i, to, per = sims[(sp, h, sname)]
                net, per_net = apply_costs(days, gross, ex_i, to, per, c)
                m = metrics(net, per_net, to, rf[days], h, None if sname == "EW" else ew_ex)
                name, _, layer = sname.partition("|")
                met_rows.append(dict(split=sp, h=h, strategy=name, layer=layer or "allocation benchmark", cost_bp=c_bp, **m))
                if c_bp == COST0:
                    daily_rows.append(pd.DataFrame(dict(split=sp, h=h, strategy=sname, date=[cal[i] for i in days], gross=gross, net_10bp=net)))
        for sname in strategies:
            days, gross, ex_i, to, per = sims[(sp, h, sname)]
            for k, e in enumerate(ex_i):
                reb_rows.append(dict(split=sp, h=h, strategy=sname, decision_date=reb[k], execution_date=cal[e], turnover=to[k], period_gross_return=per[k]))
        if blocking:
            break
    if blocking:
        W.text("REPORT_STEP8.md", "# REPORT STEP 8\n\n## BLOCKING\n\n" + "\n".join(f"- {b}" for b in blocking) + "\n")
        raise SystemExit("BLOCKING: " + "; ".join(blocking))
    M = pd.DataFrame(met_rows)

    # ---------------------------------------------------------------- Static vs Adaptive
    sva, grs_rows = [], []
    layers = ["L1_top3", "L1_top5", f"L2_g{GAMMA0}"]
    for (h, sp) in sorted(anchors):
        seeds_s = {k[0].split("_s")[1] for k in sets if k[0].startswith("static_s") and k[1] == h and k[2] == sp}
        seeds_a = {k[0].split("_s")[1] for k in sets if k[0].startswith("adaptive_s") and k[1] == h and k[2] == sp}
        pairs = [(f"seed {s}", f"adaptive_s{s}", f"static_s{s}") for s in sorted(seeds_s & seeds_a, key=int)] + [("ensembles", "adaptive_ens", "static_ens")]
        for layer in layers:
            for label, an, sn in pairs:
                if (sp, h, f"{an}|{layer}") not in sims or (sp, h, f"{sn}|{layer}") not in sims:
                    continue
                res = {}
                for side, nm in (("a", an), ("s", sn)):
                    days, gross, ex_i, to, per = sims[(sp, h, f"{nm}|{layer}")]
                    res[side] = apply_costs(days, gross, ex_i, to, per, COST0 / 1e4)[0]
                ma = M[(M.split == sp) & (M.h == h) & (M.strategy == an) & (M.layer == layer) & (M.cost_bp == COST0)].iloc[0]
                ms = M[(M.split == sp) & (M.h == h) & (M.strategy == sn) & (M.layer == layer) & (M.cost_bp == COST0)].iloc[0]
                bt = cbb_diff(res["a"], res["s"], rf[days], h, a.boot_b, f"{sp}|{h}|{layer}|{label}")
                sva.append(dict(split=sp, h=h, layer=layer, pair=label, cost_bp=COST0, sharpe_adaptive=ma.sharpe, sharpe_static=ms.sharpe,
                                ann_geo_ret_adaptive=ma.ann_geo_return, ann_geo_ret_static=ms.ann_geo_return,
                                ann_geo_ret_diff=ma.ann_geo_return - ms.ann_geo_return, B=a.boot_b, block_days=h, **bt))
            seed_rows = [r for r in sva if r["split"] == sp and r["h"] == h and r["layer"] == layer and r["pair"].startswith("seed")]
            if seed_rows:
                S_ = pd.DataFrame(seed_rows)
                sva.append(dict(split=sp, h=h, layer=layer, pair=f"seed average (n={len(S_)})", cost_bp=COST0,
                                sharpe_adaptive=S_.sharpe_adaptive.mean(), sharpe_static=S_.sharpe_static.mean(), sharpe_diff=S_.sharpe_diff.mean(),
                                ann_geo_ret_adaptive=S_.ann_geo_ret_adaptive.mean(), ann_geo_ret_static=S_.ann_geo_ret_static.mean(),
                                ann_geo_ret_diff=S_.ann_geo_ret_diff.mean(), n_seed_pairs_adaptive_higher_sharpe=int((S_.sharpe_diff > 0).sum())))
        # GRS: Layer 2, gamma 5, 10 bp vs 1/N
        for label, members in (("ensembles", ["static_ens", "adaptive_ens"]),
                               ("all seeds", sorted([k[0] for k in sets if k[1] == h and k[2] == sp and (k[0].startswith("static_s") or k[0].startswith("adaptive_s"))]))):
            days = sims[(sp, h, "EW")][0]
            dE, gE, eE, tE, pE = sims[(sp, h, "EW")]
            f_ex = apply_costs(dE, gE, eE, tE, pE, COST0 / 1e4)[0] - rf[dE]
            cols = []
            for nm in members:
                d_, g_, e_, t_, p_ = sims[(sp, h, f"{nm}|L2_g{GAMMA0}")]
                cols.append(apply_costs(d_, g_, e_, t_, p_, COST0 / 1e4)[0] - rf[d_])
            stat, p, T, N = grs(np.column_stack(cols), f_ex)
            grs_rows.append(dict(split=sp, h=h, assets=label, members=",".join(members), N=N, T_days=T, GRS_F=stat, p_value=p))
    SVA, GRS = pd.DataFrame(sva), pd.DataFrame(grs_rows)
    # across-seed mean / SD per mechanism
    ss = M[M.strategy.str.match(r"^(static|adaptive)_s\d+$") & (M.cost_bp == COST0)].copy()
    ss["mechanism"] = ss.strategy.str.split("_").str[0]
    mcols = ["ann_geo_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover_ex_initial", "hit_rate_periods", "ce_A5", "alpha_vs_ew_ann"]
    SEEDSTAT = ss.groupby(["split", "h", "layer", "mechanism"])[mcols].agg(["mean", "std", "count"])
    SEEDSTAT.columns = [f"{a_}_{b_}" for a_, b_ in SEEDSTAT.columns]
    SEEDSTAT = SEEDSTAT.reset_index()

    # ---------------------------------------------------------------- write CSVs
    W.csv("metrics_all.csv", M)
    W.csv("static_vs_adaptive.csv", SVA)
    W.csv("grs_tests.csv", GRS)
    W.csv("seed_dispersion.csv", SEEDSTAT)
    W.csv("long_short_diagnostic.csv", pd.DataFrame(ls_rows))
    W.csv("rebalances.csv", pd.DataFrame(reb_rows))
    W.csv("weights.csv", pd.DataFrame(w_rows))
    D = pd.concat(daily_rows, ignore_index=True)
    W.csv("daily_returns_gross_and_net10bp.csv", D)
    W.csv("qp_solver_log.csv", pd.DataFrame(qp_log, columns=["split", "h", "strategy", "date", "message", "action"]))
    RP = pd.DataFrame(rp_rows)
    W.csv("risk_parity_check.csv", RP)

    # ---------------------------------------------------------------- figure
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    series = [("static_ens|L2_g5", "Static ensemble (MV, γ=5)", "Static", "#2a78d6"), ("adaptive_ens|L2_g5", "Adaptive ensemble (MV, γ=5)", "Adaptive", "#eb6834"),
              ("EW", "1/N", "1/N", "#1baf7a"), ("MINVAR", "Minimum variance", "MinVar", "#eda100"), ("RISKPARITY", "Risk parity", "RiskPar", "#e87ba4")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), facecolor="#fcfcfb")
    handles = []
    for ax, h in zip(axes, HORIZONS):
        ax.set_facecolor("#fcfcfb")
        ends = []
        for key, lab, short, col in series:
            g = D[(D.split == "test") & (D.h == h) & (D.strategy == key)]
            if g.empty:
                continue
            v = np.cumprod(1 + g.net_10bp.to_numpy()) - 1
            x = pd.to_datetime(g.date)
            ln, = ax.plot(x, 100 * v, color=col, lw=2, label=lab)
            if h == HORIZONS[0]:
                handles.append(ln)
            ends.append([100 * v[-1], f"{short} {100 * v[-1]:+.0f}%", x.iloc[-1]])
            cum_rows.append(pd.DataFrame(dict(h=h, strategy=key, label=lab, date=g.date.to_numpy(), cum_return_net_10bp=v)))
        # de-overlap end labels (minimum vertical gap in data units)
        lo_, hi_ = ax.get_ylim()
        gap = 0.06 * (hi_ - lo_)
        ends.sort(key=lambda e: e[0])
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for (yv, txt, xe), yl in zip(ends, ys):
            ax.annotate(txt, (xe, yv), xytext=(xe, yl), textcoords="data", fontsize=8, color="#3d3d3a", va="center",
                        ha="left", annotation_clip=False)
            ax.annotate("", (xe, yv), xytext=(6, 0), textcoords="offset points")
        ax.margins(x=0.02)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.grid(axis="y", color="#e8e7e2", lw=0.6)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        loc = mdates.AutoDateLocator(minticks=4, maxticks=6)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.set_title(f"h = {h} days", fontsize=11, color="#1f1f1d", loc="left")
        ax.set_ylabel("Cumulative return, net of 10 bp (%)" if h == HORIZONS[0] else "", color="#5f5e5a")
        ax.tick_params(colors="#5f5e5a", labelsize=8)
    fig.suptitle("TEST period: cumulative net returns (17-stock surviving universe)", fontsize=12, color="#1f1f1d", x=0.01, ha="left", y=0.99)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.94), ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.97, 0.88))
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    W.bytes("fig_cumulative_test_net10bp.png", buf.getvalue())
    W.csv("fig_cumulative_test_net10bp_data.csv", pd.concat(cum_rows, ignore_index=True))

    # ---------------------------------------------------------------- report
    from report8 import build_report, build_report_subset
    if a.report_subset:
        W.text("REPORT_STEP8.md", build_report_subset(a, M, RP, notes, anchors, n_ffill, n_rf_missing, blocking, a.report_subset.split(",")))
    else:
        W.text("REPORT_STEP8.md", build_report(a, M, SVA, GRS, SEEDSTAT, pd.DataFrame(ls_rows), pd.DataFrame(qp_log), notes, meta, anchors, gtm,
                                           n_ffill, n_rf_missing, blocking))
    W.text("run_info.json", pd.Series(dict(created=utc(), run_label=a.run_label, preds=a.preds, baseline_preds=a.baseline_preds,
                                           adaptive_formal_csv=a.adaptive_formal_csv, boot_b=a.boot_b, notes=notes,
                                           inputs={f"{k}": v for k, v in meta.items()})).to_json(indent=2, default_handler=str))
    print("done", W.root)


if __name__ == "__main__":
    main()
