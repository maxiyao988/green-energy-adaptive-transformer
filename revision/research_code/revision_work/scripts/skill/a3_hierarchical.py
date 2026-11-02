"""A3 - hierarchical inference for Adaptive - Static (seed x date).

D[s,t] = mean over 17 stocks of (Adaptive - Static) squared loss for seed s on date t, computed in
float32 then cast to float64, exactly as the archived Experiment 4 inference. Same for the Rank IC
difference dIC[s,t] = IC_A[s,t] - IC_S[s,t].
  1. Date-level MBB on the seed-averaged series (L=h primary, 2h sensitivity, B=20,000,
     SeedSequence([20260826,h,L])) - compared with the archived summary and replicates.
  2. Joint seed x date bootstrap: per replicate draw S seed indices with replacement (Adaptive/Static
     stay paired) and ONE set of moving date blocks applied to every drawn seed; statistic = mean over
     drawn seeds and resampled dates. RNG SeedSequence([20260826,h,L,1]) (seeds drawn first, then
     block starts). Percentile 95% CI, centred two-sided p, Holm across horizons (primary L only).
  3. Seed-level tests on per-seed mean differences: paired t (df=S-1), exact sign test, exact
     Wilcoxon signed-rank.
  4. Steps 1-3 repeated for the Rank IC difference.

Usage: python -B a3_hierarchical.py --adaptive GLOB [...] --static GLOB [...] --out-dir DIR [--archived-check]
"""
import argparse
from skill_common import *

E4I = ROOT / "dataset" / "audit" / "experiment4_inference"
IC_ARCHIVES = [
    (ROOT / "dataset/audit/experiment4_multiseed/experiment4_rank_ic_by_date.csv", None),
    (ROOT / "dataset/audit/experiment3_pathformer_ablation/experiment3_rank_ic_by_date.csv", None),
    (ROOT / "dataset/audit/main_v2/pathformer/panel_pathformer_rank_ic_by_date.csv", "adaptive"),
]


def load_cells(patterns, h):
    cells = {}
    for f in expand_globs(patterns):
        info = parse_name(f)
        if info["h"] != h or info["split"] != "test":
            continue
        if info["seed"] in cells:
            raise SystemExit(f"BLOCKING: duplicate seed {info['seed']} for h={h}: {f}")
        cells[info["seed"]] = load_pred(f)
    return cells


def series_stats(boot, theta):
    lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
    return dict(theta_hat=theta, bootstrap_mean=float(boot.mean()), bootstrap_se=float(np.std(boot, ddof=1)),
                ci_lower_95=float(lo), ci_upper_95=float(hi), ci_width=float(hi - lo),
                ci_excludes_zero=bool(lo > 0 or hi < 0),
                centered_p_two_sided=float((1 + np.sum(np.abs(boot - theta) >= abs(theta))) / (len(boot) + 1)))


def seed_tests(x):
    x = np.asarray(x, np.float64)
    n = len(x)
    t = stats.ttest_1samp(x, 0.0)
    k = int(np.sum(x > 0))
    nz = int(np.sum(x != 0))
    sign_p = float(stats.binomtest(k, nz, 0.5).pvalue) if nz else np.nan
    try:
        w = stats.wilcoxon(x, alternative="two-sided", method="exact")
        wstat, wp = float(w.statistic), float(w.pvalue)
    except ValueError:
        wstat, wp = np.nan, np.nan
    return dict(n_seeds=n, mean=float(x.mean()), sd=float(x.std(ddof=1)), n_positive=k,
                paired_t=float(t.statistic), paired_t_df=n - 1, paired_t_p_two_sided=float(t.pvalue),
                sign_test_p_two_sided=sign_p, wilcoxon_stat=wstat, wilcoxon_p_two_sided=wp,
                min_attainable_p_two_sided_sign_wilcoxon=float(2 * 0.5 ** n),
                min_attainable_p_one_sided_sign_wilcoxon=float(0.5 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive", nargs="+", required=True)
    ap.add_argument("--static", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--archived-check", action="store_true", help="compare with archived Experiment 4 inference")
    ap.add_argument("--boot-b", type=int, default=BOOT_B)
    a = ap.parse_args()
    out = Path(a.out_dir)
    B = a.boot_b
    sd_rows, date_rows, joint_rows, seed_rows, rep_rows, check = [], [], [], [], [], {}
    p_holm = {}
    for h in HORIZONS:
        A, S = load_cells(a.adaptive, h), load_cells(a.static, h)
        if sorted(A) != sorted(S) or len(A) < 2:
            raise SystemExit(f"BLOCKING: h={h} Adaptive seeds {sorted(A)} vs Static seeds {sorted(S)}")
        seeds = sorted(A, key=int)
        Dm, ICd, ICa, ICs = [], [], [], []
        for s in seeds:
            da, ds = A[s], S[s]
            if not (da[["ticker", "anchor_date"]].equals(ds[["ticker", "anchor_date"]])
                    and np.array_equal(da.y_true.to_numpy(np.float32), ds.y_true.to_numpy(np.float32))):
                raise SystemExit(f"BLOCKING: h={h} seed {s} Adaptive/Static not pairable")
            y32 = da.y_true.to_numpy(np.float32)
            ld = ((y32 - da.y_pred.to_numpy(np.float32)) ** 2 - (y32 - ds.y_pred.to_numpy(np.float32)) ** 2).astype(np.float64)
            Dm.append(ld.reshape(N_TEST_DATES, N_TICKERS).mean(1))
            Y, dates, _ = to_panel(da, "y_true")
            ia, is_ = rank_ic_rows(to_panel(da, "y_pred")[0], Y), rank_ic_rows(to_panel(ds, "y_pred")[0], Y)
            ICa.append(ia); ICs.append(is_); ICd.append(ia - is_)
            sd_rows.append(pd.DataFrame(dict(h=h, seed=int(s), anchor_date=dates, D_loss=Dm[-1], ic_adaptive=ia, ic_static=is_, D_ic=ia - is_)))
        D, DIC = np.vstack(Dm), np.vstack(ICd)
        if not (np.isfinite(D).all() and np.isfinite(DIC).all()):
            raise SystemExit(f"BLOCKING: non-finite D or IC difference at h={h}")
        nS = len(seeds)
        for metric, M in (("loss", D), ("rank_ic", DIC)):
            theta = float(M.mean())
            seed_rows.append(dict(h=h, metric=metric, seeds=",".join(seeds), **seed_tests(M.mean(1))))
            for spec, L in (("PRIMARY", h), ("SENSITIVITY", 2 * h)):
                # (1) date-only MBB on the seed-averaged series, archived RNG stream
                rng = np.random.default_rng(np.random.SeedSequence([BOOT_BASE_SEED, h, L]))
                series = M.mean(0)
                b_date = series[mbb_index(N_TEST_DATES, L, B, rng)].mean(1, dtype=np.float64)
                rd = dict(h=h, metric=metric, specification=spec, block_length=L, n_dates=N_TEST_DATES, n_seeds=nS, B=B,
                          rng="SeedSequence([20260826,h,L])", **series_stats(b_date, float(series.mean())))
                m_, se_, t_, _ = nw_t(series, h - 1)
                rd.update(hac_se=se_, hac_z=t_, hac_p_two_sided=float(2 * stats.norm.sf(abs(t_))))
                date_rows.append(rd)
                # (2) joint seed x date bootstrap
                rng = np.random.default_rng(np.random.SeedSequence([BOOT_BASE_SEED, h, L, 1]))
                sidx = rng.integers(0, nS, size=(B, nS))
                didx = mbb_index(N_TEST_DATES, L, B, rng)
                W = np.zeros((B, nS))
                np.add.at(W, (np.repeat(np.arange(B), nS), sidx.ravel()), 1.0 / nS)
                Sb = W @ M  # (B, T): seed-resampled date series
                b_joint = np.take_along_axis(Sb, didx, axis=1).mean(1)
                rj = dict(h=h, metric=metric, specification=spec, block_length=L, n_dates=N_TEST_DATES, n_seeds=nS, B=B,
                          rng="SeedSequence([20260826,h,L,1]); seeds then blocks", **series_stats(b_joint, theta))
                rj["ci_width_ratio_joint_over_date_only"] = rj["ci_width"] / rd["ci_width"]
                rj["se_ratio_joint_over_date_only"] = rj["bootstrap_se"] / rd["bootstrap_se"]
                joint_rows.append(rj)
                if spec == "PRIMARY":
                    p_holm[(metric, "joint", h)] = rj["centered_p_two_sided"]
                    p_holm[(metric, "date_only", h)] = rd["centered_p_two_sided"]
                rep_rows.append(pd.DataFrame(dict(h=h, metric=metric, specification=spec, block_length=L,
                                                  replicate=np.arange(B), date_only=b_date, joint=b_joint)))
        # --- archived checks
        if a.archived_check:
            arch_sd = pd.read_csv(E4I / "experiment4_loss_diff_by_seed_date.csv")
            arch_sd = arch_sd[arch_sd.horizon == f"{h}d"]
            mine = pd.concat([r for r in sd_rows if r.h.iloc[0] == h])
            mm = mine.merge(arch_sd.assign(seed=arch_sd.seed.astype(int), anchor_date=canon_dates(arch_sd.anchor_date)),
                            on=["seed", "anchor_date"], how="outer", validate="one_to_one")
            check[f"h{h}_seed_date_rows_matched"] = int(mm[["D_loss", "mean_loss_diff"]].notna().all(1).sum())
            check[f"h{h}_seed_date_max_abs_diff"] = float(np.nanmax(np.abs(mm.D_loss - mm.mean_loss_diff)))
            ic_arch = []
            for path, mech in IC_ARCHIVES:
                x = pd.read_csv(path)
                if mech:
                    x = x[x.frequency_setting == "daily_weekly"].assign(mechanism=mech)
                x = x.assign(h=x.horizon.astype(str).str.removesuffix("d").astype(int), anchor_date=canon_dates(x.anchor_date))
                ic_arch.append(x[(x.h == h) & x.mechanism.isin(["adaptive", "static"])][["mechanism", "seed", "anchor_date", "rank_ic"]])
            ic_arch = pd.concat(ic_arch)
            mi = pd.concat([mine.assign(mechanism="adaptive", rank_ic=mine.ic_adaptive),
                            mine.assign(mechanism="static", rank_ic=mine.ic_static)])[["mechanism", "seed", "anchor_date", "rank_ic"]]
            icm = mi.merge(ic_arch.assign(seed=ic_arch.seed.astype(int)), on=["mechanism", "seed", "anchor_date"], validate="one_to_one")
            check[f"h{h}_rank_ic_rows_matched"] = len(icm)
            check[f"h{h}_rank_ic_max_abs_diff_vs_archived"] = float(np.max(np.abs(icm.rank_ic_x - icm.rank_ic_y)))
    # Holm across horizons (primary specification only), per metric and bootstrap type
    for metric in ("loss", "rank_ic"):
        for kind, rows in (("joint", joint_rows), ("date_only", date_rows)):
            adj = holm({h: p_holm[(metric, kind, h)] for h in HORIZONS})
            for r in rows:
                if r["metric"] == metric:
                    r["centered_p_holm_primary"] = adj[r["h"]] if r["specification"] == "PRIMARY" else np.nan
    date_df, joint_df = pd.DataFrame(date_rows), pd.DataFrame(joint_rows)
    reps = pd.concat(rep_rows, ignore_index=True)
    if a.archived_check:
        arch = pd.read_csv(E4I / "experiment4_block_bootstrap_summary.csv")
        arch["h"] = arch.horizon.str.removesuffix("d").astype(int)
        arch = arch.rename(columns={c: f"archived_{c}" for c in arch.columns if c not in ("h", "specification")})
        comp = date_df[date_df.metric == "loss"].merge(arch, on=["h", "specification"], validate="one_to_one")
        pairs = {"theta_hat": "theta_hat", "bootstrap_mean": "bootstrap_mean", "bootstrap_se": "bootstrap_se",
                 "ci_lower_95": "ci_lower_95", "ci_upper_95": "ci_upper_95", "centered_p_two_sided": "centered_p_value",
                 "centered_p_holm_primary": "centered_p_value_holm_primary"}
        for mine_c, arch_c in pairs.items():
            comp[f"absdiff_{mine_c}"] = np.abs(comp[mine_c] - comp[f"archived_{arch_c}"]).fillna(0.0)
        comp = comp.rename(columns={"archived_classification": "classification"})
        arep = pd.read_csv(E4I / "experiment4_block_bootstrap_replicates.csv")
        arep["h"] = arep.horizon.str.removesuffix("d").astype(int)
        rm = reps[reps.metric == "loss"].merge(arep, on=["h", "specification", "replicate"], validate="one_to_one")
        check["replicates_matched"] = len(rm)
        check["replicates_max_abs_diff"] = float(np.max(np.abs(rm.date_only - rm.bootstrap_mean_loss_diff)))
        ah = pd.read_csv(E4I / "experiment4_hac_summary.csv")
        ah["h"] = ah.horizon.str.removesuffix("d").astype(int)
        hm = date_df[(date_df.metric == "loss") & (date_df.specification == "PRIMARY")].merge(ah, on="h")
        check["hac_z_max_abs_diff"] = float(np.max(np.abs(hm.hac_z - hm.z_stat)))
        check["summary_max_abs_diff"] = float(comp[[c for c in comp.columns if c.startswith("absdiff_")]].to_numpy().max())
        check["classification_match"] = bool((comp.classification == np.where(comp.ci_lower_95 > 0, "MBB_STATIC_LOWER_LOSS",
                                              np.where(comp.ci_upper_95 < 0, "MBB_ADAPTIVE_LOWER_LOSS", "MBB_NO_DISTINGUISHABLE_DIFFERENCE"))).all())
        check["REPRODUCED"] = bool(check["summary_max_abs_diff"] < 1e-12 and check["replicates_max_abs_diff"] < 1e-12)
        write_csv(out / "a3_date_mbb_vs_archived.csv", comp)
    write_csv(out / "a3_seed_date_differences.csv", pd.concat(sd_rows, ignore_index=True))
    write_csv(out / "a3_date_only_mbb.csv", date_df)
    write_csv(out / "a3_joint_seed_date_bootstrap.csv", joint_df)
    write_csv(out / "a3_seed_level_tests.csv", pd.DataFrame(seed_rows))
    write_csv(out / "a3_bootstrap_replicates.csv", reps)
    write_json(out / "a3_run_info.json", dict(created=utc(), adaptive=a.adaptive, static=a.static, B=B, checks=check))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(json.dumps(check, indent=1))
    print(date_df.to_string()); print(joint_df.to_string()); print(pd.DataFrame(seed_rows).to_string())


if __name__ == "__main__":
    main()
