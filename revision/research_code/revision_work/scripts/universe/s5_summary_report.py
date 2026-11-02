"""Step 6.5 - yearly survivorship summary and REPORT_STEP6.md (descriptive only).

Per calendar year 2016-2026 (2026 = Q1-Q2 only):
  n_constituents_union / per ETF          U.S.-listed securities held at any quarter-end of the year
  n_later_exited_{type}, n_unknown        among them, by evidence-based exit_type
  exited_weight_max_etf / _mean_etf       per ETF: mean over that ETF's quarters in the year of the summed % of net
                                          assets in securities that later exited; then max / mean across ETFs
  n_study_tickers_member                  distinct study tickers (share-class-exact) held that year
  n_nonpanel_still_listed                 constituents not in the 17-name panel with exit_type still_listed
Outputs: survivorship_summary.csv, exited_securities.csv, data_gaps.csv, REPORT_STEP6.md
"""
import pandas as pd
from uni_common import *

FUNDS = ["ICLN", "TAN", "QCLN", "PBW"]
EXIT_TYPES = ["delisted", "acquired/merged", "bankruptcy", "moved_to_otc"]


def md(df, cols=None):
    cols = cols or list(df.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join("" if pd.isna(r[c]) else (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + " |\n"
    return out


def main():
    funds = pd.read_csv(OUT / "funds.csv", dtype=str, keep_default_na=False)
    cov = pd.read_csv(OUT / "coverage_by_quarter.csv", dtype=str, keep_default_na=False)
    plog = pd.read_csv(OUT / "parse_log.csv", dtype=str, keep_default_na=False)
    M = pd.read_csv(OUT / "membership_quarterly.csv", dtype=str, keep_default_na=False)
    S = pd.read_csv(OUT / "securities_master.csv", dtype=str, keep_default_na=False)
    C = pd.read_csv(OUT / "holdings_classified.csv", dtype=str, keep_default_na=False)
    L13 = pd.read_csv(OUT / "sec13f_lists_log.csv", dtype=str, keep_default_na=False)
    rlog = pd.read_csv(CACHE / "request_log.csv", dtype=str, keep_default_na=False)
    M["year"] = M.quarter.str[:4].astype(int)
    for f in FUNDS:
        M[f"weight_{f}"] = M[f"weight_{f}"].astype(float)
    st = S.set_index("security_id")
    M["exit_type"] = M.security_id.map(st.exit_type)
    M["in_panel"] = M.security_id.map(st.in_study_panel) == "True"
    M["study_ticker"] = M.security_id.map(st.study_ticker)
    M["issuer_key"] = [c if c and c != "NOT FOUND" else s for c, s in zip(M.security_id.map(st.cik), M.security_id)]
    rows = []
    for y, g in M.groupby("year"):
        secs = g.drop_duplicates("security_id")
        rec = dict(year=y, quarters_covered=g.quarter.nunique(), n_constituents_union=len(secs),
                   n_issuers_union=secs.issuer_key.nunique())
        for f in FUNDS:
            rec[f"n_{f}"] = g[g[f"held_{f}"] == "True"].security_id.nunique()
        for t in EXIT_TYPES:
            rec[f"n_later_{t.replace('/', '_')}"] = int((secs.exit_type == t).sum())
        rec["n_later_exited_total"] = int(secs.exit_type.isin(EXIT_TYPES).sum())
        rec["n_unknown"] = int((secs.exit_type == "unknown").sum())
        rec["n_still_listed"] = int((secs.exit_type == "still_listed").sum())
        ex = g[g.exit_type.isin(EXIT_TYPES)]
        w = {}
        for f in FUNDS:
            fq = g[g[f"held_{f}"] == "True"].quarter.unique()
            if len(fq):
                per_q = ex.groupby("quarter")[f"weight_{f}"].sum().reindex(fq, fill_value=0.0)
                w[f] = float(per_q.mean())
        rec["exited_weight_pct_max_etf"] = max(w.values()) if w else 0.0
        rec["exited_weight_pct_mean_etf"] = sum(w.values()) / len(w) if w else 0.0
        for f in FUNDS:
            rec[f"exited_weight_pct_{f}"] = w.get(f, float("nan"))
        rec["n_study_tickers_member"] = g[g.in_panel].study_ticker.nunique()
        rec["study_tickers_member"] = ",".join(sorted(g[g.in_panel].study_ticker.unique()))
        rec["n_nonpanel_still_listed"] = int(((~secs.in_panel) & (secs.exit_type == "still_listed")).sum())
        rows.append(rec)
    Y = pd.DataFrame(rows)
    write_csv(OUT / "survivorship_summary.csv", Y)
    EX = S[S.exit_type.isin(EXIT_TYPES)].copy()
    ev = pd.read_csv(OUT / "exit_evidence.csv", dtype=str, keep_default_na=False)
    first_ev = ev.sort_values("filing_date").groupby("security_id").agg(evidence=("url", lambda s: " ; ".join(s)),
                                                                          evidence_forms=("form", lambda s: ",".join(s)))
    EX = EX.join(first_ev, on="security_id")
    EX["ticker"] = [a or b or c for a, b, c in zip(EX.edgar_tickers, EX.nport_tickers, EX.display_tickers)]
    EX = EX.sort_values(["exit_date", "names"])
    write_csv(OUT / "exited_securities.csv", EX[["security_id", "names", "ticker", "cik", "exit_type", "exit_date", "exit_date_basis",
                                                  "first_quarter_held", "last_quarter_held", "etfs", "max_weight_pct", "evidence_forms", "evidence"]])
    # ---- data gaps
    gaps = []
    for _, r in cov[cov.obtained != "True"].iterrows():
        gaps.append(dict(kind="holdings quarter missing", item=f"{r.fund} {r.quarter}", detail="NOT FOUND"))
    for _, r in plog[~plog.status.isin(["PARSED"])].iterrows():
        gaps.append(dict(kind="filing not parsed", item=f"{r.fund} {r.form} {r.get('report_date', '')}", detail=f"{r.status} {r.url}"))
    for _, r in L13[L13.status != "PARSED"].iterrows():
        gaps.append(dict(kind="13(f) list missing", item=r.quarter, detail=r.status))
    for _, r in S[S.cik == "NOT FOUND"].iterrows():
        gaps.append(dict(kind="issuer CIK NOT FOUND", item=f"{r.security_id} {r.names[:80]}", detail=f"exit_type={r.exit_type}"))
    for _, r in S[S.exit_type == "unknown"].iterrows():
        gaps.append(dict(kind="exit unknown", item=f"{r.security_id} {r.names[:80]}", detail=r.exit_date_basis))
    for _, r in S[S.current_listing != "SUCCESS"].iterrows():
        gaps.append(dict(kind="yfinance current listing FAILED", item=f"{r.security_id} {r.names[:80]}", detail=f"tickers tried: {r.yf_ticker or 'none'}; exit_type={r.exit_type}"))
    unres = C[(C.equity_line == "True") & (C.us_listed != "True") & C.exclusion_reason.str.contains("without identifier")]
    for (f, n), g in unres.groupby(["fund", "name"]):
        gaps.append(dict(kind="legacy equity line without identifier, no 13(f) name match (treated non-U.S.)", item=f"{f}: {n}",
                         detail=f"{g.quarter.min()}-{g.quarter.max()} ({len(g)} quarters)"))
    amb = C[C.exclusion_reason.str.startswith("ambiguous")]
    for (f, n), g in amb.groupby(["fund", "name"]):
        gaps.append(dict(kind="ambiguous 13(f) name match (excluded)", item=f"{f}: {n}", detail=g.exclusion_reason.iloc[0][:200]))
    bad = rlog[rlog.status.isin(["403", "429", "EXC"])]
    for _, r in bad.iterrows():
        gaps.append(dict(kind=f"HTTP {r.status} (backoff logged)", item=r.url, detail=r.note))
    non200 = rlog[rlog.status != "200"]
    last = rlog.groupby("url").tail(1)
    still_failed = last[last.status != "200"]
    for _, r in still_failed.iterrows():
        gaps.append(dict(kind=f"request whose latest attempt failed (HTTP {r.status})", item=r.url,
                         detail=f"last attempt {r.utc}; not cached; a later run re-requests it"))
    G = pd.DataFrame(gaps)
    write_csv(OUT / "data_gaps.csv", G)

    # ---- report
    Ld = []
    Ld.append("# REPORT STEP 6 — Survivorship quantification from historical clean-energy ETF holdings\n")
    Ld.append(f"Generated {utc()} (UTC). Descriptive quantification only. No model was trained, and nothing here eliminates or corrects survivorship bias. Code: `revision_work/scripts/universe/`. Outputs and the download cache: `revision_work/outputs/universe/`. All SEC requests sent `User-Agent: {UA}` at ≤ 4 requests/s, and every file was cached and never re-downloaded ({len(rlog)} logged requests over all development and final runs; non-200 responses: "
              + (", ".join(f"HTTP {k} × {v}" for k, v in non200.status.value_counts().items()) or "none")
              + f"; {len(still_failed)} URLs whose latest attempt failed are listed in `data_gaps.csv`. Failed responses are never cached, so they cannot enter the outputs).\n")
    Ld.append("## BLOCKING / DATA GAPS\n")
    n_missing = int((cov.obtained != "True").sum())
    Ld.append(f"- **Blocking: {'none' if n_missing == 0 else str(n_missing) + ' fund-quarters missing'}.** Holdings were obtained for {int((cov.obtained == 'True').sum())}/{len(cov)} fund-quarters (4 ETFs × 2016Q1–2026Q2). All {len(L13)} quarterly 13(f) lists were parsed.\n"
              f"- **Pre-N-PORT holdings (2016Q1–2019Q2/Q3)** come from N-Q / N-CSR / N-CSRS schedules and the one N-PORT Part F exhibit (NPORT-EX) per fund. These carry no CUSIP/ISIN, so securities were identified by name: first against names that carry a CUSIP in later N-PORT filings, then by a unique normalized-name match to the quarter's 13(f) list. Weights for these quarters are value ÷ sum of parsed equity values, not % of net assets. Every filing attempt is in `parse_log.csv`.\n"
              f"- U.S. listing = the security (or, for a local foreign line, its issuer's U.S. ADR/share) appears on the SEC 13(f) list for the report quarter or the preceding quarter. OTC-only securities are not on that list and are therefore excluded.\n"
              f"- Issuer CIK NOT FOUND: {int((S.cik == 'NOT FOUND').sum())}; exit_type unknown: {int((S.exit_type == 'unknown').sum())}; yfinance current listing FAILED: {int((S.current_listing != 'SUCCESS').sum())} of {len(S)} securities. "
              f"Legacy equity lines without identifier and no 13(f) match (treated as non-U.S.): {unres.groupby(['fund', 'name']).ngroups} fund-name pairs. Full list in `data_gaps.csv` ({len(G)} rows).\n"
              "- An exit is assigned only from EDGAR evidence: Form 25/25-NSE, Form 15, an 8-K with item 1.03 (bankruptcy) or items 2.01/5.01, or merger/going-private filings (DEFM14A/C, SC 13E3, SC TO-T, SC 14D9). Foreign private issuers file 6-Ks, which carry no item codes. So an ADR that went private without those forms can be classed only `delisted`, not `acquired/merged`.\n"
              "- Security IDs are CUSIP-based, so a CUSIP change (reverse split, redomicile, reorganised equity) yields a new security ID for the same issuer. Year counts are shown both by security and by issuer (CIK).\n")
    Ld.append("## 1. Funds\n")
    Ld.append(md(funds, ["fund", "fund_name", "role", "registrant_cik", "registrant_name", "series_id", "series_name_on_edgar", "class_id", "class_ticker_on_edgar", "first_filing", "last_filing"]))
    Ld.append("Source of the current IDs: `https://www.sec.gov/files/company_tickers_mf.json`. TAN's pre-2018 predecessor series was found via EDGAR full-text search and the Series/Classes section of that filing's index page (`funds.csv`, column `id_source`).\n")
    Ld.append("## 2. Holdings coverage by quarter\n")
    pv = cov.assign(cell=cov.source_form.str.replace("NPORT-P/A", "NPORT-P*") + " (" + cov.n_equity_lines + "/" + cov.n_holdings + ")").pivot(index="quarter", columns="fund", values="cell")[FUNDS].reset_index()
    Ld.append("Cell = source form (equity lines / all lines). NPORT-P* = amended N-PORT.\n")
    Ld.append(md(pv))
    Ld.append(f"`holdings_long.csv`: {len(C):,} holding lines. U.S.-listed equity securities ever held: **{len(S)}** (CUSIP-based), from **{S.cik[S.cik != 'NOT FOUND'].nunique()}** identified issuers. Exclusions with reasons: `exclusions.csv`.\n")
    Ld.append("## 4. Securities that exited (evidence-based)\n")
    ex_tab = EX[["names", "ticker", "security_id", "exit_type", "exit_date", "first_quarter_held", "last_quarter_held", "etfs", "evidence"]].copy()
    ex_tab["names"] = ex_tab.names.str.split(" \\| ").str[0]
    ex_tab["evidence"] = ex_tab.evidence.str.split(" ; ").str[0]
    Ld.append(f"{len(EX)} of {len(S)} securities have evidence of an exit. Counts: " + ", ".join(f"{t}: {int((S.exit_type == t).sum())}" for t in EXIT_TYPES + ["unknown", "still_listed"]) + ". The exit date is the filing date of the evidence form (`exit_date_basis` in `securities_master.csv`); a Form 25 delisting becomes effective 10 days after filing. All evidence links are in `exit_evidence.csv`.\n")
    Ld.append(md(ex_tab, ["names", "ticker", "security_id", "exit_type", "exit_date", "first_quarter_held", "last_quarter_held", "etfs", "evidence"]))
    unk = S[S.exit_type == "unknown"][["names", "security_id", "cik", "first_quarter_held", "last_quarter_held", "exit_date_basis"]].copy()
    if len(unk):
        unk["names"] = unk.names.str.split(" \\| ").str[0]
        Ld.append("Securities with exit_type **unknown** (no qualifying evidence found):\n")
        Ld.append(md(unk))
    Ld.append("Study-panel identity (share-class exact):\n")
    sp = S[(S.in_study_panel == "True") | (S.share_class_note != "")][["study_ticker", "names", "security_id", "cik", "share_class_note", "first_quarter_held", "last_quarter_held", "exit_type"]].copy()
    sp["names"] = sp.names.str.split(" \\| ").str[0]
    Ld.append(md(sp.sort_values("study_ticker")))
    missing_panel = sorted(set(STUDY_TICKERS) - set(S[S.in_study_panel == "True"].study_ticker))
    Ld.append(f"Study tickers never held by any of the four ETFs in 2016Q1–2026Q2 (as identified): {', '.join(missing_panel) if missing_panel else 'none'}.\n")
    Ld.append("## 5. Survivorship summary by calendar year\n")
    cols = ["year", "quarters_covered", "n_constituents_union", "n_issuers_union", "n_ICLN", "n_TAN", "n_QCLN", "n_PBW", "n_later_exited_total",
            "n_later_delisted", "n_later_acquired_merged", "n_later_bankruptcy", "n_later_moved_to_otc", "n_unknown", "n_still_listed",
            "exited_weight_pct_max_etf", "exited_weight_pct_mean_etf", "n_study_tickers_member", "n_nonpanel_still_listed"]
    Ld.append(md(Y, cols))
    Ld.append("Weights are % of ETF net assets (approximate value shares before N-PORT). Per-ETF values (`exited_weight_pct_{ETF}`) and the study tickers present each year are in `survivorship_summary.csv`. 2026 covers Q1–Q2 only.\n")
    # ---- plain-language paragraph (numbers computed)
    tot = len(S)
    nex = len(EX)
    allyrs = M.drop_duplicates("security_id")
    nonpanel_live = int(((S.in_study_panel != "True") & (S.exit_type == "still_listed")).sum())
    y16 = Y[Y.year == 2016].iloc[0] if (Y.year == 2016).any() else None
    Ld.append("## Summary\n")
    Ld.append(f"Between 2016Q1 and 2026Q2 the four clean-energy ETFs held **{tot} distinct U.S.-listed securities**. Of these, **{nex} ({100 * nex / tot:.0f}%)** have EDGAR evidence of a later exit: "
              + ", ".join(f"{int((S.exit_type == t).sum())} {t}" for t in EXIT_TYPES if (S.exit_type == t).sum()) + f". Another {int((S.exit_type == 'unknown').sum())} have no determinable status. "
              + (f"Among the {int(y16.n_constituents_union)} U.S.-listed constituents held in 2016, {int(y16.n_later_exited_total)} later exited. Names that later exited took up on average {y16.exited_weight_pct_mean_etf:.1f}% (up to {y16.exited_weight_pct_max_etf:.1f}% in one ETF) of ETF net assets that year. " if y16 is not None else "")
              + f"A panel of 17 names that survive to today therefore omits a material share of what a sector investor held at the time, and the omitted names are disproportionately the ones that failed, were acquired or were taken private. "
              f"The panel also omits {nonpanel_live} securities that were held and are still listed, so it is not merely 'all survivors'. It is a selected subset of survivors. "
              "These counts describe the exposure only. They do not measure, correct or eliminate the effect of survivorship on any forecast result.\n")
    write_text(OUT / "REPORT_STEP6.md", "\n".join(Ld))
    print("\n".join(Ld)[:6000])


if __name__ == "__main__":
    main()
