"""Step 6.4 - securities master and evidence-based exits for every U.S.-listed security ever held.

Issuer identification (never guessed):
  * EDGAR full-text search for the CUSIP in Schedule 13D/13G filings (cover pages carry the subject CUSIP);
    the subject is the name-similar result entity (token overlap >= 0.6) that appears in the most hits
    (>= half of them, strictly more than the runner-up). Otherwise NOT FOUND.
Issuer history: data.sec.gov/submissions/CIK##########.json (all filing pages).
exit_type (only from evidence, evaluated after the first quarter held):
  bankruptcy      8-K item 1.03
  still_listed    CUSIP on the latest 13(f) list (2026Q2) or the issuer currently has a U.S. exchange listing on EDGAR
  acquired/merged Form 25/25-NSE or Form 15 together with merger evidence (8-K item 2.01/5.01, DEFM14A/DEFM14C,
                  SC 13E3, SC TO-T, SC 14D9) within 180 days
  moved_to_otc    Form 25 with an 8-K item 3.01 whose text mentions OTC trading
  delisted        Form 25/25-NSE (or Form 15) without the above
  unknown         none of the above
current_listing: yfinance price history for tickers EDGAR currently associates with the CIK (or N-PORT ticker);
SUCCESS requires the yfinance name to match the issuer name.
Outputs: securities_master.csv, exit_evidence.csv, cik_resolution_log.csv
"""
import re
from collections import defaultdict
from datetime import datetime
import pandas as pd
from uni_common import *

LATEST_Q = "2026Q2"
US_EXCH = {"NYSE", "Nasdaq", "NYSE American", "NYSE Arca", "NYSE MKT", "CBOE", "BATS", "Cboe BZX"}
DELIST = {"25-NSE", "25", "25-NSE/A", "25/A"}
DEREG = {"15-12B", "15-12G", "15-15D", "15F-12B", "15F-12G", "15F-15D", "15-12B/A", "15-12G/A", "15-15D/A"}
MERGER_FORMS = {"DEFM14A", "DEFM14C", "PREM14A", "PREM14C", "SC 13E3", "SC 13E3/A", "SC TO-T", "SC TO-T/A", "SC 14D9", "SC 14D9/A"}
SCHED_FORMS = "SC%2013G,SC%2013G%2FA,SC%2013D,SC%2013D%2FA,SCHEDULE%2013G,SCHEDULE%2013G%2FA,SCHEDULE%2013D,SCHEDULE%2013D%2FA"
STUDY_CLASS = {"CWEN": "18539C204", "BEP": "G16258108"}  # share-class identity required for these two
# the same class under an earlier CUSIP (e.g. NRG Yield Class C before the 2018 rename to Clearway) also counts
STUDY_CLASS_TEXT = {"CWEN": r"\bCL(ASS)?\s*C\b", "BEP": r"UNIT|PARTNERSHIP"}


def resolve_cik(cusip, names):
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{cusip}%22&forms={SCHED_FORMS}"
    j, st = fetch_json(url)
    hits = (j or {}).get("hits", {}).get("hits", [])
    if not hits:  # some issuers have no 13D/13G hits -> any form
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{cusip}%22"
        j, st = fetch_json(url)
        hits = (j or {}).get("hits", {}).get("hits", [])
    best, freq = {}, defaultdict(int)
    for h in hits:
        seen = set()
        for dn in h["_source"].get("display_names", []):
            m = re.match(r"^(.*?)\s+(?:\(([^)]*)\)\s+)?\(CIK (\d{10})\)$", dn.strip())
            if not m:
                continue
            nm, tick, cik = m.group(1).strip(), m.group(2) or "", m.group(3)
            if cik not in seen:
                freq[cik] += 1
                seen.add(cik)
            sc = max(name_overlap(nm, n) for n in names)
            if sc > best.get(cik, (0,))[0]:
                best[cik] = (sc, nm, tick)
    # name-similar candidates (>= 0.6); the subject company appears in (almost) every hit, so rank by hit frequency
    cand = sorted([(freq[c], v[0], c, v) for c, v in best.items() if v[0] >= 0.6], reverse=True)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], -freq[kv[0]]))
    if cand and cand[0][0] >= 0.5 * len(hits) and (len(cand) == 1 or cand[1][0] < cand[0][0]):
        _, _, cik, (sc, nm, tick) = cand[0]
        return dict(cik=cik, edgar_name=nm, display_tickers=tick, cik_method="EFTS CUSIP search (13D/13G subject)", cik_score=sc, cik_query=url, n_hits=len(hits))
    return dict(cik="NOT FOUND", edgar_name="", display_tickers="", cik_method="NOT FOUND", cik_score=ranked[0][1][0] if ranked else 0.0,
                cik_query=url, n_hits=len(hits), cik_candidates="; ".join(f"{c}:{v[1]}:{v[0]:.2f}:hits={freq[c]}" for c, v in ranked[:4]))


def submissions(cik):
    j, st = fetch_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if not j:
        return None, []
    rows = []

    def add(block):
        n = len(block.get("form", []))
        for i in range(n):
            rows.append(dict(form=block["form"][i], filing_date=block["filingDate"][i], accession=block["accessionNumber"][i],
                             items=block.get("items", [""] * n)[i] or "", primary_doc=block.get("primaryDocument", [""] * n)[i] or ""))
    add(j.get("filings", {}).get("recent", {}))
    for f in j.get("filings", {}).get("files", []):
        jj, _ = fetch_json(f"https://data.sec.gov/submissions/{f['name']}")
        if jj:
            add(jj)
    return j, rows


def idx_url(cik, acc):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}-index.htm"


def q_start(q):
    y, k = int(q[:4]), int(q[-1])
    return f"{y}-{3 * (k - 1) + 1:02d}-01"


def otc_mentioned(cik, f):
    if not f["primary_doc"]:
        return False
    t, _ = fetch(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{f['accession'].replace('-', '')}/{f['primary_doc']}")
    return bool(t and re.search(r"(?i)\bOTC\b|over-the-counter|OTCQX|OTCQB|OTC Pink|pink sheets", re.sub(r"<[^>]+>", " ", t)))


def classify(sec, sub, filings, on_latest):
    """Returns exit_type, exit_date, basis, evidence rows."""
    start = q_start(sec["first_quarter_held"])
    F = [f for f in filings if f["filing_date"] >= start]
    ev = []
    bk = sorted([f for f in F if f["form"].startswith("8-K") and "1.03" in f["items"]], key=lambda f: f["filing_date"])
    for f in bk[:1]:
        ev.append(("bankruptcy", f))
    exch = [e for e in (sub or {}).get("exchanges", []) if e]
    listed_now = any(e in US_EXCH for e in exch)
    d25 = sorted([f for f in F if f["form"] in DELIST], key=lambda f: f["filing_date"])
    d15 = sorted([f for f in F if f["form"] in DEREG], key=lambda f: f["filing_date"])
    if bk:
        last = d25[-1] if d25 else None
        if last:
            ev.append(("delisting (Form 25)", last))
        return "bankruptcy", bk[0]["filing_date"], "8-K item 1.03 filing date", ev
    if on_latest or listed_now:
        return "still_listed", "", ("CUSIP on 13(f) list " + LATEST_Q if on_latest else "") + \
            ("; " if on_latest and listed_now else "") + (f"EDGAR exchanges={','.join(exch)}" if listed_now else ""), ev
    anchor = (d25 or d15 or [None])[-1]
    if anchor is None:
        return "unknown", "", "no Form 25/15, 8-K 1.03 or current listing found on EDGAR", ev
    ev.append(("delisting (Form 25)" if anchor["form"] in DELIST else "deregistration (Form 15)", anchor))
    if d15 and anchor["form"] in DELIST:
        ev.append(("deregistration (Form 15)", d15[-1]))
    ad = datetime.strptime(anchor["filing_date"], "%Y-%m-%d")
    near = [f for f in F if abs((datetime.strptime(f["filing_date"], "%Y-%m-%d") - ad).days) <= 180]
    mg = [f for f in near if (f["form"].startswith("8-K") and re.search(r"\b(2\.01|5\.01)\b", f["items"])) or f["form"] in MERGER_FORMS]
    if mg:
        for f in sorted(mg, key=lambda f: f["filing_date"])[-2:]:
            ev.append(("merger evidence", f))
        return "acquired/merged", anchor["filing_date"], f"Form {anchor['form']} filing date (merger evidence within 180 days)", ev
    k301 = [f for f in near if f["form"].startswith("8-K") and "3.01" in f["items"]]
    for f in k301:
        if otc_mentioned(sub["cik"], f):
            ev.append(("8-K item 3.01 mentioning OTC", f))
            return "moved_to_otc", anchor["filing_date"], f"Form {anchor['form']} filing date; 8-K 3.01 text mentions OTC", ev
    return "delisted", anchor["filing_date"], f"Form {anchor['form']} filing date", ev


def yf_check(tickers, names):
    try:
        import yfinance as yf
    except ImportError:
        return "FAILED", "", "", "yfinance not importable"
    for t in tickers:
        if not t:
            continue
        try:
            time.sleep(1.0)
            tk = yf.Ticker(t)
            h = tk.history(period="max", auto_adjust=False)
            if h is None or h.empty:
                continue
            nm = ""
            try:
                info = tk.get_info()
                nm = info.get("longName") or info.get("shortName") or ""
            except Exception:
                pass
            if nm and max(name_overlap(nm, n) for n in names) < 0.5:
                continue  # ticker now belongs to another issuer
            return "SUCCESS", t, str(h.index.max().date()), nm
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)[:120]
            continue
    return "FAILED", ",".join(t for t in tickers if t), "", ""


def main():
    M = pd.read_csv(OUT / "membership_quarterly.csv", dtype=str, keep_default_na=False)
    C = pd.read_csv(OUT / "holdings_classified.csv", dtype=str, keep_default_na=False)
    C = C[C.us_listed == "True"]
    F13 = pd.read_csv(OUT / "sec13f_lists_parsed.csv", dtype=str, keep_default_na=False)
    latest = set(F13[(F13.quarter == LATEST_Q) & (F13.deleted != "True")].cusip)
    ct, _ = fetch_json("https://www.sec.gov/files/company_tickers.json")
    study_cik = {}
    for v in (ct or {}).values():
        if v["ticker"] in STUDY_TICKERS:
            study_cik[v["ticker"]] = f"{int(v['cik_str']):010d}"
    sec_rows, ev_rows, log = [], [], []
    for sid, g in M.groupby("security_id"):
        h = C[C.security_id == sid]
        names = sorted(set(h.name))
        qs = sorted(g.quarter)
        rec = dict(security_id=sid, cusips=sid, names=" | ".join(names)[:500],
                   isins=",".join(sorted(set(x for x in h["isin"] if x))), nport_tickers=",".join(sorted(set(x for x in h.ticker if x))),
                   us_listing_type=g.us_listing_type.iloc[0], sec13f_text=g.sec13f_text.iloc[0],
                   held_via_local_foreign_line=bool(((h["isin"].str[:2] != "US") & (h["isin"] != "") & (h.cusip9 != h.cusip)).any()),
                   first_quarter_held=qs[0], last_quarter_held=qs[-1], n_quarters_held=len(qs),
                   etfs=",".join(f for f in ["ICLN", "TAN", "QCLN", "PBW"] if (g[f"held_{f}"] == "True").any()),
                   max_weight_pct=float(g.max_weight_pct.astype(float).max()), on_13f_list_latest=sid in latest)
        r = resolve_cik(sid, names)
        rec.update({k: r.get(k, "") for k in ["cik", "edgar_name", "display_tickers", "cik_method", "cik_score"]})
        log.append(dict(security_id=sid, names=rec["names"][:200], **r))
        sub, filings = (None, [])
        if r["cik"] != "NOT FOUND":
            sub, filings = submissions(r["cik"])
        if sub:
            sub["cik"] = r["cik"]
            rec.update(edgar_current_name=sub.get("name", ""), edgar_tickers=",".join(sub.get("tickers", [])),
                       edgar_exchanges=",".join(e for e in sub.get("exchanges", []) if e), state_of_incorporation=sub.get("stateOfIncorporation", ""),
                       business_country=((sub.get("addresses") or {}).get("business") or {}).get("stateOrCountry", ""),
                       former_names="; ".join(fn.get("name", "") for fn in sub.get("formerNames", []))[:300])
            et, ed, basis, ev = classify(rec, sub, filings, sid in latest)
        else:
            rec.update(edgar_current_name="NOT FOUND", edgar_tickers="", edgar_exchanges="", state_of_incorporation="", business_country="", former_names="")
            et, ed, basis, ev = (("still_listed", "", f"CUSIP on 13(f) list {LATEST_Q}", []) if sid in latest
                                 else ("unknown", "", "issuer CIK NOT FOUND on EDGAR", []))
        rec.update(exit_type=et, exit_date=ed, exit_date_basis=basis)
        for kind, f in ev:
            ev_rows.append(dict(security_id=sid, names=rec["names"][:120], cik=rec["cik"], exit_type=et, evidence_kind=kind, form=f["form"],
                                filing_date=f["filing_date"], items=f["items"], accession=f["accession"], url=idx_url(rec["cik"], f["accession"])))
        # domicile refinement for listing type
        if rec["us_listing_type"].startswith("common (domicile"):
            st = rec.get("state_of_incorporation", "")
            if re.fullmatch(r"[A-Z]{2}", st or ""):
                rec["us_listing_type"] = "domestic common"
            elif st:
                rec["us_listing_type"] = f"foreign ordinary with U.S. listing (incorporation code {st})"
        # study panel identity
        tick_cands = [t for t in (rec.get("edgar_tickers", "").split(",") + rec["nport_tickers"].split(",") + rec["display_tickers"].split(",")) if t.strip()]
        rec["in_study_panel"] = False
        rec["study_ticker"] = ""
        rec["share_class_note"] = ""
        for t, cik in study_cik.items():
            if rec["cik"] == cik:
                same_class = sid == STUDY_CLASS.get(t) or bool(re.search(STUDY_CLASS_TEXT.get(t, "$^"), rec["sec13f_text"] + " " + rec["names"], re.I))
                if t in STUDY_CLASS and not same_class:
                    rec["share_class_note"] = f"same issuer as study ticker {t} but different share class/security (study uses CUSIP {STUDY_CLASS[t]})"
                else:
                    rec["in_study_panel"], rec["study_ticker"] = True, t
                    if t == "CWEN":
                        rec["share_class_note"] = ("Clearway Energy Class C (CWEN); Class A (CWEN.A) is CUSIP 18539C105" if sid == STUDY_CLASS[t]
                                                   else f"Class C under earlier CUSIP {sid} (NRG Yield before the 2018 rename to Clearway Energy)")
                    if t == "BEP":
                        rec["share_class_note"] = "Brookfield Renewable Partners LP units (BEP); BEPC (Brookfield Renewable Corp) is a separate issuer/security"
        status, used, last, yname = yf_check([t.strip() for t in rec.get("edgar_tickers", "").split(",") if t.strip()] or
                                             [t for t in rec["nport_tickers"].split(",") if t], names + [rec.get("edgar_current_name", "")])
        rec.update(current_listing=status, yf_ticker=used, yf_last_date=last, yf_name=yname)
        sec_rows.append(rec)
        print(f"{sid} {rec['names'][:40]:40s} cik={rec['cik']} {et:16s} {ed} yf={status} {last}", flush=True)
    S = pd.DataFrame(sec_rows)
    write_csv(OUT / "securities_master.csv", S)
    write_csv(OUT / "exit_evidence.csv", pd.DataFrame(ev_rows))
    write_csv(OUT / "cik_resolution_log.csv", pd.DataFrame(log))
    print(S.exit_type.value_counts())


if __name__ == "__main__":
    main()
