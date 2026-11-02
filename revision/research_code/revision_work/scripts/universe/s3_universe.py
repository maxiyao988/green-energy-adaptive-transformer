"""Step 6.3 - U.S.-listed equity universe by quarter.

A holding line is kept when (i) it is an equity line (N-PORT assetCat EC, or a common-stock / equity-interest
section of a legacy schedule) and (ii) its security is on the SEC Official List of Section 13(f) Securities
for that quarter (= traded on a U.S. national exchange / Nasdaq), not marked DELETED, as an equity entry
(options, debt, warrants, rights excluded). Identification, in order:
  cusip        N-PORT CUSIP (check digit valid) found on the quarter's list
  isin_cusip   CUSIP/CINS embedded in the ISIN found on the list
  nport_name   legacy (no identifiers) name equal (normalised) to a name carrying a CUSIP in N-PORT holdings
  13f_name     normalised name uniquely matching one equity issuer on that quarter's list
The quarter's list is tried first, then the preceding quarter's list (flagged "(prior-quarter list)").
Everything else is excluded and recorded (non-equity, not on the 13(f) list, ambiguous name match).
Outputs: holdings_classified.csv, membership_quarterly.csv, exclusions.csv
"""
import re
import pandas as pd
from uni_common import *

FUNDS = ["ICLN", "TAN", "QCLN", "PBW"]
NON_EQUITY_DESC = re.compile(r"(\*W\b|\bW EXP\b|%|\b(CALL|PUT|DEBT|NOTE|NOTES|DBCV|BOND|WTS?|WARRANTS?|RIGHTS?|RT|"
                             r"CONV|SR NT|FLOAT|ETF|FUND|TR UNIT|SHS BEN INT)\b)")
def pfd_not_adr(t):  # preferred shares are kept only when listed as an ADR/ADS
    return bool(re.search(r"\b(PFD|PREFERRED)\b", t)) and not re.search(r"\b(ADR|ADS)\b", t)
ABBR = {"ENRGY": "ENERGY", "ENGY": "ENERGY", "INTL": "INTERNATIONAL", "TECHNOLOGIES": "TECH", "TECHNOLOGY": "TECH", "TECHS": "TECH",
        "SYS": "SYSTEMS", "PWR": "POWER", "MTRS": "MOTORS", "RENEWABL": "RENEWABLE", "RENEWABLES": "RENEWABLE", "SOLUTIONS": "SOLUTNS",
        "SVCS": "SERVICES", "MGMT": "MANAGEMENT", "INDS": "INDUSTRIES", "MATLS": "MATERIALS", "COS": "COMPANIES", "ELEC": "ELECTRIC",
        "AMER": "AMERICA", "AMERN": "AMERICAN", "SUSTAINBL": "SUSTAINABLE", "INFRAST": "INFRASTRUCTURE", "INFRASTRUCTUR": "INFRASTRUCTURE",
        "CAP": "CAPITAL", "NAT": "NATIONAL", "PRTNRS": "PARTNERS", "ENVIRONMENTAL": "ENVIRONMENT", "SEMICONDUCTOR": "SEMICONDUCTOR",
        "SEMICONDUCTR": "SEMICONDUCTOR", "PHOTONIC": "PHOTONICS", "SCIENTIFIC": "SCIENTIFIC", "SCIENTIF": "SCIENTIFIC"}
STOP = {"INC", "CORP", "CORPORATION", "CO", "LTD", "LIMITED", "PLC", "SA", "AG", "NV", "LP", "L", "P", "THE", "HOLDINGS", "HLDGS",
        "HOLDING", "HLDG", "GROUP", "GRP", "CLASS", "CL", "A", "B", "C", "COM", "COMMON", "STOCK", "SHS", "SH", "NEW", "ADR", "ADS",
        "SPONSORED", "SPON", "ORD", "ORDINARY", "UNITS", "UNIT", "PARTNERSHIP", "PARTN", "LTD", "INT", "TRUST", "SE", "ASA", "OYJ",
        "COMPANY", "DE", "N", "V", "AS", "AB", "SPA", "NPV", "PAR", "VALUE", "REIT", "OF", "AND", "&", "REPSTG", "REP", "REPSTG",
        "PREFERENCE", "SHARES", "SHARE", "H", "BEARER", "REGISTERED", "REG", "NAMEN", "AKT", "PUB", "LLC", "S", "BHD", "TBK", "PT",
        "SPONS", "SPNSRD", "SPONSRD", "GDR", "DEPOSITARY", "DEPOSITORY", "RECEIPT", "RECEIPTS", "RCPT", "ORDINARY", "ORDS", "SUB",
        "VTG", "VOTING", "SER", "SERIES", "LTD", "PARTNRSHP", "INTERESTS", "INTS", "LIMITED", "DEP", "NE", "NVS", "PFD", "N-V"}


def toks(s):
    s = re.sub(r"\([^)]*\)", " ", (s or "").upper())  # drop parentheticals such as (Canada), (a)
    s = re.sub(r"\bU\.?\s?S\.?(?=\s)", "US", s)  # "U S GEOTHERMAL" / "U.S." -> "US"
    s = re.sub(r"[^A-Z0-9 ]", " ", s.replace("&", " AND "))
    out = []
    for w in s.split():
        w = ABBR.get(w, w)
        if w not in STOP and not re.fullmatch(r"\d{1,2}", w):
            out.append(w)
    return out


def name_match(h, l):
    """Holding tokens h vs 13(f) tokens l. Tokens must be equal (after abbreviation mapping), except that
    the LAST 13(f) token may be a truncated prefix (>= 3 chars) of the holding token (30-character issuer
    field). All tokens of the shorter sequence must match the other's leading tokens in order; at least
    2 tokens, or exactly one token on both sides (>= 3 chars)."""
    if not h or not l:
        return False
    n = min(len(h), len(l))
    if n < 2 and not (len(h) == len(l) == 1 and len(h[0]) >= 3):
        return False
    for i in range(n):
        if h[i] == l[i]:
            continue
        if i == len(l) - 1 and len(l[i]) >= 3 and h[i].startswith(l[i]):
            continue
        return False
    return True


def class_letter(s):
    m = re.search(r"(?i)\b(class|cl)\s*-?\s*([A-C])\b", s or "")
    return m.group(2).upper() if m else None


def main():
    H = pd.read_csv(OUT / "holdings_long.csv", dtype={"cusip": str, "isin": str, "ticker": str}, keep_default_na=False, na_values=[""])
    F = pd.read_csv(OUT / "sec13f_lists_parsed.csv", dtype=str, keep_default_na=False)
    F = F[(F.deleted != "True") & F.cusip.map(valid_cusip) & ~F.text.str.contains(NON_EQUITY_DESC) & ~F.text.map(pfd_not_adr)]
    F["toks"] = F.text.map(toks)
    on_list = set(zip(F.quarter, F.cusip))
    qs = sorted(F.quarter.unique())
    prevq = {q: (qs[i - 1] if i else None) for i, q in enumerate(qs)}
    by_q = {q: g for q, g in F.groupby("quarter")}
    desc_of = F.drop_duplicates("cusip", keep="last").set_index("cusip").text.to_dict()

    H["is_nport"] = H.source_form.str.startswith("NPORT-P")
    H["equity_line"] = H.asset_cat.astype(str).str.match(r"^EC") & (H.payoff.fillna("Long") != "Short")
    # names seen with a CUSIP in N-PORT (for identifier-less legacy rows)
    npn = H[H.is_nport & H.cusip.fillna("").map(valid_cusip)]
    name2cusip = {}
    for nm, c in zip(npn.name, npn.cusip):
        name2cusip.setdefault(" ".join(toks(nm)), set()).add(c)

    out = []
    for r in H.itertuples(index=False):
        rec = r._asdict()
        rec.update(security_id="", cusip9="", match_method="", us_listed=False, exclusion_reason="", sec13f_text="")
        if not r.equity_line or re.search(r"(?i)\b(ETF|index fund|trust fund)\b", r.name or ""):
            rec["exclusion_reason"] = (f"non-equity line (asset category {r.asset_cat})" if not r.equity_line
                                       else "investment fund / ETF held as equity line (excluded: funds)")
            out.append(rec)
            continue
        q = r.quarter
        cands = []
        c = (r.cusip or "").strip().upper() if isinstance(r.cusip, str) else ""
        if valid_cusip(c):
            cands.append((c, "cusip"))
        ic = cusip_from_isin(r.isin if isinstance(r.isin, str) else "")
        if ic and ic != c:
            cands.append((ic, "isin_cusip"))
        if not r.is_nport:
            for cc in sorted(name2cusip.get(" ".join(toks(r.name)), [])):
                cands.append((cc, "nport_name"))
        # the list for the report quarter, else the list in force for the preceding quarter (report dates
        # fall inside the quarter, before that quarter's list is compiled)
        hit = None
        for qq, tag in ((q, ""), (prevq.get(q), " (prior-quarter list)")):
            if qq is None:
                continue
            hit = next(((cc, m + tag) for cc, m in cands if (qq, cc) in on_list), None)
            if hit is None and qq in by_q:
                ht = toks(r.name)
                g = by_q[qq]
                m = g[g.toks.map(lambda lt: name_match(ht, lt))]
                if m.cusip.nunique() > 1:
                    cl = class_letter(r.name) or class_letter(r.title if isinstance(r.title, str) else "")
                    if cl:
                        m = m[m.text.str.contains(rf"\bCL(ASS)?\s*-?\s*{cl}\b")]
                if m.cusip.nunique() == 1:
                    hit = (m.cusip.iloc[0], "13f_name" + tag)
                elif m.cusip.nunique() > 1 and not rec["exclusion_reason"]:
                    rec["exclusion_reason"] = "ambiguous 13(f) name match: " + "; ".join(sorted(set(m.cusip + " " + m.text)))[:300]
            if hit:
                rec["exclusion_reason"] = ""
                break
        if hit:
            rec.update(cusip9=hit[0], security_id=hit[0], match_method=hit[1], us_listed=True, sec13f_text=desc_of.get(hit[0], ""))
        elif not rec["exclusion_reason"]:
            rec["exclusion_reason"] = ("equity line not on the quarter's 13(f) list (not U.S.-exchange-listed)" if cands
                                       else "equity line without identifier and no 13(f) name match (treated as not U.S.-listed; unverified)")
        out.append(rec)
    C = pd.DataFrame(out)
    # listing type from the 13(f) description and the identifiers observed for the CUSIP
    isin_by_cusip = C[C.us_listed & C["isin"].notna()].groupby("cusip9")["isin"].agg(lambda s: sorted(set(x for x in s if isinstance(x, str) and x)))

    def ltype(cusip9, text, name):
        t = (text or "").upper()
        if re.search(r"\bUNIT|PARTNERSHIP|PARTN|LP INT|COM UNIT", t) or re.search(r"(?i)\bL\.?P\.?$|\bpartners,? l\.?p", name or ""):
            return "partnership unit"
        if re.search(r"\b(ADR|ADS|SPONSORED|SPON|DEPOSITARY)\b", t) or re.search(r"\b(ADR|ADS)\b", (name or "").upper()):
            return "ADR"
        isins = isin_by_cusip.get(cusip9, [])
        if any(i.startswith("US") for i in isins):
            return "domestic common"
        if isins or cusip9[:1].isalpha():
            return "foreign ordinary with U.S. listing"
        return "common (domicile not identified from holdings; see securities_master)"
    C["us_listing_type"] = [ltype(c, t, n) if u else "" for c, t, n, u in zip(C.cusip9, C.sec13f_text, C.name, C.us_listed)]
    write_csv(OUT / "holdings_classified.csv", C.drop(columns=["is_nport"]))
    write_csv(OUT / "exclusions.csv", C[~C.us_listed][["fund", "quarter", "report_date", "source_form", "name", "cusip", "isin",
                                                         "asset_cat", "inv_country", "pct_net_assets", "exclusion_reason"]])
    K = C[C.us_listed]
    rows = []
    for (q, sid), g in K.groupby(["quarter", "security_id"]):
        rec = dict(quarter=q, security_id=sid, names=" | ".join(sorted(set(g.name))), us_listing_type=g.us_listing_type.iloc[0],
                   match_methods=",".join(sorted(set(g.match_method))), sec13f_text=g.sec13f_text.iloc[0])
        for f in FUNDS:
            gf = g[g.fund == f]
            rec[f"held_{f}"] = len(gf) > 0
            rec[f"weight_{f}"] = float(gf.pct_net_assets.sum()) if len(gf) else 0.0
        rec["max_weight_pct"] = max(rec[f"weight_{f}"] for f in FUNDS)
        rec["n_etfs"] = sum(rec[f"held_{f}"] for f in FUNDS)
        rows.append(rec)
    M = pd.DataFrame(rows).sort_values(["quarter", "security_id"])
    write_csv(OUT / "membership_quarterly.csv", M)
    s = C[C.equity_line].groupby(["fund", "match_method"]).size().unstack(fill_value=0)
    print(s)
    print(C[C.equity_line & ~C.us_listed].groupby("fund").size())
    print("securities:", M.security_id.nunique(), "quarters:", M.quarter.nunique())
    print(M.groupby("quarter").security_id.nunique().to_string())


if __name__ == "__main__":
    main()
