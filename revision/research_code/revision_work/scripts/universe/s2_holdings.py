"""Step 6.2 - quarterly holdings for ICLN, TAN, QCLN, PBW (calendar quarters 2016Q1-2026Q2).

Source priority per fund x calendar quarter (quarter = calendar quarter containing the report date):
  1. Form N-PORT-P (latest amendment for the report date): primary_doc.xml, every holding parsed.
  2. Otherwise N-Q / N-CSR / N-CSRS / NPORT-EX schedule exhibit (original first; amendment only if the original fails to parse):
     the fund's "Schedule/Portfolio of Investments" section is located in the (multi-fund) document and
     its rows are parsed with a token grammar. These schedules carry no CUSIP/ISIN; parsing may be
     imperfect and every attempt is logged in parse_log.csv.
Outputs: holdings_long.csv, coverage_by_quarter.csv, parse_log.csv
"""
import html as _html
import re
import xml.etree.ElementTree as ET
import pandas as pd
from bs4 import BeautifulSoup
from uni_common import *

QUARTERS = [f"{y}Q{q}" for y in range(2016, 2027) for q in range(1, 5)][:42]  # 2016Q1..2026Q2
FUND_KEY = {
    "ICLN": r"(?i)ishares\s*(®|\(r\))?\s*global\s+clean\s+energy",
    "TAN": r"(?i)(guggenheim|invesco|claymore)\s+solar\s+etf",
    "QCLN": r"(?i)clean\s*edge\s*(®|\(r\))?\s*green\s+energy",
    "PBW": r"(?i)wilderhill\s+clean\s+energy\s+(portfolio|etf)",
}
FUND_TITLE = re.compile(r"(?i)(\b(etf|portfolio|fund)\b.*\([A-Z]{2,5}\)\s*$)|^(invesco|powershares|guggenheim|ishares|first trust)\b.*\b(etf|portfolio|fund)\b")
CATEGORY = re.compile(r"(?i)^(common\s+stocks?(\s+(and|&)\s+other\s+equity\s+interests)?|preferred\s+stocks?|rights|warrants|"
                      r"short-term\s+investments|money\s+market\s+funds?|investment\s+compan(y|ies)|master\s+limited\s+partnerships?|"
                      r"securities\s+lending\s+collateral|exchange-traded\s+funds?)(\s*\(continued\))?$")
SUMMARY_NAME = re.compile(r"(?i)^(assets\b|liabilities\b|total\b|common\s+stocks?\b|preferred\s+stocks?\b|money\s+market\b|"
                          r"securities\s+lending\b|investments\b|level\s+\d|net\s+assets)")
HEADER = re.compile(r"(?i)^(consolidated\s+)?(schedule|portfolio)\s+of\s+investments")
STOP = re.compile(r"(?i)^(total\s+investments|total\s+investments\s+in\s+securities|net\s+assets|valuation\s+inputs|"
                  r"the\s+following\s+table|level\s+1|summary\s+of\s+(the\s+)?inputs|investments\s+in\s+securities|"
                  r"see\s+notes|notes\s+to|abbreviations|portfolio\s+abbreviations|footnote\s+legend|\(a\)\s+non-income)")
NUM = re.compile(r"^\(?\$?\s*[\d,]+(\.\d+)?\)?$")
EQUITY_SECTION = re.compile(r"(?i)(common\s+stock|other\s+equity\s+interests|master\s+limited\s+partnership|"
                            r"^equities|equity\s+securities|^stocks)")
NON_EQUITY_SECTION = re.compile(r"(?i)(short-term|money\s+market|investment\s+compan|securities\s+lending|"
                                r"rights|warrants|preferred|exchange-traded\s+fund|repurchase|collateral|cash\s+equivalent)")


# ---------------------------------------------------------------- N-PORT

def strip_ns(root):
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def parse_nport(xml_text):
    root = strip_ns(ET.fromstring(xml_text.encode("utf-8")))
    gi = root.find(".//genInfo")
    fi = root.find(".//fundInfo")
    meta = dict(series_id=gi.findtext("seriesId"), series_name=gi.findtext("seriesName"),
                rep_pd_date=gi.findtext("repPdDate"), rep_pd_end=gi.findtext("repPdEnd"),
                net_assets=float(fi.findtext("netAssets") or "nan") if fi is not None else float("nan"))
    rows = []
    for s in root.iter("invstOrSec"):
        ids = s.find("identifiers")
        isin = ticker = other = ""
        if ids is not None:
            if ids.find("isin") is not None:
                isin = ids.find("isin").get("value", "")
            if ids.find("ticker") is not None:
                ticker = ids.find("ticker").get("value", "")
            others = [f"{o.get('otherDesc', '')}={o.get('value', '')}" for o in ids.findall("other")]
            other = ";".join(others)
        ac = s.findtext("assetCat") or (s.find("assetConditional").get("assetCat", "") + ":" + s.find("assetConditional").get("desc", "") if s.find("assetConditional") is not None else "")
        ic = s.findtext("issuerCat") or (s.find("issuerConditional").get("issuerCat", "") + ":" + s.find("issuerConditional").get("desc", "") if s.find("issuerConditional") is not None else "")
        cur = s.findtext("curCd") or (s.find("currencyConditional").get("curCd", "") if s.find("currencyConditional") is not None else "")
        f = lambda t: float(s.findtext(t)) if s.findtext(t) not in (None, "") else float("nan")
        rows.append(dict(name=(s.findtext("name") or "").strip(), title=(s.findtext("title") or "").strip(),
                         cusip=(s.findtext("cusip") or "").strip(), isin=isin, ticker=ticker, other_ids=other,
                         lei=s.findtext("lei") or "", asset_cat=ac, issuer_cat=ic, inv_country=s.findtext("invCountry") or "",
                         currency=cur, balance=f("balance"), units=s.findtext("units") or "", value_usd=f("valUSD"),
                         pct_net_assets=f("pctVal"), payoff=s.findtext("payoffProfile") or ""))
    return meta, rows


# ---------------------------------------------------------------- legacy schedules

def doc_lines(raw):
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    if re.search(r"(?i)<(td|p|div|br)\b", t):
        t = re.sub(r"(?i)</(td|th|p|div|tr|li|h\d)>|<br\s*/?>", "\n", t)
        t = re.sub(r"<[^>]+>", " ", t)
        is_html = True
    else:
        is_html = False
    t = _html.unescape(t).replace("\xa0", " ")
    out = []
    for line in t.split("\n"):
        # HTML: one token per cell/paragraph; plain text: split fixed-width columns on runs of >= 2 spaces
        for tok in ([line.strip()] if is_html else re.split(r"\s{2,}", line.strip())):
            tok = re.sub(r"\s+", " ", tok).strip()
            if not tok:
                continue
            m = None if is_html else re.match(r"^([\d,]{1,15})\s+([A-Za-z0-9].*)$", tok)
            if m and any(c.isalpha() for c in m.group(2)):
                out.extend([m.group(1), m.group(2)])
            else:
                out.append(tok)
    return out


def index_main_doc(index_url, form):
    t, st = fetch(index_url)
    if not t:
        return None, None, f"index {st}"
    s = BeautifulSoup(t, "html.parser")
    per = re.search(r"Period of Report\s*</div>\s*<div class=\"info\">([0-9-]+)", t)
    for tr in s.select("table.tableFile tr")[1:]:
        td = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
        a = tr.find("a")
        if len(td) >= 4 and a and td[3].split("/")[0] == form.split("/")[0]:
            href = a["href"]
            if href.startswith("/ix?doc="):
                href = href[len("/ix?doc="):]
            return "https://www.sec.gov" + href, (per.group(1) if per else None), "ok"
    return None, (per.group(1) if per else None), "main document not found"


def is_num(tok):
    return bool(NUM.match(tok.replace(" ", ""))) and any(c.isdigit() for c in tok)


def to_num(tok):
    tok = tok.replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
    return float(tok)


def find_sections(L, key):
    heads = [i for i, l in enumerate(L) if HEADER.match(l)]
    def win(a, b):
        return " ".join(L[max(0, a):min(len(L), b)])
    # strict: fund name within 6 tokens of the header (names may be split over several tokens)
    ours = [i for i in heads if re.search(key, win(i - 6, i + 7))]
    if not ours:
        # fallback: fund name up to 40 tokens before the header (e.g. a composition table comes first),
        # never reaching back past the previous header
        ours = []
        for n, i in enumerate(heads):
            lo = max(i - 40, (heads[n - 1] + 1) if n else 0)
            titles = [j for j in range(lo, i) if FUND_TITLE.search(L[j])]
            if titles and re.search(key, win(titles[-1] - 1, titles[-1] + 2)):
                ours.append(i)
    spans = []
    for i in ours:
        nxt = [j for j in heads if j > i]
        end = nxt[0] if nxt else min(len(L), i + 3000)
        spans.append((i, end))
    return spans


def parse_schedule(L, spans):
    """Token grammar over the fund's section(s): returns holding rows and diagnostics."""
    toks = []
    for a, b in spans:
        seg = L[a:b]
        for k, tok in enumerate(seg):
            if STOP.match(tok) and k > 0:
                break
            toks.append(tok)
    # orientation: first data token after the first category heading (contains %) is a number => shares-first
    first_data = None
    for k, tok in enumerate(toks):
        if "%" in tok:
            for t2 in toks[k + 1:k + 6]:
                if "%" in t2 or t2 == "$":
                    continue
                first_data = t2
                break
            if first_data:
                break
    shares_first = bool(first_data and is_num(first_data))
    rows, section = [], ""
    text_buf, nums, pending = [], [], None

    def heading(tok):
        return "%" in tok and not is_num(tok)

    def name():
        return re.sub(r"\s+", " ", " ".join(text_buf)).strip()

    for tok in toks:
        if CATEGORY.match(tok):
            section = re.sub(r"\s*\(continued\)$", "", tok, flags=re.I).strip()
            text_buf, nums, pending = [], [], None
            continue
        if heading(tok):
            if (EQUITY_SECTION.search(tok) or NON_EQUITY_SECTION.search(tok)) and not re.match(r"(?i)^total\b", tok):
                section = re.sub(r"\s*[-—–]+\s*[\d.]+%.*$", "", tok).strip()
            text_buf, nums, pending = [], [], None
            continue
        if tok == "$" or re.fullmatch(r"[—–-]+", tok):
            continue
        if re.match(r"(?i)^(total\b|shares$|number$|number of$|of shares$|value$|description$|security$|principal|\(unaudited\)|"
                    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d)", tok) \
                or HEADER.match(tok) or re.search(FUND_ANY, tok):
            text_buf, nums, pending = [], [], None
            continue
        if is_num(tok):
            v = to_num(tok)
            if shares_first:
                # grammar: NUM(shares) TEXT+ NUM(value); unlabeled subtotals are overwritten by the next shares
                if text_buf and pending is not None:
                    rows.append(dict(raw_name=name(), shares=pending, value=v, section=section, shares_first=True))
                    text_buf, pending = [], None
                else:
                    text_buf, pending = [], v
            else:
                # grammar: TEXT+ NUM(shares) NUM(value)
                if text_buf:
                    nums.append(v)
                    if len(nums) == 2:
                        rows.append(dict(raw_name=name(), shares=nums[0], value=nums[1], section=section, shares_first=False))
                        text_buf, nums = [], []
            continue
        if not shares_first and nums:
            text_buf, nums = [], []
        text_buf.append(tok)
    rows = [r for r in rows if r["raw_name"] and clean_name(r["raw_name"]) and not SUMMARY_NAME.match(clean_name(r["raw_name"]))
            and not re.match(r"^\(cost", r["raw_name"], re.I)]
    return rows, shares_first


FUND_ANY = r"(?i)(ishares\s*(®|\(r\))?\s*global\s+clean|solar\s+etf|clean\s*edge|wilderhill)"


def clean_name(n):
    n = re.sub(r"^.*\(continued\)\s*", "", n, flags=re.I)  # page-continuation headings glued to the first name
    n = re.sub(r"\s*\((a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|\d{1,2})\)", "", n)
    n = re.sub(r"\s+[*†‡§#\d,]+\s*$", "", n)
    n = re.sub(r"\s+[a-z](,[a-z])*\s*$", "", n)
    n = re.sub(r"(?<=[A-Za-z.])\d{1,2}$", "", n)  # footnote digit glued to the name (e.g. "LP1")
    n = n.replace("*", "").strip(" ,;")
    return re.sub(r"\s+", " ", n)


# ---------------------------------------------------------------- main

def main():
    funds = pd.read_csv(OUT / "funds.csv", dtype=str)
    fl = pd.read_csv(OUT / "filings_index.csv", dtype=str)
    fl = fl[fl.index_url.notna()]
    hold, log = [], []
    # ---- N-PORT
    npf = fl[fl.form.isin(["NPORT-P", "NPORT-P/A"])].copy()
    np_meta = []
    for _, r in npf.iterrows():
        base = r.index_url.rsplit("/", 1)[0]
        url = base + "/primary_doc.xml"
        t, st = fetch(url)
        if not t:
            log.append(dict(fund=r.fund, form=r.form, accession=r.accession, filing_date=r.filing_date, url=url, status=f"FETCH {st}", n_holdings=0))
            continue
        try:
            meta, rows = parse_nport(t)
        except ET.ParseError as e:
            log.append(dict(fund=r.fund, form=r.form, accession=r.accession, filing_date=r.filing_date, url=url, status=f"XML ERROR {e}", n_holdings=0))
            continue
        ok = meta["series_id"] == r.series_id
        np_meta.append(dict(fund=r.fund, form=r.form, accession=r.accession, filing_date=r.filing_date, url=url, ok=ok, rows=rows, **meta))
        log.append(dict(fund=r.fund, form=r.form, accession=r.accession, filing_date=r.filing_date, url=url, report_date=meta["rep_pd_date"],
                        status="PARSED" if ok else f"SERIES MISMATCH {meta['series_id']}", n_holdings=len(rows)))
    npm = pd.DataFrame(np_meta)
    npm = npm[npm.ok].sort_values(["fund", "rep_pd_date", "filing_date", "accession"])
    chosen_np = npm.groupby(["fund", "rep_pd_date"]).tail(1)  # latest amendment per report date
    chosen_np = chosen_np.assign(quarter=chosen_np.rep_pd_date.map(quarter_of))
    chosen_np = chosen_np[chosen_np.quarter.isin(QUARTERS)]  # study window 2016Q1-2026Q2
    chosen_np = chosen_np.sort_values("rep_pd_date").groupby(["fund", "quarter"]).tail(1)
    for _, r in chosen_np.iterrows():
        for h in r.rows:
            hold.append(dict(fund=r.fund, quarter=r.quarter, report_date=r.rep_pd_date, source_form=r.form, accession=r.accession,
                             source_url=r.url, net_assets=r.net_assets, weight_basis="N-PORT pctVal (% of net assets)", **h))
    covered = set(zip(chosen_np.fund, chosen_np.quarter))
    # ---- legacy forms for the remaining fund-quarters
    # NPORT-EX = N-PORT Part F schedule-of-investments exhibit (covers the N-Q -> public N-PORT transition quarter)
    leg = fl[fl.form.isin(["N-Q", "N-Q/A", "N-CSR", "N-CSR/A", "N-CSRS", "N-CSRS/A", "NPORT-EX", "NPORT-EX/A"]) & (fl.filing_date >= "2016-01-01")].copy()
    info = []
    for _, r in leg.iterrows():
        idx_t, _ = fetch(r.index_url)
        per = re.search(r"Period of Report\s*</div>\s*<div class=\"info\">([0-9-]+)", idx_t or "")
        info.append(per.group(1) if per else None)
    leg["period"] = info
    leg = leg[leg.period.notna()]
    leg["quarter"] = leg.period.map(quarter_of)
    leg = leg[leg.quarter.isin(QUARTERS)]
    leg["is_amend"] = leg.form.str.endswith("/A")
    for (fund, q), g in leg.groupby(["fund", "quarter"]):
        if (fund, q) in covered:
            continue
        g = g.sort_values(["is_amend", "period", "filing_date"], ascending=[True, False, True])
        for _, r in g.iterrows():
            url, period, st = index_main_doc(r.index_url, r.form)
            entry = dict(fund=fund, form=r.form, accession=r.accession, filing_date=r.filing_date, url=url, report_date=r.period, quarter=q)
            if not url:
                log.append(dict(entry, status=f"NO MAIN DOC ({st})", n_holdings=0))
                continue
            raw, st2 = fetch(url)
            if not raw:
                log.append(dict(entry, status=f"FETCH {st2}", n_holdings=0))
                continue
            L = doc_lines(raw)
            spans = find_sections(L, FUND_KEY[fund])
            if not spans:
                log.append(dict(entry, status="FUND SECTION NOT FOUND", n_holdings=0))
                continue
            rows, sf = parse_schedule(L, spans)
            eq = [x for x in rows if x["section"] == "" or EQUITY_SECTION.search(x["section"])]
            status = "PARSED" if len(eq) >= 5 else "PARSE FAILED (fewer than 5 equity rows)"
            log.append(dict(entry, status=status, n_holdings=len(rows), n_equity_rows=len(eq), n_sections=len(spans), shares_first=sf))
            if status != "PARSED":
                continue
            tot = sum(x["value"] for x in eq)
            for x in rows:
                hold.append(dict(fund=fund, quarter=q, report_date=r.period, source_form=r.form, accession=r.accession, source_url=url,
                                 net_assets=float("nan"), weight_basis="value / sum of parsed equity-section values (approximate)",
                                 name=clean_name(x["raw_name"]), title=x["raw_name"], cusip="", isin="", ticker="", other_ids="", lei="",
                                 asset_cat=("EC (from schedule section)" if (x["section"] == "" or EQUITY_SECTION.search(x["section"])) else f"OTHER ({x['section']})"),
                                 issuer_cat="", inv_country="", currency="USD", balance=x["shares"], units="NS", value_usd=x["value"],
                                 pct_net_assets=100 * x["value"] / tot if tot else float("nan"), payoff="Long", schedule_section=x["section"]))
            covered.add((fund, q))
            break
    H = pd.DataFrame(hold)
    write_csv(OUT / "holdings_long.csv", H)
    write_csv(OUT / "parse_log.csv", pd.DataFrame(log))
    cov = []
    for fund in ["ICLN", "TAN", "QCLN", "PBW"]:
        for q in QUARTERS:
            g = H[(H.fund == fund) & (H.quarter == q)]
            cov.append(dict(fund=fund, quarter=q, obtained=len(g) > 0, source_form=";".join(sorted(g.source_form.unique())) if len(g) else "NOT FOUND",
                            report_date=";".join(sorted(g.report_date.unique())) if len(g) else "", n_holdings=len(g),
                            n_equity_lines=int(g.asset_cat.astype(str).str.startswith("EC").sum()) if len(g) else 0,
                            accession=";".join(sorted(g.accession.unique())) if len(g) else ""))
    C = pd.DataFrame(cov)
    write_csv(OUT / "coverage_by_quarter.csv", C)
    print(C.pivot(index="quarter", columns="fund", values="n_holdings").to_string())
    print(pd.DataFrame(log).groupby(["fund", "status"]).size())


if __name__ == "__main__":
    main()
