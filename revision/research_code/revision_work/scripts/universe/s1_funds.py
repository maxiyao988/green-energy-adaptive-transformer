"""Step 6.1 - identify the four ETFs (registrant CIK, series ID, class ID) and list each series' filings.

Identity sources (all EDGAR):
  * https://www.sec.gov/files/company_tickers_mf.json (current CIK / series / class per fund ticker)
  * predecessor series (TAN before its 2018 reorganisation into Invesco): EDGAR full-text search for
    "Guggenheim Solar ETF" N-Q filings, then that filing's index page (Series and Classes section).
Filings per series: EDGAR browse-edgar Atom feed with the series ID as CIK.
Outputs: funds.csv, filings_index.csv
"""
import re
from bs4 import BeautifulSoup
import pandas as pd
from uni_common import *

FUNDS = [("ICLN", "iShares Global Clean Energy ETF"), ("TAN", "Invesco Solar ETF"),
         ("QCLN", "First Trust NASDAQ Clean Edge Green Energy Index Fund"), ("PBW", "Invesco WilderHill Clean Energy ETF")]
KEEP_FORMS = {"NPORT-P", "NPORT-P/A", "N-Q", "N-Q/A", "N-CSR", "N-CSR/A", "N-CSRS", "N-CSRS/A"}


def series_index_info(index_url, series_id):
    """Series name / class / ticker / registrant from a filing index page."""
    t, st = fetch(index_url)
    if not t:
        return None
    soup = BeautifulSoup(t, "html.parser")
    txt = re.sub(r"\s+", " ", soup.get_text(" "))
    m = re.search(rf"Series\s+{series_id}\s+(.+?)\s+Class/Contract (C\d{{9}})\s+(.+?)\s+([A-Z]{{2,5}})\s+(?:Status|Mailing)", txt)
    return dict(series_name=m.group(1).strip() if m else "NOT FOUND", class_id=m.group(2) if m else "NOT FOUND",
                class_ticker=m.group(4) if m else "NOT FOUND")


def list_series_filings(series_id):
    rows, start, reg = [], 0, "NOT FOUND"
    while True:
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={series_id}&type=&dateb=&owner=include&start={start}&count=100&output=atom"
        t, st = fetch(url)
        if not t:
            rows.append(dict(series_id=series_id, form="FETCH_FAILED", note=f"{url} status {st}"))
            break
        soup = BeautifulSoup(t, "html.parser")
        entries = soup.find_all("entry")
        if start == 0 and soup.find("conformed-name"):
            reg = soup.find("conformed-name").get_text().strip()
        for e in entries:
            g = lambda tag: (e.find(tag).get_text().strip() if e.find(tag) else "")
            rows.append(dict(series_id=series_id, form=g("filing-type"), filing_date=g("filing-date"),
                             accession=g("accession-number") or g("accession-nunber"), index_url=g("filing-href"),
                             size=g("size"), registrant_name=reg))
        if len(entries) < 100:
            break
        start += 100
    return rows


def main():
    mf, st = fetch_json("https://www.sec.gov/files/company_tickers_mf.json")
    if not mf:
        raise SystemExit("BLOCKING: company_tickers_mf.json unavailable")
    by_t = {r[3]: dict(zip(mf["fields"], r)) for r in mf["data"]}
    # predecessor of TAN (Guggenheim Solar ETF, Claymore Exchange-Traded Fund Trust 2)
    q = "https://efts.sec.gov/LATEST/search-index?q=%22Guggenheim%20Solar%20ETF%22&forms=N-Q&dateRange=custom&startdt=2016-01-01&enddt=2018-12-31"
    j, _ = fetch_json(q)
    pred = None
    for h in (j or {}).get("hits", {}).get("hits", []):
        if "Claymore Exchange-Traded Fund Trust 2" in " ".join(h["_source"]["display_names"]):
            adsh = h["_source"]["adsh"]
            cik = int(h["_source"]["ciks"][0])
            idx = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{adsh}-index.htm"
            t, _ = fetch(idx)
            m = re.search(r"Series\s+(S\d{9})\s*(?:&nbsp;|\s)*Guggenheim Solar ETF", re.sub(r"<[^>]+>", " ", t or ""))
            if m:
                pred = dict(series_id=m.group(1), registrant_cik=f"{cik:010d}", source=idx, search=q)
                break
    funds, filings = [], []
    for tk, name in FUNDS:
        r = by_t.get(tk)
        series = []
        if r:
            series.append(dict(series_id=r["seriesId"], registrant_cik=f"{int(r['cik']):010d}", class_id=r["classId"], role="current",
                               id_source="https://www.sec.gov/files/company_tickers_mf.json"))
        if tk == "TAN":
            series.append(dict(series_id=pred["series_id"] if pred else "NOT FOUND", registrant_cik=pred["registrant_cik"] if pred else "NOT FOUND",
                               class_id=None, role="predecessor (pre-2018 reorganisation)", id_source=pred["source"] if pred else q))
        for s in series:
            fl = list_series_filings(s["series_id"]) if s["series_id"] != "NOT FOUND" else []
            for f_ in fl:
                f_["fund"] = tk
            filings += fl
            kept = [f_ for f_ in fl if f_.get("form") in KEEP_FORMS and f_.get("index_url")]
            info = series_index_info(kept[0]["index_url"], s["series_id"]) if kept else None
            funds.append(dict(fund=tk, fund_name=name, role=s["role"], registrant_cik=s["registrant_cik"],
                              registrant_name=(fl[0].get("registrant_name", "NOT FOUND") if fl else "NOT FOUND"),
                              series_id=s["series_id"], series_name_on_edgar=(info or {}).get("series_name", "NOT FOUND"),
                              class_id=s["class_id"] or (info or {}).get("class_id", "NOT FOUND"),
                              class_ticker_on_edgar=(info or {}).get("class_ticker", "NOT FOUND"),
                              first_filing=min((f_["filing_date"] for f_ in fl if f_.get("filing_date")), default="NOT FOUND"),
                              last_filing=max((f_["filing_date"] for f_ in fl if f_.get("filing_date")), default="NOT FOUND"),
                              n_holdings_forms=len(kept), id_source=s["id_source"],
                              series_filings_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={s['series_id']}&type=&dateb=&owner=include&count=100"))
            print(funds[-1])
    write_csv(OUT / "funds.csv", pd.DataFrame(funds))
    fdf = pd.DataFrame(filings)
    write_csv(OUT / "filings_index.csv", fdf)
    print(fdf.groupby(["fund", "series_id", "form"]).size().to_string())


if __name__ == "__main__":
    main()
