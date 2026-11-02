"""Step 6.3a - SEC Official Lists of Section 13(f) Securities, 2016Q1-2026Q2.

The list for quarter q contains the equity securities (incl. ADRs, foreign shares and units) traded on a
U.S. national securities exchange or Nasdaq, by CUSIP; it is the evidence used for "trades on a U.S.
exchange" in quarter q. 2016-2019 lists are PDF (parsed with pypdf; two page layouts handled), 2020+ are fixed-width text.
Output: sec13f_lists_parsed.csv (quarter, cusip, has_options_flag, issuer, description, deleted, text, source_url)
"""
import io
import re
import pandas as pd
from uni_common import *

PAGE = "https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/official-list-section-13f-securities"
QUARTERS = [(y, q) for y in range(2016, 2027) for q in range(1, 5)][:42]
CUSIP_PDF = re.compile(r"([0-9A-Z]{6}) ([0-9A-Z]{2}) ([0-9])\s?(\*?)")


def links():
    t, st = fetch(PAGE)
    if not t:
        raise SystemExit(f"BLOCKING: 13(f) list page unavailable ({st})")
    hrefs = set(re.findall(r'href="([^"]*13flist[^"]*)"', t))
    out = {}
    for y, q in QUARTERS:
        cand = [h for h in hrefs if re.search(rf"13flist{y}q{q}(-txt)?\.(txt|pdf)$", h)]
        txt = [h for h in cand if h.endswith(".txt")]
        pdf = [h for h in cand if h.endswith(".pdf")]
        pick = (txt or pdf or [None])[0]
        out[(y, q)] = ("https://www.sec.gov" + pick) if pick and pick.startswith("/") else pick
    return out


def parse_txt(t):
    rows = []
    for line in t.splitlines():
        if len(line) < 12 or not re.match(r"^[0-9A-Z]{9}[ *]", line):
            continue
        rest = line[10:]
        issuer = rest[:30].strip()
        desc = rest[30:].strip()
        deleted = "*D*" in desc or "DELETED" in desc
        desc = re.sub(r"\s+(\*[A-Z]\*)?\s*[A-Z]?$", "", desc.replace("*D*", "").replace("*A*", "")).strip()
        rows.append(dict(cusip=line[:9], has_options_flag=line[9] == "*", issuer=issuer, description=desc,
                         deleted=deleted, text=f"{issuer} {desc}".strip()))
    return rows


def parse_pdf(b):
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(b))
    rows = []
    for page in r.pages:
        txt = page.extract_text() or ""
        txt = re.sub(r"Run Date:.*?Page \d+", " ", txt)
        # layout A (most years): one entry per line, CUSIP as a single 9-character block
        lineform = []
        for line in txt.split("\n"):
            m = re.match(r"^([0-9A-Z]{9})(\*?)\s*(\S.*)$", line.strip())
            if m and re.search(r"\d", m.group(1)):
                body = m.group(3)
                deleted = bool(re.search(r"\bDELETED\b", body))
                body = re.sub(r"\b(ADDED|DELETED)\b", "", body).strip()
                lineform.append(dict(cusip=m.group(1), has_options_flag=m.group(2) == "*", issuer="", description="",
                                     deleted=deleted, text=re.sub(r"\s+", " ", body)))
        txt = txt.replace("\n", " ")
        ms = list(CUSIP_PDF.finditer(txt))
        if len(lineform) > len(ms):
            rows += lineform
            continue
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(txt)
            body = re.sub(r"\s+", " ", txt[m.end():end]).strip()
            deleted = bool(re.search(r"\bDELETED\b", body))
            body = re.sub(r"\b(ADDED|DELETED)\b", "", body).strip()
            body = re.sub(r"(CUSIP NO ISSUER NAME ISSUER DESCRIPTION STATUS|Year: Run Time:.*$|Total Count:.*$)", "", body).strip()
            rows.append(dict(cusip=m.group(1) + m.group(2) + m.group(3), has_options_flag=m.group(4) == "*", issuer="",
                             description="", deleted=deleted, text=body))
    return rows


def main():
    L = links()
    allrows, info = [], []
    for (y, q), url in L.items():
        qq = f"{y}Q{q}"
        if not url:
            info.append(dict(quarter=qq, url="NOT FOUND", status="NOT FOUND", n_entries=0))
            continue
        b, st = fetch(url, binary=True)
        if not b:
            info.append(dict(quarter=qq, url=url, status=f"FETCH {st}", n_entries=0))
            continue
        rows = parse_txt(b.decode("latin-1")) if url.endswith(".txt") else parse_pdf(b)
        valid = sum(valid_cusip(r["cusip"]) for r in rows)
        for r in rows:
            r.update(quarter=qq, source_url=url)
        allrows += rows
        info.append(dict(quarter=qq, url=url, status="PARSED", n_entries=len(rows), n_valid_check_digit=valid,
                         share_valid=valid / max(len(rows), 1)))
        print(info[-1], flush=True)
    write_csv(OUT / "sec13f_lists_parsed.csv", pd.DataFrame(allrows))
    write_csv(OUT / "sec13f_lists_log.csv", pd.DataFrame(info))


if __name__ == "__main__":
    main()
