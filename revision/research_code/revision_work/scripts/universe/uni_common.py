"""Shared helpers for Step 6 (survivorship quantification from ETF holdings). No training.

* Every SEC request sends the fair-access User-Agent, stays <= 4 requests/second, and is cached
  under revision_work/outputs/universe/cache/; a cached URL is never downloaded again.
* HTTP 403/429 (and transient 5xx) -> back off 10 s, 30 s, 60 s (logged), then give up (recorded as a data gap).
* Outputs are created exclusively (mode 'x'); nothing is overwritten.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys
sys.dont_write_bytecode = True
import csv
import gzip
import hashlib
import json
import re
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[3]
OUT_BASE = ROOT / "revision_work" / "outputs" / "universe"
CACHE = OUT_BASE / "cache"  # the download cache always lives in the real output tree
# UNI_OUT_DIR may redirect result files (dry runs only); the cache is never redirected.
OUT = Path(os.environ.get("UNI_OUT_DIR", OUT_BASE))
DEPS = OUT_BASE / "_deps"
if DEPS.exists():
    sys.path.append(str(DEPS))
FORBIDDEN = ROOT / "revision_work" / "outputs" / "causal_features"
UA = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")
SEC_HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
SEC_HOSTS = ("sec.gov", "www.sec.gov", "efts.sec.gov", "data.sec.gov")
MIN_INTERVAL = 0.25  # <= 4 requests per second
BACKOFF = (10, 30, 60)
STUDY_TICKERS = ["AES", "BEP", "BLDP", "BLNK", "CSIQ", "CWEN", "DQ", "ENPH", "FCEL", "FSLR", "HASI",
                 "JKS", "NEE", "ORA", "PLUG", "RUN", "SEDG"]
_last = [0.0]
_session = requests.Session()


def utc():
    return datetime.now(timezone.utc).isoformat()


def rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def target(p):
    p = Path(p).resolve()
    if not (p.is_relative_to(OUT.resolve()) or p.is_relative_to(OUT_BASE.resolve())) or p.is_relative_to(FORBIDDEN.resolve()):
        raise ValueError(f"Output outside permitted directory: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"Refusing overwrite: {p}")
    return p


def write_bytes(p, b):
    with target(p).open("xb") as f:
        f.write(b)


def write_text(p, s):
    write_bytes(p, s.encode("utf-8"))


def write_csv(p, df):
    write_bytes(p, df.to_csv(index=False).encode("utf-8"))


def write_json(p, obj):
    write_text(p, json.dumps(obj, indent=2, default=str))


def cache_path(url):
    u = urlparse(url)
    path = u.path.strip("/") or "index"
    if u.query:
        path += "__q_" + hashlib.sha1(u.query.encode()).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9._/\-]", "_", path)
    if safe.endswith("/") or "." not in Path(safe).name:
        safe += ".dat"
    return CACHE / u.netloc / safe


def _log(row):
    lp = CACHE / "request_log.csv"
    lp.parent.mkdir(parents=True, exist_ok=True)
    new = not lp.exists()
    with lp.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "url", "status", "bytes", "note"])
        w.writerow(row)


def fetch(url, binary=False, headers=None):
    """Return (content or None, status). Cached content is returned without any network call."""
    cp = cache_path(url)
    if cp.exists():
        b = cp.read_bytes()
        return (b if binary else b.decode("utf-8", errors="replace")), "CACHED"
    host = urlparse(url).netloc
    hdr = dict(SEC_HEADERS) if host.endswith("sec.gov") else {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    if headers:
        hdr.update(headers)
    for attempt in range(len(BACKOFF) + 1):
        wait = MIN_INTERVAL - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            r = _session.get(url, headers=hdr, timeout=60)
        except requests.RequestException as e:
            _log([utc(), url, "EXC", 0, repr(e)[:200]])
            if attempt < len(BACKOFF):
                time.sleep(BACKOFF[attempt])
                continue
            return None, "EXC"
        if r.status_code in (403, 429, 500, 502, 503, 504):  # rate limits and transient server errors
            _log([utc(), url, r.status_code, 0, f"backoff {BACKOFF[attempt] if attempt < len(BACKOFF) else 'give-up'} s"])
            if attempt < len(BACKOFF):
                time.sleep(BACKOFF[attempt])
                continue
            return None, str(r.status_code)
        _log([utc(), url, r.status_code, len(r.content), ""])
        if r.status_code != 200:
            return None, str(r.status_code)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(cp.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.rename(cp)
        return (r.content if binary else r.content.decode("utf-8", errors="replace")), "200"
    return None, "FAILED"


def fetch_json(url):
    t, st = fetch(url)
    return (json.loads(t) if t else None), st


def norm_name(s):
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    stop = {"INC", "CORP", "CORPORATION", "CO", "LTD", "LIMITED", "PLC", "SA", "AG", "NV", "LP", "L", "P", "THE", "HOLDINGS",
            "HOLDING", "GROUP", "CLASS", "CL", "A", "B", "C", "COM", "COMMON", "STOCK", "SHS", "NEW", "ADR", "ADS", "SPONSORED",
            "SPON", "ORD", "ORDINARY", "UNITS", "UNIT", "PARTNERSHIP", "PARTNERS", "TRUST", "SE", "ASA", "OYJ", "COMPANY", "DE", "THE"}
    return " ".join(w for w in s.split() if w not in stop)


def name_overlap(a, b):
    A, B = set(norm_name(a).split()), set(norm_name(b).split())
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def quarter_of(date_str):
    d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def cusip_check_digit(c8):
    total = 0
    for i, ch in enumerate(c8):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch.upper()) - 55
        elif ch == "*":
            v = 36
        elif ch == "@":
            v = 37
        elif ch == "#":
            v = 38
        else:
            return None
        if i % 2 == 1:
            v *= 2
        total += v // 10 + v % 10
    return str((10 - total % 10) % 10)


def valid_cusip(c):
    c = (c or "").strip().upper()
    return len(c) == 9 and c[:8] != "00000000" and cusip_check_digit(c[:8]) == c[8]


def cusip_from_isin(isin):
    isin = (isin or "").strip().upper()
    if len(isin) == 12 and valid_cusip(isin[2:11]):
        return isin[2:11]
    return None
