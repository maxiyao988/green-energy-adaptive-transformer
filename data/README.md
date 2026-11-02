# Data access

Daily OHLCV data were obtained from Yahoo Finance through `yfinance` for
17 tickers: AES, BEP, BLDP, BLNK, CSIQ, CWEN, DQ, ENPH, FCEL, FSLR, HASI,
JKS, NEE, ORA, PLUG, RUN and SEDG. The primary snapshot was collected on
2026-07-08. Weekly observations are reconstructed from daily observations.
Raw vendor prices are not redistributed, and a new download may differ
because of vendor revisions and corporate-action adjustments.

The revision download script is
`revision/research_code/revision_audit/download_revision_data.py`.
Set `SEC_USER_AGENT` before running scripts that query SEC EDGAR:

```sh
export SEC_USER_AGENT="Your Name your.email@example.com"
```

Research scripts retain their original project layout; adjust paths and provide
local data before running them. See `revision/INDEX.md`.
