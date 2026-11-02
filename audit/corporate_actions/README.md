# Corporate-action SEC source lists

Issuer filings were retrieved from SEC EDGAR. The vendor comparison uses two
Yahoo Finance snapshots: the original 2026-07-08 snapshot and the second
retrieval on 2026-10-16. The original acquisition date is user-supplied; these
snapshots are not an authenticated point-in-time vendor archive. Raw prices
and cached filing bodies are not included.

| File | Contents | Producer in this repository |
|---|---|---|
| `source_catalog.csv` | SEC URLs, filing identifiers, retrieval status, hashes and original cache/text locations | `revision/research_code/revision_work/outputs/data_audit/scripts/assemble_inventory.py` |
| `A_event_verification.csv` | Eight vendor split-type events compared with issuer terms, ratios and dates | Same `assemble_inventory.py` |
| `B_handling_rules.csv` | Treatment of corporate actions in level inputs, window-relative inputs, labels and return calculations | Same `assemble_inventory.py` |
| `A_cash_dividend_spotchecks.csv` | Dividend coverage and issuer spot checks, including amount/date limitations | Same `assemble_inventory.py` |

The associated retrieval code is
`revision/research_code/revision_work/outputs/data_audit/scripts/collect_sources.py`
and `collect_followup.py` in that directory. Cache paths document the original
research layout; the caches are not distributed. Source URLs, accession numbers,
quoted issuer terms and numerical results are preserved. Set `SEC_USER_AGENT`
before querying EDGAR. These lists retain the documented BEP event mismatch
and the limits of vendor event classification.
