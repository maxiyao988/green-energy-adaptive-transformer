# Adaptive Multi-Scale Transformer for Green-Energy Stock Return Forecasting

Code, audit materials and result tables for the paper

> **An Adaptive Multi-Scale Transformer for Green-Energy Stock Return Forecasting: A Controlled Evaluation of Static Scale Aggregation and Sample-Dependent Routing**
> Xiyao Ma and Chenxue Yang. Manuscript computers-4613491, *Computers* (MDPI), under review (revised version, November 2026).

The study asks whether sample-dependent routing over temporal scales improves stock return forecasts beyond fixed or globally learned (static) scale aggregation. A multi-scale Transformer combines Daily and Weekly price–volume inputs to forecast 5-, 10- and 20-trading-day returns for a retrospectively assembled panel of 17 green-energy stocks (43,571 stock–date samples).

## Main findings

- **Forecast skill is limited.** Under the baseline training configuration in the primary test period, every evaluated neural configuration has a higher mean squared error, averaged across training runs, than the global training-mean predictor.
- **Routing adds no consistent value.** Adaptive (sample-dependent) routing does not show a consistent advantage over static aggregation across training settings, targets and test windows. In frozen-representation controls it does not significantly outperform a parameter-matched constant-input gate.
- **Training randomness matters for inference.** Apparent advantages supported by inference conditional on the fitted models weaken once variation across training runs is included (seed × date bootstrap).
- **Input construction matters more than the weighting rule.** Window-relative inputs improve accuracy and reduce between-seed dispersion relative to price-level inputs.
- **A revised training configuration helps against the historical mean, not against zero.** In retrospective evaluations of two earlier windows, its ensemble forecasts improve pooled accuracy relative to the historical mean (one-sided Clark–West tests), but not relative to the zero-return forecast.
- **Portfolio results are descriptive.** Mean–variance (Ledoit–Wolf) and top-5 portfolios net of transaction costs vary with input representation, horizon and allocation rule, without a consistent economic advantage for adaptive routing.

The conclusions are specific to this retrospective panel and the evaluated periods.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/green_energy_transformer/` | Clean, installable implementation of the data pipeline, the multi-scale model (Single, Fixed, Static, Adaptive), the benchmark models, training and evaluation, refactored from the original-submission pipeline. |
| `experiments/01–05_*.py` | Reproduction scripts for the original-submission experiments (benchmark, input configurations, mechanism ablation, multi-seed inference, router diagnostics). |
| `revision/` | Research code for every analysis added in the revision, kept as it was run (see `revision/INDEX.md`), plus the scripts that draw Figures 3–5 and S1 and build the supplementary workbook. |
| `results/supplementary/` | Supplementary materials of the paper: Tables S1–S18 (a formatted workbook and a full-records workbook with per-seed results), and Tables S19–S20 with Figure S1. |
| `audit/` | SEC EDGAR source lists: the issuer filings checked in the corporate-action audit, and the fund filings (168 fund-quarters) used to reconstruct the historical ETF-holdings universe. |
| `tests/` | Temporal-integrity tests for the clean pipeline. |
| `data/` | Data-access notes. Raw vendor prices are not redistributed. |

`revision/INDEX.md` maps the analyses of the paper to the scripts that produced them.

## Data

Daily OHLCV data were obtained from Yahoo Finance through the `yfinance` package. The primary snapshot was collected on 8 July 2026; a second snapshot (16 October 2026) was used for the corporate-action audit. Raw prices are not included; the download scripts are provided, and later downloads can differ because of vendor revisions and corporate-action adjustments. The dataset is retrospectively assembled and is not a point-in-time archive.

The SEC EDGAR filings used in the corporate-action audit and in the historical ETF-holdings reconstruction (ICLN, TAN, QCLN and PBW; 168 fund-quarters, 2016Q1–2026Q2) are listed in `audit/`. Scripts that query EDGAR read the required contact string from the environment variable `SEC_USER_AGENT` (for example `export SEC_USER_AGENT="Your Name your.email@example.com"`).

## Environment

The revision analyses were run on CPU with Python 3.12.1, PyTorch 2.2.0, NumPy 1.26.4, pandas 2.2.0, SciPy 1.13.0 and scikit-learn 1.5.0 (`requirements-revision.txt`). The clean package requires Python 3.10 or later:

```bash
git clone https://github.com/maxiyao988/green-energy-adaptive-transformer.git
cd green-energy-adaptive-transformer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest
```

## Reproducibility notes

- The five training seeds {0, 1, 21, 42, 3407} were fixed before the experiments.
- The research code in `revision/` is published as run. Paths inside it refer to the original project layout and may need adjusting; it has not been refactored into the package.
- Pre-specified protocols for the revision analyses are included next to their scripts (`PROTOCOL_*.md`).
- Model checkpoints, per-draw bootstrap arrays and per-observation prediction files are not included because of their size.

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). Volume, pages and DOI will be added after publication.
