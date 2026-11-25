# Hybrid Structural + ML Corporate Bankruptcy Prediction

AI in Finance (Project 11), IIT Hyderabad, 2026
Team: Pushkar Gudla, Vijaya Sreyas Badde, Adya Thikkavarapu

We predict whether a US non-financial company will **file for bankruptcy within 12 months** of a 10-K/10-Q filing. We compare three families of models:

- **Structural:** Merton-type distance to default (DD) computed from SEC filings.
- **Machine learning:** logistic regression, random forest, XGBoost (static and with history), LSTM.
- **Hybrid:** ML models combined with the structural model in three ways (H1–H3).

Everything is built from free, official SEC EDGAR data.

## Research questions

1. **RQ1:** Do structural features add predictive value on top of a well-featured ML model?
2. **RQ2:** Do they help more under distribution shift (new period, new industry, unseen firms) or with limited data?
3. **RQ3:** How sensitive are the results to how the DD is constructed?
4. **RQ4:** Does history (lag features or an LSTM) help beyond a single filing, and does structure still help once history is included?

## Quick start

```bash
pip install -r requirements.txt

# 1. data (needs internet + an SEC contact e-mail in the user agent)
python -m data_pipeline.download_sec_companyfacts --user-agent "Your Name your.email@domain"
python -m data_pipeline.augment_companyfacts_panel          # builds the v2 panel used by every model

# 2. experiments: open and run from the repository root
jupyter notebook implementation_notebook.ipynb

# tests
python -m unittest tests.test_pipeline -v
```

The notebook writes all figures and tables to `results/`. Run-time switches are at the top of the notebook (`TUNE_TRIALS`, `RUN_LSTM`, ...). A full run takes about 1–3 hours on a laptop CPU.

## Data

| | |
|---|---|
| Unit | one 10-K / 10-Q filing (point in time = filing date) |
| Sample | 252,943 filings, 10,743 non-financial firms, 2010–2025 |
| Label | `y = 1` if the firm's first **8-K Item 1.03** (bankruptcy/receivership) is filed within 12 months after the filing |
| Positives | 2,314 filings from 684 firms (~0.9%) |
| Features | balance-sheet ratios, income-statement and cash-flow ratios, cover-page public float, filing metadata, history (lag) features, structural DD features |

**Pipeline** (all sources are SEC bulk files; raw downloads go to `sec_bulk/`, which git ignores):

1. `data_pipeline/audit_event_inventory_bulk.py` + `data_pipeline/sec_labels.py` → bankruptcy event table (`artifacts/sec_event_validation/`)
2. `data_pipeline/build_companyfacts_panel.py` → **v1 panel**: balance-sheet ratios per filing (FSDS filing metadata + Company Facts values)
3. `data_pipeline/validate_companyfacts_panel.py` → integrity checks on v1 (duplicates, label timing, maturity cutoff)
4. `data_pipeline/augment_companyfacts_panel.py` → **v2 panel** (`artifacts/sec_companyfacts_bankruptcy_v2/`): v1 + income/cash-flow ratios, public float, structural features

Steps 1–3 are only needed to rebuild the panel from scratch: `data_pipeline/download_sec_fsds.py` and `data_pipeline/download_sec_submissions.py` fetch their inputs, several GB. Step 4 needs only `companyfacts.zip`.

## Structural features without stock prices

Free price sources drop delisted firms, and those are exactly the bankrupt ones. So the Merton model is fed with SEC data (`src/structural/sec_structural.py`):

- **Accounting DD:**
  - V = book assets, D = current liabilities + ½ long-term liabilities.
  - σ_A = volatility of the firm's own asset history, shrunk towards its industry median.
  - μ = asset growth.
  - Same idea as the KMV private-firm model.
- **Public-float naive DD:** the Bharath–Shumway (2008) naive DD, with the 10-K cover-page public float as market equity.

`src/structural/merton.py` contains the full market-data Merton/KMV solver (four DD variants). `market_structural_pipeline/` holds the original Yahoo-price pipeline that uses it. It covers only 27 bankrupt firms, so it is kept for reference, not used in the final results.

## Models and evaluation

| Group | Models |
|---|---|
| Structural only | logit on the DD features |
| ML only | logit, random forest, **XGB-static** (one filing), **XGB-lag** (+ history of the last 8 filings), **LSTM** (last 8 filings) |
| Hybrid | **H1** structural features as inputs · **H2** structural PD as a prior (XGBoost `base_margin` / LSTM output offset) · **H3** stacking |
| Control | XGBoost with the raw Merton *inputs* but no DD formula |

- **Main split (out-of-time):** train 2010–18 → gap 2019 → validate 2020 → gap 2021 → test 2022–25. The gap years stop 12-month labels from leaking across splits.
- **Leakage control:** preprocessing is fitted on training rows only, and probabilities are Platt-calibrated on the validation year. Tuning uses an equal budget per model family, on time-ordered folds inside the training years.
- **Other regimes:**
  - rolling year-by-year tests (2015–25),
  - leave-one-industry-out (with and without out-of-time),
  - unseen firms,
  - limited data (2–100% of training firms),
  - prediction horizon (3 / 6 / 12 months).
- **Metrics:** AUROC, average precision (AP), AP lift (AP / base rate), Brier, top-decile capture, calibration and PD-bucket backtests. Model differences use firm-cluster bootstrap 95% CIs.
- **Interpretation:** permutation importance (single features and feature groups), SHAP, feature-group ablation.

## Key results (test 2022–2025, 700 positive filings from 212 firms)

| Model | AUROC | AP | AP lift |
|---|---|---|---|
| Structural only (DD logit) | 0.800 | 0.041 | 3.6× |
| Logit, ML features | 0.827 | 0.051 | 4.6× |
| Logit, hybrid (H1) | 0.848 | 0.064 | 5.7× |
| Random forest, ML features | 0.862 | 0.086 | 7.6× |
| XGB-static, ML features | 0.862 | 0.088 | 7.9× |
| XGB-static, hybrid (H1) | 0.855 | 0.087 | 7.8× |
| XGB-static + structural prior (H2) | 0.865 | 0.093 | 8.3× |
| **XGB-lag, ML features** | **0.872** | 0.100 | 8.9× |
| LSTM, ML features | 0.854 | 0.068 | 6.0× |
| **Stacking (structural + XGB-lag + LSTM)** | **0.872** | **0.103** | **9.2×** |

*From the run of 18 Nov 2026; numbers can shift slightly between runs. Full tables are in `results/tables/`.*

- **ML vs structural:** tree models beat the structural model by about 0.06 AUROC, and beat logit by about 0.035. That matches the ~2.6-point gain Moscatelli et al. (2020) report on public data.
- **RQ1:** structural features clearly help the **linear** model: +0.021 AUROC, CI [0.013, 0.030]. They add **nothing measurable to the tree models**: XGB and RF differences are within ±0.007, with CIs covering 0. SHAP gives structural inputs and outputs about 27% of the attribution, yet removing them costs no accuracy. The trees recover the same information from the raw ratios, consistent with Campbell et al. (2008) and Bharath & Shumway (2008).
- **RQ2:** the structural prior helps most where data are unfamiliar:
  - out-of-industry: AP lift 11.8× (ML) vs 14.9× (H1) and 16.2× (H2);
  - unseen firms: H2 0.865 vs ML 0.855 AUROC.
  - There is no gain out-of-industry + out-of-time, and no consistent gain with limited data.
- **RQ4:** history helps. XGB-lag beats XGB-static (+0.010 AUROC, CI [0.001, 0.020]) and is the best single model. The LSTM is *worse* than XGB-lag (−0.018 AUROC).
- **Calibration:** calibrated PDs are too low in 2022–25, because the 2020 calibration year had unusually few bankruptcies. This is the "slow adaptation" problem also reported by Moscatelli et al.

## Repository layout

```
implementation_notebook.ipynb   all experiments, figures and tables
data_pipeline/                  SEC download, label audit, panel build/validation (run with python -m)
src/pipeline/                   shared experiment code
    data.py         load panel, feature groups and feature sets
    lags.py         history features for XGB-lag
    splits.py       purged time splits + regime splits (rolling, industry, firm, limited data)
    preprocess.py   train-only clipping / imputation / scaling
    calibration.py  class weights, downsampling + Bayes correction, Platt / isotonic
    models.py       structural logit, logit, random forest, XGBoost (+ prior)
    sequence.py     LSTM
    hybrid.py       structural prior (H2), stacking (H3)
    metrics.py      metrics, firm bootstrap, DeLong test, PD-bucket backtest
    runner.py       train -> calibrate -> test for any model; equal-budget tuning
    plots.py        figure helpers
src/structural/                 Merton toolbox + SEC-only structural features
artifacts/                      built datasets, manifests, validation reports
market_structural_pipeline/     original Yahoo-price Merton pipeline (reference only)
results/                        notebook outputs: figures/, tables/, predictions/
reports/                        mid-term report (PDF) and final report (LaTeX source + PDF)
tests/test_pipeline.py          unit tests (splits, preprocessing, calibration, metrics, runner)
```

## Limitations

- The label is an **SEC 8-K Item 1.03 disclosure**, a proxy for court bankruptcy. Firms that never file the 8-K, and out-of-court restructurings, count as non-events.
- Only standard XBRL tags are used. Custom and dimensional facts are omitted, and debt facts are missing for about half of the filings.
- The structural features use **book values and an annual, stale public float**, not daily market prices. They are therefore weaker than a true market-based Merton DD.
- There are about 200 bankrupt firms per test period, so firm-level confidence intervals are wide.
