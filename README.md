# Corporate Bankruptcy Prediction — Team Handoff

Internal project guide and status log for the team's shift from generic default prediction to SEC bankruptcy-event prediction.

## Start here

| Purpose | Location |
| --- | --- |
| Current validated SEC panel, provenance, rules, and limitations | [`artifacts/sec_companyfacts_bankruptcy_v1/README.md`](artifacts/sec_companyfacts_bankruptcy_v1/README.md) |
| SEC label audit and local inventory comparison | [`artifacts/sec_event_validation/sec_bulk_event_audit.json`](artifacts/sec_event_validation/sec_bulk_event_audit.json) |
| Current OOT evaluation and predictions | [`artifacts/sec_companyfacts_final_eval/README.md`](artifacts/sec_companyfacts_final_eval/README.md) |
| Shared feature and temporal split contract | [`src/edgar_panel.py`](src/edgar_panel.py) |
| SEC data builder | [`build_companyfacts_panel.py`](build_companyfacts_panel.py) |
| Reproducible dataset integrity audit | [`validate_companyfacts_panel.py`](validate_companyfacts_panel.py) |
| SEC bulk download scripts and manifests | [`download_sec_fsds.py`](download_sec_fsds.py), [`download_sec_submissions.py`](download_sec_submissions.py), [`download_sec_companyfacts.py`](download_sec_companyfacts.py) |
| Tree/LSTM evaluator | [`run_sec_filing_eval.py`](run_sec_filing_eval.py) |
| Original prototype pipeline and its OOT runner | [`build_panel.py`](build_panel.py), [`run_evaluation.py`](run_evaluation.py) |

## Current data state

The active dataset is `artifacts/sec_companyfacts_bankruptcy_v1/sec_bankruptcy_panel.csv`, built from official SEC EDGAR sources. It contains **252,943 filing snapshots from 10,743 firms**, including **2,314 positive filing rows from 684 firms**. SEC CompanyFacts supplies normalized standard US-GAAP values, joined to FSDS SUB filing metadata on exact accession and exact reporting-period end date. The model observation date is the filing date.

The target is the first SEC 8-K/8-K-A Item 1.03 event within 12 months after each filing. The label inventory was rebuilt from SEC Submissions bulk data: 978 local rows match the SEC-selected accession, 12 local rows use a different verified accession with the same event date, and SEC has three additional recent event firms. There are no date mismatches or unverified local accessions. The three newer events are beyond the mature label window.

**Do not use** `artifacts/sec_fsds_bankruptcy_v1/sec_bankruptcy_panel.csv` for modeling. SEC FSDS NUM omits XBRL scale attributes; its raw values created impossible ratios. That artifact is retained for traceability and marked quarantined. The active panel uses CompanyFacts values instead.

Coverage is materially better than the old prototype, but not complete: 778 of the 993 event firms in the refreshed event inventory appear in the accounting panel; some event companies lack usable standard CompanyFacts/FSDS filings. The target is an SEC Item 1.03 disclosure proxy, not a court-confirmed bankruptcy label. Standard non-custom, entity-wide facts are used; custom/dimensional facts are omitted. See the dataset manifest for ratio rules, hashes, and missingness.

## Current model state

Random Forest, XGBoost, and a two-layer LSTM have been run on the active panel with the same OOT dates: train through 2018, purge 2019, validate 2020, purge 2021, test from 2022 through the mature cutoff (2025-10-31). Firms may appear in more than one time partition, so this is OOT, not unseen-firm validation. The LSTM uses up to eight prior filing observations, training-only imputation/scaling/clipping, class weighting, gradient clipping, early stopping on validation AP, and a three-seed probability ensemble.

On the same LSTM-eligible test endpoints, AP was 0.0527 for Random Forest, 0.0520 for XGBoost, and 0.0474 for the LSTM ensemble. The positive rate was about 1.09%; all rank above prevalence, but none clearly beats the others. Firm-cluster bootstrap intervals overlap. The class-weighted outputs are not calibrated event probabilities yet; use ranking metrics for now and calibrate on validation data before interpreting probabilities or thresholds.

These are first full-panel results, not final report claims. Test average precision is about 4.4–4.8 times the test positive rate, but the firm-cluster bootstrap intervals overlap across models, so there is no clear model winner. The LSTM is a valid implemented baseline, but it did not outperform the trees in this run. Freeze the target, model choices, and split before any further test-set decisions. Tuning belongs on validation.

## Next steps for our work

1. Review and lock the teammate's structural feature handoff: firm key, date meaning, units, feature/status semantics, and missing values.
2. Join the final structural outputs to this SEC panel using point-in-time-safe keys, then rerun the common OOT model comparison including tree and LSTM variants with and without structural features.
3. Use filing SIC/SIC2 to predeclare an out-of-industry holdout. Check positive-firm counts before interpreting OOI results; report which groups are held out.
4. Inspect the tagged fact mappings against source filings for a small targeted sample, especially unusual balance-sheet observations and missing debt facts.
5. Calibrate model probabilities on validation data, then freeze evaluation choices and report OOT/OOI uncertainty and sample limitations in the project report.

## Rebuild commands

Run every Python command with the team's designated environment:

```powershell
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' audit_event_inventory_bulk.py
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' build_companyfacts_panel.py
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' validate_companyfacts_panel.py
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_sec_filing_eval.py --epochs 15 --lstm-seeds 42 43 44 --bootstrap-replicates 500
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' audit_matched_sequence_support.py
```

Raw SEC downloads are stored under ignored `sec_bulk/`; manifests record source hashes. The download scripts require a `SEC_USER_AGENT` environment variable containing a real contact. SEC sources: [Financial Statement Data Sets](https://www.sec.gov/dera/data/financial-statement-data-sets.html), [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), and [Financial Statement Data documentation](https://www.sec.gov/files/financial-statement-data-sets.pdf).
