# SEC EDGAR bankruptcy panel, v0.1.0

## Contents

- `bankruptcy_panel.csv`: firm-month modelling panel, 57,749 rows and 599 firms.
- `edgar_event_inventory.csv`: all 990 event firms in the local SEC-derived label table, including coverage outcomes.
- `firm_label_coverage.csv`: firm-level observation period, positive-month count, and no-positive explanation.
- `audit.json` and `dataset_manifest.json`: checks, source hashes, definitions, and provenance.

## Event and target definitions

An event is the first 8-K or 8-K/A filing with SEC Item 1.03 (Bankruptcy or Receivership) since 2010. The upstream label code uses `reportDate` if it is valid and not after `filingDate`; otherwise it uses `filingDate`. For a firm-month observation at date *t*, `y=1` if the event date is in *(t, t + 12 months]*. Rows at or after the event date are not included in the panel.

## Construction and provenance

This artifact packages the existing SEC-derived label tables and the existing `panel.csv`. It does not download or re-parse SEC bulk archives during this run. The event inventory records how the 990 local event labels pass through fundamentals and price coverage; the modelling panel has a narrower market-data eligibility filter. See the manifest for SHA-256 hashes and input paths.

The panel features originate from the point-in-time routines in `data.py` and `build_panel.py`: SEC XBRL debt and share facts, Yahoo Finance prices, and the FRED DGS1 one-year Treasury series. SEC-reported facts are used only after filing availability; prices are split-adjusted and not dividend-adjusted.

## Rebuild

Run from the project root in the specified virtual environment:

```powershell
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' dataset_generation.py --mode audit --panel 'C:\Users\vijay\Downloads\panel.csv'
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' dataset_generation.py --mode build --panel 'C:\Users\vijay\Downloads\panel.csv'
```

## Current coverage and limitations

- 990 bankruptcy labels are in the local event inventory; 489 have a fundamentals/price-check record.
- 29 firms pass the saved price-coverage check; 27 event firms appear in the modelling panel.
- EDGAR label fundamentals statuses: ok=489, no_debt=237, no_xbrl=143, stale=99, no_shares=22. The parser currently checks a limited list of standard US-GAAP debt concepts; `no_debt` can reflect missing usable debt facts or tag coverage, not necessarily zero debt. SEC Company Facts APIs aggregate standard taxonomies and omit issuer custom taxonomies.
- The saved control table contains 10 firms that also occur in the full EDGAR event inventory. The modelling-panel builder now excludes the full event inventory from controls on future rebuilds; this packaging step also reconciles any such firm against the event dates.
- The panel contains 186 positive firm-months from 17 firms. 10 event firms have no positive rows because their event falls more than 12 months after their final observation.
- The source panel had 617 firm-month rows whose `bankrupt_firm` flag disagreed with the full EDGAR event table. This builder removes 608 post-event rows and recomputes the event flag from the full inventory. The corrected target has 0 mismatches against that inventory.
- Controls were sampled from the current SEC ticker list and therefore favor surviving registrants. The builder removes firms with a known event from the control set when such firms appear in the EDGAR inventory. Many failed firms lack historical tickers or price coverage; the modelling panel is not a representative sample of all EDGAR bankruptcy events.
- The raw SEC bulk archives are not included here. Rebuilding the label inventory from primary SEC files requires the upstream bulk archives and a valid SEC User-Agent in `edgar_labels.py`.
- Nonzero structural-model status rows indicate invalid or insufficient inputs; inspect the corresponding `*_status` fields before selecting structural features.

Do not use `y`, `bankrupt_firm`, `firm`, `date`, or status fields as ordinary predictors. This is a dataset construction artifact, not evidence of model performance.
