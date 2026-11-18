# SEC EDGAR bankruptcy panel

Each row is a 10-K/10-Q filing snapshot. Numeric features use standard SEC CompanyFacts values from the same accession and exact reporting-period end date; `date` is the EDGAR filing date. SEC FSDS SUB provides filing metadata and filing-time SIC.

This build contains 252,943 rows from 10,743 firms, including 2,314 positive filing rows from 684 firms. The refreshed Item 1.03 inventory includes 993 non-financial event firms; 778 appear in the CompanyFacts filing panel. The gap is a coverage limit, not a claim that those firms did not experience bankruptcy.

Predictors: `log_assets_usd, liabilities_to_assets, current_ratio, cash_to_assets, debt_to_assets, equity_to_assets, working_capital_to_assets, cash_to_current_liabilities, debt_to_equity, debt_partial`. The 12-month label uses the independently rebuilt SEC Item 1.03 event table. The mature label cutoff is 2025-10-31.

Ratios with denominators below 1,000,000 reporting-currency units are left missing. Negative liabilities/cash and current assets or cash exceeding 105% of total assets are omitted from affected ratios. No dataset-level winsorization is applied. See `dataset_manifest.json` for source hashes, feature mapping, counts, and limitations. Raw CompanyFacts values are SEC-normalized; custom and dimensional concepts are omitted.

Run `validate_companyfacts_panel.py` to reproduce the structural audit. Its current result is `validation_report.json` (PASS): no duplicate firm/date or firm/accession rows, invalid/missed labels, post-event rows, rows beyond the mature cutoff, non-finite predictors, or rows missing all model features. Review high ratio tail values and inspect a targeted sample of original filings before final claims.
