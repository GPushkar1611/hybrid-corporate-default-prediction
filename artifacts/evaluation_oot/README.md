# Shared out-of-time comparison

This run orchestrates the tree and LSTM pipeline checks using the same packaged panel and temporal cutoffs. It writes model-specific metrics plus `comparison.csv` for the common core feature set.

- Tree snapshots score each eligible firm-month. LSTM scores only months with a complete 12-month continuous history; see `lstm/metrics.json` for endpoint coverage.
- Tree runs also include the structural feature set, which is not yet included in the LSTM comparison.
- Split: train through 2018-12-31, validate in 2020, test from 2022, with 2019 and 2021 purged for the 12-month label horizon.
- Firms may occur across time partitions, so this is not an unseen-company test.
- Current test data has only four positive firms. This is a pipeline smoke comparison, not a reliable performance evaluation.

## Out-of-industry evaluation

Not run: the packaged panel contains no SIC or industry field. Before an OOI experiment, join a historically appropriate firm-to-industry mapping, predeclare held-out industries, and report held-out firm and event counts. Do not infer industry from target outcomes or use current classifications without noting survivorship/reclassification limitations.

See `evaluation.json`, `comparison.csv`, and the nested model artifacts.
