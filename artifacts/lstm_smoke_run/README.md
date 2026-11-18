# EDGAR LSTM smoke run

One fixed CPU run to verify sequence construction, preprocessing, training, and prediction. It is not tuned and is not a final evaluation.

- Each sample is a 12-month sequence from one firm; sequences cannot cross firms or missing calendar months.
- Only endpoints with a complete 12-month continuous history are retained. See `metrics.json` `sequence_coverage` for how many rows and positive endpoints this excludes in each partition.
- The endpoint month supplies the target. All input features are point-in-time features at or before that endpoint.
- Imputation and scaling are fitted on raw observations dated on/before 2018-12-31 only.
- Training ends in 2018-12-31; 2019 is purged; validation is calendar year 2020; 2021 is purged; test begins 2022-01-01.
- Validation has 2 positive firms; test has 4 positive firms.
- Firms may recur across time partitions. This is not an unseen-firm evaluation.

See `metrics.json` and `test_predictions.csv`. Metrics are for pipeline verification only.
