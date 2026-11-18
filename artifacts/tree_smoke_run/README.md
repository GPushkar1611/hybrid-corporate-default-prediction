# EDGAR tree smoke run

This is one fixed, untuned pipeline run on the packaged SEC EDGAR bankruptcy panel. It is not a final model evaluation.

## Split

- Train through 2018-12-31.
- Purge all 2019 rows so training labels have matured before validation.
- Validate on 2020.
- Purge all 2021 rows so validation labels have matured before test.
- Test on 2022-2024.

Firms can appear in more than one time partition. This is a temporal holdout for future observations of the observed firm universe, not an unseen-firm evaluation.

## Features and models

The run compares Random Forest and XGBoost on six core market/debt features, then adds distance-to-default, probability-of-default, and debt-to-asset features from the four structural variants. Preprocessing is fitted on training data. Infinite structural values are converted to missing values and median-imputed from training data.

## Read the result carefully

The training partition contains 10 positive firms; validation contains 3; test contains 4. Those event counts are too small for reliable comparisons. Use these results to confirm the data/model path runs; do not use them to select a final model or claim generalization.

`metrics.json` contains partition counts and metrics. `test_predictions.csv` contains all four model/feature-set test scores and labels for inspection.
