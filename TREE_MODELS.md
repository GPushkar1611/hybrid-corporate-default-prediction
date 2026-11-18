# Tree models: quick guide

This package provides reusable tree-based models. It supports the EDGAR monthly panel and older annual/cross-sectional examples; use the EDGAR smoke runners below for the current bankruptcy-prediction direction.

- `RandomForestModel`: a Random Forest default-risk classifier.
- `XGBoostModel`: an XGBoost default-risk classifier.
- `build_snapshot_features`: one model row per firm-time observation.
- `build_window_features`: a flattened history of `L` observations for tree models.
- `temporal_split`: chronological train/validation/test splitting.
- `evaluate_binary`: shared probability-based metrics for the final notebook.

## Files

| File | Purpose |
| --- | --- |
| `src/tree_models/config.py` | Defines `DatasetSpec`, the column names and positive label for a dataset. |
| `src/tree_models/validation.py` | Validates required columns, target labels, duplicate firm-years, data types, missing values, and suspicious constant-within-firm labels. |
| `src/tree_models/features.py` | Produces snapshot and historical-window feature matrices. It never combines records from different firms. |
| `src/tree_models/splits.py` | Creates train/validation/test partitions in chronological order. |
| `src/tree_models/random_forest.py` | Random Forest wrapper with sensible defaults and class weighting. |
| `src/tree_models/xgboost_model.py` | XGBoost wrapper with binary-probability defaults. |
| `src/tree_models/base.py` | Shared train-only preprocessing, model persistence, and the common API. |
| `src/tree_models/evaluation.py` | AUROC, PR-AUC, Brier score, log loss, top-decile capture, and threshold metrics. |
| `tests/test_tree_models.py` | Small synthetic tests for validation, windows, splitting, and RF probability output. |

## Install

```powershell
pip install -r requirements.txt
```

`xgboost` is only needed when using `XGBoostModel`.

## Snapshot example

```python
import sys
sys.path.append("src")

import pandas as pd
from tree_models import (
    DatasetSpec, RandomForestModel, build_snapshot_features,
    temporal_split, validate_dataset, evaluate_binary,
)

df = pd.read_csv("american_bankruptcy_dataset.csv")
spec = DatasetSpec(
    entity_column="company_name",
    time_column="fyear",
    target_column="status_label",
    feature_columns=[f"X{i}" for i in range(1, 19)] + ["Division", "MajorGroup"],
    positive_label="failed",
    categorical_columns=["Division", "MajorGroup"],
)

# This returns a warning for the current CSV: investigate target semantics
# before interpreting results as a year-ahead default forecast.
print(validate_dataset(df, spec))

snapshot = build_snapshot_features(df, spec)
split = temporal_split(snapshot, "fyear", train_end=2011, validation_end=2014)

model = RandomForestModel(
    categorical_columns=spec.categorical_columns,
    params={"n_estimators": 500, "max_depth": 8, "random_state": 42},
).fit(split.train)

probability = model.predict_proba(split.test)
print(evaluate_binary(split.test.y, probability))
model.save("rf_static.joblib")
```

## EDGAR panel smoke run

The project-specific first pass uses the corrected panel and a one-year-purged time split. Run it from the repository root with the project's configured environment:

```powershell
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_tree_smoke.py
```

The script compares Random Forest and XGBoost on the core market/debt columns, then adds structural features. Training ends in 2018, 2019 is purged, validation is 2020, 2021 is purged, and test is 2022-2024. It writes partition counts and metrics to `artifacts/tree_smoke_run/metrics.json` and test predictions to `artifacts/tree_smoke_run/test_predictions.csv`.

This is only a pipeline check: validation has three positive firms and test has four. Firms can appear in more than one time partition, so it is not an unseen-firm evaluation. The existing preprocessing converts infinite feature values to missing and fits median imputers on training data only.

## LSTM sequence smoke run

The monthly sequence path uses the same EDGAR feature contract and purged split. Each sample is a trailing sequence within one firm; it is skipped if a calendar month is missing. Fit preprocessing on the training period only, then use a fixed CPU LSTM run:

```powershell
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' run_lstm_smoke.py
```

The default sequence length is 12 months. The run trains up to ten epochs and restores the checkpoint with the best validation average precision, using class weighting, dropout, weight decay, and gradient clipping. It writes metrics and test predictions to `artifacts/lstm_smoke_run/`. The current panel has very few positive firms, so the LSTM metrics remain a pipeline check only.

## XGBoost or a historical-window experiment

Use the exact same interface. The model only changes in one line; the feature set changes in one line.

```python
from tree_models import XGBoostModel, build_window_features

# Four annual observations, ending at the prediction year.  Feature names
# become X1_lag0 ... X1_lag3, etc. Incomplete histories are excluded.
windowed = build_window_features(df, spec, window=4)
split = temporal_split(windowed, "fyear", train_end=2011, validation_end=2014)

model = XGBoostModel(
    params={"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05},
).fit(split.train)
probability = model.predict_proba(split.test)
metrics = evaluate_binary(split.test.y, probability)
```

The windowed output contains only numeric lag columns when the input features are numeric. For now, omit categorical sector columns from a window specification; static models can include them directly.

## Tuning in the final notebook

Keep tuning outside the model package. Loop over parameter dictionaries and select using **validation** metrics only; run the test set once after selecting a setting.

```python
for depth in [3, 4, 6]:
    candidate = XGBoostModel(params={"max_depth": depth, "n_estimators": 300})
    candidate.fit(split.train)
    score = evaluate_binary(split.validation.y, candidate.predict_proba(split.validation))["average_precision"]
```

## Legacy American Bankruptcy CSV example

This warning applies to `american_bankruptcy_dataset.csv` and the annual example above, not the current EDGAR panel. The package does not invent label semantics: the legacy `status_label` is constant for each observed company and must be verified before claiming it predicts default in the next year. The current EDGAR panel has a separate documented 12-month forward event label; see `artifacts/edgar_bankruptcy_v1/README.md`.

## Run tests

```powershell
$env:PYTHONPATH = "src"
& 'D:\IITH\Sem 5\AI3005 - AI Project\.venv_B\Scripts\python.exe' -m unittest discover -s tests -v
```
