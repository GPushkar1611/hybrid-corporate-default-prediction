# Tree models: quick guide

This package provides the tree-based part of the project:

- `RandomForestModel`: a Random Forest default-risk classifier.
- `XGBoostModel`: an XGBoost default-risk classifier.
- `build_snapshot_features`: one model row per firm-year.
- `build_window_features`: a flattened history of `L` firm-years for tree models.
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

## Important limitation of the supplied CSV

The package can model the CSV, but it does not invent label semantics. The present `status_label` is constant for each observed company, so it must be verified before claiming that it represents default in the next year. This is why `validate_dataset` emits a warning. A later EDGAR panel with a documented forward default label can use the same code unchanged.

## Run tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
