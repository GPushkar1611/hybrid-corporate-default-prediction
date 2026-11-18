"""Run a one-pass, purged-time tree smoke test on the EDGAR panel.

This is a pipeline check, not a tuned or statistically reliable model study.
The 2019 and 2021 calendar years are purged so every scored label horizon is
settled before the next partition begins.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from src.tree_models import (
    RandomForestModel,
    XGBoostModel,
    build_snapshot_features,
    evaluate_binary,
    purged_temporal_split,
)
from src.edgar_panel import (
    FEATURE_SETS,
    LABEL_HORIZON_MONTHS,
    TEST_START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    edgar_dataset_spec,
)


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def partition_counts(dataset) -> dict:
    positive = dataset.y.eq(1)
    numeric_features = dataset.X.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return {
        "rows": int(len(dataset.y)),
        "firms": int(dataset.metadata["firm"].nunique()),
        "positive_rows": int(positive.sum()),
        "positive_firms": int(dataset.metadata.loc[positive, "firm"].nunique()),
        "positive_rate": float(positive.mean()),
        "features": int(dataset.X.shape[1]),
        "missing_feature_cells": int(dataset.X.isna().sum().sum()),
        "infinite_numeric_feature_cells": int(np.isinf(numeric_features).sum()),
    }


def clean_metrics(y_true, probabilities) -> dict:
    metrics = evaluate_binary(y_true, probabilities)
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in metrics.items()
    }


def run(panel_path: Path, out_dir: Path) -> dict:
    frame = pd.read_csv(panel_path, parse_dates=["date"])
    feature_sets = FEATURE_SETS
    results = {
        "dataset": str(panel_path.resolve()),
        "split": {
            "train_end": TRAIN_END,
            "validation_start": VALIDATION_START,
            "validation_end": VALIDATION_END,
            "test_start": TEST_START,
            "label_horizon_months": LABEL_HORIZON_MONTHS,
            "purge_1": "2019-01-01 through 2019-12-31",
            "purge_2": "2021-01-01 through 2021-12-31",
        },
        "packages": {
            name: package_version(name)
            for name in ("pandas", "scikit-learn", "xgboost")
        },
        "feature_sets": {},
        "warning": (
            "One fixed run only. Validation has three positive firms and the test has four; "
            "metrics are for pipeline verification, not model selection or performance claims."
        ),
    }
    prediction_rows = []

    for feature_name, columns in feature_sets.items():
        spec = edgar_dataset_spec(columns)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Panel lacks {feature_name} features: {missing}")
        dataset = build_snapshot_features(frame, spec)
        split = purged_temporal_split(
            dataset,
            spec.time_column,
            train_end=TRAIN_END,
            validation_start=VALIDATION_START,
            validation_end=VALIDATION_END,
            test_start=TEST_START,
            horizon_months=LABEL_HORIZON_MONTHS,
        )
        train, validation, test = split.train, split.validation, split.test
        dates = pd.to_datetime(dataset.metadata[spec.time_column])
        purged = {
            "2019_rows_purged": int(((dates >= "2019-01-01") & (dates <= "2019-12-31")).sum()),
            "2021_rows_purged": int(((dates >= "2021-01-01") & (dates <= "2021-12-31")).sum()),
        }
        train_positive = int(train.y.sum())
        train_negative = int(len(train.y) - train_positive)
        imbalance_weight = train_negative / train_positive
        results["feature_sets"][feature_name] = {
            "features": columns,
            "partitions": {
                "train": partition_counts(train),
                "validation": partition_counts(validation),
                "test": partition_counts(test),
                "purged": purged,
            },
            "models": {},
        }

        models = {
            "random_forest": RandomForestModel(
                params={"n_estimators": 300, "max_depth": 8, "random_state": 42}
            ),
            "xgboost": XGBoostModel(
                params={
                    "n_estimators": 300,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "scale_pos_weight": imbalance_weight,
                    "tree_method": "hist",
                    "random_state": 42,
                }
            ),
        }
        for model_name, model in models.items():
            model.fit(train)
            val_prob = model.predict_proba(validation)
            test_prob = model.predict_proba(test)
            results["feature_sets"][feature_name]["models"][model_name] = {
                "validation": clean_metrics(validation.y, val_prob),
                "test": clean_metrics(test.y, test_prob),
                "training_scale_pos_weight": float(imbalance_weight) if model_name == "xgboost" else None,
            }
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "feature_set": feature_name,
                        "model": model_name,
                        "firm": test.metadata["firm"].to_numpy(),
                        "date": test.metadata["date"].to_numpy(),
                        "y": test.y.to_numpy(),
                        "probability": test_prob,
                    }
                )
            )
            print(
                f"{feature_name} / {model_name}: "
                f"validation AP={results['feature_sets'][feature_name]['models'][model_name]['validation']['average_precision']:.4f}, "
                f"test AP={results['feature_sets'][feature_name]['models'][model_name]['test']['average_precision']:.4f}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.concat(prediction_rows, ignore_index=True).to_csv(out_dir / "test_predictions.csv", index=False)
    readme = f"""# EDGAR tree smoke run

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

The training partition contains {results['feature_sets']['core']['partitions']['train']['positive_firms']} positive firms; validation contains {results['feature_sets']['core']['partitions']['validation']['positive_firms']}; test contains {results['feature_sets']['core']['partitions']['test']['positive_firms']}. Those event counts are too small for reliable comparisons. Use these results to confirm the data/model path runs; do not use them to select a final model or claim generalization.

`metrics.json` contains partition counts and metrics. `test_predictions.csv` contains all four model/feature-set test scores and labels for inspection.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("artifacts/edgar_bankruptcy_v1/bankruptcy_panel.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/tree_smoke_run"))
    args = parser.parse_args()
    results = run(args.panel, args.out_dir)
    print(f"Results written to {args.out_dir.resolve()}")
    print(results["warning"])


if __name__ == "__main__":
    main()
