"""Run a fixed, firm-isolated LSTM smoke test on the EDGAR monthly panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.edgar_panel import (
    CORE_FEATURES,
    LABEL_HORIZON_MONTHS,
    TEST_START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    edgar_dataset_spec,
)
from src.sequence_models import LSTMClassifier, build_sequences
from src.tree_models import evaluate_binary, purged_temporal_split


def _clean_metrics(y_true, probabilities) -> dict:
    metrics = evaluate_binary(y_true, probabilities)
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in metrics.items()
    }


def _partition_counts(dataset) -> dict:
    positive = dataset.y.eq(1)
    return {
        "sequences": int(len(dataset.y)),
        "firms": int(dataset.metadata["firm"].nunique()),
        "positive_endpoints": int(positive.sum()),
        "positive_firms": int(dataset.metadata.loc[positive, "firm"].nunique()),
        "positive_rate": float(positive.mean()),
    }


def _sequence_coverage(frame: pd.DataFrame, sequences, spec) -> dict:
    """Compare eligible panel endpoints with endpoints having a full sequence."""
    date = pd.to_datetime(frame[spec.time_column])
    partitions = {
        "train": date <= pd.Timestamp(TRAIN_END),
        "validation": date.between(VALIDATION_START, VALIDATION_END),
        "test": date >= pd.Timestamp(TEST_START),
    }
    available = sequences.metadata.set_index([spec.entity_column, spec.time_column]).index
    output = {}
    for name, mask in partitions.items():
        candidate = frame.loc[mask, [spec.entity_column, spec.time_column, spec.target_column]]
        candidate_index = pd.MultiIndex.from_frame(candidate[[spec.entity_column, spec.time_column]])
        covered = candidate_index.isin(available)
        positives = pd.to_numeric(candidate[spec.target_column], errors="raise").eq(1)
        output[name] = {
            "panel_endpoints": int(len(candidate)),
            "sequence_endpoints": int(covered.sum()),
            "endpoints_without_full_window": int((~covered).sum()),
            "positive_panel_endpoints": int(positives.sum()),
            "positive_endpoints_without_full_window": int((positives & ~covered).sum()),
        }
    return output


def run(panel_path: Path, out_dir: Path, *, window: int = 12, epochs: int = 10) -> dict:
    frame = pd.read_csv(panel_path, parse_dates=["date"])
    spec = edgar_dataset_spec(CORE_FEATURES)
    if frame.duplicated([spec.entity_column, spec.time_column]).any():
        raise ValueError("Input panel has duplicate firm-month rows")

    # Fit missing-value handling and scaling only on rows available by the
    # training cutoff. The embargo years can supply later sequence history,
    # but never contribute to the fitted preprocessing statistics or labels.
    raw_features = frame[list(CORE_FEATURES)].replace([np.inf, -np.inf], np.nan)
    train_rows = frame[spec.time_column] <= pd.Timestamp(TRAIN_END)
    if not train_rows.any():
        raise ValueError("No panel observations before the training cutoff")
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_imputed = imputer.fit_transform(raw_features.loc[train_rows])
    scaler = StandardScaler()
    scaler.fit(train_imputed)
    all_imputed = imputer.transform(raw_features)
    all_scaled = scaler.transform(all_imputed).astype(np.float32)
    transformed_columns = list(imputer.get_feature_names_out(CORE_FEATURES))
    sequence_frame = frame[[spec.entity_column, spec.time_column, spec.target_column]].copy()
    for index, column in enumerate(transformed_columns):
        sequence_frame[column] = all_scaled[:, index]
    sequence_spec = edgar_dataset_spec(transformed_columns)

    sequences = build_sequences(sequence_frame, sequence_spec, window=window)
    split = purged_temporal_split(
        sequences,
        spec.time_column,
        train_end=TRAIN_END,
        validation_start=VALIDATION_START,
        validation_end=VALIDATION_END,
        test_start=TEST_START,
        horizon_months=LABEL_HORIZON_MONTHS,
    )
    model = LSTMClassifier(
        hidden_size=32,
        dropout=0.25,
        learning_rate=1e-3,
        epochs=epochs,
        batch_size=256,
        random_state=42,
        device="cpu",
        num_layers=2,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        patience=4,
    ).fit(
        split.train.X,
        split.train.y,
        validation_data=(split.validation.X, split.validation.y),
    )
    validation_probabilities = model.predict_proba(split.validation.X)
    test_probabilities = model.predict_proba(split.test.X)
    results = {
        "dataset": str(panel_path.resolve()),
        "features": list(CORE_FEATURES),
        "sequence_features_after_missing_indicators": transformed_columns,
        "sequence_window_months": window,
        "label_horizon_months": LABEL_HORIZON_MONTHS,
        "split": {
            "train_end": TRAIN_END,
            "validation_start": VALIDATION_START,
            "validation_end": VALIDATION_END,
            "test_start": TEST_START,
            "purge_1": "2019-01-01 through 2019-12-31",
            "purge_2": "2021-01-01 through 2021-12-31",
            "firms_can_overlap_across_time_partitions": True,
        },
        "model": {
            "architecture": "two-layer unidirectional LSTM, hidden_size=32, inter-layer/output dropout=0.25",
            "epochs": epochs,
            "epochs_run": len(model.training_history),
            "early_stopping": "validation average precision; best checkpoint restored",
            "best_epoch": model.best_epoch,
            "best_validation_average_precision": model.best_validation_average_precision,
            "validation_average_precision_by_epoch": model.validation_average_precision,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "random_state": 42,
            "training_scale_pos_weight": model.scale_pos_weight,
            "training_loss_by_epoch": model.training_history,
        },
        "preprocessing": {
            "imputer": "median with missingness indicators",
            "scaler": "StandardScaler",
            "fit_rows": f"observations dated on/before {TRAIN_END}",
        },
        "partitions": {
            "train": _partition_counts(split.train),
            "validation": _partition_counts(split.validation),
            "test": _partition_counts(split.test),
        },
        "sequence_coverage": _sequence_coverage(frame, sequences, spec),
        "metrics": {
            "validation": _clean_metrics(split.validation.y, validation_probabilities),
            "test": _clean_metrics(split.test.y, test_probabilities),
        },
        "warning": (
            "One fixed pipeline run only. Event counts are small; the test contains only "
            f"{_partition_counts(split.test)['positive_firms']} positive firms. Do not interpret metrics as model performance evidence."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "firm": split.test.metadata[spec.entity_column].to_numpy(),
            "date": split.test.metadata[spec.time_column].to_numpy(),
            "y": split.test.y.to_numpy(),
            "probability": test_probabilities,
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)
    readme = f"""# EDGAR LSTM smoke run

One fixed CPU run to verify sequence construction, preprocessing, training, and prediction. Validation average precision selects the checkpoint/epoch; no broad hyperparameter search is performed, and this is not a final evaluation.

- Each sample is a {window}-month sequence from one firm; sequences cannot cross firms or missing calendar months.
- Only endpoints with a complete {window}-month continuous history are retained. See `metrics.json` `sequence_coverage` for how many rows and positive endpoints this excludes in each partition.
- The endpoint month supplies the target. All input features are point-in-time features at or before that endpoint.
- Imputation and scaling are fitted on raw observations dated on/before {TRAIN_END} only.
- Training ends in {TRAIN_END}; 2019 is purged; validation is calendar year 2020; 2021 is purged; test begins {TEST_START}.
- Validation has {_partition_counts(split.validation)['positive_firms']} positive firms; test has {_partition_counts(split.test)['positive_firms']} positive firms.
- Firms may recur across time partitions. This is not an unseen-firm evaluation.

See `metrics.json` and `test_predictions.csv`. Metrics are for pipeline verification only.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(
        "LSTM smoke run: "
        f"validation AP={results['metrics']['validation']['average_precision']:.4f}, "
        f"test AP={results['metrics']['test']['average_precision']:.4f}"
    )
    print(f"Results written to {out_dir.resolve()}")
    print(results["warning"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("artifacts/edgar_bankruptcy_v1/bankruptcy_panel.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/lstm_smoke_run"))
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run(args.panel, args.out_dir, window=args.window, epochs=args.epochs)


if __name__ == "__main__":
    main()
