"""Run a leak-aware tree/LSTM evaluation on the SEC as-filed panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler

from build_companyfacts_panel import FEATURE_COLUMNS
from src.edgar_panel import (
    LABEL_HORIZON_MONTHS,
    TEST_START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    edgar_dataset_spec,
)
from src.sequence_models import LSTMClassifier, build_sequences
from src.tree_models import (
    RandomForestModel,
    XGBoostModel,
    build_snapshot_features,
    evaluate_binary,
    purged_temporal_split,
)


def _metric_dict(y, probability) -> dict:
    result = evaluate_binary(y, probability)
    return {key: value.tolist() if isinstance(value, np.ndarray) else float(value) for key, value in result.items()}


def _counts(dataset) -> dict:
    positive = dataset.y.eq(1)
    return {
        "rows": int(len(dataset.y)),
        "firms": int(dataset.metadata["firm"].nunique()),
        "positive_rows": int(positive.sum()),
        "positive_firms": int(dataset.metadata.loc[positive, "firm"].nunique()),
        "positive_rate": float(positive.mean()),
    }


def _firm_cluster_bootstrap(y, probability, firm, *, replicates: int = 500, seed: int = 42) -> dict:
    """Percentile intervals resampling firms, not correlated filing rows."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    firm = np.asarray(firm)
    unique = pd.unique(firm)
    by_firm = {key: np.flatnonzero(firm == key) for key in unique}
    rng = np.random.default_rng(seed)
    scores = {key: [] for key in ("average_precision", "auroc", "brier_score")}
    valid = 0
    for _ in range(replicates):
        sampled_firms = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_firm[key] for key in sampled_firms])
        target = y[rows]
        if len(np.unique(target)) < 2:
            continue
        valid += 1
        scores["average_precision"].append(average_precision_score(target, probability[rows]))
        scores["auroc"].append(roc_auc_score(target, probability[rows]))
        scores["brier_score"].append(brier_score_loss(target, probability[rows]))
    return {
        "unit": "firm cluster",
        "requested_replicates": replicates,
        "valid_replicates": valid,
        "interval_level": 0.95,
        "percentile_intervals": {
            key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else [None, None]
            for key, values in scores.items()
        },
    }


def _prepare_sequences(frame: pd.DataFrame, feature_columns, spec, *, window: int):
    raw = frame[list(feature_columns)].replace([np.inf, -np.inf], np.nan)
    train_mask = frame[spec.time_column].le(pd.Timestamp(TRAIN_END))
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    train_values = imputer.fit_transform(raw.loc[train_mask])
    all_values = imputer.transform(raw)
    # Ratio features are heavy-tailed. Fit clipping bounds and a robust scale
    # from training observations only, then apply those frozen transforms.
    lower, upper = np.quantile(train_values, [0.01, 0.99], axis=0)
    train_clipped = np.clip(train_values, lower, upper)
    all_clipped = np.clip(all_values, lower, upper)
    scaler = RobustScaler().fit(train_clipped)
    scaled = scaler.transform(all_clipped).astype(np.float32)
    transformed_columns = list(imputer.get_feature_names_out(feature_columns))
    sequence_frame = frame[[spec.entity_column, spec.time_column, spec.target_column]].copy()
    for index, column in enumerate(transformed_columns):
        sequence_frame[column] = scaled[:, index]
    sequence_spec = edgar_dataset_spec(transformed_columns)
    sequences = build_sequences(
        sequence_frame,
        sequence_spec,
        window=window,
        require_consecutive_months=False,
    )
    return sequences, {
        "imputer": "training-only median with missingness indicators",
        "scaler": "training-only RobustScaler",
        "training_only_clipping_quantiles": [0.01, 0.99],
        "fit_rows": int(train_mask.sum()),
        "sequence_window_filing_observations": window,
        "consecutive_calendar_periods_required": False,
    }


def run(
    panel_path: Path,
    out_dir: Path,
    *,
    window: int = 8,
    epochs: int = 30,
    lstm_seeds: tuple[int, ...] = (42, 43, 44),
    bootstrap_replicates: int = 500,
) -> dict:
    frame = pd.read_csv(panel_path, parse_dates=["date"])
    if frame.duplicated(["firm", "date"]).any():
        raise ValueError("SEC filing panel has duplicate firm-date rows")
    spec = edgar_dataset_spec(FEATURE_COLUMNS)
    missing_features = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing_features:
        raise ValueError(f"Panel is missing model features: {missing_features}")
    tabular = build_snapshot_features(frame, spec)
    split = purged_temporal_split(
        tabular,
        spec.time_column,
        train_end=TRAIN_END,
        validation_start=VALIDATION_START,
        validation_end=VALIDATION_END,
        test_start=TEST_START,
        horizon_months=LABEL_HORIZON_MONTHS,
    )
    train_positive = int(split.train.y.sum())
    if train_positive == 0:
        raise ValueError("Training set contains no positive filing observations")
    pos_weight = (len(split.train.y) - train_positive) / train_positive
    results = {
        "dataset": str(panel_path.resolve()),
        "features": list(FEATURE_COLUMNS),
        "split": {
            "train_end": TRAIN_END,
            "validation_start": VALIDATION_START,
            "validation_end": VALIDATION_END,
            "test_start": TEST_START,
            "label_horizon_months": LABEL_HORIZON_MONTHS,
            "purge_years": [2019, 2021],
            "firms_can_overlap_across_time": True,
        },
        "partitions": {
            "train": _counts(split.train),
            "validation": _counts(split.validation),
            "test": _counts(split.test),
        },
        "models": {},
        "warnings": [],
    }
    predictions = []
    models = {
        "random_forest": RandomForestModel(
            params={"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 5, "random_state": 42}
        ),
        "xgboost": XGBoostModel(
            params={
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.03,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": pos_weight,
                "tree_method": "hist",
                "random_state": 42,
            }
        ),
    }
    for name, model in models.items():
        model.fit(split.train)
        validation_probability = model.predict_proba(split.validation)
        test_probability = model.predict_proba(split.test)
        results["models"][name] = {
            "validation": _metric_dict(split.validation.y, validation_probability),
            "test": _metric_dict(split.test.y, test_probability),
            "test_firm_cluster_bootstrap": _firm_cluster_bootstrap(
                split.test.y, test_probability, split.test.metadata["firm"],
                replicates=bootstrap_replicates,
            ),
            "parameters": model.params,
        }
        predictions.append(pd.DataFrame({
            "model": name,
            "firm": split.test.metadata["firm"].to_numpy(),
            "date": split.test.metadata["date"].to_numpy(),
            "y": split.test.y.to_numpy(),
            "probability": test_probability,
        }))

    sequences, sequence_preprocessing = _prepare_sequences(
        frame, FEATURE_COLUMNS, spec, window=window
    )
    sequence_split = purged_temporal_split(
        sequences,
        spec.time_column,
        train_end=TRAIN_END,
        validation_start=VALIDATION_START,
        validation_end=VALIDATION_END,
        test_start=TEST_START,
        horizon_months=LABEL_HORIZON_MONTHS,
    )
    seed_results = []
    validation_ensemble = []
    test_ensemble = []
    for seed in lstm_seeds:
        model = LSTMClassifier(
            hidden_size=64,
            dropout=0.25,
            learning_rate=5e-4,
            epochs=epochs,
            batch_size=512,
            random_state=seed,
            device="cpu",
            num_layers=2,
            weight_decay=1e-3,
            gradient_clip_norm=1.0,
            patience=5,
            min_delta=1e-5,
        ).fit(
            sequence_split.train.X,
            sequence_split.train.y,
            validation_data=(sequence_split.validation.X, sequence_split.validation.y),
        )
        val_probability = model.predict_proba(sequence_split.validation.X)
        test_probability = model.predict_proba(sequence_split.test.X)
        validation_ensemble.append(val_probability)
        test_ensemble.append(test_probability)
        seed_results.append({
            "seed": seed,
            "best_epoch": model.best_epoch,
            "best_validation_average_precision": model.best_validation_average_precision,
            "epochs_run": len(model.training_history),
            "training_loss": model.training_history,
            "validation_average_precision_by_epoch": model.validation_average_precision,
            "scale_pos_weight": model.scale_pos_weight,
        })
    val_lstm = np.mean(validation_ensemble, axis=0)
    test_lstm = np.mean(test_ensemble, axis=0)
    results["sequence_partitions"] = {
        "train": _counts(sequence_split.train),
        "validation": _counts(sequence_split.validation),
        "test": _counts(sequence_split.test),
    }
    results["sequence_preprocessing"] = sequence_preprocessing
    results["models"]["lstm_ensemble"] = {
        "validation": _metric_dict(sequence_split.validation.y, val_lstm),
        "test": _metric_dict(sequence_split.test.y, test_lstm),
        "test_firm_cluster_bootstrap": _firm_cluster_bootstrap(
            sequence_split.test.y, test_lstm, sequence_split.test.metadata["firm"],
            replicates=bootstrap_replicates,
        ),
        "architecture": "two-layer unidirectional LSTM, hidden_size=64, dropout=0.25",
        "epochs_max": epochs,
        "batch_size": 512,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
        "early_stopping": "best validation average precision per seed; restore best checkpoint",
        "seed_runs": seed_results,
    }
    predictions.append(pd.DataFrame({
        "model": "lstm_ensemble",
        "firm": sequence_split.test.metadata["firm"].to_numpy(),
        "date": sequence_split.test.metadata["date"].to_numpy(),
        "y": sequence_split.test.y.to_numpy(),
        "probability": test_lstm,
    }))
    results["warnings"].append(
        "Before final claims, inspect a targeted sample of CompanyFacts tag mappings against original filings; custom and dimensional concepts are omitted."
    )
    if results["partitions"]["test"]["positive_firms"] < 30:
        results["warnings"].append(
            "Fewer than 30 positive firms in the test period; firm-cluster uncertainty will remain wide and metrics are exploratory."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(out_dir / "test_predictions.csv", index=False)
    comparison_rows = []
    for model_name, model_results in results["models"].items():
        for partition in ("validation", "test"):
            metric = model_results[partition]
            count_block = results["sequence_partitions" if model_name == "lstm_ensemble" else "partitions"][partition]
            comparison_rows.append({
                "model": model_name,
                "partition": partition,
                "rows": count_block["rows"],
                "firms": count_block["firms"],
                "positive_rows": count_block["positive_rows"],
                "positive_firms": count_block["positive_firms"],
                "auroc": metric["auroc"],
                "average_precision": metric["average_precision"],
                "brier_score": metric["brier_score"],
                "top_decile_default_capture": metric["top_decile_default_capture"],
            })
    pd.DataFrame(comparison_rows).to_csv(out_dir / "comparison.csv", index=False)
    (out_dir / "README.md").write_text(
        "# SEC as-filed OOT evaluation\n\n"
        "Tree and filing-sequence LSTM models use the same point-in-time SEC filing panel, "
        "feature definitions, and purged temporal boundaries. Their row counts differ because "
        "the LSTM requires eight prior filing observations. Use `audit_matched_sequence_support.py` "
        "to compare all models on the same LSTM-eligible test endpoints. The LSTM uses eight prior "
        "filing observations, not calendar months, and averages three seeded models. "
        "Preprocessing and outlier limits are fitted on training dates only.\n\n"
        "Metrics include filing-row results and firm-cluster bootstrap intervals to reflect "
        "within-company dependence. The test covers the mature label horizon only. Class-weighted "
        "probabilities are not calibrated; rank by AP/AUROC until calibration is fit on validation. "
        "Inspect a targeted sample of CompanyFacts tag mappings against original filings before "
        "final claims.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "partitions": results["partitions"],
        "sequences": results["sequence_partitions"],
        "model_test_ap": {name: block["test"]["average_precision"] for name, block in results["models"].items()},
        "warnings": results["warnings"],
        "out_dir": str(out_dir.resolve()),
    }, indent=2), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("artifacts/sec_companyfacts_bankruptcy_v1/sec_bankruptcy_panel.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sec_companyfacts_final_eval"))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lstm-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    run(
        args.panel, args.out_dir,
        window=args.window,
        epochs=args.epochs,
        lstm_seeds=tuple(args.lstm_seeds),
        bootstrap_replicates=args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
