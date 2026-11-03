import warnings

import numpy as np
import pandas as pd

from .config import DatasetSpec


def validate_dataset(frame: pd.DataFrame, spec: DatasetSpec, *, fail_on_constant_entity_label: bool = False) -> dict:
    """Validate panel shape and target semantics before creating features.

    A constant target per entity is a warning, not an error by default: some
    datasets are legitimate cross-sectional labels, but it is not enough
    evidence for a year-ahead default target.
    """
    required = {
        spec.entity_column,
        spec.time_column,
        spec.target_column,
        *spec.feature_columns,
        *spec.categorical_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame[[spec.entity_column, spec.time_column]].isna().any().any():
        raise ValueError("Entity and time columns cannot contain missing values.")
    if frame.duplicated([spec.entity_column, spec.time_column]).any():
        raise ValueError("Duplicate entity-time observations found.")

    labels = frame[spec.target_column]
    if labels.isna().any():
        raise ValueError("Target column contains missing values.")
    observed = set(labels.unique())
    if spec.positive_label not in observed:
        raise ValueError(f"positive_label={spec.positive_label!r} is not present in target.")
    if len(observed) != 2:
        raise ValueError(f"Binary classification requires exactly two target labels; found {sorted(observed, key=str)}.")

    numeric = frame[list(spec.feature_columns)].select_dtypes(include=[np.number]).columns
    non_numeric = set(spec.feature_columns).difference(numeric).difference(spec.categorical_columns)
    if non_numeric:
        raise ValueError(f"Non-numeric features must be declared categorical: {sorted(non_numeric)}")

    distinct_labels = frame.groupby(spec.entity_column)[spec.target_column].nunique(dropna=False)
    constant_entities = int((distinct_labels == 1).sum())
    all_constant = constant_entities == len(distinct_labels)
    if all_constant:
        message = (
            "Target is constant within every entity. Verify that it is a valid "
            "time-specific, forward-looking default label before temporal forecasting."
        )
        if fail_on_constant_entity_label:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    return {
        "rows": len(frame),
        "entities": frame[spec.entity_column].nunique(),
        "positive_rate": float((labels == spec.positive_label).mean()),
        "constant_label_entities": constant_entities,
        "missing_feature_values": int(frame[list(spec.feature_columns)].isna().sum().sum()),
    }
