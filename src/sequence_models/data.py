"""Firm-isolated rolling sequence construction for panel data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.tree_models.config import DatasetSpec


@dataclass
class SequenceDataset:
    """Fixed-length feature sequences and the label/date at each endpoint."""

    X: np.ndarray
    y: pd.Series
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            raise ValueError("X must have shape (samples, time steps, features)")
        if len(self.X) != len(self.y) or len(self.X) != len(self.metadata):
            raise ValueError("X, y, and metadata must have the same number of samples")

    def subset(self, mask) -> "SequenceDataset":
        selected = np.asarray(mask, dtype=bool)
        if selected.ndim != 1 or len(selected) != len(self.y):
            raise ValueError("mask must be one-dimensional and match the sequence count")
        return SequenceDataset(
            X=self.X[selected],
            y=self.y.iloc[selected].reset_index(drop=True),
            metadata=self.metadata.iloc[selected].reset_index(drop=True),
        )


def build_sequences(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    *,
    window: int,
    require_consecutive_months: bool = True,
) -> SequenceDataset:
    """Build trailing monthly sequences without mixing firms or future rows.

    Each sequence ends at one firm-month. Its target and prediction date come
    from that endpoint. If continuity is required, every adjacent observation
    must be one calendar month apart; weekends and variable month-end dates do
    not affect the check.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    required = {
        spec.entity_column,
        spec.time_column,
        spec.target_column,
        *spec.feature_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot build sequences from an empty panel")
    if frame[[spec.entity_column, spec.time_column]].isna().any().any():
        raise ValueError("Entity and time values cannot be missing")
    if frame.duplicated([spec.entity_column, spec.time_column]).any():
        raise ValueError("Duplicate entity-time observations found")

    ordered = frame.copy()
    ordered[spec.time_column] = pd.to_datetime(ordered[spec.time_column], errors="raise")
    labels = pd.to_numeric(ordered[spec.target_column], errors="raise")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Sequence target must be complete and binary (0/1)")
    ordered["_month_period"] = ordered[spec.time_column].dt.to_period("M")
    feature_columns = list(spec.feature_columns)
    samples: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict] = []

    for entity, group in ordered.sort_values([spec.entity_column, spec.time_column]).groupby(
        spec.entity_column, sort=False
    ):
        group = group.reset_index(drop=True)
        values = group[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        periods = group["_month_period"].astype("int64").to_numpy()
        group_labels = pd.to_numeric(group[spec.target_column], errors="raise").to_numpy()
        for end in range(window - 1, len(group)):
            start = end - window + 1
            if require_consecutive_months and not (np.diff(periods[start : end + 1]) == 1).all():
                continue
            samples.append(values[start : end + 1])
            targets.append(int(group_labels[end] == spec.positive_label))
            metadata.append(
                {spec.entity_column: entity, spec.time_column: group.loc[end, spec.time_column]}
            )

    if not samples:
        raise ValueError("No eligible sequences; check window length and monthly continuity")
    return SequenceDataset(
        X=np.stack(samples).astype(np.float32, copy=False),
        y=pd.Series(targets, name=spec.target_column, dtype="int64"),
        metadata=pd.DataFrame(metadata),
    )
