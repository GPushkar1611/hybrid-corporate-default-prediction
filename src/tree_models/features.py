import pandas as pd

from .config import DatasetSpec
from .data import ModelDataset


def _binary_target(frame: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    return (frame[spec.target_column] == spec.positive_label).astype(int)


def build_snapshot_features(frame: pd.DataFrame, spec: DatasetSpec) -> ModelDataset:
    """Create one feature row per firm-date observation."""
    ordered = frame.sort_values([spec.time_column, spec.entity_column]).reset_index(drop=True)
    return ModelDataset(
        X=ordered[list(spec.feature_columns)].copy(),
        y=_binary_target(ordered, spec),
        metadata=ordered[[spec.entity_column, spec.time_column]].copy(),
    )


def build_window_features(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    window: int,
    *,
    require_consecutive_periods: bool = True,
) -> ModelDataset:
    """Flatten each firm's trailing ``window`` observations into one row.

    The final observation supplies the target and prediction date. Lag 0 is
    the current row, lag 1 the immediately preceding row, and so on.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if window == 1:
        return build_snapshot_features(frame, spec)

    records: list[dict] = []
    targets: list[int] = []
    metadata: list[dict] = []
    columns = list(spec.feature_columns)
    ordered = frame.sort_values([spec.entity_column, spec.time_column])
    for entity, group in ordered.groupby(spec.entity_column, sort=False):
        group = group.reset_index(drop=True)
        for end in range(window - 1, len(group)):
            chunk = group.iloc[end - window + 1 : end + 1]
            if require_consecutive_periods:
                periods = pd.to_numeric(chunk[spec.time_column], errors="coerce")
                if periods.isna().any() or not periods.diff().iloc[1:].eq(1).all():
                    continue
            row = {}
            # Current values use lag 0; previous values use lag 1, 2, ...
            for lag, (_, observation) in enumerate(reversed(list(chunk.iterrows()))):
                for column in columns:
                    row[f"{column}_lag{lag}"] = observation[column]
            final = chunk.iloc[-1]
            records.append(row)
            targets.append(int(final[spec.target_column] == spec.positive_label))
            metadata.append({spec.entity_column: entity, spec.time_column: final[spec.time_column]})

    if not records:
        raise ValueError("No eligible windows were created. Check window length and time continuity.")
    return ModelDataset(pd.DataFrame(records), pd.Series(targets, name=spec.target_column), pd.DataFrame(metadata))
