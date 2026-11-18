from .data import ModelDataset, TemporalSplit
import pandas as pd


def temporal_split(dataset: ModelDataset, time_column: str, *, train_end, validation_end) -> TemporalSplit:
    """Split a prepared dataset into chronological train/validation/test sets."""
    time = dataset.metadata[time_column]
    train = dataset.subset(time <= train_end)
    validation = dataset.subset((time > train_end) & (time <= validation_end))
    test = dataset.subset(time > validation_end)
    if min(len(train.X), len(validation.X), len(test.X)) == 0:
        raise ValueError("Temporal split produced an empty train, validation, or test partition.")
    return TemporalSplit(train=train, validation=validation, test=test)


def purged_temporal_split(
    dataset: ModelDataset,
    time_column: str,
    *,
    train_end,
    validation_start,
    validation_end,
    test_start,
    horizon_months: int,
) -> TemporalSplit:
    """Chronological split with explicit gaps long enough to settle labels.

    A training label at ``train_end`` can use outcomes through
    ``train_end + horizon_months``. Validation must start after that date, and
    test must start after the validation label horizon. Pass boundaries as
    explicit dates so the intended embargo is visible and can be audited.
    """
    if horizon_months < 1:
        raise ValueError("horizon_months must be positive")
    train_end = pd.Timestamp(train_end)
    validation_start = pd.Timestamp(validation_start)
    validation_end = pd.Timestamp(validation_end)
    test_start = pd.Timestamp(test_start)
    if not train_end < validation_start <= validation_end < test_start:
        raise ValueError("Split dates must satisfy train_end < validation_start <= validation_end < test_start")
    if train_end + pd.DateOffset(months=horizon_months) >= validation_start:
        raise ValueError("Training label horizon overlaps validation; increase the first purge gap")
    if validation_end + pd.DateOffset(months=horizon_months) >= test_start:
        raise ValueError("Validation label horizon overlaps test; increase the second purge gap")

    time = pd.to_datetime(dataset.metadata[time_column])
    train = dataset.subset(time <= train_end)
    validation = dataset.subset((time >= validation_start) & (time <= validation_end))
    test = dataset.subset(time >= test_start)
    if min(len(train.y), len(validation.y), len(test.y)) == 0:
        raise ValueError("Purged temporal split produced an empty partition.")
    return TemporalSplit(train=train, validation=validation, test=test)
