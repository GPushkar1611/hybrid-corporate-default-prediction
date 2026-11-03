from .data import ModelDataset, TemporalSplit


def temporal_split(dataset: ModelDataset, time_column: str, *, train_end, validation_end) -> TemporalSplit:
    """Split a prepared dataset into chronological train/validation/test sets."""
    time = dataset.metadata[time_column]
    train = dataset.subset(time <= train_end)
    validation = dataset.subset((time > train_end) & (time <= validation_end))
    test = dataset.subset(time > validation_end)
    if min(len(train.X), len(validation.X), len(test.X)) == 0:
        raise ValueError("Temporal split produced an empty train, validation, or test partition.")
    return TemporalSplit(train=train, validation=validation, test=test)
