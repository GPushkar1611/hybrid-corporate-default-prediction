from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelDataset:
    """Features, binary target, and identifiers for one modelling sample."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame

    def subset(self, mask) -> "ModelDataset":
        return ModelDataset(
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y.loc[mask].reset_index(drop=True),
            metadata=self.metadata.loc[mask].reset_index(drop=True),
        )


@dataclass
class TemporalSplit:
    train: ModelDataset
    validation: ModelDataset
    test: ModelDataset
