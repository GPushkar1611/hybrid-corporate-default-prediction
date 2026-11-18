from abc import ABC, abstractmethod
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import ModelDataset


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class TreeModel(ABC):
    """Common fit/predict/save interface for tabular tree classifiers."""

    def __init__(self, *, categorical_columns=(), params=None):
        self.categorical_columns = tuple(categorical_columns)
        self.params = dict(params or {})
        self.pipeline: Pipeline | None = None

    @abstractmethod
    def _estimator(self):
        """Return an unfitted sklearn-compatible binary classifier."""

    def fit(self, dataset: ModelDataset):
        missing = set(self.categorical_columns).difference(dataset.X.columns)
        if missing:
            raise ValueError(f"Configured categorical columns are absent: {sorted(missing)}")
        numeric_columns = [column for column in dataset.X.columns if column not in self.categorical_columns]
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", SimpleImputer(strategy="median"), numeric_columns),
                ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("one_hot", _one_hot_encoder())]), list(self.categorical_columns)),
            ],
            remainder="drop",
        )
        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", self._estimator())])
        # XBRL/structural ratios can contain +/-inf after division by values
        # near zero. Treat those as missing so train-only imputers can handle
        # them consistently at fit and prediction time.
        self.pipeline.fit(_replace_infinities(dataset.X), dataset.y)
        return self

    def predict_proba(self, dataset: ModelDataset):
        if self.pipeline is None:
            raise RuntimeError("Fit the model before prediction.")
        return self.pipeline.predict_proba(_replace_infinities(dataset.X))[:, 1]

    def save(self, path: str | Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("Fit the model before saving.")
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path):
        return joblib.load(path)


def _replace_infinities(features):
    return features.replace([float("inf"), float("-inf")], float("nan"))
