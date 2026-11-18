"""Train-only preprocessing for tabular (and, later, sequence) models.

Every statistic -- clipping bounds, medians, means, standard deviations -- is
fitted on training rows only and then frozen, so validation/test data never
influence the transformation.

Recommended settings per model family:
    logit / elastic net / LSTM : clip + impute (+ missing indicators) + scale
    random forest              : impute (+ missing indicators), no clip/scale needed
    XGBoost                    : passthrough (handles NaN natively; trees are scale-free)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRESETS = {
    "linear": dict(clip=(0.01, 0.99), impute=True, indicators=True, scale=True),
    "neural": dict(clip=(0.01, 0.99), impute=True, indicators=True, scale=True),
    "forest": dict(clip=None, impute=True, indicators=True, scale=False),
    "boosting": dict(clip=None, impute=False, indicators=False, scale=False),
}


class Preprocessor:
    def __init__(self, columns, *, clip=(0.01, 0.99), impute=True, indicators=True, scale=True):
        self.columns = list(columns)
        self.clip = clip
        self.impute = impute
        self.indicators = indicators
        self.scale = scale
        self.fitted = False

    @classmethod
    def preset(cls, columns, kind: str) -> "Preprocessor":
        return cls(columns, **PRESETS[kind])

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].to_numpy(dtype="float64", copy=True)
        values[~np.isfinite(values)] = np.nan
        return values

    def fit(self, frame: pd.DataFrame) -> "Preprocessor":
        x = self._raw(frame)
        if self.clip is not None:
            self.lower_, self.upper_ = np.nanquantile(x, list(self.clip), axis=0)
            x = np.clip(x, self.lower_, self.upper_)
        self.median_ = np.nanmedian(x, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)  # all-missing column
        # indicators only for columns that are actually missing in training
        self.indicator_columns_ = np.flatnonzero(np.isnan(x).any(axis=0)) if self.indicators else np.array([], int)
        if self.scale:
            filled = np.where(np.isnan(x), self.median_, x)
            self.mean_ = filled.mean(axis=0)
            std = filled.std(axis=0)
            self.std_ = np.where(std > 0, std, 1.0)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit the preprocessor on training rows first")
        x = self._raw(frame)
        missing = np.isnan(x)
        if self.clip is not None:
            x = np.clip(x, self.lower_, self.upper_)  # NaN stays NaN
        if self.impute:
            x = np.where(missing, self.median_, x)
        if self.scale:
            x = (x - self.mean_) / self.std_
        if len(self.indicator_columns_):
            x = np.hstack([x, missing[:, self.indicator_columns_].astype("float64")])
        return x.astype("float32")

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    @property
    def feature_names_(self) -> list[str]:
        names = list(self.columns)
        names += [f"{self.columns[i]}__missing" for i in self.indicator_columns_]
        return names
