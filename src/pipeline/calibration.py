"""Rare-event handling and probability calibration.

Two ways to deal with ~1% positives during training:
  * class weights   -- every positive row weighted by n_neg / n_pos;
  * downsampling    -- keep all positives and a random share of negatives, then
                       undo the distortion with the Bayes correction of
                       Moscatelli et al. (2020, Appendix 2):
                           p = beta * p_s / (beta * p_s - p_s + 1),
                       beta = share of negatives kept.
Either way the raw scores are not reliable probabilities, so a calibrator
(Platt or isotonic) is fitted on the validation year and applied to test.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-7


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype="float64")))


def balanced_weights(y) -> np.ndarray:
    """Sample weights giving both classes equal total weight."""
    y = np.asarray(y).astype(int)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return np.where(y == 1, n_neg / max(n_pos, 1), 1.0)


def downsample(y, *, negatives_per_positive: float = 1.0, seed: int = 0) -> tuple[np.ndarray, float]:
    """Indices of a training sample with all positives and k negatives per positive.

    Returns (indices, beta) where beta = share of negatives kept (for prior_correct).
    """
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_keep = min(len(neg), int(round(negatives_per_positive * len(pos))))
    kept = rng.choice(neg, size=n_keep, replace=False)
    return np.sort(np.concatenate([pos, kept])), n_keep / len(neg)


def prior_correct(p_sampled, beta: float) -> np.ndarray:
    """Moscatelli et al. (2020) eq. A2: map downsampled-model PDs back to the population."""
    p = np.asarray(p_sampled, dtype="float64")
    return beta * p / (beta * p - p + 1.0)


class Calibrator:
    """Platt (logistic on the score's logit) or isotonic calibration, fitted on validation."""

    def __init__(self, method: str = "platt"):
        if method not in {"platt", "isotonic", "none"}:
            raise ValueError("method must be 'platt', 'isotonic' or 'none'")
        self.method = method

    def fit(self, p_raw, y) -> "Calibrator":
        y = np.asarray(y).astype(int)
        if self.method == "platt":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000)
            self.model_.fit(logit(p_raw).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
            self.model_.fit(np.asarray(p_raw, dtype="float64"), y)
        return self

    def transform(self, p_raw) -> np.ndarray:
        if self.method == "none":
            return np.asarray(p_raw, dtype="float64")
        if self.method == "platt":
            return self.model_.predict_proba(logit(p_raw).reshape(-1, 1))[:, 1]
        return self.model_.predict(np.asarray(p_raw, dtype="float64"))
