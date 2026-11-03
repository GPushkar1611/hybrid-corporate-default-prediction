import warnings

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(y_true, probabilities, *, threshold: float = 0.5) -> dict:
    """Return probability, ranking, and threshold metrics for binary default risk."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p):
        raise ValueError("y_true and probabilities must have equal length.")
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty sample.")
    if not ((p >= 0).all() and (p <= 1).all()):
        raise ValueError("Probabilities must be between 0 and 1.")

    if len(np.unique(y)) < 2:
        warnings.warn("AUROC and average precision are undefined with one class.", UserWarning, stacklevel=2)
        auroc = np.nan
        average_precision = np.nan
    else:
        auroc = roc_auc_score(y, p)
        average_precision = average_precision_score(y, p)
    predicted = (p >= threshold).astype(int)
    top_n = max(1, int(np.ceil(len(p) * 0.10)))
    top_indices = np.argsort(p)[-top_n:]
    defaults = y.sum()
    top_decile_capture = float(y[top_indices].sum() / defaults) if defaults else np.nan
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "top_decile_default_capture": top_decile_capture,
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]),
    }
