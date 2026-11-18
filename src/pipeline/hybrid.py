"""Hybrid structural + ML models.

H1  feature augmentation  -- no special code: use the ``hybrid`` / ``hybrid_lag``
                             feature sets (structural outputs as extra columns).
H2  structural prior      -- ``structural_prior`` fits the structural-only logit
                             on training rows and returns its log-odds for every
                             row; pass it as ``offset`` to ``fit_predict`` with the
                             ``xgboost_offset`` spec (or to the LSTM). The ML model
                             then learns a correction to the Merton-type PD; with
                             little data it stays close to the structural model.
H3  stacking              -- ``stack`` fits a logistic meta-learner on the
                             validation-year predictions of several base models
                             and applies it to their test predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .calibration import logit
from .data import ENTITY, TARGET, TIME, feature_columns
from .metrics import evaluate
from .preprocess import Preprocessor
from .runner import FitResult
from .splits import Split


def structural_prior(frame: pd.DataFrame, split: Split, columns=None) -> np.ndarray:
    """Log-odds of the structural-only logit for every row of ``frame`` (fit on train rows).

    The logit is *unweighted*, so its log-odds are on the population scale and can
    serve directly as a base margin. Rows with missing DD inputs are handled by the
    preprocessor (train-median imputation + missing indicators), so every row gets
    a finite prior.
    """
    columns = list(columns or feature_columns("structural"))
    prep = Preprocessor.preset(columns, "linear").fit(frame.loc[split.train])
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(prep.transform(frame.loc[split.train]), frame.loc[split.train, TARGET])
    return model.decision_function(prep.transform(frame)).astype("float64")


def stack(results: dict[str, FitResult], *, C: float = 1.0, name: str = "stack") -> FitResult:
    """H3: logistic meta-learner on base-model log-odds, fitted on the validation year.

    All base results must come from the same split and score the same rows in
    the same order (true for every model run through ``fit_predict`` /
    ``fit_predict_sequence`` on one split).
    """
    names = list(results)
    first = results[names[0]]
    for other in names[1:]:
        for part in ("validation", "test"):
            a = getattr(first, part)[[ENTITY, TIME]].reset_index(drop=True)
            b = getattr(results[other], part)[[ENTITY, TIME]].reset_index(drop=True)
            if not a.equals(b):
                raise ValueError(f"{other} scored different {part} rows than {names[0]}; cannot stack")

    def design(part):
        return np.column_stack([logit(getattr(results[n], part)["p_raw"]) for n in names])

    y_val = first.validation[TARGET].to_numpy()
    meta = LogisticRegression(C=C, max_iter=2000).fit(design("validation"), y_val)
    outputs = {}
    for part in ("validation", "test"):
        sub = getattr(first, part).drop(columns=["p_raw", "p"]).copy()
        sub["p_raw"] = sub["p"] = meta.predict_proba(design(part))[:, 1]
        outputs[part] = sub
    metrics = {
        "test": evaluate(outputs["test"][TARGET], outputs["test"]["p"]),
        # in-sample for the meta-learner: reported for completeness only
        "validation_in_sample": evaluate(outputs["validation"][TARGET], outputs["validation"]["p"]),
    }
    return FitResult(
        model_name=name, feature_set=names, split_name=first.split_name,
        params={"C": C, "weights": dict(zip(names, meta.coef_[0].round(4))),
                "intercept": float(meta.intercept_[0])},
        validation=outputs["validation"], test=outputs["test"], metrics=metrics,
    )
