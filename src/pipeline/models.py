"""Model zoo: one ``ModelSpec`` per model family, all run through ``runner.fit_predict``.

    structural_logit  sigma(a + b'DD) on the structural columns only (Merton as a scoring model)
    logit             L2 logistic regression, the statistical benchmark (Moscatelli's LOG/PLR)
    random_forest     Breiman RF
    xgboost           gradient-boosted trees. "XGB-static" = run on a snapshot feature set
                      (ml, hybrid); "XGB-lag" = the same model on a *_lag feature set.
    xgboost_offset    XGBoost started from the structural prior (hybrid H2): the
                      structural log-odds are passed as base_margin, so the trees
                      learn only a correction to the Merton-type PD.

The LSTM lives in ``sequence.py`` (it needs 3-D inputs) and stacking (H3) in
``hybrid.py``.

Search spaces are deliberately the same size for every family (equal tuning
budget via ``runner.tune``).
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .runner import ModelSpec


def _logistic(params: dict) -> LogisticRegression:
    """L2-penalised logistic regression with the fast lbfgs solver."""
    return LogisticRegression(**{"C": 1.0, "max_iter": 2000, **params})


def _xgb(params: dict):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xgboost is required: pip install xgboost") from exc
    return XGBClassifier(**params)


XGB_DEFAULTS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "n_jobs": -1,
    "random_state": 42,
}
XGB_SPACE = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 20.0],
}


def structural_logit() -> ModelSpec:
    """Structural-only baseline: logistic map from DD (+ D/V) to PD, as in Bharath-Shumway."""
    return ModelSpec(
        name="structural_logit",
        make=_logistic,
        preprocessing="linear",
        imbalance="weights",
        search_space={"C": [0.01, 0.1, 1.0, 10.0]},
    )


def logit() -> ModelSpec:
    """Statistical benchmark: L2-penalised logistic regression (Moscatelli's LOG/PLR).

    An elastic net needs the 'saga' solver, which is extremely slow with the large
    class weights used here; Moscatelli et al. (2020, Table 2) find plain and
    penalised logit perform identically, so the L2 version is used.
    """
    return ModelSpec(
        name="logit",
        make=_logistic,
        preprocessing="linear",
        imbalance="weights",
        params={"C": 0.1},
        search_space={"C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
    )


def random_forest() -> ModelSpec:
    return ModelSpec(
        name="random_forest",
        make=lambda p: RandomForestClassifier(**p),
        preprocessing="forest",
        imbalance="weights",
        params={"n_estimators": 500, "min_samples_leaf": 20, "max_features": "sqrt",
                "max_depth": None, "n_jobs": -1, "random_state": 42},
        search_space={"max_depth": [None, 8, 12, 16], "min_samples_leaf": [5, 20, 50, 100],
                      "max_features": ["sqrt", 0.3, 0.5]},
    )


def xgboost() -> ModelSpec:
    """XGB-static on snapshot feature sets; XGB-lag when given a *_lag feature set."""
    return ModelSpec(
        name="xgboost",
        make=_xgb,
        preprocessing="boosting",
        imbalance="weights",          # balanced sample weights == scale_pos_weight
        params=dict(XGB_DEFAULTS),
        search_space=XGB_SPACE,
    )


def xgboost_offset() -> ModelSpec:
    """Hybrid H2. No class weights: the base margin is an unweighted log-odds prior,
    and re-weighting would shift it. Calibration on validation happens afterwards."""
    return ModelSpec(
        name="xgboost_offset",
        make=_xgb,
        preprocessing="boosting",
        imbalance="none",
        params=dict(XGB_DEFAULTS),
        search_space=XGB_SPACE,
        supports_offset=True,
    )


MODELS = {
    "structural_logit": structural_logit,
    "logit": logit,
    "random_forest": random_forest,
    "xgboost": xgboost,
    "xgboost_offset": xgboost_offset,
}
