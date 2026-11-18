"""Train any tabular model on a split, calibrate on validation, score test.

A model is described by a ``ModelSpec``: a factory that turns a parameter dict
into an unfitted estimator with sklearn's ``fit`` / ``predict_proba``
(sklearn models and xgboost.XGBClassifier both qualify), the preprocessing
preset it needs, and how it handles class imbalance. ``fit_predict`` then runs
the same leak-free recipe for every model:

    1. preprocessing fitted on training rows only
    2. imbalance handling (class weights or downsampling + Bayes correction)
    3. fit (optionally with an offset / base margin -- hybrid H2)
    4. raw scores on validation and test
    5. calibrator fitted on validation, applied to validation and test

``tune`` gives every model the same budget: random search over its search
space, scored by mean average precision over time-ordered CV folds inside the
training period.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .calibration import Calibrator, balanced_weights, downsample, prior_correct
from .data import ENTITY, TARGET, TIME
from .metrics import METRICS, evaluate
from .preprocess import Preprocessor
from .splits import Split, time_cv_folds

META_COLUMNS = [ENTITY, TIME, TARGET, "sic2", "sic_division", "log_assets_usd"]


@dataclass
class ModelSpec:
    name: str
    make: Callable[[dict], Any]
    preprocessing: str                      # key of preprocess.PRESETS
    imbalance: str = "weights"              # 'weights' | 'downsample' | 'none'
    params: dict = field(default_factory=dict)
    search_space: dict = field(default_factory=dict)
    supports_offset: bool = False           # can take a base margin (XGBoost)


@dataclass
class FitResult:
    model_name: str
    feature_set: list[str]
    split_name: str
    params: dict
    validation: pd.DataFrame                # META_COLUMNS + p_raw + p
    test: pd.DataFrame
    metrics: dict
    model: Any = None
    preprocessor: Preprocessor | None = None
    calibrator: Calibrator | None = None
    seconds: float = 0.0

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics).T


def _fit_model(spec: ModelSpec, params: dict, X, y, *, seed: int, offset=None,
               negatives_per_positive: float = 1.0):
    """Fit with the spec's imbalance strategy. Returns (model, beta or None)."""
    model = spec.make({**spec.params, **params})
    fit_kwargs = {}
    beta = None
    rows = np.arange(len(y))
    if spec.imbalance == "downsample":
        rows, beta = downsample(y, negatives_per_positive=negatives_per_positive, seed=seed)
    elif spec.imbalance == "weights":
        fit_kwargs["sample_weight"] = balanced_weights(y)
    if offset is not None:
        if not spec.supports_offset:
            raise ValueError(f"{spec.name} does not support an offset/base margin")
        fit_kwargs["base_margin"] = offset[rows]
    if "sample_weight" in fit_kwargs:
        fit_kwargs["sample_weight"] = fit_kwargs["sample_weight"][rows]
    model.fit(X[rows], np.asarray(y)[rows], **fit_kwargs)
    return model, beta


def _predict(spec: ModelSpec, model, X, beta, offset=None) -> np.ndarray:
    kwargs = {"base_margin": offset} if offset is not None else {}
    p = model.predict_proba(X, **kwargs)[:, 1]
    return prior_correct(p, beta) if beta is not None else p


def fit_predict(
    spec: ModelSpec,
    frame: pd.DataFrame,
    features: list[str],
    split: Split,
    *,
    params: dict | None = None,
    calibration: str = "platt",
    seed: int = 0,
    offset: np.ndarray | None = None,
    negatives_per_positive: float = 1.0,
    keep_model: bool = False,
) -> FitResult:
    """Train on split.train, calibrate on split.validation, score split.test.

    ``offset`` (full-length array aligned with ``frame``) is a log-odds prior
    passed as XGBoost's base_margin -- the trees then learn a correction to it.
    """
    start = time.time()
    params = dict(params or {})
    prep = Preprocessor.preset(features, spec.preprocessing).fit(frame.loc[split.train])
    X_train = prep.transform(frame.loc[split.train])
    y_train = frame.loc[split.train, TARGET].to_numpy()
    off = (lambda mask: None) if offset is None else (lambda mask: np.asarray(offset, float)[mask])
    model, beta = _fit_model(spec, params, X_train, y_train, seed=seed, offset=off(split.train),
                             negatives_per_positive=negatives_per_positive)

    outputs = {}
    for part in ("validation", "test"):
        mask = getattr(split, part)
        sub = frame.loc[mask, [c for c in META_COLUMNS if c in frame.columns]].reset_index(drop=True)
        sub["p_raw"] = _predict(spec, model, prep.transform(frame.loc[mask]), beta, off(mask))
        outputs[part] = sub
    calibrator = Calibrator(calibration).fit(outputs["validation"]["p_raw"], outputs["validation"][TARGET])
    for part in outputs.values():
        part["p"] = calibrator.transform(part["p_raw"])

    metrics = {}
    for part, sub in outputs.items():
        metrics[part] = evaluate(sub[TARGET], sub["p"])
        metrics[f"{part}_uncalibrated"] = evaluate(sub[TARGET], sub["p_raw"])
    return FitResult(
        model_name=spec.name, feature_set=list(features), split_name=split.name,
        params={**spec.params, **params}, validation=outputs["validation"], test=outputs["test"],
        metrics=metrics, model=model if keep_model else None,
        preprocessor=prep if keep_model else None, calibrator=calibrator,
        seconds=time.time() - start,
    )


def results_table(results: dict[str, FitResult], part: str = "test",
                  columns=("auroc", "average_precision", "ap_lift", "brier", "top_decile_capture", "ece")) -> pd.DataFrame:
    """One row per experiment with the chosen metrics of one partition."""
    rows = {}
    for name, r in results.items():
        m = r.metrics.get(part) or r.metrics.get("test")
        rows[name] = {"model": r.model_name, **{c: m[c] for c in columns},
                      "rows": m["rows"], "positives": m["positives"]}
    return pd.DataFrame(rows).T


def permutation_importance(result: FitResult, frame: pd.DataFrame, split: Split, *,
                           n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """AUROC drop on the test set when one feature is randomly permuted
    (Moscatelli et al. 2020, Table 9). Needs ``fit_predict(..., keep_model=True)``
    and a model without an offset."""
    from sklearn.metrics import roc_auc_score

    if result.model is None or result.preprocessor is None:
        raise ValueError("refit with keep_model=True")
    rng = np.random.default_rng(seed)
    test = frame.loc[split.test, result.feature_set].reset_index(drop=True)
    y = frame.loc[split.test, TARGET].to_numpy()
    X = result.preprocessor.transform(test)
    base = roc_auc_score(y, result.model.predict_proba(X)[:, 1])
    rows = []
    for column in result.feature_set:
        drops = []
        for _ in range(n_repeats):
            shuffled = test.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            p = result.model.predict_proba(result.preprocessor.transform(shuffled))[:, 1]
            drops.append(base - roc_auc_score(y, p))
        rows.append({"feature": column, "auroc_drop": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("auroc_drop", ascending=False).reset_index(drop=True)


def cv_score(spec: ModelSpec, frame: pd.DataFrame, features: list[str], params: dict,
             folds, *, metric: str = "average_precision", seed: int = 0, offset=None) -> float:
    """Mean validation metric over time-ordered folds (preprocessing refit per fold)."""
    scores = []
    for fit_mask, score_mask in folds:
        prep = Preprocessor.preset(features, spec.preprocessing).fit(frame.loc[fit_mask])
        y_fit = frame.loc[fit_mask, TARGET].to_numpy()
        off_fit = None if offset is None else np.asarray(offset, float)[fit_mask]
        off_score = None if offset is None else np.asarray(offset, float)[score_mask]
        model, beta = _fit_model(spec, params, prep.transform(frame.loc[fit_mask]), y_fit,
                                 seed=seed, offset=off_fit)
        p = _predict(spec, model, prep.transform(frame.loc[score_mask]), beta, off_score)
        scores.append(METRICS[metric](frame.loc[score_mask, TARGET], p))
    return float(np.mean(scores))


def tune(spec: ModelSpec, frame: pd.DataFrame, features: list[str], split: Split, *,
         n_trials: int = 12, seed: int = 0, metric: str = "average_precision",
         validation_years=(2015, 2016, 2017, 2018), offset=None, verbose: bool = True):
    """Random search with a fixed budget; trial 0 is always the spec's default params.

    Uses only rows of ``split.train`` (time-ordered folds), never validation/test.
    Returns (best_params, trials DataFrame).
    """
    rng = np.random.default_rng(seed)
    folds = time_cv_folds(frame, within=split.train, validation_years=validation_years)
    candidates = [{}]
    keys = list(spec.search_space)
    seen = set()
    while len(candidates) < n_trials and keys:
        trial = {k: spec.search_space[k][rng.integers(len(spec.search_space[k]))] for k in keys}
        key = tuple(sorted((k, repr(v)) for k, v in trial.items()))
        if key in seen:
            if len(seen) >= np.prod([len(v) for v in spec.search_space.values()]):
                break
            continue
        seen.add(key)
        candidates.append(trial)
    rows = []
    for i, trial in enumerate(candidates):
        t0 = time.time()
        score = cv_score(spec, frame, features, trial, folds, metric=metric, seed=seed, offset=offset)
        rows.append({"trial": i, **trial, f"cv_{metric}": score, "seconds": time.time() - t0})
        if verbose:
            print(f"  [{spec.name}] trial {i + 1}/{len(candidates)}  cv {metric}={score:.4f}  {trial}", flush=True)
    trials = pd.DataFrame(rows).sort_values(f"cv_{metric}", ascending=False)
    best = dict(candidates[int(trials.iloc[0]["trial"])])
    return best, trials
