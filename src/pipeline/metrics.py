"""Evaluation metrics, uncertainty and backtesting.

* ``evaluate``            -- AUROC, average precision (PR-AUC), lift over the base
                             rate, Brier, log loss, top-decile capture, ECE.
* ``firm_bootstrap``      -- percentile CIs resampling *firms* (filings of one firm
                             are correlated, so resampling rows would understate
                             uncertainty).
* ``paired_bootstrap_delta`` -- CI for the difference between two models scored
                             on the same rows (e.g. hybrid - ML): the key RQ1/RQ2 test.
* ``delong_test``         -- DeLong et al. (1988) test for two correlated AUROCs
                             (treats rows as independent; report alongside the
                             firm bootstrap).
* ``calibration_table``   -- reliability curve data (predicted vs observed by bin).
* ``backtest_buckets``    -- realised bankruptcy rate by PD bucket with a binomial
                             traffic light, as in Moscatelli et al. (2020, Table 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom, norm, rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

METRICS = {
    "auroc": roc_auc_score,
    "average_precision": average_precision_score,
    "brier": brier_score_loss,
}


def top_fraction_capture(y, p, fraction: float = 0.10) -> float:
    """Share of all positives found among the riskiest ``fraction`` of rows."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return np.nan
    k = max(1, int(np.ceil(fraction * len(y))))
    top = np.argsort(-np.asarray(p, dtype="float64"), kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def expected_calibration_error(y, p, bins: int = 10) -> float:
    table = calibration_table(y, p, bins=bins)
    return float((table["rows"] * (table["mean_predicted"] - table["observed_rate"]).abs()).sum() / table["rows"].sum())


def evaluate(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype="float64"), 1e-7, 1 - 1e-7)
    prevalence = y.mean()
    out = {"rows": len(y), "positives": int(y.sum()), "prevalence": float(prevalence)}
    if 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
        out["ap_lift"] = out["average_precision"] / prevalence
    else:
        out["auroc"] = out["average_precision"] = out["ap_lift"] = np.nan
    out["brier"] = float(brier_score_loss(y, p))
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    out["top_decile_capture"] = top_fraction_capture(y, p, 0.10)
    out["ece"] = expected_calibration_error(y, p)
    return out


def _firm_index(firm) -> list[np.ndarray]:
    codes, _ = pd.factorize(np.asarray(firm))
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, bounds)


def firm_bootstrap(y, p, firm, *, metrics=("auroc", "average_precision", "brier"),
                   replicates: int = 500, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Percentile CIs from resampling firms with replacement."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    groups = _firm_index(firm)
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in metrics}
    for _ in range(replicates):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[rows].min() == y[rows].max():
            continue
        for m in metrics:
            draws[m].append(METRICS[m](y[rows], p[rows]))
    a = (1 - level) / 2
    return pd.DataFrame({
        m: {"estimate": METRICS[m](y, p), "low": np.quantile(v, a), "high": np.quantile(v, 1 - a),
            "valid_replicates": len(v)}
        for m, v in draws.items()
    }).T


def paired_bootstrap_delta(y, p_a, p_b, firm, *, metrics=("auroc", "average_precision"),
                           replicates: int = 500, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """CI for metric(p_a) - metric(p_b) on the same rows, resampling firms.

    ``share_le_0`` is the fraction of replicates where A is not better than B
    (a one-sided bootstrap p-value for 'A beats B').
    """
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, dtype="float64")
    p_b = np.asarray(p_b, dtype="float64")
    groups = _firm_index(firm)
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in metrics}
    for _ in range(replicates):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[rows].min() == y[rows].max():
            continue
        for m in metrics:
            draws[m].append(METRICS[m](y[rows], p_a[rows]) - METRICS[m](y[rows], p_b[rows]))
    a = (1 - level) / 2
    out = {}
    for m, v in draws.items():
        v = np.asarray(v)
        out[m] = {"delta": METRICS[m](y, p_a) - METRICS[m](y, p_b), "low": np.quantile(v, a),
                  "high": np.quantile(v, 1 - a), "share_le_0": float((v <= 0).mean()),
                  "valid_replicates": len(v)}
    return pd.DataFrame(out).T


def delong_test(y, p_a, p_b) -> dict:
    """Two-sided DeLong test for AUROC(p_a) == AUROC(p_b) (fast version, Sun & Xu 2014)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")  # positives first
    y = y[order]
    scores = np.vstack([np.asarray(p_a, float)[order], np.asarray(p_b, float)[order]])
    m = int(y.sum())
    n = len(y) - m
    tx = np.vstack([rankdata(s[:m]) for s in scores])
    ty = np.vstack([rankdata(s[m:]) for s in scores])
    tz = np.vstack([rankdata(s) for s in scores])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auroc_a": float(aucs[0]), "auroc_b": float(aucs[1]), "delta": float(aucs[0] - aucs[1]),
            "z": float(z), "p_value": float(2 * norm.sf(abs(z)))}


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Mean predicted PD vs observed rate within quantile bins of predicted PD."""
    frame = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype="float64")})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    return frame.groupby("bin").agg(rows=("y", "size"), mean_predicted=("p", "mean"),
                                    observed_rate=("y", "mean"))


# Credit quality steps used by Moscatelli et al. (2020): upper PD of each bucket
CQS_THRESHOLDS = {"CQS1-2": 0.001, "CQS3": 0.004, "CQS4": 0.01, "CQS5": 0.015,
                  "CQS6": 0.03, "CQS7": 0.05, "CQS8+": 1.0}


def backtest_buckets(y, p, thresholds: dict = CQS_THRESHOLDS) -> pd.DataFrame:
    """Realised rate per PD bucket and a binomial traffic light.

    p-value = P(X >= observed | n, upper PD of bucket): red < 0.01, yellow < 0.10,
    else green. Rows of one firm are not independent, so treat colours as indicative.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    edges = [0.0] + list(thresholds.values())
    rows = []
    for (name, upper), lower in zip(thresholds.items(), edges[:-1]):
        mask = (p > lower) & (p <= upper) if lower > 0 else (p <= upper)
        n, k = int(mask.sum()), int(y[mask].sum())
        p_value = float(binom.sf(k - 1, n, upper)) if n else np.nan
        light = "n/a" if not n else ("red" if p_value < 0.01 else "yellow" if p_value < 0.10 else "green")
        rows.append({"bucket": name, "upper_pd": upper, "rows": n, "positives": k,
                     "realised_rate": k / n if n else np.nan,
                     "mean_predicted": float(p[mask].mean()) if n else np.nan,
                     "p_value": p_value, "light": light})
    return pd.DataFrame(rows).set_index("bucket")
