"""Small matplotlib helpers used by the implementation notebook. Every function
draws one figure, saves it to ``path`` (PNG, 150 dpi) and returns the figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .data import TARGET, TIME
from .metrics import calibration_table


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def roc_pr_curves(results: dict, path, title: str = ""):
    """ROC and precision-recall curves of several FitResults on their test sets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, r in results.items():
        y, p = r.test[TARGET], r.test["p"]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, label=f"{name} ({r.metrics['test']['auroc']:.3f})")
        ax2.plot(rec, prec, label=f"{name} ({r.metrics['test']['average_precision']:.3f})")
    ax1.plot([0, 1], [0, 1], "k:", lw=1)
    first = next(iter(results.values()))
    ax2.axhline(first.test[TARGET].mean(), color="k", ls=":", lw=1, label="base rate")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (AUROC)")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (AP)", ylim=(0, None))
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.suptitle(title)
    return _save(fig, path)


def calibration_plot(results: dict, path, bins: int = 10):
    """Mean predicted PD vs realised bankruptcy rate in deciles of predicted PD (log axes)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = 1.0
    for name, r in results.items():
        t = calibration_table(r.test[TARGET], r.test["p"], bins=bins)
        t = t[(t.observed_rate > 0) & (t.mean_predicted > 0)]
        lo = min(lo, t.mean_predicted.min(), t.observed_rate.min())
        ax.plot(t.mean_predicted, t.observed_rate, "o-", label=name)
    ax.plot([lo, 1], [lo, 1], "k:", lw=1, label="perfect calibration")
    ax.set(xscale="log", yscale="log", xlabel="Mean predicted PD (decile)", ylabel="Realised rate",
           title="Calibration on the test set")
    ax.legend(fontsize=8)
    return _save(fig, path)


def delta_forest(deltas: pd.DataFrame, path, metric: str = "auroc"):
    """Point estimate and 95% firm-bootstrap CI for each 'A - B' comparison."""
    d = deltas[deltas["metric"] == metric].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.2))
    y = np.arange(len(d))
    colors = ["tab:green" if lo > 0 else "tab:red" if hi < 0 else "tab:gray" for lo, hi in zip(d["low"], d["high"])]
    for yi, (delta, low, high, color) in enumerate(zip(d["delta"], d["low"], d["high"], colors)):
        ax.errorbar(delta, yi, xerr=[[delta - low], [high - delta]], fmt="o", color=color, capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y, [f"{a}  vs  {b}" for a, b in zip(d["a"], d["b"])], fontsize=8)
    ax.set(xlabel=f"Difference in test {metric} (95% firm bootstrap CI)",
           title="Head-to-head comparisons (green = A significantly better)")
    return _save(fig, path)


def line_plot(table: pd.DataFrame, x: str, y: str, hue: str, path, *, title: str = "",
              ylabel: str | None = None, band: tuple[str, str] | None = None, logx: bool = False):
    """One line per ``hue`` value; optional shaded band from two columns (low, high)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in table.groupby(hue, sort=False):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], "o-", label=name)
        if band:
            ax.fill_between(g[x], g[band[0]], g[band[1]], alpha=0.15)
    if logx:
        ax.set_xscale("log")
    ax.set(xlabel=x, ylabel=ylabel or y, title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def grouped_bars(table: pd.DataFrame, x: str, y: str, hue: str, path, *, title: str = "",
                 ylabel: str | None = None, ylim=None):
    pivot = table.pivot_table(index=x, columns=hue, values=y, sort=False)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(pivot) * max(1, pivot.shape[1] / 2)), 5))
    pivot.plot.bar(ax=ax, rot=30, width=0.8)
    ax.set(xlabel="", ylabel=ylabel or y, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def horizontal_bars(table: pd.DataFrame, label: str, value: str, path, *, title: str = "", xlabel: str | None = None,
                    error: str | None = None):
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(t) + 1.2))
    ax.barh(t[label], t[value], xerr=t[error] if error else None, color="tab:blue")
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel=xlabel or value, title=title)
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, path)


def event_time_plot(frame: pd.DataFrame, columns: dict, path, *, max_months: int = 36,
                    title: str = "Average value before bankruptcy"):
    """Median of each column vs months until the bankruptcy event (bankrupt firms only).

    ``columns`` maps column name -> label. Shows how early each signal moves.
    """
    sub = frame[frame["bankruptcy_date"].notna()].copy()
    sub["months_to_event"] = ((sub["bankruptcy_date"] - sub[TIME]).dt.days / 30.44).round()
    sub = sub[sub["months_to_event"].between(0, max_months)]
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4), squeeze=False)
    for ax, (col, label) in zip(axes[0], columns.items()):
        g = sub.groupby("months_to_event")[col].median()
        ax.plot(g.index, g.values, "o-", ms=3)
        ax.invert_xaxis()
        ax.set(xlabel="Months until bankruptcy", ylabel=f"median {label}", title=label)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    return _save(fig, path)


def class_distributions(frame: pd.DataFrame, columns: dict, path, *, clip=(0.01, 0.99)):
    """Histograms of each column for y=1 vs y=0 rows (clipped to the 1st-99th percentile)."""
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4), squeeze=False)
    for ax, (col, label) in zip(axes[0], columns.items()):
        x = frame[col].dropna()
        lo, hi = x.quantile(list(clip))
        bins = np.linspace(lo, hi, 50)
        for value, name in ((0, "no bankruptcy within 12m"), (1, "bankruptcy within 12m")):
            v = frame.loc[frame[TARGET].eq(value), col].clip(lo, hi).dropna()
            ax.hist(v, bins=bins, density=True, alpha=0.5, label=name)
        ax.set(title=label, xlabel=label, ylabel="density")
        ax.legend(fontsize=8)
    return _save(fig, path)
