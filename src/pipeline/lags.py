"""History ("lag") features for the XGB-lag model, built from each firm's past filings only.

For every base column x and filing t of a firm (filings ordered by date):

    x__d1      x_t - x_{t-1}               change since the previous filing
    x__d4      x_t - x_{t-4}               change over ~one year (four quarterly filings)
    x__mean8   mean of the last 8 filings   (incl. t)
    x__min8    min  of the last 8 filings
    x__max8    max  of the last 8 filings
    x__std8    std  of the last 8 filings
    x__slope8  OLS slope of x on filing number over the last 8 filings

plus ``losses_last4`` = number of loss-making filings among the last four.

All windows end at the current filing, so no future information is used. The
midterm plan motivates this: the tree baselines of Pellegrino et al. (2024)
saw only raw levels, so part of the LSTM's advantage may have come from weak
inputs. XGB-lag gives the trees the same history the LSTM sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENTITY = "firm"

LAG_BASE_COLUMNS = [
    "log_assets_usd", "liabilities_to_assets", "equity_to_assets", "current_ratio",
    "cash_to_assets", "working_capital_to_assets", "roa", "ebit_to_assets",
    "cfo_to_assets", "retained_earnings_to_assets", "sales_to_assets", "log_public_float",
]
LAG_STRUCTURAL_COLUMNS = ["acct_dd", "pf_naive_dd"]
WINDOW = 8
SUFFIXES = ("d1", "d4", f"mean{WINDOW}", f"min{WINDOW}", f"max{WINDOW}", f"std{WINDOW}", f"slope{WINDOW}")


def lag_names(columns) -> list[str]:
    return [f"{c}__{s}" for c in columns for s in SUFFIXES]


def _rolling_sum(values: pd.Series, firm: pd.Series, window: int) -> pd.Series:
    """Sum of the last ``window`` rows within each firm (rows already firm/date sorted)."""
    cs = values.groupby(firm, sort=False).cumsum()
    lagged = cs.groupby(firm, sort=False).shift(window).fillna(0.0)
    return cs - lagged


def add_lag_features(frame: pd.DataFrame, columns=None, *, window: int = WINDOW) -> pd.DataFrame:
    """Return a copy of ``frame`` with lag features appended. ``frame`` must be sorted by firm, date."""
    columns = list(columns or LAG_BASE_COLUMNS + LAG_STRUCTURAL_COLUMNS)
    if not frame[ENTITY].is_monotonic_increasing:
        raise ValueError("frame must be sorted by firm, date before adding lag features")
    out = frame.copy()
    firm = out[ENTITY]
    group = out.groupby(ENTITY, sort=False)
    position = group.cumcount().astype("float64")  # filing number within firm
    new = {}
    for c in columns:
        x = out[c].astype("float64")
        g = x.groupby(firm, sort=False)
        new[f"{c}__d1"] = x - g.shift(1)
        new[f"{c}__d4"] = x - g.shift(4)

        valid = x.notna().astype("float64")
        xv = x.fillna(0.0)
        tv = position * valid
        n = _rolling_sum(valid, firm, window)
        sx = _rolling_sum(xv, firm, window)
        sxx = _rolling_sum(xv * xv, firm, window)
        st = _rolling_sum(tv, firm, window)
        stt = _rolling_sum(tv * position, firm, window)
        stx = _rolling_sum(tv * xv, firm, window)
        with np.errstate(all="ignore"):
            mean = sx / n
            var = (sxx - sx * sx / n) / (n - 1)
            denom = stt - st * st / n
            slope = (stx - st * sx / n) / denom
        new[f"{c}__mean{window}"] = mean.where(n >= 1)
        new[f"{c}__std{window}"] = np.sqrt(var.clip(lower=0)).where(n >= 2)
        new[f"{c}__slope{window}"] = slope.where((n >= 3) & (denom.abs() > 1e-12))

        rolled = g.rolling(window, min_periods=1)
        new[f"{c}__min{window}"] = rolled.min().reset_index(level=0, drop=True)
        new[f"{c}__max{window}"] = rolled.max().reset_index(level=0, drop=True)

    if "loss_flag" in out:
        losses = out["loss_flag"].fillna(0.0)
        new["losses_last4"] = _rolling_sum(losses, firm, 4)
    lagged = pd.DataFrame(new, index=out.index).replace([np.inf, -np.inf], np.nan)
    return pd.concat([out, lagged], axis=1)
