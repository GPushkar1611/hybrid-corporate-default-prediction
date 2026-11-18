"""Load the v2 SEC bankruptcy panel and define the feature sets used by every experiment.

One row = one 10-K/10-Q filing. ``date`` is the EDGAR filing date (the moment
the information became public); ``y`` = 1 if the firm's first 8-K Item 1.03
bankruptcy event falls within the 12 months after that date.

Feature sets are built from named groups so that every experiment and ablation
uses exactly the same column lists:

    ml_base        v1 balance-sheet ratios only (reproduces the first evaluation)
    ml             full ML feature set: balance sheet + income/cash flow
                   + cover-page market data + filing metadata
    ml_raw_merton  ml + the raw *inputs* of the Merton formula (default point,
                   asset volatility, asset drift, equity volatility) but no DD.
                   Control for RQ1: if `hybrid` beats this, the gain comes from
                   the structural functional form, not from extra variables.
    hybrid         ml_raw_merton + the structural outputs (DD, D/V_A, ...)   [H1]
    structural     structural outputs only (for the structural-only baseline)

and their history versions for XGB-lag (see ``lags.py``):

    ml_lag, ml_raw_merton_lag   + lags of the core accounting/market ratios
    hybrid_lag                  + lags of the ratios and of the DDs

This gives the 2x2 design of the midterm plan:
                 no structure        with structure
    snapshot     ml                  hybrid
    history      ml_lag / LSTM(ml)   hybrid_lag / LSTM(hybrid)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.structural.sec_structural import default_point_liabilities

from .lags import LAG_BASE_COLUMNS, LAG_STRUCTURAL_COLUMNS, add_lag_features, lag_names

PANEL_V2 = Path("artifacts/sec_companyfacts_bankruptcy_v2/sec_bankruptcy_panel.csv")

ENTITY, TIME, TARGET = "firm", "date", "y"
HORIZON_MONTHS = 12

GROUPS: dict[str, list[str]] = {
    # balance-sheet ratios from v1
    "accounting_base": [
        "log_assets_usd", "liabilities_to_assets", "current_ratio", "cash_to_assets",
        "debt_to_assets", "equity_to_assets", "working_capital_to_assets",
        "cash_to_current_liabilities", "debt_to_equity", "debt_partial",
    ],
    # income statement / cash flow (annualised)
    "accounting_flow": [
        "roa", "ebit_to_assets", "sales_to_assets", "retained_earnings_to_assets",
        "cfo_to_assets", "interest_coverage", "loss_flag",
    ],
    # 10-K cover page: public float as a (stale, annual) market value of equity
    "market_cover_page": ["log_public_float", "public_float_to_liabilities", "public_float_age_days"],
    "filing_metadata": ["days_since_prev_filing", "is_amendment", "is_annual"],
    # raw Merton inputs (no DD formula) -- the RQ1 control
    "merton_inputs": ["log_default_point", "acct_sigma_a", "acct_mu", "pf_sigma_e"],
    # structural outputs (Merton functional form applied to SEC inputs)
    "structural_dd": ["acct_dd", "acct_d_va", "pf_naive_dd", "pf_e_to_d"],
    # RQ3 sensitivity variants of the accounting DD
    "structural_sensitivity": ["acct_dd_mu0", "acct_dd_debt", "acct_d_va_debt"],
    # history features (created by load_panel(lags=True))
    "lags_accounting": lag_names(LAG_BASE_COLUMNS) + ["losses_last4"],
    "lags_structural": lag_names(LAG_STRUCTURAL_COLUMNS),
}
_SNAPSHOT_GROUPS = [g for g in GROUPS if not g.startswith("lags_")]

FEATURE_SETS: dict[str, list[str]] = {
    "ml_base": GROUPS["accounting_base"],
    "ml": GROUPS["accounting_base"] + GROUPS["accounting_flow"]
    + GROUPS["market_cover_page"] + GROUPS["filing_metadata"],
}
FEATURE_SETS["ml_raw_merton"] = FEATURE_SETS["ml"] + GROUPS["merton_inputs"]
FEATURE_SETS["hybrid"] = FEATURE_SETS["ml_raw_merton"] + GROUPS["structural_dd"]
FEATURE_SETS["structural"] = GROUPS["structural_dd"]
FEATURE_SETS["ml_lag"] = FEATURE_SETS["ml"] + GROUPS["lags_accounting"]
FEATURE_SETS["ml_raw_merton_lag"] = FEATURE_SETS["ml_raw_merton"] + GROUPS["lags_accounting"]
FEATURE_SETS["hybrid_lag"] = FEATURE_SETS["hybrid"] + GROUPS["lags_accounting"] + GROUPS["lags_structural"]

# Probability from the structural model used as the H2 prior / structural-only score
STRUCTURAL_PD_COLUMNS = ["acct_pd", "pf_naive_pd"]

# Columns that must never be used as predictors (identifiers, label, outcome
# dates, diagnostics). Checked by ``feature_columns``.
_FORBIDDEN = {"firm", "date", "y", "bankruptcy_date", "adsh", "period", "accepted", "cik"}
_FORBIDDEN_SUFFIXES = ("_status", "_months", "_date", "_date_ffill", "_hist_intervals")


def feature_columns(name_or_groups: str | list[str]) -> list[str]:
    """Columns for a named feature set, or the union of a list of group names."""
    if isinstance(name_or_groups, str):
        columns = list(FEATURE_SETS[name_or_groups])
    else:
        columns = [c for group in name_or_groups for c in GROUPS[group]]
    columns = list(dict.fromkeys(columns))  # de-duplicate, keep order
    bad = [c for c in columns if c in _FORBIDDEN or c.endswith(_FORBIDDEN_SUFFIXES)]
    if bad:
        raise ValueError(f"Identifier/label/diagnostic columns cannot be predictors: {bad}")
    return columns


def load_panel(path: str | Path = PANEL_V2, *, lags: bool = True) -> pd.DataFrame:
    """Read the v2 panel, add derived columns (and lag features), run integrity checks."""
    frame = pd.read_csv(
        path,
        parse_dates=["date", "period", "bankruptcy_date"],
        low_memory=False,
    )
    if frame.duplicated([ENTITY, TIME]).any():
        raise ValueError("Panel has duplicate firm-date rows")
    if not set(frame[TARGET].unique()) <= {0, 1}:
        raise ValueError("Target must be binary 0/1")

    default_point = default_point_liabilities(frame)
    frame["log_default_point"] = np.log(default_point.where(default_point > 0))
    frame["sic2"] = pd.to_numeric(frame["sic2"], errors="coerce")
    frame["year"] = frame[TIME].dt.year

    numeric = [c for g in _SNAPSHOT_GROUPS for c in GROUPS[g]] + STRUCTURAL_PD_COLUMNS
    missing = sorted(set(numeric) - set(frame.columns))
    if missing:
        raise ValueError(f"Panel is missing expected columns: {missing}")
    frame[numeric] = frame[numeric].astype("float64").replace([np.inf, -np.inf], np.nan)
    frame = frame.sort_values([ENTITY, TIME], kind="stable").reset_index(drop=True)
    if lags:
        frame = add_lag_features(frame, LAG_BASE_COLUMNS + LAG_STRUCTURAL_COLUMNS)
    return frame


def relabel(frame: pd.DataFrame, months: int) -> pd.DataFrame:
    """Copy of ``frame`` with y = 1 if bankruptcy falls within ``months`` after the filing.

    Horizons up to 12 months are safe: the panel's mature-label cutoff guarantees a
    full 12-month outcome window for every row.
    """
    if months > HORIZON_MONTHS:
        raise ValueError("horizons beyond 12 months are not fully observed in this panel")
    out = frame.copy()
    event = out["bankruptcy_date"]
    out[TARGET] = (event.notna() & (event > out[TIME])
                   & (event <= out[TIME] + pd.DateOffset(months=months))).astype("int8")
    return out


def describe_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows, firms and bankruptcies per filing year (for the report's data table)."""
    positive = frame[TARGET].eq(1)
    return frame.assign(_pos=positive).groupby("year").agg(
        filings=(TARGET, "size"),
        firms=(ENTITY, "nunique"),
        positive_filings=("_pos", "sum"),
        positive_firms=(ENTITY, lambda s: s[frame.loc[s.index, TARGET].eq(1)].nunique()),
        positive_rate=(TARGET, "mean"),
    )
