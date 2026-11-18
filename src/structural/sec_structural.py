"""Structural (Merton-type) features computable from SEC filings alone.

No daily stock prices are available for the SEC panel (free sources drop
delisted firms), so the market-based KMV iteration in ``merton.py`` cannot be
run for every firm. This module keeps the Merton functional form and replaces
the unobservable inputs with SEC-reported quantities:

1. Accounting DD  (``acct_*``) -- in the spirit of the KMV private-firm model:
       V_A     = book total assets
       sigma_A = volatility of log book assets across the firm's own past
                 filings, shrunk towards the SIC2 median (training rows only)
       mu      = trailing asset growth (main) or 0 (sensitivity variant)
       D       = default point (see ``default_point_*``)
       DD      = [ln(V_A/D) + (mu - sigma_A^2/2) T] / (sigma_A sqrt(T)),  PD = Phi(-DD)

2. Public-float naive DD (``pf_*``) -- Bharath & Shumway (2008) naive DD with
   the annual 10-K cover-page public float (dei:EntityPublicFloat) as market
   equity E, sigma_E from the history of float-implied price changes, and the
   past-year float-implied return as mu.

Every value at row t uses only filings dated on or before t (point-in-time).
Sector fallbacks are fitted on rows dated on or before ``train_end`` only.
Book-asset volatility is smoother than market asset volatility, so accounting
DD levels are larger than market DD levels; use them for ranking/as features,
and calibrate any PD on validation data before reading it as a probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .merton import dd_from_inputs, naive_dd

try:  # scipy is a project dependency; the fallback only keeps tests light
    from scipy.special import ndtr as _ndtr
except ImportError:  # pragma: no cover
    from math import erfc, sqrt

    _ndtr = np.vectorize(lambda x: 0.5 * erfc(-x / sqrt(2.0)) if np.isfinite(x) else np.nan)

OK, BAD_INPUT = 0, 1
MIN_INTERVAL_YEARS, MAX_INTERVAL_YEARS = 0.15, 1.10


# --------------------------------------------------------------------------
# default points
# --------------------------------------------------------------------------
def default_point_liabilities(frame: pd.DataFrame) -> pd.Series:
    """KMV-style default point from liabilities: CL + 0.5 * (L - CL).

    Total liabilities fall back to assets - equity when the Liabilities tag is
    absent (a common XBRL omission). Covers ~92% of filings vs ~50% for debt.
    """
    liabilities = frame["liabilities"].where(frame["liabilities"].notna(), frame["assets"] - frame["equity"])
    current = frame["current_liabilities"]
    long_term = (liabilities - current).clip(lower=0)
    point = current + 0.5 * long_term
    return point.where(point > 0)


def default_point_debt(frame: pd.DataFrame) -> pd.Series:
    """Bharath-Shumway default point: short-term debt + 0.5 * long-term debt."""
    short, long_ = frame["short_debt"], frame["long_debt"]
    point = short.fillna(0) + 0.5 * long_.fillna(0)
    return point.where((short.notna() | long_.notna()) & (point > 0))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _history_moments(times: np.ndarray, log_values: np.ndarray, max_points: int):
    """Annualised drift/volatility of a log series observed at irregular times.

    Uses the last ``max_points`` observations; intervals outside
    [MIN_INTERVAL_YEARS, MAX_INTERVAL_YEARS] (amendments, long gaps) are skipped.
    Returns (mu, sigma, n_intervals).
    """
    t = times[-max_points:]
    x = log_values[-max_points:]
    dt = np.diff(t)
    dx = np.diff(x)
    keep = (dt >= MIN_INTERVAL_YEARS) & (dt <= MAX_INTERVAL_YEARS) & np.isfinite(dx)
    dt, dx = dt[keep], dx[keep]
    n = len(dt)
    if n == 0:
        return np.nan, np.nan, 0
    mu = dx.sum() / dt.sum()
    if n < 2:
        return mu, np.nan, n
    resid = dx - mu * dt
    sigma = np.sqrt((resid ** 2).sum() / dt.sum() * n / (n - 1))
    return mu, sigma, n


def _sector_prior(values: pd.Series, sic2: pd.Series, eligible: pd.Series) -> tuple[pd.Series, float]:
    """Median of ``values`` by SIC2 over eligible (training, enough history) rows."""
    pool = pd.DataFrame({"v": values, "s": sic2})[eligible & values.notna()]
    by_sector = pool.groupby("s")["v"].median()
    overall = float(pool["v"].median()) if len(pool) else np.nan
    return sic2.map(by_sector).fillna(overall), overall


def _blend(firm_value, n, prior, k: float):
    """Shrink a firm estimate towards its sector prior: weight n / (n + k)."""
    w = n / (n + k)
    firm = np.where(np.isfinite(firm_value), firm_value, prior)
    w = np.where(np.isfinite(firm_value), w, 0.0)
    return w * firm + (1 - w) * prior


# --------------------------------------------------------------------------
# 1. accounting DD
# --------------------------------------------------------------------------
def add_accounting_dd(
    frame: pd.DataFrame,
    *,
    train_end,
    max_points: int = 9,
    min_intervals: int = 4,
    shrink_k: float = 4.0,
    sigma_floor: float = 0.05,
    mu_clip: float = 0.5,
    min_assets: float = 1_000_000,
    T: float = 1.0,
) -> pd.DataFrame:
    """Return accounting-DD columns aligned to ``frame.index``.

    Required columns: firm, date (filing date), period (balance-sheet date),
    sic2, assets, current_liabilities, liabilities, equity, short_debt, long_debt.
    """
    f = frame.sort_values(["firm", "date", "period"], kind="stable")
    years = (f["period"] - pd.Timestamp("2000-01-01")).dt.days.to_numpy() / 365.25
    log_assets = np.log(f["assets"].where(f["assets"] > 0)).to_numpy()
    firms = f["firm"].to_numpy()
    n_rows = len(f)
    mu_hat = np.full(n_rows, np.nan)
    sigma_hat = np.full(n_rows, np.nan)
    n_int = np.zeros(n_rows, dtype=int)

    starts = np.flatnonzero(np.r_[True, firms[1:] != firms[:-1]])
    ends = np.r_[starts[1:], n_rows]
    for a, b in zip(starts, ends):
        hist: dict[float, float] = {}  # period -> latest filed log assets (amendments overwrite)
        for i in range(a, b):
            if np.isfinite(log_assets[i]):
                hist[years[i]] = log_assets[i]
            if len(hist) >= 2:
                t = np.fromiter(sorted(hist), float)
                x = np.array([hist[k] for k in t])
                mu_hat[i], sigma_hat[i], n_int[i] = _history_moments(t, x, max_points)

    out = pd.DataFrame(index=f.index)
    out["acct_hist_intervals"] = n_int
    sigma_hat = pd.Series(sigma_hat, index=f.index)
    eligible = (f["date"] <= pd.Timestamp(train_end)) & (out["acct_hist_intervals"] >= min_intervals)
    prior, _ = _sector_prior(sigma_hat, f["sic2"], eligible)
    sigma_a = np.maximum(_blend(sigma_hat.to_numpy(), n_int, prior.to_numpy(), shrink_k), sigma_floor)
    mu_g = np.clip(np.where(np.isfinite(mu_hat), mu_hat, 0.0), -mu_clip, mu_clip)

    # same rule as the panel ratios: denominators below $1m are unreliable (shells, cents)
    V = f["assets"].where(f["assets"] >= min_assets).to_numpy(float)
    d_liab = default_point_liabilities(f).to_numpy(float)
    d_debt = default_point_debt(f).to_numpy(float)
    with np.errstate(all="ignore"):
        dd_liab_g = dd_from_inputs(V, d_liab, sigma_a, mu_g, T)
        dd_liab_0 = dd_from_inputs(V, d_liab, sigma_a, 0.0, T)
        dd_debt_g = dd_from_inputs(V, d_debt, sigma_a, mu_g, T)
    out["acct_sigma_a"] = sigma_a
    out["acct_mu"] = mu_g
    out["acct_dd"] = dd_liab_g                      # main variant
    out["acct_pd"] = _ndtr(-dd_liab_g)
    out["acct_d_va"] = d_liab / V
    out["acct_dd_mu0"] = dd_liab_0                  # sensitivity: no drift
    out["acct_dd_debt"] = dd_debt_g                 # sensitivity: debt default point
    out["acct_d_va_debt"] = d_debt / V
    for column in ("acct_dd", "acct_pd", "acct_d_va", "acct_dd_mu0", "acct_dd_debt", "acct_d_va_debt"):
        out[column] = out[column].replace([np.inf, -np.inf], np.nan)
    out["acct_status"] = np.where(np.isfinite(out["acct_dd"]), OK, BAD_INPUT).astype("int8")
    return out.reindex(frame.index)


# --------------------------------------------------------------------------
# 2. public-float naive DD
# --------------------------------------------------------------------------
def add_public_float_dd(
    frame: pd.DataFrame,
    *,
    train_end,
    max_points: int = 6,
    min_intervals: int = 2,
    shrink_k: float = 2.0,
    sigma_floor: float = 0.10,
    T: float = 1.0,
) -> pd.DataFrame:
    """Return public-float naive-DD columns aligned to ``frame.index``.

    Required columns: firm, date, sic2, public_float, public_float_date,
    shares_outstanding, plus the default-point inputs. ``public_float`` is
    non-missing only on filings whose cover page reports it (10-Ks); the latest
    reported value is carried forward to later filings of the same firm.
    """
    f = frame.sort_values(["firm", "date", "period"], kind="stable")
    firms = f["firm"].to_numpy()
    pf = f["public_float"].where(f["public_float"] > 0).to_numpy(float)
    pf_time = f["public_float_date"]
    pf_years = ((pf_time - pd.Timestamp("2000-01-01")).dt.days / 365.25).to_numpy(float)
    shares = f["shares_outstanding"].where(f["shares_outstanding"] > 0).to_numpy(float)
    n_rows = len(f)
    E = np.full(n_rows, np.nan)
    E_date = np.full(n_rows, np.nan)
    sig = np.full(n_rows, np.nan)
    ret = np.full(n_rows, np.nan)
    n_int = np.zeros(n_rows, dtype=int)

    starts = np.flatnonzero(np.r_[True, firms[1:] != firms[:-1]])
    ends = np.r_[starts[1:], n_rows]
    for a, b in zip(starts, ends):
        hist: dict[float, float] = {}       # float date -> log implied price
        last_e = last_t = np.nan
        for i in range(a, b):
            if np.isfinite(pf[i]) and np.isfinite(pf_years[i]):
                last_e, last_t = pf[i], pf_years[i]
                if np.isfinite(shares[i]):
                    hist[pf_years[i]] = np.log(pf[i] / shares[i])
            E[i], E_date[i] = last_e, last_t
            if len(hist) >= 2:
                t = np.fromiter(sorted(hist), float)
                x = np.array([hist[k] for k in t])
                # annual observations: allow gaps up to 1.6 years
                dt, dx = np.diff(t[-max_points:]), np.diff(x[-max_points:])
                keep = (dt >= 0.5) & (dt <= 1.6)
                dt, dx = dt[keep], dx[keep]
                n_int[i] = len(dt)
                if len(dt):
                    ret[i] = dx[-1] / dt[-1]
                if len(dt) >= 2:
                    z = dx / np.sqrt(dt)
                    sig[i] = z.std(ddof=1)

    out = pd.DataFrame(index=f.index)
    out["pf_hist_intervals"] = n_int
    sig = pd.Series(sig, index=f.index)
    eligible = (f["date"] <= pd.Timestamp(train_end)) & (out["pf_hist_intervals"] >= min_intervals)
    prior, _ = _sector_prior(sig, f["sic2"], eligible)
    sigma_e = np.maximum(_blend(sig.to_numpy(), n_int, prior.to_numpy(), shrink_k), sigma_floor)
    mu = np.clip(np.where(np.isfinite(ret), ret, 0.0), -2.0, 2.0)
    D = default_point_liabilities(f).to_numpy(float)
    result = naive_dd(E, sigma_e, D, mu, T)
    date_years = ((f["date"] - pd.Timestamp("2000-01-01")).dt.days / 365.25).to_numpy(float)
    out["pf_log_e"] = np.log(E)
    out["pf_age_days"] = (date_years - E_date) * 365.25
    out["pf_sigma_e"] = np.where(np.isfinite(E), sigma_e, np.nan)
    out["pf_e_to_d"] = E / D
    out["pf_naive_dd"] = result["dd"].replace([np.inf, -np.inf], np.nan).to_numpy()
    out["pf_naive_pd"] = _ndtr(-out["pf_naive_dd"].to_numpy())
    out["pf_status"] = np.where(np.isfinite(out["pf_naive_dd"]), OK, BAD_INPUT).astype("int8")
    for column in ("pf_log_e", "pf_e_to_d"):
        out[column] = out[column].replace([np.inf, -np.inf], np.nan)
    return out.reindex(frame.index)
