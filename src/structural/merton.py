"""
merton.py  --  Merton structural-credit-risk toolbox (Project 11, Stage 2)
==========================================================================

Implements the four distance-to-default (DD) variants of the mid-term report
(following Bharath & Shumway 2008):

    "kmv"            (i)   iterative KMV-style: back out daily asset values,
                           re-estimate sigma_A, repeat to convergence;
                           mu from the asset-return series
    "simul_mu_r"     (ii)  simultaneous solve of Eqs. (1) and (3), mu = r
    "simul_mu_hist"  (iii) simultaneous solve of Eqs. (1) and (3), mu from the
                           history of implied asset returns
    "naive"          (iv)  no solving: V = E + D, sigma_D = 0.05 + 0.25 sigma_E

Notation (same as the report)
-----------------------------
E        market value of equity (price x shares)            [currency]
D        face value of debt = default point                 [currency]
sigma_E  annualised equity volatility                       [decimal, 0.4 = 40%]
r        continuously-compounded risk-free rate             [decimal]
T        horizon in years (default 1.0)
V_A      asset value, sigma_A asset volatility, mu drift of assets

Eq. (1)  E = V Phi(d1) - D exp(-rT) Phi(d2)
Eq. (3)  sigma_E E = Phi(d1) sigma_A V
Eq. (2)  DD = [ln(V/D) + (mu - sigma_A^2/2) T] / (sigma_A sqrt(T)),  PD = Phi(-DD)

Numerical method (why it is robust)
-----------------------------------
Instead of a 2-D Newton iteration (which can leave the feasible region), the
system is reduced to nested 1-D bisections with PROVEN brackets:

  * inner, for fixed sigma_A:  V in [E, E + D exp(-rT)]
        (call(V) is increasing in V; call(E) < E; call(E + D e^{-rT}) >= E)
  * outer, on sigma_A:         sigma_A in [sigma_E E/(E + D e^{-rT}), sigma_E]
        (because E <= Phi(d1) V  =>  sigma_A <= sigma_E, and
               Phi(d1) <= 1, V <= E + D e^{-rT}  =>  lower bound)

Bisection always converges inside the bracket, so there are no "wild" roots.
All loops are compiled with numba when it is installed (pip install numba);
without numba the module still runs, just much more slowly.

Status codes (column "status" in every result)
----------------------------------------------
0 ok | 1 invalid input (NaN, E<=0, D<=0, sigma<=0) | 2 not converged
3 bracket failure | 4 insufficient price history
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr  # standard normal CDF (vectorised)

try:  # ------------------------------------------------------------------
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pure-python fallback so the code still runs / tests
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(f):
            return f

        return deco


OK, BAD_INPUT, NO_CONVERGE, BAD_BRACKET, SHORT_HISTORY = 0, 1, 2, 3, 4
VARIANTS = ("kmv", "simul_mu_r", "simul_mu_hist", "naive")
SQRT2 = 1.4142135623730951


# =============================================================================
# 1. Low-level compiled kernels (scalar maths)
# =============================================================================
@njit(cache=True)
def _ncdf(x):
    return 0.5 * math.erfc(-x / SQRT2)


@njit(cache=True)
def _call_and_d1(V, D, s, r, T):
    """Black-Scholes call on assets (= model equity) and d1."""
    sq = s * math.sqrt(T)
    d1 = (math.log(V / D) + (r + 0.5 * s * s) * T) / sq
    d2 = d1 - sq
    return V * _ncdf(d1) - D * math.exp(-r * T) * _ncdf(d2), d1


@njit(cache=True)
def _solve_va(E, D, s, r, T, tol, maxit):
    """Eq. (1) for V given sigma_A = s.  Bisection on [E, E + D e^{-rT}]."""
    lo = E
    hi = E + D * math.exp(-r * T)
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        c, _d1 = _call_and_d1(mid, D, s, r, T)
        if c > E:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * E:
            break
    return 0.5 * (lo + hi)


@njit(cache=True)
def _g(s, E, sigE, D, r, T, tol, maxit):
    """Residual of Eq. (3) after eliminating V through Eq. (1)."""
    V = _solve_va(E, D, s, r, T, tol * 0.01, maxit)
    _c, d1 = _call_and_d1(V, D, s, r, T)
    return _ncdf(d1) * s * V - sigE * E, V


@njit(cache=True)
def _solve_simul(E, sigE, D, r, T, tol, maxit):
    """Solve Eqs. (1)+(3) jointly. Returns (V_A, sigma_A, status)."""
    if not (math.isfinite(E) and math.isfinite(sigE) and math.isfinite(D)
            and math.isfinite(r)):
        return np.nan, np.nan, BAD_INPUT
    if E <= 0.0 or D <= 0.0 or sigE <= 0.0 or T <= 0.0:
        return np.nan, np.nan, BAD_INPUT
    disc = D * math.exp(-r * T)
    lo = sigE * E / (E + disc)
    hi = sigE
    g_lo, V_lo = _g(lo, E, sigE, D, r, T, tol, maxit)
    g_hi, V_hi = _g(hi, E, sigE, D, r, T, tol, maxit)
    # tiny slack for floating-point noise at the (analytically exact) ends
    eps = 1e-9 * sigE * E
    if g_lo > eps or g_hi < -eps:
        return V_hi, hi, BAD_BRACKET
    if hi - lo < tol:                       # (almost) debt-free firm
        return V_hi, hi, OK
    V = V_hi
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        gm, V = _g(mid, E, sigE, D, r, T, tol, maxit)
        if gm > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            sA = 0.5 * (lo + hi)
            V = _solve_va(E, D, sA, r, T, tol * 0.01, maxit)
            return V, sA, OK
    sA = 0.5 * (lo + hi)
    V = _solve_va(E, D, sA, r, T, tol * 0.01, maxit)
    return V, sA, NO_CONVERGE


@njit(cache=True)
def _compact(x):
    """Drop NaN / non-positive entries; return (buffer, count)."""
    out = np.empty(x.shape[0])
    m = 0
    for i in range(x.shape[0]):
        v = x[i]
        if math.isfinite(v) and v > 0.0:
            out[m] = v
            m += 1
    return out, m


@njit(cache=True)
def _logret_stats(x, m):
    """mean and sample std (ddof=1) of daily log returns of x[:m]."""
    n = m - 1
    s1 = 0.0
    for i in range(n):
        s1 += math.log(x[i + 1] / x[i])
    mean = s1 / n
    s2 = 0.0
    for i in range(n):
        d = math.log(x[i + 1] / x[i]) - mean
        s2 += d * d
    return mean, math.sqrt(s2 / (n - 1))


# =============================================================================
# 2. Window-level kernels (variants i and iii) -- one firm-date per call
# =============================================================================
@njit(cache=True)
def _kmv_one(Ehist, D, r, T, tol, maxit, ann, min_obs):
    """
    Variant (i): iterative KMV procedure on one window of daily equity values.
    Returns (V_A, sigma_A, mu, sigma_E, status).
    """
    x, m = _compact(Ehist)
    if m < min_obs:
        return np.nan, np.nan, np.nan, np.nan, SHORT_HISTORY
    if not (math.isfinite(D) and math.isfinite(r)) or D <= 0.0:
        return np.nan, np.nan, np.nan, np.nan, BAD_INPUT
    E = x[m - 1]
    mean_e, std_e = _logret_stats(x, m)
    sigE = std_e * math.sqrt(ann)
    if sigE <= 0.0:
        return np.nan, np.nan, np.nan, sigE, BAD_INPUT
    sA = sigE * E / (E + D)                   # B&S starting value
    Vs = np.empty(m)
    status = NO_CONVERGE
    mean_v = 0.0
    for _ in range(maxit):
        for j in range(m):
            Vs[j] = _solve_va(x[j], D, sA, r, T, 1e-10, 200)
        mean_v, std_v = _logret_stats(Vs, m)
        s_new = std_v * math.sqrt(ann)
        done = abs(s_new - sA) < tol
        sA = s_new
        if done:
            status = OK
            break
    # mean log return of V = (mu - sigma^2/2)  ->  arithmetic drift mu:
    mu = mean_v * ann + 0.5 * sA * sA
    V_now = _solve_va(E, D, sA, r, T, 1e-10, 200)
    return V_now, sA, mu, sigE, status


@njit(cache=True)
def _hist_mu_one(Ehist, D, r, T, tol, maxit, ann, min_obs):
    """
    Variant (iii): (V_A, sigma_A) from the simultaneous solve at the last
    date; mu from the daily asset values implied by that sigma_A.
    Returns (V_A, sigma_A, mu, sigma_E, status).
    """
    x, m = _compact(Ehist)
    if m < min_obs:
        return np.nan, np.nan, np.nan, np.nan, SHORT_HISTORY
    E = x[m - 1]
    _mean_e, std_e = _logret_stats(x, m)
    sigE = std_e * math.sqrt(ann)
    V, sA, st = _solve_simul(E, sigE, D, r, T, tol, maxit)
    if st == BAD_INPUT or st == BAD_BRACKET:
        return V, sA, np.nan, sigE, st
    Vs = np.empty(m)
    for j in range(m):
        Vs[j] = _solve_va(x[j], D, sA, r, T, 1e-10, 200)
    mean_v, _std_v = _logret_stats(Vs, m)
    mu = mean_v * ann + 0.5 * sA * sA
    return V, sA, mu, sigE, st


# =============================================================================
# 3. Batch kernels (parallel over firm-dates)
# =============================================================================
@njit(parallel=True, cache=True)
def _simul_batch(E, sigE, D, r, T, tol, maxit):
    n = E.shape[0]
    V = np.empty(n)
    sA = np.empty(n)
    st = np.empty(n, np.int8)
    for i in prange(n):
        v, s, k = _solve_simul(E[i], sigE[i], D[i], r[i], T, tol, maxit)
        V[i] = v
        sA[i] = s
        st[i] = k
    return V, sA, st


@njit(parallel=True, cache=True)
def _kmv_batch(W, D, r, T, tol, maxit, ann, min_obs):
    n = W.shape[0]
    V = np.empty(n)
    sA = np.empty(n)
    mu = np.empty(n)
    sE = np.empty(n)
    st = np.empty(n, np.int8)
    for i in prange(n):
        a, b, c, d, k = _kmv_one(W[i], D[i], r[i], T, tol, maxit, ann, min_obs)
        V[i] = a
        sA[i] = b
        mu[i] = c
        sE[i] = d
        st[i] = k
    return V, sA, mu, sE, st


@njit(parallel=True, cache=True)
def _hist_batch(W, D, r, T, tol, maxit, ann, min_obs):
    n = W.shape[0]
    V = np.empty(n)
    sA = np.empty(n)
    mu = np.empty(n)
    sE = np.empty(n)
    st = np.empty(n, np.int8)
    for i in prange(n):
        a, b, c, d, k = _hist_mu_one(W[i], D[i], r[i], T, tol, maxit, ann, min_obs)
        V[i] = a
        sA[i] = b
        mu[i] = c
        sE[i] = d
        st[i] = k
    return V, sA, mu, sE, st


# =============================================================================
# 4. Public helpers
# =============================================================================
def _arr(x, n: Optional[int] = None) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        a = np.full(n if n is not None else 1, float(a))
    return np.ascontiguousarray(a)


def dd_from_inputs(V, D, sigma_a, mu, T=1.0):
    """Eq. (2):  DD = [ln(V/D) + (mu - sigma_A^2/2) T] / (sigma_A sqrt(T))."""
    V, D, sigma_a, mu = map(np.asarray, (V, D, sigma_a, mu))
    return (np.log(V / D) + (mu - 0.5 * sigma_a ** 2) * T) / (sigma_a * np.sqrt(T))


def default_point(short_term_debt, long_term_debt=None, method: str = "kmv",
                  ltd_weight: float = 0.5, total_liabilities=None):
    """
    Face value D of the Merton debt.
      method="kmv"               D = STD + ltd_weight * LTD   (default, [2])
      method="total_liabilities" D = total liabilities        (alternative)
    NaNs propagate (they are NOT silently set to zero); the solver then flags
    the row with status = 1.
    """
    if method == "kmv":
        return np.asarray(short_term_debt, float) + ltd_weight * np.asarray(
            long_term_debt, float)
    if method == "total_liabilities":
        return np.asarray(total_liabilities, float)
    raise ValueError("method must be 'kmv' or 'total_liabilities'")


def merton_forward(V, D, sigma_a, r, T=1.0):
    """
    Forward direction (useful for testing / the simulator).
    Returns dict with equity E, equity vol sigma_E, delta Phi(d1), d1, d2.
    """
    V, D, s, r = (np.asarray(v, float) for v in (V, D, sigma_a, r))
    sq = s * np.sqrt(T)
    d1 = (np.log(V / D) + (r + 0.5 * s ** 2) * T) / sq
    d2 = d1 - sq
    E = V * ndtr(d1) - D * np.exp(-r * T) * ndtr(d2)
    sigma_e = ndtr(d1) * s * V / E
    return {"E": E, "sigma_E": sigma_e, "delta": ndtr(d1), "d1": d1, "d2": d2}


def _pack(V, sA, mu, status, D, T, sigma_e=None) -> pd.DataFrame:
    with np.errstate(all="ignore"):
        dd = dd_from_inputs(V, D, sA, mu, T)
        out = pd.DataFrame({
            "dd": dd,
            "pd": ndtr(-dd),
            "ln_va": np.log(V),
            "sigma_a": sA,
            "d_va": D / V,
            "va": V,
            "mu": mu,
            "status": np.asarray(status, dtype=np.int8),
        })
    if sigma_e is not None:
        out["sigma_e"] = sigma_e
    return out


# =============================================================================
# 5. Cross-section API: one row per firm-date, you supply E, sigma_E, D, r
# =============================================================================
def solve_simultaneous(E, sigma_E, D, r, T: float = 1.0, mu="r",
                       tol: float = 1e-8, max_iter: int = 200) -> pd.DataFrame:
    """
    Variant (ii) (mu = "r") or a user-supplied drift: solve Eqs. (1)+(3).

    Parameters are scalars or 1-D arrays/Series of the same length.
    mu : "r"  -> drift = risk-free rate (report variant ii)
        array/scalar -> use this drift (e.g. past stock return, or the
                        asset drift from variant iii)
    """
    E = _arr(E)
    n = E.shape[0]
    sigma_E, D, r = _arr(sigma_E, n), _arr(D, n), _arr(r, n)
    V, sA, st = _simul_batch(E, sigma_E, D, r, float(T), float(tol), int(max_iter))
    mu_arr = r if isinstance(mu, str) and mu == "r" else _arr(mu, n)
    return _pack(V, sA, mu_arr, st, D, T, sigma_e=sigma_E)


def naive_dd(E, sigma_E, D, mu, T: float = 1.0) -> pd.DataFrame:
    """
    Variant (iv), Bharath-Shumway naive DD (no solving):
        V = E + D,  sigma_D = 0.05 + 0.25 sigma_E,
        sigma_A = E/(E+D) sigma_E + D/(E+D) sigma_D,
        mu = past-year stock return.
    """
    E = _arr(E)
    n = E.shape[0]
    sigma_E, D, mu = _arr(sigma_E, n), _arr(D, n), _arr(mu, n)
    with np.errstate(all="ignore"):
        V = E + D
        sigma_D = 0.05 + 0.25 * sigma_E
        sA = E / V * sigma_E + D / V * sigma_D
    bad = ~(np.isfinite(V) & np.isfinite(sA) & np.isfinite(mu) & (E > 0) & (D > 0))
    st = np.where(bad, BAD_INPUT, OK)
    return _pack(V, sA, mu, st, D, T, sigma_e=sigma_E)


# =============================================================================
# 6. History API: you supply a window of daily equity values per firm-date
# =============================================================================
def solve_kmv(E_window, D, r, T: float = 1.0, tol: float = 1e-3,
              max_iter: int = 100, trading_days: int = 252,
              min_obs: int = 60) -> pd.DataFrame:
    """
    Variant (i), iterative KMV.  E_window: 2-D array (n_obs, window_len) of
    daily equity values, oldest -> newest (last column = observation date).
    NaNs allowed (dropped). tol is on sigma_A (B&S use 1e-3).
    """
    W = np.ascontiguousarray(np.asarray(E_window, dtype=np.float64))
    n = W.shape[0]
    D, r = _arr(D, n), _arr(r, n)
    V, sA, mu, sE, st = _kmv_batch(W, D, r, float(T), float(tol), int(max_iter),
                                   float(trading_days), int(min_obs))
    return _pack(V, sA, mu, st, D, T, sigma_e=sE)


def solve_simultaneous_hist(E_window, D, r, T: float = 1.0, tol: float = 1e-8,
                            max_iter: int = 200, trading_days: int = 252,
                            min_obs: int = 60) -> pd.DataFrame:
    """
    Variant (iii): simultaneous solve at the last date, drift from the
    implied asset-return history.
    """
    W = np.ascontiguousarray(np.asarray(E_window, dtype=np.float64))
    n = W.shape[0]
    D, r = _arr(D, n), _arr(r, n)
    V, sA, mu, sE, st = _hist_batch(W, D, r, float(T), float(tol), int(max_iter),
                                    float(trading_days), int(min_obs))
    return _pack(V, sA, mu, st, D, T, sigma_e=sE)


def build_windows(daily_equity: pd.DataFrame, firms: Sequence,
                  dates: Sequence, window_days: int = 252) -> np.ndarray:
    """
    Cut trailing windows out of a wide daily table.

    daily_equity : DataFrame, index = sorted DatetimeIndex (trading days),
                   columns = firm ids, values = market cap (same units as D).
    firms, dates : equal-length sequences, one entry per observation.
    Returns array (n_obs, window_days); newest value in the last column.
    The value at `date` is the last trading day on or before `date`
    (so no look-ahead). Missing early history is NaN-padded.
    """
    idx = daily_equity.index
    pos = idx.searchsorted(pd.DatetimeIndex(dates), side="right") - 1
    col = daily_equity.columns.get_indexer(pd.Index(firms))
    arr = daily_equity.to_numpy(dtype=np.float64)
    rows = pos[:, None] + np.arange(-window_days + 1, 1)[None, :]
    ok = (rows >= 0) & (pos[:, None] >= 0) & (col[:, None] >= 0)
    out = arr[np.clip(rows, 0, len(idx) - 1), np.clip(col, 0, None)[:, None]]
    return np.where(ok, out, np.nan)


# =============================================================================
# 7. One-call driver: panel in, structural features out
# =============================================================================
def compute_structural_features(obs: pd.DataFrame, daily_equity: pd.DataFrame,
                                variants: Iterable[str] = VARIANTS,
                                T: float = 1.0, window_days: int = 252,
                                min_obs: int = 60, tol_simul: float = 1e-8,
                                tol_kmv: float = 1e-3, chunk: int = 20000,
                                firm_col: str = "firm", date_col: str = "date",
                                debt_col: str = "D", rate_col: str = "r",
                                ret_col: Optional[str] = None) -> pd.DataFrame:
    """
    Adds structural features to a firm-month panel.

    obs          one row per firm-month with columns firm_col, date_col,
                 debt_col (D, face value), rate_col (r, decimal). Optional
                 ret_col: past-year stock return for the naive variant (if
                 None it is computed from the first/last market cap in the
                 window -- supply a price return if shares change a lot).
    daily_equity wide DataFrame (dates x firms) of daily market cap.
    variants     subset of VARIANTS.

    Returns a DataFrame with the same index as `obs` and columns
    "E", "sigma_E" plus, for every variant v:
        v_dd, v_pd, v_ln_va, v_sigma_a, v_d_va, v_va, v_mu, v_status
    """
    variants = list(variants)
    bad = set(variants) - set(VARIANTS)
    if bad:
        raise ValueError(f"unknown variants {bad}; choose from {VARIANTS}")
    firms = obs[firm_col].to_numpy()
    dates = pd.DatetimeIndex(obs[date_col])
    D_all = obs[debt_col].to_numpy(dtype=float)
    r_all = obs[rate_col].to_numpy(dtype=float)
    ret_all = obs[ret_col].to_numpy(dtype=float) if ret_col else None

    pieces = []
    for a in range(0, len(obs), chunk):
        sl = slice(a, min(a + chunk, len(obs)))
        W = build_windows(daily_equity, firms[sl], dates[sl], window_days)
        D, r = D_all[sl], r_all[sl]
        x_last = W[:, -1]

        # shared inputs (E, realised sigma_E, past return) from the window
        with np.errstate(all="ignore"):
            lr = np.diff(np.log(np.where(W > 0, W, np.nan)), axis=1)
            sigE = np.nanstd(lr, axis=1, ddof=1) * np.sqrt(252.0)
            nobs = np.sum(np.isfinite(lr), axis=1)
            sigE = np.where(nobs >= min_obs - 1, sigE, np.nan)
            valid = np.isfinite(W) & (W > 0)
            first = np.where(valid.any(axis=1),
                             W[np.arange(len(W)), valid.argmax(axis=1)], np.nan)
            ret = ret_all[sl] if ret_all is not None else x_last / first - 1.0
        out = pd.DataFrame({"E": x_last, "sigma_E": sigE})

        for v in variants:
            if v == "kmv":
                df = solve_kmv(W, D, r, T, tol_kmv, 100, 252, min_obs)
            elif v == "simul_mu_r":
                df = solve_simultaneous(x_last, sigE, D, r, T, "r", tol_simul)
            elif v == "simul_mu_hist":
                df = solve_simultaneous_hist(W, D, r, T, tol_simul, 200, 252, min_obs)
            else:  # naive
                df = naive_dd(x_last, sigE, D, ret, T)
            df = df.drop(columns=[c for c in ("sigma_e",) if c in df])
            df.columns = [f"{v}_{c}" for c in df.columns]
            out = pd.concat([out, df], axis=1)
        pieces.append(out)

    res = pd.concat(pieces, ignore_index=True)
    res.index = obs.index
    return res
