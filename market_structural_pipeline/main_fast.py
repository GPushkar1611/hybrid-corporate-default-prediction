import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root, for src/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import ndtr

# Fast numba kernels (merton.py must be in the same folder)
from src.structural import merton as m

# Import data pipeline from data.py (unchanged)
from data import build_firm_panel, fetch_risk_free

# =====================================================================
# 1. ROLLING WINDOW ESTIMATION  (vectorised + numba, replaces the loop)
# =====================================================================

def run_rolling_merton(df_panel, window=252, T=1.0, kmv_tol=1e-5):
    """
    Same outputs as the original run_rolling_merton, but ALL windows are
    solved in one parallel numba call instead of a Python loop of brentq
    root-finds.

    window = number of daily RETURNS (252) -> each window has window+1 prices,
    exactly like df.iloc[i-window : i+1] in the original code.
    """
    df = df_panel.sort_values('date').reset_index(drop=True)
    n = len(df)
    if n <= window:
        raise ValueError(f"Need more than {window} rows, got {n}")

    E_all = df['Market_Equity_E'].to_numpy(dtype=float)
    F_all = df['Default_Point_F'].to_numpy(dtype=float)
    r_all = df['r_rf'].to_numpy(dtype=float)

    # (n-window, window+1) matrix: row k = prices of the window ending at day window+k
    W = np.ascontiguousarray(sliding_window_view(E_all, window + 1))
    F, r = F_all[window:], r_all[window:]
    print(f"Calculating Merton variants over {len(W)} rolling windows "
          f"(numba: {m.HAVE_NUMBA})...")

    # ---- Variant (i): iterative KMV, one parallel call ----------------
    it = m.solve_kmv(W, F, r, T=T, tol=kmv_tol, max_iter=100,
                     trading_days=252, min_obs=window // 2)

    # Risk-neutral DD with the same asset path (mu = r)
    dd_rn = m.dd_from_inputs(it['va'].to_numpy(), F, it['sigma_a'].to_numpy(), r, T)

    # Original-code DD (used mean LOG return as mu AND subtracted 0.5*sigma^2
    # again). Kept only so you can compare with your old results.
    mean_log_252 = it['mu'].to_numpy() - 0.5 * it['sigma_a'].to_numpy() ** 2
    dd_legacy = m.dd_from_inputs(it['va'].to_numpy(), F, it['sigma_a'].to_numpy(),
                                 mean_log_252, T)

    # ---- Variant (iv): naive DD (closed form, vectorised) -------------
    lr = np.diff(np.log(W), axis=1)
    sigma_E = lr.std(axis=1, ddof=1) * np.sqrt(252)
    past_log_ret = np.log(W[:, -1] / W[:, 0])        # same mu as your original code
    nv = m.naive_dd(W[:, -1], sigma_E, F, past_log_ret, T=T)

    out = pd.DataFrame({
        'date': df['date'].to_numpy()[window:],
        'Close': df['Close'].to_numpy()[window:],
        'Market_Equity_E': W[:, -1],
        'Default_Point_F': F,
        'r_rf': r,
        # corrected iterative DD (no double-counted -0.5*sigma^2)
        'DD_iterative': it['dd'].to_numpy(),
        'pi_iterative': it['pd'].to_numpy(),
        'DD_iterative_orig_formula': dd_legacy,
        'DD_rn': dd_rn,
        'pi_rn': ndtr(-dd_rn),
        'sigma_V_iterative': it['sigma_a'].to_numpy(),
        'mu_iterative': it['mu'].to_numpy(),
        'DD_naive': nv['dd'].to_numpy(),
        'pi_naive': nv['pd'].to_numpy(),
        'sigma_V_naive': nv['sigma_a'].to_numpy(),
        'status_iterative': it['status'].to_numpy(),
        'status_naive': nv['status'].to_numpy(),
    })
    bad = (out.status_iterative != 0) | (out.status_naive != 0)
    if bad.any():
        print(f"WARNING: {bad.sum()} windows flagged (status != 0) -> set to NaN. "
              f"Counts: iterative {out.status_iterative.value_counts().to_dict()}, "
              f"naive {out.status_naive.value_counts().to_dict()}")
        num = out.columns.difference(['date', 'status_iterative', 'status_naive'])
        out.loc[bad, num] = np.nan
    return out

# =====================================================================
# 2. EXECUTION PIPELINE & VISUALIZATION  (data section unchanged)
# =====================================================================

if __name__ == "__main__":
    start_date = '2019-01-01'  # 1 extra year back for rolling window warmup
    end_date = '2023-12-31'
    ticker_symbol = "AAL"
    cik_aal = "0000006201"

    print("Fetching raw data (SEC, FRED, Yahoo)...")
    df_rf = fetch_risk_free(start_date, end_date)
    df_panel = build_firm_panel(ticker_symbol, cik_aal, start_date, end_date, df_rf)
    print(f"Panel: {len(df_panel)} trading days, "
          f"{df_panel['debt_partial'].mean():.0%} of days use partly-missing debt data, "
          f"max debt age {df_panel['debt_age_days'].max()} days")

    # Run Rolling Merton Model (fast)
    df_merton_results = run_rolling_merton(df_panel, window=252)

    print("\n=== ROLLING MERTON MODEL RESULTS ===")
    print(df_merton_results[['date', 'DD_iterative', 'DD_naive', 'pi_iterative', 'pi_naive']].tail())
    df_merton_results.to_csv('merton_results.csv', index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df_merton_results['date'], df_merton_results['DD_iterative'], label='Iterative Merton DD', color='blue')
    plt.plot(df_merton_results['date'], df_merton_results['DD_naive'], label='Naive Merton DD (Bharath & Shumway)', color='orange', linestyle='--')
    plt.title(f'Distance to Default (DD) Time Series - {ticker_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Distance to Default (Standard Deviations)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
