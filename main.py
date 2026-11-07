import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# Import data pipeline from data.py
from data import get_sec_debt_data, yf, web

# =====================================================================
# 1. MERTON MODEL SOLVERS
# =====================================================================

def bs_equity_price(V, F, r, T, sigma_V):
    """Calculates Black-Scholes Equity Value E for a given Firm Value V."""
    if V <= 0 or sigma_V <= 0:
        return 0.0
    d1 = (np.log(V / F) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    return V * norm.cdf(d1) - F * np.exp(-r * T) * norm.cdf(d2)

def invert_bs_firm_value(E_target, F, r, T, sigma_V):
    """Inverts Black-Scholes formula to solve for implied asset value V."""
    obj = lambda V: bs_equity_price(V, F, r, T, sigma_V) - E_target
    lower_bound = max(1e-5, E_target)
    upper_bound = E_target + 5.0 * F
    try:
        V_sol = brentq(obj, lower_bound, upper_bound, xtol=1e-6)
    except ValueError:
        V_sol = E_target + F
    return V_sol

def compute_iterative_merton(daily_equity_prices, F, r, T=1.0, max_iter=100, tol=1e-5):
    """Iterative KMV-Merton algorithm."""
    E_series = np.array(daily_equity_prices)
    E_latest = E_series[-1]
    
    eq_returns = np.diff(np.log(E_series))
    sigma_E = np.std(eq_returns, ddof=1) * np.sqrt(252)
    
    sigma_V = sigma_E * (E_latest / (E_latest + F))
    N = len(E_series)
    V_series = np.zeros(N)
    
    for iteration in range(max_iter):
        for t in range(N):
            V_series[t] = invert_bs_firm_value(E_series[t], F, r, T, sigma_V)
        
        asset_returns = np.diff(np.log(V_series))
        sigma_V_new = np.std(asset_returns, ddof=1) * np.sqrt(252)
        
        if abs(sigma_V_new - sigma_V) < tol:
            sigma_V = sigma_V_new
            break
        sigma_V = sigma_V_new

    mu = np.mean(asset_returns) * 252
    V_latest = V_series[-1]
    
    # Real-World Merton DD & PI
    dd_num = np.log(V_latest / F) + (mu - 0.5 * sigma_V**2) * T
    dd_den = sigma_V * np.sqrt(T)
    DD_merton = dd_num / dd_den
    pi_merton = norm.cdf(-DD_merton)
    
    # Risk-Neutral DD & PI
    dd_rn_num = np.log(V_latest / F) + (r - 0.5 * sigma_V**2) * T
    DD_rn = dd_rn_num / dd_den
    pi_rn = norm.cdf(-DD_rn)
    
    return {
        'V': V_latest, 'sigma_V': sigma_V, 'mu': mu,
        'DD_merton': DD_merton, 'pi_merton': pi_merton,
        'DD_rn': DD_rn, 'pi_rn': pi_rn
    }

def compute_naive_merton(daily_equity_prices, F, r, T=1.0):
    """Bharath & Shumway (2008) closed-form Naive Merton shortcut."""
    E_series = np.array(daily_equity_prices)
    E_latest = E_series[-1]
    
    eq_returns = np.diff(np.log(E_series))
    sigma_E = np.std(eq_returns, ddof=1) * np.sqrt(252)
    r_t_minus_1 = np.log(E_series[-1] / E_series[0])
    
    V_naive = E_latest + F
    sigma_D_naive = 0.05 + 0.25 * sigma_E
    sigma_V_naive = (E_latest / V_naive) * sigma_E + (F / V_naive) * sigma_D_naive
    
    dd_num = np.log(V_naive / F) + (r_t_minus_1 - 0.5 * sigma_V_naive**2) * T
    dd_den = sigma_V_naive * np.sqrt(T)
    DD_naive = dd_num / dd_den
    pi_naive = norm.cdf(-DD_naive)
    
    return {
        'V_naive': V_naive, 'sigma_V_naive': sigma_V_naive,
        'DD_naive': DD_naive, 'pi_naive': pi_naive
    }

# =====================================================================
# 2. ROLLING WINDOW ESTIMATION WRAPPER
# =====================================================================

def run_rolling_merton(df_panel, window=252):
    """
    Applies the Merton Model variants across df_panel using a 252-day rolling window.
    """
    df = df_panel.sort_values('date').reset_index(drop=True)
    results = []
    
    print(f"Calculating Merton variants over {len(df) - window} rolling windows...")
    
    for i in range(window, len(df)):
        # Trailing 252-day window slice
        window_slice = df.iloc[i - window : i + 1]
        
        date_t = window_slice.iloc[-1]['date']
        E_series = window_slice['Market_Equity_E'].values
        F_t = window_slice.iloc[-1]['Default_Point_F']
        r_t = window_slice.iloc[-1]['r_rf']
        close_t = window_slice.iloc[-1]['Close']
        
        # Calculate variants
        try:
            iter_res = compute_iterative_merton(E_series, F=F_t, r=r_t)
            naive_res = compute_naive_merton(E_series, F=F_t, r=r_t)
            
            results.append({
                'date': date_t,
                'Close': close_t,
                'Market_Equity_E': E_series[-1],
                'Default_Point_F': F_t,
                'r_rf': r_t,
                'DD_iterative': iter_res['DD_merton'],
                'pi_iterative': iter_res['pi_merton'],
                'DD_rn': iter_res['DD_rn'],
                'pi_rn': iter_res['pi_rn'],
                'sigma_V_iterative': iter_res['sigma_V'],
                'DD_naive': naive_res['DD_naive'],
                'pi_naive': naive_res['pi_naive'],
                'sigma_V_naive': naive_res['sigma_V_naive']
            })
        except Exception as e:
            continue
            
    return pd.DataFrame(results)

# =====================================================================
# 3. EXECUTION PIPELINE & VISUALIZATION
# =====================================================================

if __name__ == "__main__":
    from data import get_sec_debt_data
    
    # 1. Fetch Panel Data (From data.py workflow)
    start_date = '2019-01-01'  # 1 extra year back for rolling window warmup
    end_date = '2023-12-31'
    ticker_symbol = "AAL"
    cik_aal = "0000006201"

    print("Fetching raw data...")
    df_debt = get_sec_debt_data(cik_aal)
    df_rf = web.DataReader('DGS1', 'fred', start_date, end_date).rename(columns={'DGS1': 'r_rf'}) / 100.0
    df_rf = df_rf.ffill()
    
    df_market = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date)[['Close', 'Volume']]
    df_market.index = df_market.index.tz_localize(None)
    shares_outstanding = yf.Ticker(ticker_symbol).info.get('sharesOutstanding', 1e8)
    df_market['Market_Equity_E'] = df_market['Close'] * shares_outstanding
    
    df_debt['available_date'] = df_debt['end'] + pd.Timedelta(days=45)
    df_panel = pd.merge(df_market, df_rf, left_index=True, right_index=True, how='left').reset_index().rename(columns={'Date': 'date'})
    
    df_panel['date'] = pd.to_datetime(df_panel['date']).astype('datetime64[ns]')
    df_debt['available_date'] = pd.to_datetime(df_debt['available_date']).astype('datetime64[ns]')

    df_panel = pd.merge_asof(
        df_panel.sort_values('date'),
        df_debt[['available_date', 'Default_Point_F']].sort_values('available_date'),
        left_on='date', right_on='available_date', direction='backward'
    ).dropna(subset=['Market_Equity_E', 'Default_Point_F', 'r_rf'])

    # 2. Run Rolling Merton Model
    df_merton_results = run_rolling_merton(df_panel, window=252)

    # 3. Inspect Output
    print("\n=== ROLLING MERTON MODEL RESULTS ===")
    print(df_merton_results[['date', 'DD_iterative', 'DD_naive', 'pi_iterative', 'pi_naive']].tail())

    # 4. Plot Distance to Default: Iterative vs Naive
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