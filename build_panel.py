"""
build_panel.py -- firm-month panel with bankruptcy labels and Merton features.

Input : bankruptcies_covered.csv (price_status == ok rows are used) and controls.csv
Output: panel.csv  (one row per firm-month: label y, D, r, structural features)

Label for firm i, month-end t:
    y = 1  if the firm's bankruptcy_date lies in (t, t + 12 months]
    y = 0  for controls, and for a bankrupt firm's months more than 12 months before
    rows with t >= bankruptcy_date are DROPPED (the stock trades on in Chapter 11)

Usage
  python build_panel.py --bankrupt bankruptcies_covered.csv --controls controls.csv \
                        --start 2009-01-01 --end 2024-12-31 --out panel.csv
"""
import argparse
import numpy as np
import pandas as pd

import data as d
import merton as m


def collect_panels(firms, start, end, df_rf, builder=None):
    """firms: DataFrame with cik, ticker. Returns (daily panels, failures)."""
    builder = builder or d.build_firm_panel
    panels, failures = [], []
    for i, r in enumerate(firms.itertuples(), 1):
        try:
            p = builder(r.ticker, str(int(r.cik)), start, end, df_rf)
            p['firm'] = int(r.cik)
            panels.append(p)
        except Exception as e:                       # keep going; report at the end
            failures.append((int(r.cik), r.ticker, str(e)[:120]))
        if i % 25 == 0:
            print(f'  {i}/{len(firms)} firms processed, {len(failures)} failures')
    return panels, failures


def to_model_inputs(panels, bankruptcy_dates, horizon_months=12, min_days=252):
    """
    panels: list of daily panels (columns incl. firm, date, Market_Equity_E, Default_Point_F,
            r_rf, debt_age_days, debt_partial).
    bankruptcy_dates: dict cik -> Timestamp (controls absent).
    Returns (obs, daily_equity) ready for merton.compute_structural_features.
    """
    allp = pd.concat(panels, ignore_index=True).sort_values(['firm', 'date'])
    allp['n_hist'] = allp.groupby('firm').cumcount() + 1
    daily_equity = allp.pivot(index='date', columns='firm', values='Market_Equity_E').sort_index()

    me = allp.groupby(['firm', allp['date'].dt.to_period('M')], as_index=False).last()
    me = me[me['n_hist'] >= min_days]
    obs = me.rename(columns={'Default_Point_F': 'D', 'r_rf': 'r'})[
        ['firm', 'date', 'D', 'r', 'debt_age_days', 'debt_partial']].copy()

    bd = obs['firm'].map(bankruptcy_dates)               # NaT for controls
    obs = obs[~(bd.notna() & (obs['date'] >= bd))].copy()      # drop post-filing rows
    bd = obs['firm'].map(bankruptcy_dates)
    horizon_end = obs['date'] + pd.DateOffset(months=horizon_months)
    obs['y'] = (bd.notna() & (bd > obs['date']) & (bd <= horizon_end)).astype(int)
    obs['bankrupt_firm'] = bd.notna().astype(int)
    return obs.reset_index(drop=True), daily_equity


def prior_correct(p_sampled, beta):
    """Bayes correction from the report: p = beta*p_s / (beta*p_s - p_s + 1),
    beta = share of non-defaults kept when sampling."""
    p_sampled = np.asarray(p_sampled, float)
    return beta * p_sampled / (beta * p_sampled - p_sampled + 1.0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bankrupt', required=True); ap.add_argument('--controls', required=True)
    ap.add_argument('--start', default='2009-01-01'); ap.add_argument('--end', default='2024-12-31')
    ap.add_argument('--out', default='panel.csv')
    a = ap.parse_args()

    bk = pd.read_csv(a.bankrupt, parse_dates=['bankruptcy_date'])
    bk = bk[bk['price_status'] == 'ok']
    ct = pd.read_csv(a.controls)
    firms = pd.concat([bk[['cik', 'ticker']], ct[['cik', 'ticker']]]).drop_duplicates('cik')
    df_rf = d.fetch_risk_free(a.start, a.end)

    panels, failures = collect_panels(firms, a.start, a.end, df_rf)
    print(f'{len(panels)} firms built, {len(failures)} failed (e.g. {failures[:3]})')
    obs, daily_eq = to_model_inputs(panels, dict(zip(bk['cik'], bk['bankruptcy_date'])))
    feats = m.compute_structural_features(obs, daily_eq, firm_col='firm')
    panel = pd.concat([obs, feats], axis=1)
    panel.to_csv(a.out, index=False)

    n_b = obs.loc[obs.bankrupt_firm == 1, 'firm'].nunique()
    n_c = obs.loc[obs.bankrupt_firm == 0, 'firm'].nunique()
    print(f'rows={len(panel)}  bankrupt firms={n_b}  control firms={n_c}  '
          f'y=1 rows={panel.y.sum()}')
    print('structural status counts:', panel['kmv_status'].value_counts().to_dict())
    print('NOTE: controls are a sample -> use prior_correct(p, beta) before reading PD levels.')
