"""
data.py -- point-in-time data layer for the Merton / hazard-model project.

What changed vs. the original data.py (and why)
-----------------------------------------------
1. PRICES: history(auto_adjust=False). The default Close is dividend-adjusted,
   which shrinks old prices and understates historical market cap E.
   'Close' here is split-adjusted but NOT dividend-adjusted.
2. SHARES: time-varying shares from the SEC cover page (dei:EntityCommon-
   StockSharesOutstanding), usable only after its filing date, and restated to
   today's split basis so that  E = Close(split-adj) x shares(split-adj)  is
   right on every date.  The old code used today's share count for all years.
3. POINT-IN-TIME: a fundamental is usable from the date it was FIRST FILED
   (the 'filed' field in the SEC JSON), not end + 45 days, and later restated
   values / prior-year comparatives are ignored.
4. XBRL TAG DRIFT: tags are coalesced per fiscal year-end in priority order
   (the old code took the first tag that existed at all, even if it covered
   only a few years).  The tag used is recorded.
5. MISSING DEBT: if BOTH short- and long-term debt are missing the row is
   dropped (NaN), not silently set to F = 0.  If only one is missing it is
   treated as 0 and flagged (debt_partial = True).
6. RATES: FRED rates are attached with merge_asof (no trading days are lost to
   holiday mismatches, so rolling windows stay contiguous).
7. SEC downloads are cached on disk, retried, and rate-limited.

SEC requires a descriptive User-Agent with a real contact: edit HEADERS.
"""
import os
import json
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

HEADERS = {'User-Agent': 'UniversityResearcher researcher@domain.edu'}   # <- EDIT
CACHE_DIR = 'sec_cache'

# priority order: first tag with a value for a given fiscal year-end wins
SHORT_TAGS = ['DebtCurrent', 'LongTermDebtCurrent', 'ShortTermBorrowings',
              'DebtAndCapitalLeaseObligationsCurrent', 'NotesPayableCurrent',
              'CommercialPaper']
LONG_TAGS = ['LongTermDebtNoncurrent', 'LongTermDebt',
             'LongTermDebtAndCapitalLeaseObligations', 'DebtLongTerm']
SHARE_TAGS_GAAP = ['CommonStockSharesOutstanding']


# =====================================================================
# 1. SEC DOWNLOAD (cached, retried, rate-limited)
# =====================================================================

def fetch_companyfacts(cik_str, cache_dir=CACHE_DIR, use_cache=True):
    cik = str(cik_str).zfill(10)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'CIK{cik}.json')
    if use_cache and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    for attempt in range(4):
        time.sleep(0.15)                          # SEC limit: <10 requests/s
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code == 200:
            data = resp.json()
            with open(path, 'w') as f:
                json.dump(data, f)
            return data
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        break
    raise ValueError(f"Failed to fetch SEC EDGAR data for CIK {cik}. "
                     f"Status Code: {resp.status_code}")


# =====================================================================
# 2. PARSING (pure functions: testable without network)
# =====================================================================

def _empty(with_tag=False):
    cols = {'end': 'datetime64[ns]', 'filed': 'datetime64[ns]', 'val': 'float64'}
    if with_tag:
        cols['tag'] = 'object'
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})


def _first_release(records, forms, annual_duration=False):
    """One row per fiscal period end, using the EARLIEST filing that reported it."""
    df = pd.DataFrame(records)
    empty = _empty()
    if df.empty or 'form' not in df:
        return empty
    df = df[df['form'].isin(forms)].copy()
    if df.empty:
        return empty
    df['end'] = pd.to_datetime(df['end'])
    df['filed'] = pd.to_datetime(df['filed'])
    if 'start' in df:
        start = pd.to_datetime(df['start'])
        if annual_duration:                       # flow items: full-year only
            dur = (df['end'] - start).dt.days
            df = df[(dur >= 350) & (dur <= 380)]
        else:                                     # stock items: instants only
            df = df[start.isna()]
    elif annual_duration:
        return empty
    if df.empty:
        return empty
    df = df.sort_values(['end', 'filed'])
    return df.groupby('end', as_index=False).first()[['end', 'filed', 'val']]


def _concept(facts, tags, forms, annual_duration=False, unit='USD'):
    """Coalesce several candidate tags per fiscal year-end (priority order)."""
    out = None
    for tag in tags:
        if tag not in facts:
            continue
        units = facts[tag].get('units', {})
        if unit not in units:
            continue
        s = _first_release(units[unit], forms, annual_duration)
        if s.empty:
            continue
        s['tag'] = tag
        out = s if out is None else pd.concat([out, s[~s['end'].isin(out['end'])]])
    if out is None:
        return _empty(with_tag=True)
    return out.sort_values('end').reset_index(drop=True)


def parse_fundamentals(companyfacts, forms=('10-K',)):
    """
    Returns one row per fiscal year-end with columns
      end, available_date, DLC, DLTT, Default_Point_F, Total_Assets,
      Net_Income, ROA, debt_partial, short_tag, long_tag
    available_date = latest first-filing date among the items used.
    Default point F = short-term debt + 0.5 x long-term debt  (KMV / [2]).
    """
    gaap = companyfacts.get('facts', {}).get('us-gaap', {})
    s = _concept(gaap, SHORT_TAGS, forms)
    l = _concept(gaap, LONG_TAGS, forms)
    a = _concept(gaap, ['Assets'], forms)
    ni = _concept(gaap, ['NetIncomeLoss'], forms, annual_duration=True)

    def ren(df, v, f, t=None):
        cols = {'val': v, 'filed': f}
        if t:
            cols['tag'] = t
        keep = ['end'] + list(cols)
        return df[keep].rename(columns=cols)

    df = ren(s, 'DLC', 'f_s', 'short_tag')
    for part in (ren(l, 'DLTT', 'f_l', 'long_tag'), ren(a, 'Total_Assets', 'f_a'),
                 ren(ni, 'Net_Income', 'f_n')):
        df = df.merge(part, on='end', how='outer')
    if df.empty:
        return pd.DataFrame(columns=['end', 'available_date', 'DLC', 'DLTT',
                                     'Default_Point_F', 'Total_Assets', 'Net_Income',
                                     'ROA', 'debt_partial'])
    both_missing = df['DLC'].isna() & df['DLTT'].isna()
    df['debt_partial'] = df['DLC'].isna() ^ df['DLTT'].isna()
    df['DLC'] = df['DLC'].fillna(0.0)
    df['DLTT'] = df['DLTT'].fillna(0.0)
    df['Default_Point_F'] = (df['DLC'] + 0.5 * df['DLTT']).where(~both_missing)
    df['ROA'] = df['Net_Income'] / df['Total_Assets']
    fcols = [c for c in ('f_s', 'f_l', 'f_a', 'f_n') if c in df]
    for c in fcols:
        df[c] = pd.to_datetime(df[c])
    df['available_date'] = df[fcols].max(axis=1)
    df = df.dropna(subset=['Default_Point_F', 'available_date'])
    return df.sort_values('end').reset_index(drop=True)


def parse_shares(companyfacts, forms=('10-K', '10-Q')):
    """
    Shares outstanding with count date `end` and filing date `filed`.
    Cover-page (dei) count preferred; balance-sheet count as fallback.
    Several share classes filed together are summed.
    """
    facts = companyfacts.get('facts', {})
    recs = facts.get('dei', {}).get('EntityCommonStockSharesOutstanding', {}) \
                .get('units', {}).get('shares')
    if recs is None:
        for tag in SHARE_TAGS_GAAP:
            recs = facts.get('us-gaap', {}).get(tag, {}).get('units', {}).get('shares')
            if recs:
                break
    df = pd.DataFrame(recs if recs else [])
    if df.empty or 'form' not in df:
        return pd.DataFrame(columns=['end', 'filed', 'shares'])
    df = df[df['form'].isin(forms)].copy()
    df['end'] = pd.to_datetime(df['end'])
    df['filed'] = pd.to_datetime(df['filed'])
    df = df.drop_duplicates(['end', 'filed', 'val'])
    df = df.groupby(['end', 'filed'], as_index=False)['val'].sum()
    df = df.sort_values(['end', 'filed']).groupby('end', as_index=False).first()
    return df.rename(columns={'val': 'shares'})[['end', 'filed', 'shares']] \
             .sort_values('filed').reset_index(drop=True)


def adjust_shares_for_splits(df_shares, splits):
    """
    Restate reported shares to TODAY's split basis (the basis of Yahoo's Close):
    multiply by the product of split ratios that happened after the count date.
    """
    df = df_shares.copy()
    if splits is None or len(splits) == 0 or df.empty:
        df['shares_adj'] = df['shares']
        return df
    sp = splits[splits > 0]
    factors = []
    for d in df['end']:
        factors.append(float(sp[sp.index > d].prod()) if (sp.index > d).any() else 1.0)
    df['shares_adj'] = df['shares'] * np.array(factors)
    return df


# =====================================================================
# 3. MARKET + MACRO DATA
# =====================================================================

def get_market_data(ticker_sym, start_date, end_date):
    """Daily Close (split-adjusted, NOT dividend-adjusted), Adj Close, splits."""
    tk = yf.Ticker(ticker_sym)
    px = tk.history(start=start_date, end=end_date, auto_adjust=False)[
        ['Close', 'Adj Close', 'Volume']]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    px.index.name = 'date'
    splits = tk.splits
    if len(splits):
        splits.index = pd.DatetimeIndex(splits.index).tz_localize(None).normalize()
    return px, splits


def fetch_risk_free(start_date, end_date):
    """1-year Treasury (FRED DGS1) as a decimal, forward-filled."""
    df = web.DataReader('DGS1', 'fred', start_date, end_date).rename(
        columns={'DGS1': 'r_rf'}) / 100.0
    df.index.name = 'date'
    return df.ffill()


# =====================================================================
# 4. ONE-FIRM DAILY PANEL
# =====================================================================

def assemble_panel(px, splits, df_debt, df_shares, df_rf, ticker_sym=None):
    """Pure assembly step (no network)."""
    sh = adjust_shares_for_splits(df_shares, splits).sort_values('filed')
    panel = px.reset_index().rename(columns={'Date': 'date'})
    panel['date'] = pd.to_datetime(panel['date']).astype('datetime64[ns]')
    panel = panel.sort_values('date')

    sh = sh[['filed', 'shares_adj']].copy()
    sh['filed'] = sh['filed'].astype('datetime64[ns]')
    panel = pd.merge_asof(panel, sh, left_on='date', right_on='filed', direction='backward')

    rf = df_rf.reset_index().rename(columns={df_rf.index.name or 'index': 'date'})
    rf['date'] = pd.to_datetime(rf['date']).astype('datetime64[ns]')
    panel = pd.merge_asof(panel, rf[['date', 'r_rf']].sort_values('date'),
                          on='date', direction='backward')

    d = df_debt[['available_date', 'end', 'Default_Point_F', 'Total_Assets', 'ROA',
                 'debt_partial']].copy()
    d['available_date'] = d['available_date'].astype('datetime64[ns]')
    panel = pd.merge_asof(panel, d.sort_values('available_date'),
                          left_on='date', right_on='available_date', direction='backward')

    panel['Market_Equity_E'] = panel['Close'] * panel['shares_adj']
    panel['debt_age_days'] = (panel['date'] - panel['available_date']).dt.days
    panel = panel.rename(columns={'shares_adj': 'shares'})
    panel = panel.dropna(subset=['Market_Equity_E', 'Default_Point_F', 'r_rf'])
    if ticker_sym is not None:
        panel['ticker'] = ticker_sym
    return panel.reset_index(drop=True)


def build_firm_panel(ticker_sym, cik_str, start_date, end_date, df_rf,
                     forms=('10-K',)):
    """Daily point-in-time panel for one firm (network: SEC + Yahoo)."""
    facts = fetch_companyfacts(cik_str)
    df_debt = parse_fundamentals(facts, forms=forms)
    df_shares = parse_shares(facts)
    if df_debt.empty:
        raise ValueError(f"{ticker_sym}: no usable debt data in SEC facts")
    if df_shares.empty:
        raise ValueError(f"{ticker_sym}: no share counts in SEC facts")
    px, splits = get_market_data(ticker_sym, start_date, end_date)
    if px.empty:
        raise ValueError(f"{ticker_sym}: no price history")
    return assemble_panel(px, splits, df_debt, df_shares, df_rf, ticker_sym)


def get_sec_debt_data(cik_str):
    """Backward-compatible wrapper (same name as the original function)."""
    return parse_fundamentals(fetch_companyfacts(cik_str))


# =====================================================================
# 5. DEMO
# =====================================================================

if __name__ == "__main__":
    start_date, end_date = '2019-01-01', '2023-12-31'
    df_rf = fetch_risk_free(start_date, end_date)
    df_panel = build_firm_panel('AAL', '0000006201', start_date, end_date, df_rf)
    print("\n=== UNIFIED DATASET READY FOR MERTON MODEL ===")
    print(df_panel[['date', 'Close', 'shares', 'Market_Equity_E',
                    'Default_Point_F', 'r_rf', 'debt_age_days']].head())
    print(df_panel[['date', 'Close', 'shares', 'Market_Equity_E',
                    'Default_Point_F', 'debt_partial']].tail())
