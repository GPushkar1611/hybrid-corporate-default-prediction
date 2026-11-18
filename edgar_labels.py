"""
edgar_labels.py -- build the bankruptcy label table from SEC EDGAR bulk files
and check whether each bankrupt firm can actually be modelled.

PIPELINE (each step answers one question)
  1. download   : get submissions.zip + companyfacts.zip (bulk, no per-firm API calls)
  2. labels     : which firms filed an 8-K with Item 1.03 (Bankruptcy or Receivership)?
  3. fundamentals coverage : does each have debt + share counts from XBRL, recent enough?
  4. price coverage        : can we get >= 252 days of prices before its label months?
  5. controls   : healthy comparison firms

Usage
  python edgar_labels.py download  --dest sec_bulk
  python edgar_labels.py labels    --submissions sec_bulk/submissions.zip \
                                   --companyfacts sec_bulk/companyfacts.zip --out bankruptcies.csv
  python edgar_labels.py prices    --table bankruptcies.csv --out bankruptcies_covered.csv
  python edgar_labels.py controls  --tickers company_tickers.json \
                                   --submissions sec_bulk/submissions.zip \
                                   --bankrupt bankruptcies.csv --n 1500 --out controls.csv

Files are LARGE (companyfacts ~ >1 GB, submissions several GB). The SEC wants a
descriptive User-Agent with a real contact: edit UA below.
"""
import argparse
import json
import os
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

import data as d     # parse_fundamentals / parse_shares / yfinance handle

def _sec_headers():
    user_agent = os.environ.get('SEC_USER_AGENT', '').strip()
    if '@' not in user_agent or 'domain.edu' in user_agent.lower():
        raise RuntimeError(
            'Set SEC_USER_AGENT to a descriptive project name and a real contact email before SEC requests.'
        )
    return {'User-Agent': user_agent}


URLS = {
    'submissions': 'https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip',
    'companyfacts': 'https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip',
}
TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
_MAIN = re.compile(r'^CIK(\d{10})\.json$')
_PAGE = re.compile(r'^CIK(\d{10})-submissions-\d+\.json$')


# ---------------------------------------------------------------------------
# 1. download
# ---------------------------------------------------------------------------
def download_bulk(dest_dir='sec_bulk', which=('submissions', 'companyfacts')):
    os.makedirs(dest_dir, exist_ok=True)
    for key in which:
        path = os.path.join(dest_dir, f'{key}.zip')
        if os.path.exists(path):
            print(f'{path} exists, skipping')
            continue
        print(f'downloading {URLS[key]} -> {path} (large; be patient)')
        with requests.get(URLS[key], headers=_sec_headers(), stream=True, timeout=(30, 300)) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)


def download_company_tickers(path='company_tickers.json'):
    """Fetch the SEC ticker -> CIK list (current registrants) if it is not on disk."""
    if os.path.exists(path):
        return path
    print(f'{path} not found; downloading {TICKERS_URL}')
    r = requests.get(TICKERS_URL, headers=_sec_headers(), timeout=(20, 90))
    r.raise_for_status()
    with open(path, 'w') as f:
        f.write(r.text)
    return path


# ---------------------------------------------------------------------------
# 2. bankruptcy table from submissions.zip
# ---------------------------------------------------------------------------
def _codes(items):
    if isinstance(items, str):
        return [c.strip() for c in items.split(',') if c.strip()]
    if isinstance(items, list):
        return [str(c).strip() for c in items]
    return []


def _hits(block, cik):
    forms = block.get('form', []) or []
    items = block.get('items', []) or []
    acc = block.get('accessionNumber', []) or []
    fdate = block.get('filingDate', []) or []
    rdate = block.get('reportDate', []) or []
    out = []
    for i, form in enumerate(forms):
        if form not in ('8-K', '8-K/A') or i >= len(items):
            continue
        if '1.03' in _codes(items[i]):
            out.append(dict(cik=cik, form=form, accession=acc[i], filing_date=fdate[i],
                            report_date=(rdate[i] if i < len(rdate) else '') or None))
    return out


def build_bankruptcy_table(submissions_zip, since='2010-01-01',
                           sic_exclude=(6000, 6999), progress_every=100000):
    """
    One row per CIK: its FIRST Item 1.03 event on/after `since`.
      bankruptcy_date = 8-K reportDate (date of the event) when sane, else filing date
    Financials (SIC 6000-6999, includes SPACs / ABS trusts) are excluded, as in the report.
    """
    since = pd.Timestamp(since)
    hits = []
    with zipfile.ZipFile(submissions_zip) as zf:
        infos = zf.infolist()
        for k, info in enumerate(infos):
            if progress_every and k % progress_every == 0:
                print(f'  scanned {k}/{len(infos)} files; {len(hits)} Item 1.03 filings so far')
            m_main, m_page = _MAIN.match(info.filename), _PAGE.match(info.filename)
            if not (m_main or m_page):
                continue
            raw = zf.read(info)
            if b'1.03' not in raw:                  # cheap pre-filter before JSON parsing
                continue
            j = json.loads(raw)
            cik = int((m_main or m_page).group(1))
            block = j.get('filings', {}).get('recent', {}) if m_main else j
            hits += _hits(block, cik)
        if not hits:
            return pd.DataFrame()
        df = pd.DataFrame(hits)
        meta = {}
        for cik in df['cik'].unique():              # name / SIC / tickers from the main file
            try:
                j = json.loads(zf.read(f'CIK{int(cik):010d}.json'))
            except KeyError:
                continue
            meta[cik] = dict(name=j.get('name'),
                             sic=pd.to_numeric(j.get('sic'), errors='coerce'),
                             entity_type=j.get('entityType'),
                             tickers='|'.join(j.get('tickers', []) or []),
                             exchanges='|'.join(str(e) for e in (j.get('exchanges', []) or [])))
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
    ok = df['report_date'].notna() & (df['report_date'] <= df['filing_date'])
    df['event_date'] = df['report_date'].where(ok, df['filing_date'])
    df = df[df['event_date'] >= since]
    # Select one actual first-event filing deterministically. Previously the
    # event date, filing date, and accession were aggregated independently,
    # so same-event-date filings could produce a mismatched accession/date row.
    ordered = df.sort_values(
        ['cik', 'event_date', 'filing_date', 'accession'], kind='stable'
    )
    first = ordered.drop_duplicates('cik', keep='first').rename(columns={
        'event_date': 'bankruptcy_date',
        'filing_date': 'first_filing_date',
        'accession': 'first_accession',
    })
    counts = ordered.groupby('cik', as_index=False).size().rename(columns={'size': 'n_1_03_filings'})
    tab = first[['cik', 'bankruptcy_date', 'first_filing_date', 'first_accession']].merge(
        counts, on='cik', how='left', validate='one_to_one'
    )
    tab = tab.merge(pd.DataFrame.from_dict(meta, orient='index').rename_axis('cik').reset_index(),
                    on='cik', how='left')
    n0 = len(tab)
    fin = tab['sic'].between(sic_exclude[0], sic_exclude[1])
    tab = tab[~fin].reset_index(drop=True)
    print(f'{n0} firms with Item 1.03 since {since.date()}; '
          f'{n0 - len(tab)} dropped as financial (SIC {sic_exclude[0]}-{sic_exclude[1]}); '
          f'{len(tab)} remain')
    return tab


# ---------------------------------------------------------------------------
# 3. fundamentals coverage from companyfacts.zip
# ---------------------------------------------------------------------------
def fundamentals_coverage(table, companyfacts_zip, max_age_days=550):
    rows = []
    with zipfile.ZipFile(companyfacts_zip) as zf:
        names = set(zf.namelist())
        for r in table.itertuples():
            rec = dict(cik=r.cik, fund_status='no_xbrl', n_fund_before=0, debt_age_days=np.nan)
            nm = f'CIK{int(r.cik):010d}.json'
            if nm in names:
                facts = json.loads(zf.read(nm))
                fund, sh = d.parse_fundamentals(facts), d.parse_shares(facts)
                f = fund[fund['available_date'] <= r.bankruptcy_date]
                s = sh[sh['filed'] <= r.bankruptcy_date]
                rec['n_fund_before'] = len(f)
                if f.empty:
                    rec['fund_status'] = 'no_debt'
                elif s.empty:
                    rec['fund_status'] = 'no_shares'
                else:
                    age = (r.bankruptcy_date - f['available_date'].max()).days
                    rec['debt_age_days'] = age
                    rec['fund_status'] = 'ok' if age <= max_age_days else 'stale'
            rows.append(rec)
    out = pd.DataFrame(rows)
    print('fundamentals coverage:\n', out['fund_status'].value_counts().to_string())
    return table.merge(out, on='cik')


# ---------------------------------------------------------------------------
# 4. price coverage
# ---------------------------------------------------------------------------
def _yahoo_fetch(ticker, start, end):
    try:
        px = d.yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)[['Close']]
    except Exception:
        return None
    if px is None or px.empty:
        return None
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    return px


def price_coverage(table, tickers_override=None, min_days=252, min_label_months=6,
                   max_gap_days=180, pause=0.3, fetch=None):
    """
    For each bankrupt firm count how many of the 12 label months have >= min_days of
    prior trading days.  `fetch(ticker, start, end)` must return a DataFrame indexed
    by date with a 'Close' column (default: Yahoo).
    status: ok | no_ticker | no_prices | ends_too_early | too_few_label_months
    """
    fetch = fetch or _yahoo_fetch
    override = tickers_override or {}
    rows = []
    for r in table.itertuples():
        bd = r.bankruptcy_date
        tk = override.get(r.cik) or ((r.tickers or '').split('|')[0] if isinstance(r.tickers, str) else '')
        rec = dict(cik=r.cik, ticker=tk or None, price_status='no_ticker',
                   n_label_months=0, first_price=pd.NaT, last_price=pd.NaT)
        if tk:
            px = fetch(tk, bd - pd.Timedelta(days=900), bd + pd.Timedelta(days=30))
            if px is None or px.empty:
                rec['price_status'] = 'no_prices'
            else:
                px = px.sort_index()
                rec['first_price'], rec['last_price'] = px.index.min(), px.index.max()
                lastday = px.index.to_series().groupby(px.index.to_period('M')).last()
                lab = lastday[(lastday > bd - pd.DateOffset(months=12)) & (lastday < bd)]
                n_prior = np.array([px.index.searchsorted(t, side='right') for t in lab])
                rec['n_label_months'] = int((n_prior >= min_days).sum())
                pre = px.index[px.index <= bd]
                gap = (bd - pre.max()).days if len(pre) else 10 ** 6
                if gap > max_gap_days:
                    rec['price_status'] = 'ends_too_early'
                elif rec['n_label_months'] < min_label_months:
                    rec['price_status'] = 'too_few_label_months'
                else:
                    rec['price_status'] = 'ok'
            time.sleep(pause)
        rows.append(rec)
    out = pd.DataFrame(rows)
    print('price coverage:\n', out['price_status'].value_counts().to_string())
    return table.drop(columns=['ticker'], errors='ignore').merge(out, on='cik')


# ---------------------------------------------------------------------------
# 5. controls
# ---------------------------------------------------------------------------
def sample_controls(company_tickers_path, submissions_zip, exclude_ciks, n,
                    seed=0, sic_exclude=(6000, 6999), candidates_factor=3):
    """
    Random non-financial firms from the SEC ticker list that are not in `exclude_ciks`.
    CAVEAT: company_tickers.json lists CURRENT registrants, so controls are survivors
    (firms later acquired/delisted for other reasons are absent). Mention in the report.
    """
    with open(download_company_tickers(company_tickers_path)) as f:
        j = json.load(f)
    cand = pd.DataFrame(list(j.values()))
    cand['cik'] = cand['cik_str'].astype(int)
    cand = cand.drop_duplicates('cik')
    cand = cand[~cand['cik'].isin(set(exclude_ciks))]
    cand = cand.sample(min(len(cand), n * candidates_factor), random_state=seed)
    keep = []
    with zipfile.ZipFile(submissions_zip) as zf:
        names = set(zf.namelist())
        for r in cand.itertuples():
            nm = f'CIK{r.cik:010d}.json'
            if nm not in names:
                continue
            jj = json.loads(zf.read(nm))
            sic = pd.to_numeric(jj.get('sic'), errors='coerce')
            if pd.notna(sic) and sic_exclude[0] <= sic <= sic_exclude[1]:
                continue
            keep.append(dict(cik=r.cik, ticker=r.ticker, name=r.title, sic=sic))
            if len(keep) >= n:
                break
    return pd.DataFrame(keep)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('download'); a.add_argument('--dest', default='sec_bulk')
    b = sub.add_parser('labels')
    b.add_argument('--submissions', required=True); b.add_argument('--companyfacts', required=True)
    b.add_argument('--since', default='2010-01-01'); b.add_argument('--out', default='bankruptcies.csv')
    c = sub.add_parser('prices')
    c.add_argument('--table', required=True); c.add_argument('--out', default='bankruptcies_covered.csv')
    c.add_argument('--overrides', help='optional CSV with columns cik,ticker for manual matches')
    e = sub.add_parser('controls')
    e.add_argument('--tickers', default='company_tickers.json', help='auto-downloaded if missing'); e.add_argument('--submissions', required=True)
    e.add_argument('--bankrupt', required=True,
                   help='full bankruptcy event table (bankruptcies.csv), not only the covered subset')
    e.add_argument('--n', type=int, default=1500)
    e.add_argument('--out', default='controls.csv')
    args = ap.parse_args()

    if args.cmd == 'download':
        download_bulk(args.dest)
    elif args.cmd == 'labels':
        t = build_bankruptcy_table(args.submissions, args.since)
        t = fundamentals_coverage(t, args.companyfacts)
        t.to_csv(args.out, index=False); print('wrote', args.out)
    elif args.cmd == 'prices':
        t = pd.read_csv(args.table, parse_dates=['bankruptcy_date', 'first_filing_date'])
        ov = None
        if args.overrides:
            o = pd.read_csv(args.overrides); ov = dict(zip(o['cik'], o['ticker']))
        t = price_coverage(t[t['fund_status'] == 'ok'], ov)
        t.to_csv(args.out, index=False); print('wrote', args.out)
    elif args.cmd == 'controls':
        bk = pd.read_csv(args.bankrupt)
        c_ = sample_controls(args.tickers, args.submissions, bk['cik'], args.n)
        c_.to_csv(args.out, index=False); print('wrote', args.out, len(c_))
