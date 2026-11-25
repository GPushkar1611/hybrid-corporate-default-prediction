"""Build the SEC 8-K Item 1.03 bankruptcy event table from the SEC submissions bulk ZIP.

Extracted from the original edgar_labels.py (now in market_structural_pipeline/) so the
active label pipeline does not depend on the old Yahoo/market-data code.
"""
import json
import re
import zipfile

import pandas as pd

_MAIN = re.compile(r'^CIK(\d{10})\.json$')
_PAGE = re.compile(r'^CIK(\d{10})-submissions-\d+\.json$')


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
