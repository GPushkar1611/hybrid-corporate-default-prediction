"""Build a point-in-time SEC bankruptcy panel with normalized CompanyFacts values.

SEC FSDS SUB files supply filing dates/SIC. Monetary values come from the SEC
Company Facts bulk API, whose standard-taxon values are already normalized.
SEC FSDS NUM is deliberately not used: its published value field omits XBRL
scale, which can produce impossible ratios if read as dollars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
FINANCIAL_SIC = (6000, 6999)
TAG_ALIASES = {
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterestAndRedeemableNoncontrollingInterest",
    ],
    "short_debt": [
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "CurrentPortionOfLongTermDebtCurrent",
        "ShortTermDebtCurrent",
        "CommercialPaper",
    ],
    "long_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
    ],
}
FEATURE_COLUMNS = (
    "log_assets_usd",
    "liabilities_to_assets",
    "current_ratio",
    "cash_to_assets",
    "debt_to_assets",
    "equity_to_assets",
    "working_capital_to_assets",
    "cash_to_current_liabilities",
    "debt_to_equity",
    "debt_partial",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_maturity_cutoff(as_of: pd.Timestamp, horizon_months: int = 12) -> pd.Timestamp:
    candidate = as_of - pd.DateOffset(months=horizon_months)
    month_end = candidate.to_period("M").to_timestamp("M")
    return month_end if month_end <= candidate else month_end - pd.offsets.MonthEnd(1)


def _submissions(archive_dir: Path, start_year: int, end_year: int, end_quarter: int) -> pd.DataFrame:
    frames = []
    quarters = [
        (year, quarter)
        for year in range(start_year, end_year + 1)
        for quarter in range(1, (end_quarter if year == end_year else 4) + 1)
    ]
    for number, (year, quarter) in enumerate(quarters, start=1):
        path = archive_dir / f"{year}q{quarter}.zip"
        with zipfile.ZipFile(path) as archive:
            frame = pd.read_csv(
                archive.open("sub.txt"), sep="\t", dtype=str, keep_default_na=False,
                usecols=["adsh", "cik", "sic", "countryba", "form", "period", "filed", "accepted"],
            )
        frame = frame[frame["form"].isin(FORMS)].copy()
        frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce")
        frame["sic"] = pd.to_numeric(frame["sic"], errors="coerce")
        frame["period"] = pd.to_datetime(frame["period"], format="%Y%m%d", errors="coerce")
        frame["filed"] = pd.to_datetime(frame["filed"], format="%Y%m%d", errors="coerce")
        frame["accepted"] = pd.to_datetime(frame["accepted"], errors="coerce")
        frame = frame.dropna(subset=["cik", "period", "filed"])
        frame["cik"] = frame["cik"].astype("int64")
        frame = frame[~frame["sic"].between(*FINANCIAL_SIC, inclusive="both")]
        frames.append(frame)
        print(f"[{number}/{len(quarters)}] loaded SEC SUB metadata {year}q{quarter}: {len(frame):,}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result["accession_normalized"] = result["adsh"].str.replace("-", "", regex=False).str.upper()
    return result


def _companyfacts_for_filer(
    companyfacts_zip: zipfile.ZipFile,
    cik: int,
    target: pd.DataFrame,
    wanted_accessions: set[str],
) -> list[dict]:
    name = f"CIK{cik:010d}.json"
    try:
        payload = json.loads(companyfacts_zip.read(name))
    except KeyError:
        return []
    us_gaap = payload.get("facts", {}).get("us-gaap", {})
    raw: dict[tuple[str, str], list[tuple[int, str, float]]] = {}
    for concept, aliases in TAG_ALIASES.items():
        for rank, tag in enumerate(aliases):
            concept_facts = us_gaap.get(tag)
            if not concept_facts:
                continue
            for unit, records in concept_facts.get("units", {}).items():
                for record in records:
                    accession = str(record.get("accn", "")).replace("-", "").upper()
                    if accession not in wanted_accessions:
                        continue
                    if record.get("start") not in (None, ""):
                        continue  # balance-sheet point-in-time facts only
                    filing = target.loc[accession]
                    if isinstance(filing, pd.DataFrame):
                        filing = filing.iloc[-1]
                    if pd.Timestamp(record.get("end")).normalize() != pd.Timestamp(filing["period"]).normalize():
                        continue
                    try:
                        value = float(record["val"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    raw.setdefault((accession, concept), []).append((rank, unit, value))
    if not raw:
        return []

    by_accession: dict[str, dict[str, list[tuple[int, str, float]]]] = {}
    for (accession, concept), values in raw.items():
        by_accession.setdefault(accession, {})[concept] = values
    output = []
    for accession, concepts in by_accession.items():
        unit_counts = Counter(unit for values in concepts.values() for _, unit, _ in values)
        selected_unit = sorted(unit_counts, key=lambda unit: (-unit_counts[unit], unit != "USD", unit))[0]
        filing = target.loc[accession]
        if isinstance(filing, pd.DataFrame):
            filing = filing.iloc[-1]
        row = {
            "adsh": filing["adsh"],
            "cik": int(filing["cik"]),
            "sic": filing["sic"],
            "countryba": filing["countryba"],
            "form": filing["form"],
            "period": filing["period"],
            "filed": filing["filed"],
            "accepted": filing["accepted"],
            "reporting_unit": selected_unit,
        }
        for concept, values in concepts.items():
            candidates = [item for item in values if item[1] == selected_unit]
            if candidates:
                row[concept] = sorted(candidates, key=lambda item: item[0])[0][2]
        output.append(row)
    return output


def _derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for concept in TAG_ALIASES:
        if concept not in result:
            result[concept] = np.nan
    for concept in TAG_ALIASES:
        result[concept] = pd.to_numeric(result[concept], errors="coerce")
    assets = result["assets"]
    current_assets = result["current_assets"]
    current_liabilities = result["current_liabilities"]
    short_debt = result["short_debt"]
    long_debt = result["long_debt"]
    result["debt_partial"] = (short_debt.isna() ^ long_debt.isna()).astype(float)
    result["total_debt"] = short_debt.fillna(0) + long_debt.fillna(0)
    result.loc[short_debt.isna() & long_debt.isna(), "total_debt"] = np.nan
    # Ratio denominators below $1m are too vulnerable to cents/shell entities,
    # incomplete statements, and tiny denominators. Keep log assets so these
    # firms remain in the panel, but treat their ratios as unavailable.
    positive_assets = assets.where(assets >= 1_000_000)
    positive_current_liabilities = current_liabilities.where(current_liabilities >= 1_000_000)
    valid_current_assets = current_assets.where(current_assets.ge(0) & current_assets.le(assets * 1.05))
    valid_cash = result["cash"].where(result["cash"].ge(0) & result["cash"].le(assets * 1.05))
    valid_liabilities = result["liabilities"].where(result["liabilities"].ge(0))

    def safe_ratio(numerator, denominator, *, nonzero_denominator=False):
        den = denominator.where(denominator.abs() >= 1_000_000) if nonzero_denominator else denominator.where(denominator >= 1_000_000)
        return numerator.div(den).replace([np.inf, -np.inf], np.nan)

    result["log_assets_usd"] = np.log1p(assets.where((assets > 0) & result["reporting_unit"].eq("USD")))
    result["liabilities_to_assets"] = safe_ratio(valid_liabilities, positive_assets)
    result["current_ratio"] = safe_ratio(valid_current_assets, positive_current_liabilities)
    result["cash_to_assets"] = safe_ratio(valid_cash, positive_assets)
    result["debt_to_assets"] = safe_ratio(result["total_debt"], positive_assets)
    result["equity_to_assets"] = safe_ratio(result["equity"], positive_assets)
    result["working_capital_to_assets"] = safe_ratio(valid_current_assets - current_liabilities, positive_assets)
    result["cash_to_current_liabilities"] = safe_ratio(valid_cash, positive_current_liabilities)
    result["debt_to_equity"] = safe_ratio(result["total_debt"], result["equity"], nonzero_denominator=True)
    return result


def build_panel(
    archive_dir: Path,
    companyfacts_path: Path,
    events_path: Path,
    out_dir: Path,
    *,
    start_year: int = 2010,
    end_year: int = 2025,
    end_quarter: int = 4,
    label_as_of: str = "2026-11-17",
) -> dict:
    events = pd.read_csv(events_path, parse_dates=["bankruptcy_date"])
    events["cik"] = pd.to_numeric(events["cik"], errors="raise").astype("int64")
    events = events.drop_duplicates("cik").set_index("cik")["bankruptcy_date"]
    as_of = pd.Timestamp(label_as_of)
    cutoff = _event_maturity_cutoff(as_of)
    metadata = _submissions(archive_dir, start_year, end_year, end_quarter)
    metadata = metadata[metadata["filed"].le(cutoff)].copy()
    # FSDS includes the same submission in a quarterly archive once; this
    # defensive drop protects future archive refreshes with duplicate entries.
    metadata = metadata.sort_values(["cik", "filed", "accepted", "adsh"], kind="stable")
    metadata = metadata.drop_duplicates(["cik", "adsh"], keep="last")
    submissions_by_cik = {int(cik): group.copy() for cik, group in metadata.groupby("cik", sort=False)}
    # Build the accession index once. Rebuilding a 40k-entry set for every
    # filer made the initial implementation needlessly quadratic.
    target = metadata.set_index("accession_normalized")
    wanted_accessions = set(target.index)
    facts_rows = []
    with zipfile.ZipFile(companyfacts_path) as companyfacts_zip:
        for index, (cik, group) in enumerate(submissions_by_cik.items(), start=1):
            facts_rows.extend(_companyfacts_for_filer(companyfacts_zip, cik, target, wanted_accessions))
            if index % 500 == 0 or index == len(submissions_by_cik):
                print(f"Loaded normalized CompanyFacts for {index:,}/{len(submissions_by_cik):,} filers", flush=True)
    if not facts_rows:
        raise ValueError("No normalized CompanyFacts matched the SEC filing metadata")
    filings = _derive_features(pd.DataFrame(facts_rows))
    filings["bankruptcy_date"] = filings["cik"].map(events)
    filings = filings[~(filings["bankruptcy_date"].notna() & filings["filed"].ge(filings["bankruptcy_date"]))].copy()
    filings["y"] = (
        filings["bankruptcy_date"].notna()
        & filings["bankruptcy_date"].gt(filings["filed"])
        & filings["bankruptcy_date"].le(filings["filed"] + pd.DateOffset(months=12))
    ).astype("int8")
    filings = filings.sort_values(["cik", "filed", "accepted", "adsh"], kind="stable")
    filings = filings.drop_duplicates(["cik", "filed"], keep="last")
    filings = filings.rename(columns={"cik": "firm", "filed": "date"})
    filings["sic"] = pd.to_numeric(filings["sic"], errors="coerce").astype("Int64")
    filings["sic2"] = (filings["sic"] // 100).astype("Int64")
    if filings.duplicated(["firm", "date"]).any():
        raise AssertionError("Duplicate firm-date rows after deduplication")
    if not set(filings["y"].unique()).issubset({0, 1}):
        raise AssertionError("Target is not binary")

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "sec_bankruptcy_panel.csv"
    filings.to_csv(panel_path, index=False, date_format="%Y-%m-%d")
    positive = filings[filings["y"].eq(1)]
    archive_manifest = archive_dir / "download_manifest.json"
    companyfacts_manifest = companyfacts_path.with_name("companyfacts_manifest.json")
    submissions_manifest = companyfacts_path.with_name("submissions_manifest.json")
    manifest = {
        "dataset": "SEC EDGAR normalized CompanyFacts bankruptcy panel",
        "version": "1.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "financial_source": "SEC Company Facts bulk API, us-gaap, values normalized by SEC; filing metadata/SIC from SEC Financial Statement Data Sets SUB files.",
        "label_source": str(events_path.resolve()),
        "label_source_sha256": sha256(events_path),
        "event_definition": "First SEC 8-K/8-K-A Item 1.03 bankruptcy date rebuilt from SEC submissions bulk; y=1 if event is after filing date and within 12 months.",
        "feature_availability": "CompanyFacts values are selected by the same accession number and exact reporting-period end date as the FSDS filing snapshot; filing date is the prediction-time boundary.",
        "sec_fsds_scale_audit": "FSDS NUM.value is documented as unscaled and was excluded from model features after sample checks produced impossible asset/liability ratios. Only SUB metadata is used from FSDS.",
        "sources": {
            "fsds_archives_manifest": str(archive_manifest.resolve()),
            "fsds_archives_manifest_sha256": sha256(archive_manifest),
            "companyfacts_archive": str(companyfacts_path.resolve()),
            "companyfacts_archive_sha256": sha256(companyfacts_path),
            "submissions_archive_manifest": str(submissions_manifest.resolve()),
            "submissions_archive_manifest_sha256": sha256(submissions_manifest),
        },
        "cutoffs": {
            "financial_submissions_through": str(cutoff.date()),
            "label_inventory_as_of": label_as_of,
            "mature_12_month_label_cutoff": str(cutoff.date()),
            "horizon_months": 12,
        },
        "features": list(FEATURE_COLUMNS),
        "tag_aliases": TAG_ALIASES,
        "feature_quality_rules": [
            "CompanyFacts instant fact end date must exactly equal the FSDS reporting period date and the accession must exactly match.",
            "Ratio denominators below 1,000,000 reporting-currency units are treated as missing; filing rows and log-assets remain.",
            "Negative liabilities and negative cash are treated as missing for ratios; current assets or cash above 105% of total assets are treated as inconsistent and omitted from the affected ratios.",
            "No arbitrary ratio winsorization is performed during dataset generation; model preprocessing fits clipping bounds on training data only.",
        ],
        "counts": {
            "rows": int(len(filings)),
            "firms": int(filings["firm"].nunique()),
            "positive_rows": int(filings["y"].sum()),
            "positive_firms": int(positive["firm"].nunique()),
            "event_firms_with_companyfacts": int(filings.loc[filings["bankruptcy_date"].notna(), "firm"].nunique()),
            "event_firms_in_source_inventory": int(events.index.nunique()),
            "date_min": str(filings["date"].min().date()),
            "date_max": str(filings["date"].max().date()),
            "duplicate_firm_date_rows": int(filings.duplicated(["firm", "date"]).sum()),
            "rows_missing_all_model_features": int(filings[list(FEATURE_COLUMNS)].isna().all(axis=1).sum()),
            "sic_missing_rows": int(filings["sic"].isna().sum()),
            "usd_reporting_rows": int(filings["reporting_unit"].eq("USD").sum()),
        },
        "limitations": [
            "CompanyFacts includes standard non-custom, entity-wide XBRL facts only; custom taxonomies and dimensional facts are not represented.",
            "CompanyFacts and FSDS are SEC-derived structured data and are not a substitute for checking original filings when a fact is implausible.",
            "Non-event controls are filers with no known Item 1.03 event in the rebuilt submissions inventory; the label means no recorded event, not necessarily economic solvency.",
            "This accounting-only panel does not contain market variables or structural distance-to-default outputs.",
        ],
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    coverage = filings.groupby("firm", as_index=False).agg(
        first_filing=("date", "min"), last_filing=("date", "max"), filing_count=("adsh", "nunique"),
        positive_filings=("y", "sum"), sic2=("sic2", "first"), bankruptcy_date=("bankruptcy_date", "first"),
    )
    coverage.to_csv(out_dir / "firm_coverage.csv", index=False)
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print(f"Wrote {panel_path.resolve()}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=Path("sec_bulk/fsds"))
    parser.add_argument("--companyfacts", type=Path, default=Path("sec_bulk/companyfacts.zip"))
    parser.add_argument("--events", type=Path, default=Path("artifacts/sec_event_validation/sec_item_1_03_events_rebuilt.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sec_companyfacts_bankruptcy_v1"))
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--end-quarter", type=int, default=4)
    parser.add_argument("--label-as-of", default="2026-11-17")
    args = parser.parse_args()
    build_panel(
        args.archives,
        args.companyfacts,
        args.events,
        args.out_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        end_quarter=args.end_quarter,
        label_as_of=args.label_as_of,
    )


if __name__ == "__main__":
    main()
