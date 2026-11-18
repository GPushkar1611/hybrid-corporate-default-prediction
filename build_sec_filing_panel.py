"""Build a point-in-time bankruptcy panel from SEC Financial Statement Data Sets.

Inputs are the official quarterly SEC FSDS archives, the local SEC-derived
Item 1.03 event inventory, and (optionally) no external price data. Each row is
a nonfinancial registrant's as-filed 10-K/10-Q snapshot dated on its filing
date. Only consolidated facts on the reported balance-sheet date are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
FINANCIAL_SIC = (6000, 6999)
FEATURE_COLUMNS = (
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

# Ordered aliases: the first usable tag in each list is preferred. The SEC
# dataset also includes tag labels, which are used as a fallback for custom
# concepts when no standard taxonomy tag is present in the filing.
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
EXACT_TAGS = {
    tag: (concept, rank)
    for concept, tags in TAG_ALIASES.items()
    for rank, tag in enumerate(tags)
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _custom_concept(tag: str, label: str) -> str | None:
    text = re.sub(r"[^a-z0-9]+", " ", f"{tag} {label}".lower()).strip()
    if "lease" in text and not any(word in text for word in ("finance", "capital")):
        return None
    if "current" in text and "asset" in text:
        return "current_assets"
    if "current" in text and "liabilit" in text:
        return "current_liabilities"
    if "cash and cash equivalent" in text or "cash cash equivalent" in text:
        return "cash"
    if any(word in text for word in ("stockholders equity", "shareholders equity", "shareholders fund")):
        return "equity"
    if "asset" in text and ("total assets" in text or text.endswith(" assets")) and "current" not in text:
        return "assets"
    if "liabilit" in text and "equity" not in text:
        if "current" in text or "short term" in text:
            return "current_liabilities" if "current" in text else None
        if "total liabilit" in text or text in {"liabilities", "total liability"}:
            return "liabilities"
    debt_words = any(word in text for word in ("debt", "borrowings", "notes payable", "commercial paper"))
    if debt_words:
        if "current" in text or "short term" in text:
            return "short_debt"
        if "long term" in text or "noncurrent" in text:
            return "long_debt"
    return None


def _event_maturity_cutoff(as_of: pd.Timestamp, horizon_months: int) -> pd.Timestamp:
    candidate = as_of - pd.DateOffset(months=horizon_months)
    month_end = candidate.to_period("M").to_timestamp("M")
    return month_end if month_end <= candidate else month_end - pd.offsets.MonthEnd(1)


def _read_submissions(archive: zipfile.ZipFile) -> pd.DataFrame:
    frame = pd.read_csv(archive.open("sub.txt"), sep="\t", dtype=str, keep_default_na=False)
    required = {"adsh", "cik", "sic", "countryba", "form", "period", "filed", "accepted"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"SEC SUB file lacks columns: {sorted(missing)}")
    frame = frame[frame["form"].isin(FORMS)].copy()
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce")
    frame["sic"] = pd.to_numeric(frame["sic"], errors="coerce")
    frame["filed"] = pd.to_datetime(frame["filed"], format="%Y%m%d", errors="coerce")
    frame["period"] = pd.to_datetime(frame["period"], format="%Y%m%d", errors="coerce")
    frame["accepted"] = pd.to_datetime(frame["accepted"], errors="coerce")
    frame = frame.dropna(subset=["cik", "filed", "period"])
    frame["cik"] = frame["cik"].astype("int64")
    frame = frame[~frame["sic"].between(*FINANCIAL_SIC, inclusive="both")]
    return frame


def _tag_lookup(archive: zipfile.ZipFile) -> dict[tuple[str, str], tuple[str, int]]:
    tags = pd.read_csv(
        archive.open("tag.txt"), sep="\t", dtype=str, keep_default_na=False,
        usecols=["tag", "version", "custom", "tlabel", "doc"],
    )
    lookup = {}
    for row in tags.itertuples(index=False):
        exact = EXACT_TAGS.get(row.tag)
        if exact:
            lookup[(row.tag, row.version)] = exact
        elif row.custom == "1":
            concept = _custom_concept(row.tag, f"{row.tlabel} {row.doc}")
            if concept:
                lookup[(row.tag, row.version)] = (concept, 100)
    return lookup


def _filing_facts(archive: zipfile.ZipFile, submissions: pd.DataFrame, tags) -> pd.DataFrame:
    accession_to_submission = submissions.set_index("adsh")[["cik", "sic", "filed", "period", "accepted", "form", "countryba"]]
    accessions = set(submissions["adsh"])
    columns = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "value", "coreg"]
    numeric_columns = pd.read_csv(archive.open("num.txt"), sep="\t", nrows=0).columns
    if "segments" in numeric_columns:
        columns.append("segments")
    pieces = []
    for chunk in pd.read_csv(
        archive.open("num.txt"), sep="\t", dtype={"adsh": str, "tag": str, "version": str, "ddate": str, "qtrs": str, "uom": str, "coreg": str},
        keep_default_na=False, usecols=columns, chunksize=750_000,
    ):
        chunk = chunk[chunk["adsh"].isin(accessions)]
        if chunk.empty:
            continue
        chunk["qtrs"] = pd.to_numeric(chunk["qtrs"], errors="coerce")
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk[chunk["qtrs"].eq(0) & chunk["value"].notna() & chunk["coreg"].eq("")]
        if "segments" in chunk:
            chunk = chunk[chunk["segments"].eq("")]
        if chunk.empty:
            continue
        chunk["tag_map"] = [tags.get((tag, version)) for tag, version in zip(chunk["tag"], chunk["version"])]
        chunk = chunk[chunk["tag_map"].notna()].copy()
        if chunk.empty:
            continue
        chunk["concept"] = chunk["tag_map"].map(lambda pair: pair[0])
        chunk["tag_rank"] = chunk["tag_map"].map(lambda pair: pair[1])
        chunk = chunk.merge(accession_to_submission, left_on="adsh", right_index=True, how="inner")
        chunk["ddate"] = pd.to_datetime(chunk["ddate"], format="%Y%m%d", errors="coerce")
        chunk = chunk[chunk["ddate"].dt.to_period("M").eq(chunk["period"].dt.to_period("M"))]
        if not chunk.empty:
            pieces.append(chunk[["adsh", "tag", "concept", "tag_rank", "uom", "value", "cik", "sic", "filed", "period", "accepted", "form", "countryba"]])
    if not pieces:
        return pd.DataFrame()
    facts = pd.concat(pieces, ignore_index=True)
    # The same reporting entity should use one monetary unit in a filing.
    # Select the most common unit, with USD preferred in ties when present.
    unit_counts = facts.groupby(["adsh", "uom"]).size().rename("n").reset_index()
    unit_counts["usd_tie"] = unit_counts["uom"].eq("USD")
    unit_counts = unit_counts.sort_values(["adsh", "n", "usd_tie", "uom"], ascending=[True, False, False, True])
    units = unit_counts.drop_duplicates("adsh").set_index("adsh")["uom"]
    facts = facts[facts["uom"].eq(facts["adsh"].map(units))]
    facts = facts.sort_values(["adsh", "concept", "tag_rank", "tag"], kind="stable")
    facts = facts.drop_duplicates(["adsh", "concept"], keep="first")
    wide = facts.pivot(index="adsh", columns="concept", values="value")
    custom = facts.assign(is_custom=facts["tag_rank"].eq(100)).pivot(
        index="adsh", columns="concept", values="is_custom"
    )
    custom.columns = [f"_custom_{name}" for name in custom.columns]
    return wide.join(custom).join(accession_to_submission, how="left").reset_index()


def _make_features(filings: pd.DataFrame) -> pd.DataFrame:
    result = filings.copy()
    for concept in TAG_ALIASES:
        if concept not in result:
            result[concept] = np.nan

    def raw(name: str) -> pd.Series:
        return pd.to_numeric(result[name], errors="coerce")

    assets = raw("assets")
    current_debt = raw("short_debt")
    long_debt = raw("long_debt")
    result["debt_partial"] = (current_debt.isna() ^ long_debt.isna()).astype(float)
    result["total_debt"] = current_debt.fillna(0) + long_debt.fillna(0)
    result.loc[current_debt.isna() & long_debt.isna(), "total_debt"] = np.nan

    def ratio(numerator, denominator):
        top = raw(numerator) if isinstance(numerator, str) else numerator
        bottom = raw(denominator) if isinstance(denominator, str) else denominator
        return top.div(bottom.where(bottom.abs() > 1e-12)).replace([np.inf, -np.inf], np.nan)

    result["liabilities_to_assets"] = ratio("liabilities", assets)
    result["current_ratio"] = ratio("current_assets", "current_liabilities")
    result["cash_to_assets"] = ratio("cash", assets)
    result["debt_to_assets"] = ratio("total_debt", assets)
    result["equity_to_assets"] = ratio("equity", assets)
    result["working_capital_to_assets"] = ratio(
        raw("current_assets") - raw("current_liabilities"), assets
    )
    result["cash_to_current_liabilities"] = ratio("cash", "current_liabilities")
    result["debt_to_equity"] = ratio("total_debt", "equity")
    return result


def build_panel(
    archive_dir: Path,
    events_path: Path,
    out_dir: Path,
    *,
    start_year: int = 2010,
    end_year: int = 2025,
    end_quarter: int = 4,
    label_as_of: str | None = None,
) -> dict:
    events = pd.read_csv(events_path, parse_dates=["bankruptcy_date"])
    events["cik"] = pd.to_numeric(events["cik"], errors="raise").astype("int64")
    events = events.drop_duplicates("cik").set_index("cik")["bankruptcy_date"]
    if events.isna().any():
        raise ValueError("Event inventory contains missing bankruptcy dates")
    if label_as_of is None:
        label_as_of = str(events.max().date())
    as_of = pd.Timestamp(label_as_of)
    cutoff = _event_maturity_cutoff(as_of, 12)
    quarters = [
        (year, quarter)
        for year in range(start_year, end_year + 1)
        for quarter in range(1, (end_quarter if year == end_year else 4) + 1)
    ]
    filing_frames = []
    counters = Counter()
    for index, (year, quarter) in enumerate(quarters, start=1):
        archive_path = archive_dir / f"{year}q{quarter}.zip"
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing SEC archive: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            submissions = _read_submissions(archive)
            counters["eligible_submissions_before_cutoff"] += int((submissions["filed"] <= cutoff).sum())
            submissions = submissions[submissions["filed"] <= cutoff]
            counters["eligible_ciks"] += int(submissions["cik"].nunique())
            mapped_tags = _tag_lookup(archive)
            facts = _filing_facts(archive, submissions, mapped_tags)
            if not facts.empty:
                filing_frames.append(facts)
        print(f"[{index}/{len(quarters)}] parsed {year}q{quarter}: {len(submissions):,} candidate submissions", flush=True)
    if not filing_frames:
        raise ValueError("No eligible financial-statement submissions/facts found")
    filings = pd.concat(filing_frames, ignore_index=True)
    filings = _make_features(filings)
    # Filing date is the prediction-time boundary. Drop observations after
    # the last date for which a full 12-month event horizon is observable.
    filings = filings[filings["filed"] <= cutoff]
    filings["bankruptcy_date"] = filings["cik"].map(events)
    filings = filings[~(filings["bankruptcy_date"].notna() & filings["filed"].ge(filings["bankruptcy_date"]))].copy()
    filings["y"] = (
        filings["bankruptcy_date"].notna()
        & filings["bankruptcy_date"].gt(filings["filed"])
        & filings["bankruptcy_date"].le(filings["filed"] + pd.DateOffset(months=12))
    ).astype("int8")
    # A company can file multiple qualifying statements on one day. Preserve
    # the last accepted submission so each entity-date is a unique sequence step.
    filings = filings.sort_values(["cik", "filed", "accepted", "adsh"], kind="stable")
    filings = filings.drop_duplicates(["cik", "filed"], keep="last")
    filings = filings.rename(columns={"cik": "firm", "filed": "date"})
    filings["sic"] = pd.to_numeric(filings["sic"], errors="coerce").astype("Int64")
    filings["sic2"] = (filings["sic"] // 100).astype("Int64")
    if filings.duplicated(["firm", "date"]).any():
        raise AssertionError("Output panel has duplicate firm-date rows")
    if not set(filings["y"].unique()).issubset({0, 1}):
        raise AssertionError("Output target is not binary")

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "sec_bankruptcy_panel.csv"
    filings.to_csv(panel_path, index=False, date_format="%Y-%m-%d")
    panel_events = set(filings.loc[filings["bankruptcy_date"].notna(), "firm"].unique())
    positives = filings.loc[filings["y"].eq(1)]
    archive_manifest_path = archive_dir / "download_manifest.json"
    if not archive_manifest_path.exists():
        raise FileNotFoundError(f"Missing archive hash manifest: {archive_manifest_path}")
    manifest = {
        "dataset": "SEC EDGAR as-filed bankruptcy prediction panel",
        "version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": "https://www.sec.gov/dera/data/financial-statement-data-sets.html",
        "financial_source": "SEC Financial Statement Data Sets NUM/SUB/TAG; only as-filed consolidated primary-statement facts",
        "financial_archive_manifest": str(archive_manifest_path),
        "financial_archive_manifest_sha256": sha256(archive_manifest_path),
        "event_source": str(events_path.resolve()),
        "event_source_sha256": sha256(events_path),
        "event_definition": "First SEC Item 1.03 event in local SEC submissions-derived event table; y=1 when event is after filing date and within 12 months.",
        "feature_availability": "Filing facts are from the same SEC filing; observation date is EDGAR filed date. No subsequent restatements are joined to earlier observations.",
        "target_horizon_months": 12,
        "label_as_of": str(as_of.date()),
        "mature_label_observation_cutoff": str(cutoff.date()),
        "financial_sic_exclusion": list(FINANCIAL_SIC),
        "features": list(FEATURE_COLUMNS),
        "custom_tag_fallback": "Label/definition heuristics, only when no standard tag mapping has priority; all mapping rules are recorded in source code and raw facts remain reproducible from archives.",
        "custom_mapped_fact_counts_by_concept": {
            concept: int(filings.get(f"_custom_{concept}", pd.Series(dtype=bool)).fillna(False).sum())
            for concept in TAG_ALIASES
        },
        "mapped_fact_coverage_by_concept": {
            concept: int(filings.get(concept, pd.Series(dtype=float)).notna().sum())
            for concept in TAG_ALIASES
        },
        "counts": {
            "rows": int(len(filings)),
            "firms": int(filings["firm"].nunique()),
            "positive_rows": int(filings["y"].sum()),
            "positive_firms": int(positives["firm"].nunique()),
            "event_firms_with_financial_statements": int(len(panel_events)),
            "event_firms_in_event_table": int(events.index.nunique()),
            "event_firms_with_positive_observations": int(positives["firm"].nunique()),
            "date_min": str(filings["date"].min().date()),
            "date_max": str(filings["date"].max().date()),
            "duplicate_firm_date_rows": int(filings.duplicated(["firm", "date"]).sum()),
            "rows_missing_all_features": int(filings[list(FEATURE_COLUMNS)].isna().all(axis=1).sum()),
            "sic_missing_rows": int(filings["sic"].isna().sum()),
            "structural_source_rows": 0,
        },
        "limitations": [
            "The local bankruptcy event table is derived from SEC submissions but must be independently revalidated against the SEC submissions API before a final claim.",
            "Controls are SEC filers without a known Item 1.03 event in the local event inventory; an unobserved or incomplete event label can create false negatives.",
            "SEC Financial Statement Data Sets are as-filed primary-statement extracts and can contain redundancies or extraction errors; see SEC documentation and inspect sample filings.",
            "Financial ratios use label-based custom-tag fallback; the normalization audit and coverage report must be reviewed before modeling.",
            "This accounting-only panel does not contain market variables or teammate structural distance-to-default features.",
        ],
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    coverage = (
        filings.groupby("firm", as_index=False)
        .agg(first_filing=("date", "min"), last_filing=("date", "max"), filings=("adsh", "nunique"),
             positive_snapshots=("y", "sum"), sic2=("sic2", "first"), bankruptcy_date=("bankruptcy_date", "first"))
    )
    coverage.to_csv(out_dir / "firm_coverage.csv", index=False)
    (out_dir / "README.md").write_text(
        "# SEC EDGAR filing-level bankruptcy panel\n\n"
        "This dataset uses SEC Financial Statement Data Sets as its financial source. "
        "Rows are 10-K/10-Q filing snapshots on the filing date, with consolidated "
        "balance-sheet facts from that same filing. The target uses the local SEC-derived "
        "8-K Item 1.03 event table, with a 12-month horizon. See `dataset_manifest.json` "
        "for source hashes, cutoffs, features, counts, and limitations.\n\n"
        "`sec_bankruptcy_panel.csv` includes predictor columns, metadata (`firm`, `date`, "
        "`sic`, `sic2`, `form`, `period`, `adsh`), target `y`, and event date. Use only "
        f"`{', '.join(FEATURE_COLUMNS)}` as predictors. Industry and event metadata are not predictors.\n\n"
        "Do not treat model metrics as final until event labels and tag mapping have been "
        "audited and the evaluation protocol is predeclared.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print(f"Wrote {panel_path.resolve()}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=Path("sec_bulk/fsds"))
    parser.add_argument("--events", type=Path, default=Path("bankruptcies.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sec_fsds_bankruptcy_v1"))
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--end-quarter", type=int, default=4)
    parser.add_argument("--label-as-of", help="Latest date through which the event inventory is known complete")
    args = parser.parse_args()
    build_panel(
        args.archives, args.events, args.out_dir,
        start_year=args.start_year, end_year=args.end_year,
        end_quarter=args.end_quarter, label_as_of=args.label_as_of,
    )


if __name__ == "__main__":
    main()
