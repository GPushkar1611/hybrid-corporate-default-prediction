"""Build panel v2: add income-statement, cash-flow, cover-page and structural features.

Input : the v1 SEC filing panel (one row per 10-K/10-Q filing, matched by
        accession) and the SEC Company Facts bulk ZIP.
Output: artifacts/sec_companyfacts_bankruptcy_v2/{sec_bankruptcy_panel.csv,
        dataset_manifest.json}

New facts are taken from the SAME accession as each panel row, so no future
information enters a row. Flow facts (income statement, cash flow) are matched
on the filing's reporting-period end date. A 10-Q reports quarter and
year-to-date values; we take the 12-month value when present, otherwise the
longest duration (year-to-date), and annualise it (x 12 / months).

``--skip-companyfacts`` builds only the filing-metadata and accounting-DD
columns from the v1 panel (useful before the ZIP is downloaded).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_companyfacts_panel import FEATURE_COLUMNS as V1_FEATURES
from src.pipeline.splits import TRAIN_END_YEAR

TRAIN_END = f"{TRAIN_END_YEAR}-12-31"  # sector priors use training rows only
from src.structural.sec_structural import add_accounting_dd, add_public_float_dd

FLOW_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "operating_income": ["OperatingIncomeLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}
INSTANT_TAGS = {
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
}
SIC_DIVISIONS = [  # standard SIC divisions (2-digit ranges)
    (1, 9, "A_agriculture"), (10, 14, "B_mining"), (15, 17, "C_construction"),
    (20, 39, "D_manufacturing"), (40, 49, "E_transport_utilities"), (50, 51, "F_wholesale"),
    (52, 59, "G_retail"), (60, 67, "H_finance"), (70, 89, "I_services"), (91, 99, "J_public_admin"),
]
MIN_DENOMINATOR = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Company Facts extraction
# --------------------------------------------------------------------------
def _months(start: str, end: str) -> float:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.4375


def _extract_firm(payload: dict, rows: pd.DataFrame) -> list[dict]:
    """rows: this firm's panel rows (adsh, period, reporting_unit)."""
    by_accession = {r.adsh: r for r in rows.itertuples(index=False)}
    gaap = payload.get("facts", {}).get("us-gaap", {})
    dei = payload.get("facts", {}).get("dei", {})
    found: dict[str, dict] = {adsh: {"adsh": adsh} for adsh in by_accession}

    def records(block, tag):
        for unit, items in block.get(tag, {}).get("units", {}).items():
            for item in items:
                row = by_accession.get(item.get("accn"))
                if row is not None:
                    yield unit, item, row

    # flow facts: same accession, end == period, choose by alias rank then duration
    for concept, aliases in FLOW_TAGS.items():
        best: dict[str, tuple] = {}
        for rank, tag in enumerate(aliases):
            for unit, item, row in records(gaap, tag):
                if unit != row.reporting_unit or not item.get("start"):
                    continue
                if pd.Timestamp(item["end"]) != row.period:
                    continue
                months = _months(item["start"], item["end"])
                if not 2.5 <= months <= 13.0:
                    continue
                try:
                    value = float(item["val"])
                except (KeyError, TypeError, ValueError):
                    continue
                full_year = 11.0 <= months <= 13.0
                key = (rank, not full_year, -months)  # lower is better
                if row.adsh not in best or key < best[row.adsh][0]:
                    best[row.adsh] = (key, value, months)
        for adsh, (_key, value, months) in best.items():
            found[adsh][concept] = value * 12.0 / months
            found[adsh][f"{concept}_months"] = months

    # instant facts: same accession, no start, end == period
    for concept, aliases in INSTANT_TAGS.items():
        best = {}
        for rank, tag in enumerate(aliases):
            for unit, item, row in records(gaap, tag):
                if unit != row.reporting_unit or item.get("start"):
                    continue
                if pd.Timestamp(item["end"]) != row.period:
                    continue
                if row.adsh not in best or rank < best[row.adsh][0]:
                    best[row.adsh] = (rank, float(item["val"]))
        for adsh, (_rank, value) in best.items():
            found[adsh][concept] = value

    # cover page: public float (USD, as of its own date) and shares outstanding
    for unit, item, row in records(dei, "EntityPublicFloat"):
        if unit != "USD":
            continue
        current = found[row.adsh].get("public_float_date")
        if current is None or item["end"] > current:
            found[row.adsh]["public_float"] = float(item["val"])
            found[row.adsh]["public_float_date"] = item["end"]
    shares: dict[str, dict[str, float]] = {}
    for unit, item, row in records(dei, "EntityCommonStockSharesOutstanding"):
        if unit != "shares":
            continue
        shares.setdefault(row.adsh, {}).setdefault(item["end"], 0.0)
        shares[row.adsh][item["end"]] += float(item["val"])  # several share classes
    for adsh, by_end in shares.items():
        latest = max(by_end)
        found[adsh]["shares_outstanding"] = by_end[latest]
    return list(found.values())


def extract_companyfacts(panel: pd.DataFrame, companyfacts_path: Path) -> pd.DataFrame:
    rows = panel[["firm", "adsh", "period", "reporting_unit"]]
    out = []
    with zipfile.ZipFile(companyfacts_path) as archive:
        names = set(archive.namelist())
        groups = list(rows.groupby("firm", sort=False))
        for index, (firm, group) in enumerate(groups, start=1):
            name = f"CIK{int(firm):010d}.json"
            if name in names:
                out.extend(_extract_firm(json.loads(archive.read(name)), group))
            if index % 1000 == 0 or index == len(groups):
                print(f"extracted extra facts for {index:,}/{len(groups):,} firms", flush=True)
    facts = pd.DataFrame(out)
    if "public_float_date" in facts:
        facts["public_float_date"] = pd.to_datetime(facts["public_float_date"])
    return facts


# --------------------------------------------------------------------------
# derived features
# --------------------------------------------------------------------------
def _ratio(numerator, denominator):
    den = denominator.where(denominator.abs() >= MIN_DENOMINATOR)
    return numerator.div(den).replace([np.inf, -np.inf], np.nan)


def clean_public_float(f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop cover-page public floats that are mis-dated or implausible.

    * float date after the filing date: a tagging error (the value was known at
      filing, but its date would corrupt the float history and age);
    * <= $100k (incl. the 0 reported by firms with no public market), > $5 trillion,
      or > 10,000 x total assets: scale/tagging errors.
    """
    pf, assets = f["public_float"], f["assets"]
    bad_date = pf.notna() & (f["public_float_date"] > f["date"])
    implausible = pf.notna() & (~pf.between(1e5, 5e12) | (pf > 1e4 * assets))
    f = f.copy()
    f.loc[bad_date | implausible, "public_float"] = np.nan
    f.loc[bad_date | implausible, "public_float_date"] = pd.NaT
    return f, {"dropped_misdated": int(bad_date.sum()), "dropped_implausible": int((implausible & ~bad_date).sum())}


def add_flow_features(f: pd.DataFrame) -> pd.DataFrame:
    assets = f["assets"].where(f["assets"] >= MIN_DENOMINATOR)
    liabilities = f["liabilities"].where(f["liabilities"].notna(), f["assets"] - f["equity"])
    f["roa"] = _ratio(f["net_income"], assets)
    f["ebit_to_assets"] = _ratio(f["operating_income"], assets)
    f["sales_to_assets"] = _ratio(f["revenue"], assets)
    f["retained_earnings_to_assets"] = _ratio(f["retained_earnings"], assets)
    f["cfo_to_assets"] = _ratio(f["cfo"], assets)
    interest = f["interest_expense"].where(f["interest_expense"] > 0)
    f["interest_coverage"] = _ratio(f["operating_income"], interest)
    f["loss_flag"] = np.where(f["net_income"].notna(), (f["net_income"] < 0).astype(float), np.nan)

    # carry the latest cover-page public float forward within each firm (point in time)
    f = f.sort_values(["firm", "date", "period"], kind="stable")
    for column in ("public_float", "public_float_date"):
        f[f"{column}_ffill"] = f.groupby("firm")[column].ffill()
    pf = f["public_float_ffill"].where(f["public_float_ffill"] > 0)
    f["log_public_float"] = np.log(pf)
    f["public_float_to_liabilities"] = _ratio(pf, liabilities)  # Altman X4 with float as market equity
    f["public_float_age_days"] = (f["date"] - f["public_float_date_ffill"]).dt.days
    return f


def add_filing_metadata(f: pd.DataFrame) -> pd.DataFrame:
    f = f.sort_values(["firm", "date", "period"], kind="stable")
    f["days_since_prev_filing"] = f.groupby("firm")["date"].diff().dt.days
    f["is_amendment"] = f["form"].str.endswith("/A").astype(float)
    f["is_annual"] = f["form"].str.startswith("10-K").astype(float)
    f["sic_division"] = None
    for low, high, label in SIC_DIVISIONS:
        f.loc[f["sic2"].between(low, high), "sic_division"] = label
    return f


FLOW_FEATURES = [
    "roa", "ebit_to_assets", "sales_to_assets", "retained_earnings_to_assets",
    "cfo_to_assets", "interest_coverage", "loss_flag",
]
MARKET_FEATURES = ["log_public_float", "public_float_to_liabilities", "public_float_age_days"]
FILING_FEATURES = ["days_since_prev_filing", "is_amendment", "is_annual"]
STRUCT_ACCT = ["acct_dd", "acct_pd", "acct_d_va", "acct_sigma_a", "acct_mu"]
STRUCT_ACCT_SENSITIVITY = ["acct_dd_mu0", "acct_dd_debt", "acct_d_va_debt"]
STRUCT_PF = ["pf_naive_dd", "pf_naive_pd", "pf_sigma_e", "pf_e_to_d"]


def build(panel_path: Path, out_dir: Path, companyfacts_path: Path | None) -> dict:
    panel = pd.read_csv(panel_path, parse_dates=["date", "period", "accepted", "bankruptcy_date"])
    n0 = len(panel)
    if companyfacts_path is not None:
        facts = extract_companyfacts(panel, companyfacts_path)
        panel = panel.merge(facts, on="adsh", how="left", validate="one_to_one")
    else:
        for column in [*FLOW_TAGS, *INSTANT_TAGS, "public_float", "shares_outstanding"]:
            panel[column] = np.nan
        panel["public_float_date"] = pd.NaT
    assert len(panel) == n0, "row count changed during merge"
    for column in [*FLOW_TAGS, *INSTANT_TAGS, "public_float", "shares_outstanding"]:
        if column not in panel:
            panel[column] = np.nan
    if "public_float_date" not in panel:
        panel["public_float_date"] = pd.NaT

    panel, public_float_cleaning = clean_public_float(panel)
    panel = add_filing_metadata(panel)
    panel = add_flow_features(panel)
    acct = add_accounting_dd(panel, train_end=TRAIN_END)
    panel = panel.join(acct)
    pf_input = panel.drop(columns=["public_float", "public_float_date"]).rename(
        columns={"public_float_ffill": "public_float", "public_float_date_ffill": "public_float_date"})
    # only the filings that themselves report the float update the history
    pf_input["public_float"] = panel["public_float"]
    pf_input["public_float_date"] = panel["public_float_date"]
    panel = panel.join(add_public_float_dd(pf_input, train_end=TRAIN_END))
    panel = panel.sort_values(["firm", "date"], kind="stable").reset_index(drop=True)

    features = {
        "accounting_base": list(V1_FEATURES),
        "accounting_flow": FLOW_FEATURES,
        "market_cover_page": MARKET_FEATURES,
        "filing_metadata": FILING_FEATURES,
        "structural_accounting": STRUCT_ACCT,
        "structural_accounting_sensitivity": STRUCT_ACCT_SENSITIVITY,
        "structural_public_float": STRUCT_PF,
    }
    all_features = [c for group in features.values() for c in group]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sec_bankruptcy_panel.csv"
    panel.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    manifest = {
        "dataset": "SEC EDGAR bankruptcy panel v2 (v1 + flows, cover page, structural)",
        "version": "2.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_panel": str(panel_path),
        "source_panel_sha256": sha256(panel_path),
        "companyfacts_archive": str(companyfacts_path) if companyfacts_path else None,
        "companyfacts_archive_sha256": sha256(companyfacts_path) if companyfacts_path else None,
        "train_end_for_sector_priors": TRAIN_END,
        "feature_groups": features,
        "flow_tags": FLOW_TAGS,
        "instant_tags": INSTANT_TAGS,
        "rules": [
            "All new facts come from the same accession as the panel row; flows must end on the reporting-period date.",
            "Flow values are annualised: 12-month value if reported, else longest year-to-date duration x 12/months.",
            "Ratio denominators below 1,000,000 are treated as missing (as in v1).",
            "Public float is carried forward from the latest filing that reports it; public_float_age_days records staleness.",
            "Public floats dated after the filing date, <= $100k, > $5T, or > 10,000 x assets are dropped as tagging/scale errors.",
            "Accounting DD: V=book assets, D=CL+0.5*(L-CL), sigma_A from past filings shrunk to SIC2 median (train rows), mu=trailing asset growth.",
            "Public-float DD: Bharath-Shumway naive DD with E=public float, sigma_E from float-implied price history (sector-shrunk), mu=past float-implied return.",
            "Columns ending in _status, _months, _date, _hist_intervals are diagnostics, not predictors.",
        ],
        "counts": {
            "rows": int(len(panel)),
            "firms": int(panel["firm"].nunique()),
            "positive_rows": int(panel["y"].sum()),
            "positive_firms": int(panel.loc[panel["y"].eq(1), "firm"].nunique()),
        },
        "public_float_cleaning": public_float_cleaning,
        "feature_coverage": {c: round(float(panel[c].notna().mean()), 4) for c in all_features},
        "feature_coverage_positive_rows": {
            c: round(float(panel.loc[panel["y"].eq(1), c].notna().mean()), 4) for c in all_features
        },
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote {out_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=Path("artifacts/sec_companyfacts_bankruptcy_v1/sec_bankruptcy_panel.csv"))
    parser.add_argument("--companyfacts", type=Path, default=Path("sec_bulk/companyfacts.zip"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sec_companyfacts_bankruptcy_v2"))
    parser.add_argument("--skip-companyfacts", action="store_true")
    args = parser.parse_args()
    build(args.panel, args.out_dir, None if args.skip_companyfacts else args.companyfacts)


if __name__ == "__main__":
    main()
