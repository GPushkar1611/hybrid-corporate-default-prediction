"""Audit and package the SEC EDGAR bankruptcy panel.

Works from the repository's SEC-derived label tables and an already-built
firm-month panel. It does not download the multi-gigabyte SEC bulk archives.
See ``--help`` for the audit and build commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PANEL_REQUIRED = {
    "firm", "date", "D", "r", "debt_age_days", "debt_partial", "y",
    "bankrupt_firm", "E", "sigma_E", "kmv_status",
}
LABEL_REQUIRED = {"cik", "bankruptcy_date"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def inspect_inputs(
    panel_path: Path,
    bankruptcies_path: Path,
    covered_path: Path,
    controls_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    bankruptcies = pd.read_csv(bankruptcies_path, parse_dates=["bankruptcy_date"])
    covered = pd.read_csv(covered_path, parse_dates=["bankruptcy_date"])
    controls = pd.read_csv(controls_path)
    _require_columns(panel, PANEL_REQUIRED, "Panel")
    _require_columns(bankruptcies, LABEL_REQUIRED, "Bankruptcy labels")
    _require_columns(covered, LABEL_REQUIRED, "Covered bankruptcy labels")
    _require_columns(controls, {"cik", "ticker"}, "Controls")

    panel["firm"] = pd.to_numeric(panel["firm"], errors="raise").astype("int64")
    panel["y"] = pd.to_numeric(panel["y"], errors="raise").astype("int64")
    panel["bankrupt_firm"] = pd.to_numeric(panel["bankrupt_firm"], errors="raise").astype("int64")
    covered["cik"] = pd.to_numeric(covered["cik"], errors="raise").astype("int64")
    bankruptcies["cik"] = pd.to_numeric(bankruptcies["cik"], errors="raise").astype("int64")
    controls["cik"] = pd.to_numeric(controls["cik"], errors="raise").astype("int64")

    if panel.empty:
        raise ValueError("Panel is empty.")
    if panel["date"].isna().any():
        raise ValueError("Panel has unparseable/missing dates.")
    if not set(panel["y"].unique()).issubset({0, 1}):
        raise ValueError("Panel y must be binary (0/1).")
    if panel.duplicated(["firm", "date"]).any():
        raise ValueError("Panel contains duplicate firm-date rows.")

    # Use all SEC-derived events. Using only covered firms lets known failures
    # with weak fundamentals slip into the sampled control set.
    event_dates = bankruptcies.drop_duplicates("cik").set_index("cik")["bankruptcy_date"]
    event_for_row = panel["firm"].map(event_dates)
    expected_bankrupt = event_for_row.notna().astype("int64")
    expected_y = (
        event_for_row.notna()
        & (event_for_row > panel["date"])
        & (event_for_row <= panel["date"] + pd.DateOffset(months=12))
    ).astype("int64")
    # Audit the source panel before correcting control contamination and rows
    # that occur after a known bankruptcy event.
    y_mismatch = int((panel["y"] != expected_y).sum())
    bankrupt_flag_mismatch = int((panel["bankrupt_firm"] != expected_bankrupt).sum())
    source_panel_rows = int(len(panel))
    source_panel_firms = int(panel["firm"].nunique())
    source_firms = set(panel["firm"].unique())
    source_panel_counts = panel["firm"].value_counts()
    post_event_mask = event_for_row.notna() & (panel["date"] >= event_for_row)
    post_event_rows = int(post_event_mask.sum())
    panel = panel.loc[~post_event_mask].copy()
    panel["y"] = expected_y.loc[~post_event_mask].to_numpy()
    panel["bankrupt_firm"] = expected_bankrupt.loc[~post_event_mask].to_numpy()

    by_firm = panel.groupby("firm", as_index=False).agg(
        first_observation=("date", "min"),
        last_observation=("date", "max"),
        n_months=("date", "size"),
        positive_months=("y", "sum"),
        bankrupt_firm=("bankrupt_firm", "max"),
        kmv_nonzero_rows=("kmv_status", lambda s: int((pd.to_numeric(s, errors="coerce") != 0).sum())),
    )
    by_firm["bankruptcy_date"] = by_firm["firm"].map(event_dates)
    by_firm["days_event_after_last_observation"] = (
        by_firm["bankruptcy_date"] - by_firm["last_observation"]
    ).dt.days
    by_firm["no_positive_reason"] = ""
    is_event = by_firm["bankruptcy_date"].notna()
    no_positive = is_event & by_firm["positive_months"].eq(0)
    right_censored = no_positive & (
        by_firm["bankruptcy_date"] > by_firm["last_observation"] + pd.DateOffset(months=12)
    )
    by_firm.loc[right_censored, "no_positive_reason"] = "event occurs more than 12 months after final panel observation"
    by_firm.loc[no_positive & ~right_censored, "no_positive_reason"] = "requires investigation: no positive row despite event within observable range"
    by_firm.loc[~is_event, "no_positive_reason"] = "no event in full EDGAR event table"

    panel_firms = set(panel["firm"].unique())
    event_firms = set(bankruptcies["cik"].unique())
    control_firms = set(controls["cik"].unique())
    # Audit the full EDGAR event inventory and trace each event through the
    # fundamentals, price-coverage, and modelling-panel filters.
    event_inventory = bankruptcies.drop_duplicates("cik").merge(
        covered.drop_duplicates("cik"), on="cik", how="left", suffixes=("", "_covered")
    )
    panel_counts = by_firm.set_index("firm")
    event_inventory["source_panel_firm"] = event_inventory["cik"].isin(source_firms)
    event_inventory["panel_firm"] = event_inventory["cik"].isin(panel_counts.index)
    event_inventory["source_panel_rows"] = event_inventory["cik"].map(
        source_panel_counts
    ).fillna(0).astype("int64")
    event_inventory["panel_positive_months"] = event_inventory["cik"].map(
        panel_counts["positive_months"]
    ).fillna(0).astype("int64")
    event_inventory["panel_last_observation"] = event_inventory["cik"].map(
        panel_counts["last_observation"]
    )
    event_inventory["coverage_reason"] = ""
    has_coverage_record = event_inventory["price_status"].notna()
    removed_post_event = event_inventory["source_panel_firm"] & ~event_inventory["panel_firm"]
    event_inventory.loc[removed_post_event, "coverage_reason"] = (
        "source panel rows were all on/after known EDGAR event date; excluded from corrected panel"
    )
    event_inventory.loc[~has_coverage_record & ~removed_post_event, "coverage_reason"] = (
        "not in price-coverage table; inspect fund_status and upstream run"
    )
    event_inventory.loc[has_coverage_record & event_inventory["price_status"].ne("ok") & ~removed_post_event, "coverage_reason"] = (
        "price coverage failed: " + event_inventory.loc[
            has_coverage_record & event_inventory["price_status"].ne("ok"), "price_status"
        ].astype(str)
    )
    event_inventory.loc[
        has_coverage_record & event_inventory["price_status"].eq("ok") & ~event_inventory["panel_firm"] & ~removed_post_event,
        "coverage_reason",
    ] = "price coverage passed but firm is absent from current panel; upstream collection/build log needed"
    event_inventory.loc[event_inventory["panel_firm"], "coverage_reason"] = "included in current modelling panel"

    files = [panel_path, bankruptcies_path, covered_path, controls_path]
    summary: dict[str, Any] = {
        "audit_version": 1,
        "inputs": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
        "label_origin": {
            "available_artifacts": [str(bankruptcies_path), str(covered_path)],
            "event_definition": "First SEC 8-K or 8-K/A with Item 1.03 (Bankruptcy or Receivership), beginning 2010; bankruptcy_date uses reportDate when not after filingDate, otherwise filingDate.",
            "raw_sec_archives_present": any(Path(name).exists() for name in (
                "sec_bulk/submissions.zip", "sec_bulk/companyfacts.zip"
            )),
            "fresh_sec_download_performed": False,
            "note": "This run audits repository artifacts generated from SEC data; it does not independently re-fetch or re-verify SEC records.",
        },
        "counts": {
            "bankruptcy_label_rows": int(len(bankruptcies)),
            "bankruptcy_firms": int(bankruptcies["cik"].nunique()),
            "fundamentals_status_counts": {
                str(k): int(v) for k, v in bankruptcies.get("fund_status", pd.Series(dtype=object)).value_counts(dropna=False).items()
            },
            "covered_label_rows": int(len(covered)),
            "covered_price_status_counts": {str(k): int(v) for k, v in covered.get("price_status", pd.Series(dtype=object)).value_counts(dropna=False).items()},
            "control_rows": int(len(controls)),
            "known_event_firms_in_control_sample": int(len(set(controls["cik"]) & set(bankruptcies["cik"]))),
            "source_panel_rows": source_panel_rows,
            "source_panel_firms": source_panel_firms,
            "panel_rows": int(len(panel)),
            "panel_firms": int(panel["firm"].nunique()),
            "panel_date_min": str(panel["date"].min().date()),
            "panel_date_max": str(panel["date"].max().date()),
            "positive_month_rows": int(panel["y"].sum()),
            "positive_firms": int(panel.loc[panel["y"].eq(1), "firm"].nunique()),
            "panel_event_firms": int(panel.loc[panel["bankrupt_firm"].eq(1), "firm"].nunique()),
            "control_firms_in_panel": int(panel.loc[panel["bankrupt_firm"].eq(0), "firm"].nunique()),
            "event_firms_in_panel": int(len(panel_firms & event_firms)),
            "source_event_firms_in_panel": int(len(source_firms & event_firms)),
            "price_eligible_event_firms_absent_from_panel": int((
                event_inventory["price_status"].eq("ok") & ~event_inventory["panel_firm"]
            ).sum()),
            "event_firms_with_no_positive_rows": int((no_positive & is_event).sum()),
            "event_firms_right_censored_beyond_12_months": int(right_censored.sum()),
            "kmv_nonzero_status_rows": int((pd.to_numeric(panel["kmv_status"], errors="coerce") != 0).sum()),
            "post_event_source_rows_removed": post_event_rows,
            "duplicate_firm_date_rows": int(panel.duplicated(["firm", "date"]).sum()),
            "source_target_mismatch_rows_against_full_edgar_inventory": y_mismatch,
            "source_bankrupt_flag_mismatch_rows_against_full_edgar_inventory": bankrupt_flag_mismatch,
            "corrected_target_mismatch_rows_against_full_edgar_inventory": int((panel["y"] != (
                panel["firm"].map(event_dates).notna()
                & (panel["firm"].map(event_dates) > panel["date"])
                & (panel["firm"].map(event_dates) <= panel["date"] + pd.DateOffset(months=12))
            ).astype("int64")).sum()),
        },
        "limitations": [
            "Controls were sampled from the current SEC ticker list and are survivor-selected by construction.",
            "The original control draw excluded only the fundamentals-covered bankruptcy subset. This builder cross-checks every sampled firm against the full EDGAR event table, corrects the event flag, and removes observations on/after known events.",
            "Panel construction requires a usable historical ticker and market-price history; many SEC bankruptcy events are therefore absent from the modelling panel.",
            "Source panel ends before current label artifacts; events after its observation range cannot contribute positive rows.",
            "Raw SEC bulk archives are not present in the repository, so the source artifacts cannot be independently rebuilt from this checkout alone.",
        ],
    }
    return summary, panel, by_firm, event_inventory


def write_audit(
    summary: dict[str, Any],
    by_firm: pd.DataFrame,
    event_inventory: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_firm.to_csv(out_dir / "firm_label_coverage.csv", index=False)
    event_inventory.to_csv(out_dir / "edgar_event_inventory.csv", index=False)


def build_dataset(
    summary: dict[str, Any],
    panel: pd.DataFrame,
    by_firm: pd.DataFrame,
    event_inventory: pd.DataFrame,
    out_dir: Path,
    input_paths: dict[str, Path],
) -> None:
    write_audit(summary, by_firm, event_inventory, out_dir)
    panel = panel.sort_values(["firm", "date"]).reset_index(drop=True)
    panel.to_csv(out_dir / "bankruptcy_panel.csv", index=False, date_format="%Y-%m-%d")

    manifest = {
        "dataset_name": "SEC EDGAR bankruptcy prediction panel",
        "dataset_version": "0.1.0",
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "files": {
            "bankruptcy_panel.csv": "Firm-month modelling panel. One row per firm and month-end.",
            "edgar_event_inventory.csv": "All bankruptcy event rows available in the local SEC-derived bankruptcy label table, with coverage path through price checks and panel inclusion.",
            "firm_label_coverage.csv": "Per-firm observations and 12-month label coverage for firms present in the modelling panel.",
            "audit.json": "Machine-readable counts, input hashes, consistency checks, origin, and limitations.",
            "README.md": "Human-readable data dictionary and construction notes.",
        },
        "source_artifacts": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in input_paths.items()
        },
        "origin": {
            "event_source": "Repository SEC-derived event table. The upstream edgar_labels.py describes SEC submissions bulk data (submissions.zip) and companyfacts bulk data (companyfacts.zip).",
            "fresh_sec_fetch": False,
            "raw_sec_archives_available_in_checkout": summary["label_origin"]["raw_sec_archives_present"],
            "event_rule": summary["label_origin"]["event_definition"],
            "feature_source": "Previously generated panel.csv; point-in-time firm data assembled by data.py and build_panel.py from SEC fundamentals/shares, Yahoo Finance prices, and FRED DGS1 rates.",
        },
        "target": {
            "column": "y",
            "meaning": "1 when the first SEC Item 1.03 event date is after the observation date and no more than 12 calendar months later; otherwise 0.",
            "censoring": "Panel observations end 2024-12-30. The event inventory extends later; events more than 12 months after a firm's final panel observation do not create positive rows in this extract.",
        },
        "keys": ["firm", "date"],
        "target_column": "y",
        "identifier_and_diagnostic_columns_not_for_model_features": [
            "firm", "date", "y", "bankrupt_firm", "*_status",
        ],
        "known_limitations": summary["limitations"],
        "input_files": {key: str(path.resolve()) for key, path in input_paths.items()},
        "counts": summary["counts"],
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = f"""# SEC EDGAR bankruptcy panel, v0.1.0

## Contents

- `bankruptcy_panel.csv`: firm-month modelling panel, {len(panel):,} rows and {panel['firm'].nunique():,} firms.
- `edgar_event_inventory.csv`: all {summary['counts']['bankruptcy_firms']:,} event firms in the local SEC-derived label table, including coverage outcomes.
- `firm_label_coverage.csv`: firm-level observation period, positive-month count, and no-positive explanation.
- `audit.json` and `dataset_manifest.json`: checks, source hashes, definitions, and provenance.

## Event and target definitions

An event is the first 8-K or 8-K/A filing with SEC Item 1.03 (Bankruptcy or Receivership) since 2010. The upstream label code uses `reportDate` if it is valid and not after `filingDate`; otherwise it uses `filingDate`. For a firm-month observation at date *t*, `y=1` if the event date is in *(t, t + 12 months]*. Rows at or after the event date are not included in the panel.

## Construction and provenance

This artifact packages the existing SEC-derived label tables and the existing `panel.csv`. It does not download or re-parse SEC bulk archives during this run. The event inventory records how the 990 local event labels pass through fundamentals and price coverage; the modelling panel has a narrower market-data eligibility filter. See the manifest for SHA-256 hashes and input paths.

The panel features originate from the point-in-time routines in `data.py` and `build_panel.py`: SEC XBRL debt and share facts, Yahoo Finance prices, and the FRED DGS1 one-year Treasury series. SEC-reported facts are used only after filing availability; prices are split-adjusted and not dividend-adjusted.

## Rebuild

Run from the project root in the specified virtual environment:

```powershell
& 'D:\\IITH\\Sem 5\\AI3005 - AI Project\\.venv_B\\Scripts\\python.exe' dataset_generation.py --mode audit --panel 'C:\\Users\\vijay\\Downloads\\panel.csv'
& 'D:\\IITH\\Sem 5\\AI3005 - AI Project\\.venv_B\\Scripts\\python.exe' dataset_generation.py --mode build --panel 'C:\\Users\\vijay\\Downloads\\panel.csv'
```

## Current coverage and limitations

- {summary['counts']['bankruptcy_firms']:,} bankruptcy labels are in the local event inventory; {summary['counts']['covered_label_rows']:,} have a fundamentals/price-check record.
- {summary['counts']['covered_price_status_counts'].get('ok', 0):,} firms pass the saved price-coverage check; {summary['counts']['event_firms_in_panel']:,} event firms appear in the modelling panel.
- EDGAR label fundamentals statuses: {', '.join(f"{key}={value:,}" for key, value in summary['counts']['fundamentals_status_counts'].items())}. The parser currently checks a limited list of standard US-GAAP debt concepts; `no_debt` can reflect missing usable debt facts or tag coverage, not necessarily zero debt. SEC Company Facts APIs aggregate standard taxonomies and omit issuer custom taxonomies.
- The saved control table contains {summary['counts']['known_event_firms_in_control_sample']:,} firms that also occur in the full EDGAR event inventory. The modelling-panel builder now excludes the full event inventory from controls on future rebuilds; this packaging step also reconciles any such firm against the event dates.
- The panel contains {summary['counts']['positive_month_rows']:,} positive firm-months from {summary['counts']['positive_firms']:,} firms. {summary['counts']['event_firms_with_no_positive_rows']:,} event firms have no positive rows because their event falls more than 12 months after their final observation.
- The source panel had {summary['counts']['source_bankrupt_flag_mismatch_rows_against_full_edgar_inventory']:,} firm-month rows whose `bankrupt_firm` flag disagreed with the full EDGAR event table. This builder removes {summary['counts']['post_event_source_rows_removed']:,} post-event rows and recomputes the event flag from the full inventory. The corrected target has {summary['counts']['corrected_target_mismatch_rows_against_full_edgar_inventory']:,} mismatches against that inventory.
- Controls were sampled from the current SEC ticker list and therefore favor surviving registrants. The builder removes firms with a known event from the control set when such firms appear in the EDGAR inventory. Many failed firms lack historical tickers or price coverage; the modelling panel is not a representative sample of all EDGAR bankruptcy events.
- The raw SEC bulk archives are not included here. Rebuilding the label inventory from primary SEC files requires the upstream bulk archives and a valid SEC User-Agent in `edgar_labels.py`.
- Nonzero structural-model status rows indicate invalid or insufficient inputs; inspect the corresponding `*_status` fields before selecting structural features.

Do not use `y`, `bankrupt_firm`, `firm`, `date`, or status fields as ordinary predictors. This is a dataset construction artifact, not evidence of model performance.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "build"), default="audit")
    parser.add_argument("--panel", type=Path, required=True, help="Existing firm-month panel CSV")
    parser.add_argument("--bankruptcies", type=Path, default=Path("bankruptcies.csv"))
    parser.add_argument("--covered", type=Path, default=Path("bankruptcies_covered.csv"))
    parser.add_argument("--controls", type=Path, default=Path("controls.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/edgar_bankruptcy_v1"))
    args = parser.parse_args()
    input_paths = {
        "panel": args.panel,
        "bankruptcies": args.bankruptcies,
        "covered": args.covered,
        "controls": args.controls,
    }
    summary, panel, by_firm, event_inventory = inspect_inputs(
        args.panel, args.bankruptcies, args.covered, args.controls
    )
    if args.mode == "build":
        build_dataset(summary, panel, by_firm, event_inventory, args.out_dir, input_paths)
    else:
        write_audit(summary, by_firm, event_inventory, args.out_dir)
    print(json.dumps(summary["counts"], indent=2))
    action = "Dataset" if args.mode == "build" else "Audit"
    print(f"{action} artifacts written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
