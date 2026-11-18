"""Run reproducible integrity checks on the normalized SEC CompanyFacts panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_companyfacts_panel import FEATURE_COLUMNS


FEATURES = list(FEATURE_COLUMNS)


def validate(panel_path: Path, manifest_path: Path, event_audit_path: Path, out_path: Path) -> dict:
    frame = pd.read_csv(panel_path, parse_dates=["date", "period", "bankruptcy_date"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_audit = json.loads(event_audit_path.read_text(encoding="utf-8"))
    required = {"firm", "date", "period", "adsh", "y", "bankruptcy_date", "reporting_unit", *FEATURES}
    missing_columns = sorted(required - set(frame.columns))
    failures: list[str] = []
    if missing_columns:
        failures.append(f"missing_columns:{missing_columns}")
    if missing_columns:
        raise ValueError("Panel is missing required columns: " + ", ".join(missing_columns))

    duplicate_rows = int(frame.duplicated(["firm", "date"]).sum())
    positive = frame["y"].eq(1)
    recomputed_positive = (
        frame["bankruptcy_date"].notna()
        & frame["bankruptcy_date"].gt(frame["date"])
        & frame["bankruptcy_date"].le(frame["date"] + pd.DateOffset(months=12))
    )
    invalid_positive = int((positive & ~recomputed_positive).sum())
    missed_positive = int((~positive & recomputed_positive).sum())
    post_event_rows = int((frame["bankruptcy_date"].notna() & frame["date"].ge(frame["bankruptcy_date"])).sum())
    cutoff = pd.Timestamp(manifest["cutoffs"]["financial_submissions_through"])
    after_cutoff = int(frame["date"].gt(cutoff).sum())
    nonfinite_features = int(np.isinf(frame[FEATURES].to_numpy(dtype=float)).sum())
    duplicate_accessions = int(frame.duplicated(["firm", "adsh"]).sum())
    feature_missingness = frame[FEATURES].isna().mean().to_dict()
    ratios = [column for column in FEATURES if column != "debt_partial" and column != "log_assets_usd"]
    quantiles = frame[ratios].quantile([0.01, 0.5, 0.95, 0.99, 1.0]).replace([np.inf, -np.inf], np.nan)

    if duplicate_rows:
        failures.append("duplicate_firm_date_rows")
    if duplicate_accessions:
        failures.append("duplicate_firm_accessions")
    if invalid_positive or missed_positive:
        failures.append("target_does_not_recompute_from_event_dates")
    if post_event_rows:
        failures.append("post_event_observations_present")
    if after_cutoff:
        failures.append("observations_after_label_maturity_cutoff")
    if nonfinite_features:
        failures.append("nonfinite_feature_values")
    if int(manifest["counts"]["rows"]) != len(frame):
        failures.append("manifest_row_count_mismatch")
    if event_audit.get("assessment") != "pass_with_new_sec_events":
        failures.append("sec_event_inventory_audit_not_passing")
    if event_audit.get("date_mismatches") != 0 or event_audit.get("local_accessions_verified") != event_audit.get("local_event_firms"):
        failures.append("event_inventory_date_or_accession_validation_failed")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "panel": str(panel_path.resolve()),
        "dataset_version": manifest.get("version"),
        "checks": {
            "rows": int(len(frame)),
            "firms": int(frame["firm"].nunique()),
            "positive_rows": int(positive.sum()),
            "positive_firms": int(frame.loc[positive, "firm"].nunique()),
            "duplicate_firm_date_rows": duplicate_rows,
            "duplicate_firm_accession_rows": duplicate_accessions,
            "invalid_positive_labels": invalid_positive,
            "missed_positive_labels": missed_positive,
            "post_event_rows": post_event_rows,
            "rows_after_maturity_cutoff": after_cutoff,
            "nonfinite_feature_cells": nonfinite_features,
            "rows_missing_all_features": int(frame[FEATURES].isna().all(axis=1).sum()),
            "event_inventory_assessment": event_audit.get("assessment"),
            "local_accessions_verified": event_audit.get("local_accessions_verified"),
            "sec_only_event_firms": event_audit.get("sec_only_events"),
        },
        "feature_missingness_fraction": feature_missingness,
        "ratio_quantiles": {
            str(level): {column: (None if pd.isna(value) else float(value)) for column, value in row.items()}
            for level, row in quantiles.iterrows()
        },
        "failures": failures,
        "scope_note": "Structural checks cover schema, point-in-time labels, duplicates, numeric finiteness, maturity cutoff, and event inventory accession/date validation. CompanyFacts accession and exact period-date matching are enforced during panel construction; original filing review is still required for a targeted sample of unusual facts.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("artifacts/sec_companyfacts_bankruptcy_v1/sec_bankruptcy_panel.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/sec_companyfacts_bankruptcy_v1/dataset_manifest.json"))
    parser.add_argument("--event-audit", type=Path, default=Path("artifacts/sec_event_validation/sec_bulk_event_audit.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/sec_companyfacts_bankruptcy_v1/validation_report.json"))
    args = parser.parse_args()
    validate(args.panel, args.manifest, args.event_audit, args.out)


if __name__ == "__main__":
    main()
