"""Rebuild Item 1.03 bankruptcy events from the official SEC submissions bulk ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_pipeline.sec_labels import _hits, build_bankruptcy_table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(submissions_path: Path, local_events_path: Path, out_dir: Path) -> dict:
    rebuilt = build_bankruptcy_table(submissions_path, since="2010-01-01")
    local = pd.read_csv(local_events_path, parse_dates=["bankruptcy_date"])
    rebuilt["cik"] = pd.to_numeric(rebuilt["cik"], errors="raise").astype("int64")
    local["cik"] = pd.to_numeric(local["cik"], errors="raise").astype("int64")
    rebuilt["bankruptcy_date"] = pd.to_datetime(rebuilt["bankruptcy_date"])
    rebuilt.to_csv(out_dir / "sec_item_1_03_events_rebuilt.csv", index=False, date_format="%Y-%m-%d")

    left = local[["cik", "bankruptcy_date", "first_accession"]].rename(
        columns={"bankruptcy_date": "local_bankruptcy_date", "first_accession": "local_accession"}
    )
    right = rebuilt[["cik", "bankruptcy_date", "first_accession"]].rename(
        columns={"bankruptcy_date": "sec_bankruptcy_date", "first_accession": "sec_accession"}
    )
    comparison = left.merge(right, on="cik", how="outer", indicator=True)
    comparison["date_match"] = comparison["local_bankruptcy_date"].eq(comparison["sec_bankruptcy_date"])
    comparison["accession_match"] = comparison["local_accession"].astype(str).eq(comparison["sec_accession"].astype(str))
    comparison["status"] = "match"
    comparison.loc[comparison["_merge"].eq("left_only"), "status"] = "local_only"
    comparison.loc[comparison["_merge"].eq("right_only"), "status"] = "sec_only"
    comparison.loc[comparison["_merge"].eq("both") & ~comparison["date_match"], "status"] = "date_mismatch"
    comparison.loc[comparison["_merge"].eq("both") & comparison["date_match"] & ~comparison["accession_match"], "status"] = "accession_mismatch"

    # Some local rows choose another filing with the same valid reportDate.
    # Verify each stored accession directly in the bulk submissions records.
    local_accessions = {}
    with zipfile.ZipFile(submissions_path) as archive:
        cik_files: dict[int, list[str]] = {}
        for name in archive.namelist():
            match = re.match(r"^CIK(\d{10})(?:-submissions-\d+)?\.json$", name)
            if match:
                cik_files.setdefault(int(match.group(1)), []).append(name)
        for row in local.itertuples(index=False):
            cik = int(row.cik)
            expected_accession = str(row.first_accession).replace("-", "")
            verified = False
            resolved = pd.NaT
            for name in cik_files.get(cik, []):
                payload = json.loads(archive.read(name))
                main_match = name == f"CIK{cik:010d}.json"
                block = payload.get("filings", {}).get("recent", {}) if main_match else payload
                for hit in _hits(block, cik):
                    if hit["accession"].replace("-", "") != expected_accession:
                        continue
                    filed = pd.to_datetime(hit["filing_date"], errors="coerce")
                    report = pd.to_datetime(hit["report_date"], errors="coerce")
                    valid_report = pd.notna(report) and pd.notna(filed) and report <= filed
                    resolved = report if valid_report else filed
                    verified = bool(
                        pd.notna(resolved)
                        and resolved.normalize() == pd.Timestamp(row.bankruptcy_date).normalize()
                    )
                    break
                if verified:
                    break
            local_accessions[cik] = verified
    comparison["local_accession_verified"] = comparison["cik"].map(local_accessions).fillna(False)
    alternate = (
        comparison["status"].eq("accession_mismatch")
        & comparison["local_accession_verified"]
    )
    comparison.loc[alternate, "status"] = "valid_alternate_accession_same_date"
    invalid = (
        comparison["_merge"].eq("both")
        & comparison["local_accession_verified"].eq(False)
    )
    comparison.loc[invalid, "status"] = "local_accession_unverified"
    comparison.to_csv(out_dir / "event_inventory_comparison.csv", index=False)
    summary = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive": str(submissions_path.resolve()),
        "source_archive_bytes": submissions_path.stat().st_size,
        "source_archive_sha256": sha256(submissions_path),
        "local_event_file": str(local_events_path.resolve()),
        "local_event_file_sha256": sha256(local_events_path),
        "source_event_definition": "All 8-K/8-K-A submissions with SEC item 1.03 since 2010; event date is reportDate when valid and not after filingDate, otherwise filingDate; financial SIC 6000-6999 excluded.",
        "local_event_firms": int(local.cik.nunique()),
        "sec_rebuilt_event_firms": int(rebuilt.cik.nunique()),
        "status_counts": {str(key): int(value) for key, value in comparison.status.value_counts().items()},
        "date_mismatches": int(comparison.status.eq("date_mismatch").sum()),
        "accession_mismatches": int(comparison.status.eq("accession_mismatch").sum()),
        "local_only_events": int(comparison.status.eq("local_only").sum()),
        "sec_only_events": int(comparison.status.eq("sec_only").sum()),
        "local_accessions_verified": int(comparison.loc[comparison["_merge"].eq("both"), "local_accession_verified"].sum()),
        "assessment": "pass_with_new_sec_events" if comparison["status"].isin(["match", "valid_alternate_accession_same_date", "sec_only"]).all() else "review mismatch rows before modeling",
    }
    (out_dir / "sec_bulk_event_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=Path, default=Path("sec_bulk/submissions.zip"))
    parser.add_argument("--events", type=Path, default=Path("market_structural_pipeline/bankruptcies.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sec_event_validation"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit(args.submissions, args.events, args.out_dir)


if __name__ == "__main__":
    main()
