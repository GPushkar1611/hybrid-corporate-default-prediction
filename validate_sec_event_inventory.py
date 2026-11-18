"""Revalidate local bankruptcy labels against SEC Submissions API records."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


API_ROOT = "https://data.sec.gov/submissions"


def _block_records(block: dict) -> list[dict]:
    if not block:
        return []
    length = max((len(values) for values in block.values() if isinstance(values, list)), default=0)
    return [
        {key: values[index] if isinstance(values, list) and index < len(values) else None for key, values in block.items()}
        for index in range(length)
    ]


def _contains_item_103(record: dict) -> bool:
    items = record.get("items") or ""
    codes = {part.strip() for part in str(items).split(",")}
    return record.get("form") in {"8-K", "8-K/A"} and "1.03" in codes


def _request_json(session: requests.Session, url: str, cache_path: Path, pause: float) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    time.sleep(pause)
    response = session.get(url, timeout=(20, 90))
    response.raise_for_status()
    payload = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def validate_inventory(
    events_path: Path,
    out_dir: Path,
    *,
    user_agent: str,
    cache_dir: Path,
    pause: float = 0.13,
) -> pd.DataFrame:
    events = pd.read_csv(events_path, parse_dates=["bankruptcy_date", "first_filing_date"])
    required = {"cik", "bankruptcy_date", "first_filing_date", "first_accession"}
    if not required.issubset(events.columns):
        raise ValueError(f"Event table lacks required columns: {sorted(required - set(events.columns))}")
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    outputs = []
    for index, row in enumerate(events.itertuples(index=False), start=1):
        cik = int(row.cik)
        cik_text = f"{cik:010d}"
        accession = str(row.first_accession).replace("-", "").upper()
        source = "recent"
        try:
            main_path = cache_dir / f"CIK{cik_text}.json"
            main = _request_json(session, f"{API_ROOT}/CIK{cik_text}.json", main_path, pause)
            recent = _block_records(main.get("filings", {}).get("recent", {}))
            filing = next((item for item in recent if str(item.get("accessionNumber", "")).replace("-", "").upper() == accession), None)
            historical_files = main.get("filings", {}).get("files", [])
            if filing is None:
                source = "historical"
                filing_date = pd.Timestamp(row.first_filing_date)
                candidate_files = [
                    item for item in historical_files
                    if pd.Timestamp(item.get("filingFrom")) <= filing_date <= pd.Timestamp(item.get("filingTo"))
                ]
                # A few CIK history ranges can be incomplete; scan remaining
                # historical files only when the date-range candidates miss.
                candidates = candidate_files + [item for item in historical_files if item not in candidate_files]
                for history in candidates:
                    name = str(history["name"])
                    history_payload = _request_json(
                        session, f"{API_ROOT}/{name}", cache_dir / name, pause
                    )
                    matches = _block_records(history_payload)
                    filing = next((item for item in matches if str(item.get("accessionNumber", "")).replace("-", "").upper() == accession), None)
                    if filing is not None:
                        break
            found = filing is not None
            has_item = _contains_item_103(filing or {})
            report_date = pd.to_datetime((filing or {}).get("reportDate"), errors="coerce")
            filing_date = pd.to_datetime((filing or {}).get("filingDate"), errors="coerce")
            valid_report_date = pd.notna(report_date) and pd.notna(filing_date) and report_date <= filing_date
            resolved_date = report_date if valid_report_date else filing_date
            date_match = bool(pd.notna(resolved_date) and resolved_date.normalize() == pd.Timestamp(row.bankruptcy_date).normalize())
            outputs.append({
                "cik": cik,
                "local_bankruptcy_date": row.bankruptcy_date,
                "local_accession": row.first_accession,
                "sec_accession_found": found,
                "sec_form": (filing or {}).get("form"),
                "sec_item_1_03": has_item,
                "sec_filing_date": filing_date,
                "sec_report_date": report_date,
                "sec_resolved_event_date": resolved_date,
                "event_date_matches": date_match,
                "lookup_source": source,
                "status": "ok" if found and has_item and date_match else "mismatch_or_missing",
            })
        except Exception as exc:
            outputs.append({
                "cik": cik,
                "local_bankruptcy_date": row.bankruptcy_date,
                "local_accession": row.first_accession,
                "sec_accession_found": False,
                "sec_item_1_03": False,
                "event_date_matches": False,
                "lookup_source": source,
                "status": f"request_error: {type(exc).__name__}: {str(exc)[:200]}",
            })
        if index % 25 == 0 or index == len(events):
            print(f"Validated {index}/{len(events)} event CIKs", flush=True)
    result = pd.DataFrame(outputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "event_inventory_validation.csv", index=False)
    counts = result["status"].value_counts(dropna=False).to_dict()
    summary = {
        "local_event_rows_checked": int(len(result)),
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "validated_fraction": float(result["status"].eq("ok").mean()),
        "scope": "Verifies the event accession stored for each local CIK, its 8-K Item 1.03 form/item, and the resolved event date. It does not discover events omitted from the local inventory.",
        "sec_user_agent_contact_provided": True,
    }
    (out_dir / "event_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dataset_manifest_path = Path("artifacts/sec_fsds_bankruptcy_v1/dataset_manifest.json")
    if dataset_manifest_path.exists():
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        dataset_manifest["event_label_validation"] = summary
        dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
        dataset_readme = dataset_manifest_path.with_name("README.md")
        if dataset_readme.exists():
            with dataset_readme.open("a", encoding="utf-8") as stream:
                stream.write(
                    "\n## SEC event source validation\n\n"
                    f"Checked {summary['local_event_rows_checked']:,} local event accessions against SEC Submissions API records; "
                    f"{summary['status_counts'].get('ok', 0):,} passed the stored accession, 8-K Item 1.03, and event-date checks. "
                    "See `../sec_event_validation/event_inventory_validation.csv` and the validation summary. "
                    "This validates stored events, not discovery completeness for filings absent from the local event inventory.\n"
                )
    print(json.dumps(summary, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("bankruptcies.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sec_event_validation"))
    parser.add_argument("--cache-dir", type=Path, default=Path("sec_bulk/submissions_api"))
    parser.add_argument("--user-agent", required=True)
    parser.add_argument("--pause", type=float, default=0.13)
    args = parser.parse_args()
    validate_inventory(args.events, args.out_dir, user_agent=args.user_agent, cache_dir=args.cache_dir, pause=args.pause)


if __name__ == "__main__":
    main()
