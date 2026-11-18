"""Download official SEC Financial Statement Data Set quarterly archives.

The SEC datasets are as-filed extracts of numeric facts shown on primary
financial statements. This downloader records SHA-256 hashes and does not
modify or extract the source archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archives(
    out_dir: Path,
    *,
    start_year: int,
    end_year: int,
    end_quarter: int,
    user_agent: str,
    pause_seconds: float = 0.2,
) -> dict:
    if start_year < 2009 or end_year < start_year or not 1 <= end_quarter <= 4:
        raise ValueError("Years must satisfy 2009 <= start_year <= end_year; end_quarter must be 1..4")
    if "@" not in user_agent:
        raise ValueError("SEC User-Agent must include the real contact email")
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    records = []
    quarters = [
        (year, quarter)
        for year in range(start_year, end_year + 1)
        for quarter in range(1, (end_quarter if year == end_year else 4) + 1)
    ]
    total = len(quarters)
    for number, (year, quarter) in enumerate(
        quarters,
        start=1,
    ):
        name = f"{year}q{quarter}.zip"
        destination = out_dir / name
        if destination.exists() and zipfile.is_zipfile(destination):
            print(f"[{number}/{total}] verified existing {name}", flush=True)
        else:
            url = f"{BASE_URL}/{name}"
            temporary = destination.with_suffix(".zip.part")
            last_error = None
            for attempt in range(4):
                try:
                    time.sleep(pause_seconds)
                    with session.get(url, stream=True, timeout=(20, 180)) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as stream:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    stream.write(chunk)
                    if not zipfile.is_zipfile(temporary):
                        raise ValueError(f"SEC response for {name} is not a ZIP archive")
                    temporary.replace(destination)
                    print(f"[{number}/{total}] downloaded {name} ({destination.stat().st_size:,} bytes)", flush=True)
                    last_error = None
                    break
                except Exception as exc:  # retry transient SEC/network failures
                    last_error = exc
                    temporary.unlink(missing_ok=True)
                    if attempt < 3:
                        time.sleep(2 ** attempt)
            if last_error is not None:
                raise RuntimeError(f"Failed to download {name}: {last_error}") from last_error
        records.append({
            "file": name,
            "url": f"{BASE_URL}/{name}",
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        })
    manifest = {
        "dataset": "SEC Financial Statement Data Sets",
        "source_page": "https://www.sec.gov/dera/data/financial-statement-data-sets.html",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"start_year": start_year, "end_year": end_year, "end_quarter": end_quarter},
        "sec_user_agent_contact_provided": True,
        "archives": records,
    }
    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path.resolve()}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("sec_bulk/fsds"))
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--end-quarter", type=int, default=3)
    parser.add_argument("--user-agent", required=True, help="Descriptive project identity and real contact email")
    args = parser.parse_args()
    download_archives(
        args.out_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        end_quarter=args.end_quarter,
        user_agent=args.user_agent,
    )


if __name__ == "__main__":
    main()
