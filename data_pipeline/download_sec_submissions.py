"""Download the official SEC Submissions API bulk ZIP for event-label audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("sec_bulk/submissions.zip"))
    parser.add_argument("--user-agent", required=True)
    args = parser.parse_args()
    if "@" not in args.user_agent:
        raise ValueError("SEC User-Agent must include the real contact email")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".part")
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept-Encoding": "gzip, deflate"})
    if not args.out.exists() or not zipfile.is_zipfile(args.out):
        last_error = None
        for attempt in range(4):
            try:
                time.sleep(0.2)
                with session.get(URL, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                if not zipfile.is_zipfile(temporary):
                    raise ValueError("SEC response is not a ZIP archive")
                temporary.replace(args.out)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt < 3:
                    time.sleep(2 ** attempt)
        if last_error is not None:
            raise RuntimeError(f"SEC submissions archive download failed: {last_error}") from last_error
    manifest = {
        "dataset": "SEC Submissions API bulk archive",
        "url": URL,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "file": str(args.out.resolve()),
        "bytes": args.out.stat().st_size,
        "sha256": sha256(args.out),
        "sec_user_agent_contact_provided": True,
    }
    manifest_path = args.out.with_name("submissions_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
