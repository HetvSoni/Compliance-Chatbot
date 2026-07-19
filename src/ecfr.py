"""eCFR downloader — saves Title 21 parts as raw XML, with change detection via
the versioner titles endpoint (latest_amended_on)."""
import logging

from src.http_util import get, download_file
from src.sources import ECFR_TITLE

log = logging.getLogger("ecfr")
BASE = "https://www.ecfr.gov"


def title_watermark(title=ECFR_TITLE):
    """Return (latest_amended_on, up_to_date_as_of) for a CFR title."""
    resp = get(f"{BASE}/api/versioner/v1/titles.json")
    for t in resp.json().get("titles", []):
        if t.get("number") == title:
            return t.get("latest_amended_on"), t.get("up_to_date_as_of")
    return None, None


def download(source, raw_dir, watermark=None):
    """
    Download one CFR part as raw XML. Returns dict:
      {status, raw_path, bytes, watermark, skipped}
    If `watermark` matches the current latest_amended_on, skip (refresh mode).
    """
    part = source["extra"]["part"]
    amended, as_of = title_watermark()
    if not as_of:
        return {"status": "pending", "raw_path": "", "note": "could not read eCFR watermark"}

    if watermark and watermark.get("latest_amended_on") == amended:
        return {"status": "collected", "skipped": True,
                "watermark": {"latest_amended_on": amended}}

    dest = raw_dir / f"{source['id']}.xml"
    url = f"{BASE}/api/versioner/v1/full/{as_of}/title-{ECFR_TITLE}.xml"
    try:
        n = download_file(url, dest, params={"part": part})
    except FileNotFoundError:
        return {"status": "pending", "raw_path": "", "note": f"404 for part {part}"}
    return {"status": "collected", "raw_path": str(dest), "bytes": n,
            "watermark": {"latest_amended_on": amended}}
