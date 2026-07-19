#!/usr/bin/env python
"""
download_sources.py  —  Phase 1 deliverable (acquisition)

Fetches every source in the registry to data/raw/, using the right method per
source, and records the outcome in docs/document_manifest.csv. Paywalled sources
(ISO, USP) and registration-gated ones (BioASQ) are documented, never silently
skipped.

Usage
-----
  python download_sources.py                 # download everything (fresh)
  python download_sources.py --refresh       # only re-pull sources that changed
  python download_sources.py --only cfr_21_211 openfda_drug_enforcement
  python download_sources.py --category primary_kb
  python download_sources.py --manifest-only # just (re)seed the manifest CSV

Set OPENFDA_API_KEY in your environment for higher openFDA rate limits.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import manifest, state
from src.sources import SOURCES
from src import ecfr, openfda, direct, warning_letters, benchmarks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("download")

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _dispatch(source, wm, refresh):
    """Route one source to its downloader. Returns a result dict."""
    watermark = wm if refresh else None
    method = source["method"]
    if method == "ecfr_api":
        return ecfr.download(source, RAW_DIR, watermark)
    if method == "openfda_api":
        return openfda.download(source, RAW_DIR, watermark)
    if method == "direct_pdf":
        return direct.download(source, RAW_DIR, watermark, kind="pdf")
    if method == "direct_html":
        return direct.download(source, RAW_DIR, watermark, kind="html")
    if method == "scrape":
        return warning_letters.download(source, RAW_DIR, watermark)
    if method == "hf_dataset":
        return benchmarks.download(source, RAW_DIR, watermark)
    if method == "manual":
        # Paywalled / registration: nothing to fetch; keep documented status.
        access = source["access"]
        status = "blocked" if access == "paywalled" else "pending"
        return {"status": status, "raw_path": "",
                "note": source["extra"].get("note", "manual collection required")}
    return {"status": "pending", "note": f"unknown method {method}"}


def _select(args):
    srcs = SOURCES
    if args.only:
        srcs = [s for s in srcs if s["id"] in set(args.only)]
    if args.category:
        srcs = [s for s in srcs if s["category"] == args.category]
    return srcs


def main():
    ap = argparse.ArgumentParser(description="Phase 1 source downloader")
    ap.add_argument("--refresh", action="store_true", help="skip sources whose watermark is unchanged")
    ap.add_argument("--only", nargs="*", help="download only these source ids")
    ap.add_argument("--category", help="primary_kb | enforcement | product_ref | benchmark")
    ap.add_argument("--manifest-only", action="store_true", help="just seed the manifest CSV and exit")
    args = ap.parse_args()

    df = manifest.init()
    if args.manifest_only:
        log.info("manifest seeded: %s", manifest.summary(df))
        return

    wm_state = state.load()
    for source in _select(args):
        sid = source["id"]
        log.info("→ %s (%s / %s)", sid, source["category"], source["method"])
        try:
            result = _dispatch(source, state.get(wm_state, sid), args.refresh)
        except Exception as exc:               # one bad source never kills the run
            log.exception("%s failed", sid)
            result = {"status": "pending", "note": f"error: {exc}"}

        fields = {"status": result.get("status", "pending")}
        if result.get("raw_path"):
            fields["raw_path"] = result["raw_path"]
        if result.get("note"):
            fields["notes"] = result["note"]
        manifest.update_row(df, sid, **fields)

        if result.get("watermark"):
            state.set(wm_state, sid, result["watermark"])

        if result.get("skipped"):
            log.info("   skipped (unchanged)")
        else:
            log.info("   %s%s", result.get("status"),
                     f" — {result.get('records', result.get('bytes',''))}"
                     if result.get("records") or result.get("bytes") else "")

    manifest.save(df)
    state.save(wm_state)
    log.info("done. manifest status counts: %s", manifest.summary(df))


if __name__ == "__main__":
    main()
