#!/usr/bin/env python
"""
extract_text.py  —  Phase 1 deliverable (text extraction)

Walks data/raw/, converts each collected file to a plain-text copy in data/text/,
and records the text path back into the manifest. Handles the four raw formats
Phase 1 produces:

  .pdf   -> pdfplumber (page-by-page)
  .html  -> BeautifulSoup get_text
  .xml   -> eCFR section flattener (heading + paragraph text)
  .jsonl -> openFDA / warning-letter records flattened to readable text

Keeping a plain-text copy means Phase 2 chunking never has to re-parse binaries
or re-scrape, satisfying the "save two copies of every document" requirement.

Usage
-----
  python extract_text.py                 # extract everything not yet extracted
  python extract_text.py --force         # re-extract even if text exists
  python extract_text.py --only cfr_21_211
"""
import re
import json
import argparse
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import manifest
from src.sources import by_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("extract")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TEXT_DIR = ROOT / "data" / "text"


# ---------------------------------------------------------------- extractors
def _from_pdf(path):
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _from_html(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))


def _from_ecfr_xml(path):
    """Flatten eCFR section XML to 'SECTION n — heading\\n text' blocks."""
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError as exc:
        log.error("XML parse failed %s: %s", path.name, exc)
        return ""
    blocks = []
    for div in root.iter():
        if div.tag.upper().endswith("DIV8") and div.get("TYPE") == "SECTION":
            num = (div.get("N") or "").strip()
            head_el = div.find(".//HEAD")
            heading = "".join(head_el.itertext()).strip() if head_el is not None else ""
            paras = [" ".join(p.itertext()).strip()
                     for p in div.iter() if p.tag.upper() in ("P", "FP")]
            body = "\n".join(x for x in paras if x)
            if body:
                blocks.append(f"§ {num} — {heading}\n{body}")
    return "\n\n".join(blocks)


def _from_jsonl(path, source_id):
    """Render openFDA / warning-letter records to readable text lines."""
    lines = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if source_id == "openfda_drug_enforcement":
                lines.append(
                    f"Recall {r.get('recall_number','')}: {r.get('reason_for_recall','')} "
                    f"| Product: {r.get('product_description','')} "
                    f"| Class: {r.get('classification','')} "
                    f"| Firm: {r.get('recalling_firm','')}")
            elif source_id == "fda_warning_letters":
                lines.append(r.get("text", ""))
            elif source_id == "openfda_orange_book":
                app = r.get("application_number", "")
                prods = r.get("products", []) or []
                names = "; ".join(p.get("brand_name", "") for p in prods)
                lines.append(f"{app}: {names}")
            elif source_id == "dailymed_spl_sample":
                ofda = r.get("openfda", {}) or {}
                brand = (ofda.get("brand_name") or [""])[0]
                ind = " ".join(r.get("indications_and_usage", []) or [])
                lines.append(f"{brand}: {ind}")
            else:
                lines.append(json.dumps(r)[:1500])
    return "\n".join(x for x in lines if x)


# ---------------------------------------------------------------- driver
def _extract_one(raw_path: Path, source_id: str):
    ext = raw_path.suffix.lower()
    if ext == ".pdf":
        return _from_pdf(raw_path)
    if ext == ".html":
        return _from_html(raw_path)
    if ext == ".xml":
        return _from_ecfr_xml(raw_path)
    if ext == ".jsonl":
        return _from_jsonl(raw_path, source_id)
    return ""


def main():
    ap = argparse.ArgumentParser(description="Phase 1 text extractor")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    df = manifest.load()
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        sid = row["id"]
        if args.only and sid not in set(args.only):
            continue
        raw_path = row.get("raw_path", "")
        if not raw_path or not Path(raw_path).exists():
            continue

        text_path = TEXT_DIR / f"{sid}.txt"
        if text_path.exists() and not args.force:
            continue

        try:
            text = _extract_one(Path(raw_path), sid)
        except Exception as exc:
            log.warning("extract failed for %s: %s", sid, exc)
            continue

        if not text.strip():
            log.info("%s: no text extracted", sid)
            continue

        text_path.write_text(text, encoding="utf-8")
        manifest.update_row(df, sid, text_path=str(text_path))
        log.info("%s → %s (%d chars)", sid, text_path.name, len(text))

    manifest.save(df)
    log.info("extraction complete")


if __name__ == "__main__":
    main()
