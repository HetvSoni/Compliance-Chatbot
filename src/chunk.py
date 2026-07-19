#!/usr/bin/env python
"""
chunk.py — Phase 2 deliverable (section-level chunking)

Splits each Phase 1 text extraction in data/text/ into section-level chunks
using each document's own native numbering, not fixed token windows, and
tags every chunk with retrieval metadata (framework, title, section, URL).

Chunking strategy is picked per source_id, since each document family uses a
different native numbering scheme:
  cfr     — CFR parts already carry "§ N — heading" blocks from eCFR XML
            extraction (see extract_text.py); split on those directly.
  dotted  — EU Annex 1 / ICH guidelines number clauses "1", "2.1", "4.27";
            split at every dotted-numeral heading found.
  roman   — FDA guidance docs use roman-numeral top sections ("IV. ...");
            split at the top level only (finer subheadings stay inside).
  lines   — openFDA/DailyMed/warning-letter text is already one record per
            line (from the JSONL extractor); each line is its own chunk.
  caps    — USP general chapters use standalone ALL-CAPS headings
            ("INTRODUCTION", "SPECIATION") instead of numbering.

Usage
-----
  python chunk.py                 # chunk every source with a text_path
  python chunk.py --only ich_q7
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import manifest
from src.sources import by_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("chunk")

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "processed" / "chunks.jsonl"

MIN_CHARS = 40          # drop chunks with less real content than this
MIN_AVG_WORD_LEN = 2.3  # below this, text is almost certainly PDF-extraction garble

FAMILY = {
    "cfr_21_210": "cfr", "cfr_21_211": "cfr", "cfr_21_600": "cfr",
    "cfr_21_601": "cfr", "cfr_21_606": "cfr", "cfr_21_610": "cfr",
    "eu_gmp_annex1_2022": "dotted",
    "ich_q7": "dotted", "ich_q8": "dotted", "ich_q9": "dotted", "ich_q10": "dotted",
    "fda_aseptic_2004": "roman", "fda_process_val_2011": "roman",
    "dailymed_spl_sample": "lines", "openfda_drug_enforcement": "lines",
    "fda_warning_letters": "lines", "openfda_orange_book": "lines",
    "usp_232": "caps", "usp_233": "caps",
}

CFR_HEAD = re.compile(r"^§\s*([\d.]+)\s*—\s*(?:§\s*[\d.]+\s*)?(.*)$")
DOTTED_HEAD = re.compile(r"^([1-9]\d{0,2}(?:\.\d{1,2}){0,3})\.?\s+(\S.*)$")
ROMAN_HEAD = re.compile(r"^([IVXLCDM]{1,6})\.\s+([A-Z][A-Z ,&/-]{2,80})$")
CAPS_HEAD = re.compile(r"^([A-Z][A-Z0-9 ,/\-]{2,60})$")
TOC_LEADER = re.compile(r"\.{4,}")  # dot-leader lines from a table of contents


def _is_garbled(text):
    """Flag PDF-extraction noise (e.g. scrambled cover-page glyphs)."""
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 20:
        return False
    return (sum(len(w) for w in words) / len(words)) < MIN_AVG_WORD_LEN


def _strip_garbled_lines(text):
    """Drop individual lines that are scrambled PDF-extraction glyphs
    (e.g. cover-page decorative text split one letter per token), so a
    garbled line doesn't get merged into an otherwise-clean chunk."""
    kept = []
    for line in text.splitlines():
        words = re.findall(r"[A-Za-z]+", line)
        if len(words) >= 4 and (sum(len(w) for w in words) / len(words)) < 2.0:
            continue
        kept.append(line)
    return "\n".join(kept)


def _chunk_cfr(text):
    for block in re.split(r"\n{2,}", text.strip()):
        block = block.strip()
        if not block:
            continue
        m = CFR_HEAD.match(block.splitlines()[0])
        section, heading = (m.group(1), m.group(2).strip()) if m else ("", "")
        yield section, heading, block


def _chunk_by_heading(text, pattern, max_bare_words=None):
    """max_bare_words guards against in-body numbered lists ("1) Preparing...")
    being mistaken for a bare (undotted) top-level section: a real section
    title is a short phrase, a list item is a full sentence."""
    cur_section, cur_heading, buf = "preamble", "", []
    for line in text.splitlines():
        stripped = line.strip()
        m = None if TOC_LEADER.search(stripped) else pattern.match(stripped)
        if m and max_bare_words and "." not in m.group(1) \
                and len(m.group(2).split()) > max_bare_words:
            m = None
        if m:
            if buf:
                yield cur_section, cur_heading, "\n".join(buf)
            cur_section, cur_heading = m.group(1), m.group(2).strip()
            buf = [line]
        else:
            buf.append(line)
    if buf:
        yield cur_section, cur_heading, "\n".join(buf)


def _chunk_lines(text):
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if line:
            yield str(i), "", line


def _chunk_caps(text):
    """USP chapters use standalone ALL-CAPS headings with no numbering
    (INTRODUCTION, SPECIATION, ...); the heading text doubles as the
    section label since there's no clause number to cite."""
    cur_heading, buf = "preamble", []
    for line in text.splitlines():
        stripped = line.strip()
        m = CAPS_HEAD.match(stripped)
        words = stripped.split()
        # multi-word heading, or a single long word (excludes short acronyms/units)
        looks_like_heading = m and (len(words) >= 2 or len(stripped) >= 6)
        if looks_like_heading:
            if buf:
                yield cur_heading, cur_heading, "\n".join(buf)
            cur_heading = stripped
            buf = [line]
        else:
            buf.append(line)
    if buf:
        yield cur_heading, cur_heading, "\n".join(buf)


def _split(family, text):
    if family == "cfr":
        yield from _chunk_cfr(text)
    elif family == "dotted":
        yield from _chunk_by_heading(text, DOTTED_HEAD, max_bare_words=10)
    elif family == "roman":
        yield from _chunk_by_heading(text, ROMAN_HEAD)
    elif family == "lines":
        yield from _chunk_lines(text)
    elif family == "caps":
        yield from _chunk_caps(text)


def build(only=None):
    df = manifest.load()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_chunks, n_docs, n_dropped = 0, 0, 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            sid = row["id"]
            if only and sid not in only:
                continue
            text_path = row.get("text_path", "")
            family = FAMILY.get(sid)
            if not text_path or not family or not Path(text_path).exists():
                continue

            src = by_id(sid)
            raw_text = Path(text_path).read_text(encoding="utf-8").replace("▲", "")
            text = _strip_garbled_lines(raw_text)
            idx = 0
            for section, heading, chunk_text in _split(family, text):
                chunk_text = re.sub(r"\n{2,}", "\n", chunk_text).strip()
                if (len(re.sub(r"\W", "", chunk_text)) < MIN_CHARS
                        or _is_garbled(chunk_text)
                        or TOC_LEADER.search(chunk_text)):
                    n_dropped += 1
                    continue
                rec = {
                    "chunk_id": f"{sid}::{idx:04d}",
                    "source_id": sid,
                    "framework": src["framework"],
                    "title": src["title"],
                    "section": section,
                    "heading": heading,
                    "source_url": src["url"],
                    "retrieval_date": row.get("retrieval_date", ""),
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                idx += 1
                n_chunks += 1
            n_docs += 1
            log.info("%s (%s) -> %d chunks", sid, family, idx)

    log.info("done: %d chunks from %d docs (%d dropped as noise) -> %s",
              n_chunks, n_docs, n_dropped, OUT_PATH)


def main():
    ap = argparse.ArgumentParser(description="Phase 2 section-level chunker")
    ap.add_argument("--only", nargs="*", help="only chunk these source ids")
    args = ap.parse_args()
    build(only=set(args.only) if args.only else None)


if __name__ == "__main__":
    main()
