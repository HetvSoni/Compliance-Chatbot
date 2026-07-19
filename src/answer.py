#!/usr/bin/env python
"""
answer.py — Phase 3 deliverable (baseline extractive answer generator)

Baseline tier for RAG answer generation: no seq2seq model, no hallucination
risk. Retrieves the top chunk via the main BM25 index (already well
calibrated over the full ~19,500-chunk corpus), then uses a small, chunk-
local BM25 index over just THAT chunk's own sentences to pick which part of
it best answers the query. Presents the runner-up chunks too, since the
gold-set evaluation shows BM25's MRR is 0.732 -- the correct chunk is
usually in the top-10-20 but not reliably #1, so surfacing alternates
matters more than trying to force a single "best" answer.

Design note: an earlier version tried to rerank across the WHOLE candidate
pool (20 very different chunks) using a single BM25 index built over all
their sentences combined. That backfired -- chunks with far more sentences
(e.g. a 30,000-char FDA guidance section vs. a one-line recall record)
dominate purely by having more chances to score well, actively burying the
correct chunk even when it was sitting right there in the pool. Comparing
sentences only *within* one already BM25-selected chunk avoids that;
apples-to-apples, lower stakes, and it's what the "baseline" tier should
be -- the real fix for cross-chunk precision is exactly what an
intermediate dense-retrieval/reranker tier is for, which is out of scope
here.

Usage
-----
  python answer.py --query "What is a Contamination Control Strategy?"
  python answer.py --query "..." --top-k 3
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rank_bm25 import BM25Okapi

from src.build_index import INDEX_DIR, _tokenize
from src.safety import DISCLAIMER, check_advice_seeking

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(0-9])")
CFR_HEAD_STRIP = re.compile(r"^§\s*[\d.]+\s*—\s*§\s*[\d.]+\s*")


def load_index():
    with open(INDEX_DIR / "bm25.pkl", "rb") as fh:
        bm25 = pickle.load(fh)
    with open(INDEX_DIR / "chunk_meta.json", encoding="utf-8") as fh:
        chunks = json.load(fh)
    return bm25, chunks


def retrieve(bm25, chunks, query, top_k):
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in ranked]


def _clean(text):
    text = CFR_HEAD_STRIP.sub("", text.strip())  # drop duplicated "§ N — § N Title" prefix
    return re.sub(r"\s+", " ", text).strip()


def _sentences(chunk_text):
    return [s.strip() for s in SENTENCE_SPLIT.split(chunk_text) if s.strip()]


def extract_best_span(chunk_text, query, context_window=1):
    """Within one chunk, find its best-matching sentence via a chunk-local
    BM25 index, and return it plus its immediate neighbors for readability."""
    sentences = _sentences(chunk_text)
    if len(sentences) <= 1:
        return _clean(chunk_text)[:600]

    local_bm25 = BM25Okapi([_tokenize(s) for s in sentences])
    scores = local_bm25.get_scores(_tokenize(query))
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    if scores[best_idx] <= 0:
        # no sentence shares real terminology with the query -- be honest
        # about it rather than presenting an arbitrary sentence as "the answer"
        return "[no sentence within this section closely matches the query] " + _clean(
            " ".join(sentences[:2]))

    lo, hi = max(0, best_idx - context_window), best_idx + context_window + 1
    return _clean(" ".join(sentences[lo:hi]))


def answer(query, top_k=3):
    bm25, chunks = load_index()
    hits = retrieve(bm25, chunks, query, top_k + 1)
    if not hits:
        return {"answer": "No matching regulatory text found.", "related_chunks": []}

    primary = hits[0]
    extracted = extract_best_span(primary["text"], query)

    citation = f"{primary['framework']} — {primary['title']}, § {primary['section']}"
    return {
        "answer": extracted,
        "primary_citation": citation,
        "primary_source_url": primary["source_url"],
        "primary_chunk_id": primary["chunk_id"],
        "related_chunks": [
            {"chunk_id": c["chunk_id"], "framework": c["framework"],
             "title": c["title"], "section": c["section"]}
            for c in hits[1:]
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 3 baseline extractive answer generator")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=3, help="how many 'also relevant' chunks to list")
    args = ap.parse_args()

    flagged, prob = check_advice_seeking(args.query)
    if flagged:
        print(f"\n{DISCLAIMER}\n(safety classifier confidence: {prob:.2f})")

    result = answer(args.query, top_k=args.top_k)
    print("\nAnswer:")
    print(" ", result["answer"])
    print(f"\nSource: {result['primary_citation']}")
    print(f"        {result['primary_source_url']}")
    if result["related_chunks"]:
        print("\nAlso relevant (BM25 didn't rank these #1, but worth checking"
              " -- MRR on the gold set is 0.732, so the right answer isn't"
              " always the top hit):")
        for c in result["related_chunks"]:
            print(f"  - {c['framework']} {c['title']} § {c['section']} ({c['chunk_id']})")


if __name__ == "__main__":
    main()
