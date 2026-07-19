#!/usr/bin/env python
"""
build_index.py — Phase 2 deliverable (BM25 retrieval index)

Builds a BM25 index (rank_bm25) over the section-level chunks produced by
chunk.py, so Phase 3's baseline retriever has something to query against
immediately. Also exposes --query for a quick manual smoke test.

Usage
-----
  python build_index.py                            # build models/bm25/
  python build_index.py --query "media fill"       # build, then search
  python build_index.py --query "..." --skip-build # search an existing index
"""
import argparse
import json
import logging
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("bm25")

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "models" / "bm25"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_chunks():
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"no chunks found at {CHUNKS_PATH} — run chunk.py first")
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build():
    chunks = load_chunks()
    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_DIR / "bm25.pkl", "wb") as fh:
        pickle.dump(bm25, fh)
    with open(INDEX_DIR / "chunk_meta.json", "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False)

    log.info("indexed %d chunks -> %s", len(chunks), INDEX_DIR)


def search(query, top_k=5):
    with open(INDEX_DIR / "bm25.pkl", "rb") as fh:
        bm25 = pickle.load(fh)
    with open(INDEX_DIR / "chunk_meta.json", encoding="utf-8") as fh:
        chunks = json.load(fh)

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i], scores[i]) for i in ranked]


def main():
    ap = argparse.ArgumentParser(description="Phase 2 BM25 index builder")
    ap.add_argument("--query", help="run a search after building")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--skip-build", action="store_true", help="search only, reuse existing index")
    args = ap.parse_args()

    if not args.skip_build:
        build()

    if args.query:
        for chunk, score in search(args.query, args.top_k):
            print(f"[{score:5.2f}] {chunk['framework']:10s} {chunk['title']} "
                  f"§ {chunk['section']}  ({chunk['chunk_id']})")
            print("   ", chunk["text"][:200].replace("\n", " "))
            print()


if __name__ == "__main__":
    main()
