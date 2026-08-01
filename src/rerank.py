#!/usr/bin/env python
"""
rerank.py — Phase 4 addition: neural cross-encoder reranker

BM25 finds the right chunk almost always (Recall@20 = 94.6%) but doesn't
reliably rank it #1 (MRR = 0.732) -- symptom: of 3 chunks shown, one might
be the best possible answer, one half-relevant, one off-topic, in no
particular order. This reranks the top BM25 candidates using a cross-
encoder that actually reads (query, passage) together instead of counting
term overlap, to fix ORDERING specifically -- it does not fix true recall
misses (if the right chunk isn't in the candidate pool at all, reranking
the wrong candidates won't conjure it).

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 via plain `transformers`
(AutoModelForSequenceClassification) -- deliberately NOT the
sentence-transformers package, since dense retrieval/FAISS was already
dropped from this project's scope; this stays a lighter addition on top of
BM25, not a replacement for it.

Usage
-----
  python rerank.py --query "..." --pool-size 20 --top-k 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.answer import _local_best_span, load_index, retrieve

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cache = {}


def load_reranker():
    if "model" not in _cache:
        _cache["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        _cache["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _cache["model"].eval()
    return _cache["tokenizer"], _cache["model"]


def rerank(query, candidates, top_k=3):
    """candidates: list of chunk dicts (already BM25-retrieved). Returns the
    same chunks re-ordered by cross-encoder relevance, truncated to top_k,
    each annotated with rerank_score and the local span the reranker judged
    it on (reusing answer.py's chunk-local extraction rather than judging
    on raw, possibly huge and mostly off-topic, chunk text)."""
    if not candidates:
        return []

    tokenizer, model = load_reranker()
    spans = [_local_best_span(c["text"], query)[0] for c in candidates]

    pairs = [(query, span) for span in spans]
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)
    scores = logits.tolist()
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(zip(candidates, spans, scores), key=lambda x: x[2], reverse=True)
    return [
        {**c, "rerank_span": span, "rerank_score": score}
        for c, span, score in ranked[:top_k]
    ]


def main():
    ap = argparse.ArgumentParser(description="Neural reranker over BM25 candidates")
    ap.add_argument("--query", required=True)
    ap.add_argument("--pool-size", type=int, default=20, help="how many BM25 candidates to rerank")
    ap.add_argument("--top-k", type=int, default=3, help="how many to keep after reranking")
    args = ap.parse_args()

    bm25, chunks = load_index()
    candidates = retrieve(bm25, chunks, args.query, args.pool_size)
    results = rerank(args.query, candidates, top_k=args.top_k)

    print(f"\nQuery: {args.query}\n")
    for r in results:
        print(f"[{r['rerank_score']:6.2f}] {r['framework']} {r['title']} § {r['section']} ({r['chunk_id']})")
        print("   ", r["rerank_span"][:200])
        print()


if __name__ == "__main__":
    main()
