#!/usr/bin/env python
"""
eval_retrieval.py — Phase 3 deliverable (retrieval evaluation)

Evaluates the Phase 2 BM25 index against the gold-passage set
(data/eval/gold_passages.csv) using Recall@K, MRR, and nDCG@K — the metrics
the phase plan tracks across retrieval tiers.

For multi-framework questions (more than one gold_chunk_id), a hit counts if
ANY of the linked chunk_ids appears in the top-K results.

Usage
-----
  python eval_retrieval.py
  python eval_retrieval.py --k 5 10 20
"""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.build_index import CHUNKS_PATH, INDEX_DIR, _tokenize  # reuse existing tokenizer/paths

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "eval" / "gold_passages.csv"
OUT_PATH = ROOT / "eval" / "retrieval_metrics.json"


def load_index():
    with open(INDEX_DIR / "bm25.pkl", "rb") as fh:
        bm25 = pickle.load(fh)
    with open(INDEX_DIR / "chunk_meta.json", encoding="utf-8") as fh:
        chunks = json.load(fh)
    return bm25, chunks


def ranked_chunk_ids(bm25, chunks, query, top_k):
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i]["chunk_id"] for i in ranked]


def evaluate(k_values):
    bm25, chunks = load_index()
    gold = pd.read_csv(GOLD_PATH, dtype=str, keep_default_na=False)
    max_k = max(k_values)

    recall_hits = {k: 0 for k in k_values}
    reciprocal_ranks = []
    ndcgs = {k: [] for k in k_values}
    misses = []

    for _, row in gold.iterrows():
        gold_ids = set(i for i in row["gold_chunk_ids"].split(";") if i)
        if not gold_ids:
            continue  # out-of-corpus rows have no gold_chunk_ids to evaluate against

        top_ids = ranked_chunk_ids(bm25, chunks, row["question"], max_k)

        rr = 0.0
        for rank, cid in enumerate(top_ids, start=1):
            if cid in gold_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        if rr == 0.0:
            misses.append(row["question_id"])

        for k in k_values:
            top_k_ids = top_ids[:k]
            if any(cid in gold_ids for cid in top_k_ids):
                recall_hits[k] += 1
            dcg = sum(1.0 / math.log2(r + 1) for r, cid in enumerate(top_k_ids, start=1) if cid in gold_ids)
            ideal_hits = min(len(gold_ids), k)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

    n = len(reciprocal_ranks)
    return {
        "n_questions": n,
        "mrr": sum(reciprocal_ranks) / n,
        "recall_at_k": {str(k): recall_hits[k] / n for k in k_values},
        "ndcg_at_k": {str(k): sum(ndcgs[k]) / n for k in k_values},
        "missed_question_ids": misses,
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 3 retrieval evaluation")
    ap.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    args = ap.parse_args()

    metrics = evaluate(k_values=tuple(sorted(args.k)))
    print(json.dumps(metrics, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
