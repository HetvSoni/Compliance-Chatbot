#!/usr/bin/env python
"""
compare_frameworks.py — Phase 3 deliverable (advanced tier: cross-framework
comparison module)

For a question that could plausibly be answered differently by different
regulatory bodies (e.g. "how many APS runs are needed to validate an
aseptic process" -- FDA vs EU vs ICH), retrieve each framework's own best
answer separately and present them side by side, instead of one global
top-1 answer that happens to bury the others.

Per-framework relevance check: a framework's best score is compared to the
BEST score found across ALL frameworks for this SAME query, not to some
fixed absolute cutoff. An earlier attempt at an absolute "is this off-topic"
threshold failed outright (a genuinely off-topic query scored higher than
several genuinely in-domain ones -- see answer.py's dev history) because
BM25 scores depend heavily on query length/specificity, not just topical
relevance. Comparing a framework's score to this query's OWN best score
sidesteps that: it only asks "does FDA have something roughly as relevant
to THIS query as EU does", which is a fair fight since it's the same query
both times.

Usage
-----
  python compare_frameworks.py --query "How many APS runs are needed to validate an aseptic process?"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.answer import extract_best_span, load_index
from src.build_index import _tokenize

FRAMEWORKS = ["FDA", "EU", "ICH", "USP"]
RELATIVE_COVERAGE_THRESHOLD = 0.5  # a framework's best score must be within
                                    # 50% of this query's global best to count
                                    # as "this framework has real coverage"
                                    # (calibrated empirically -- see dev notes)


def compare(query, frameworks=FRAMEWORKS, pool_size=200):
    bm25, chunks = load_index()
    scores = bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:pool_size]

    best_per_framework = {}
    for i in ranked_idx:
        fw = chunks[i]["framework"]
        if fw not in frameworks or fw in best_per_framework:
            continue
        best_per_framework[fw] = (chunks[i], scores[i])
        if len(best_per_framework) == len(frameworks):
            break

    if not best_per_framework:
        return {"query": query, "rows": []}

    global_best_score = max(s for _, s in best_per_framework.values())
    rows = []
    for fw in frameworks:
        if fw not in best_per_framework:
            rows.append({"framework": fw, "covered": False, "answer": None})
            continue
        chunk, score = best_per_framework[fw]
        has_coverage = global_best_score > 0 and (score / global_best_score) >= RELATIVE_COVERAGE_THRESHOLD
        rows.append({
            "framework": fw,
            "covered": has_coverage,
            "answer": extract_best_span(chunk["text"], query) if has_coverage else None,
            "citation": f"{chunk['title']}, § {chunk['section']}",
            "source_url": chunk["source_url"],
            "chunk_id": chunk["chunk_id"],
            "score": score,
            "relative_score": score / global_best_score if global_best_score > 0 else 0,
        })
    return {"query": query, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description="Phase 3 cross-framework comparison")
    ap.add_argument("--query", required=True)
    ap.add_argument("--frameworks", nargs="*", default=FRAMEWORKS)
    args = ap.parse_args()

    result = compare(args.query, frameworks=args.frameworks)
    print(f"\nComparing frameworks for: {result['query']}\n")
    for row in result["rows"]:
        print(f"=== {row['framework']} " + "=" * (70 - len(row["framework"])))
        if not row.get("covered"):
            note = "no matching document collected" if "score" not in row else \
                f"no strong match (relative score {row['relative_score']:.2f} vs. best framework)"
            print(f"  [{note}]\n")
            continue
        print(" ", row["answer"])
        print(f"  Source: {row['citation']}")
        print(f"          {row['source_url']}\n")


if __name__ == "__main__":
    main()
