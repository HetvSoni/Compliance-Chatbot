#!/usr/bin/env python
"""
pipeline.py — Phase 4 deliverable (chatbot integration)

Wires every Phase 3 component into one query -> answer flow:

  query -> intent/framework classifier
        -> framework-scoped retrieval (falls back to global if the
           predicted framework doesn't have enough/strong hits)
        -> relative-score relevance filter (drops chunks that don't meet a
           threshold relative to this query's own best score, instead of
           always forcing a fixed count -- see filter_relevant)
        -> generator (baseline extractive, intermediate RAG, or the
           default "polished" single-answer synthesis)
        -> citation formatter
        -> safety-flag check

This is the single entry point the Streamlit UI (app.py) and any future
API endpoint should call -- nothing else should re-implement this wiring.

Usage
-----
  python pipeline.py --query "..." --generator polished|baseline|rag
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib

from src.answer import extract_best_span, extract_multi_span, load_index
from src.build_index import _tokenize
from src.safety import DISCLAIMER, check_advice_seeking

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = ROOT / "models" / "baseline_classifier"

RERANK_POOL_SIZE = 15   # candidates fetched before optional reranking / relevance filtering

# A chunk's BM25 score must be within this fraction of the query's own best
# score in the pool to count as "actually relevant" -- same technique
# (calibrated the same way) as compare_frameworks.py's cross-framework
# coverage check. BM25 scores are always >= 0, so a ratio is well-behaved
# here (unlike the cross-encoder's possibly-negative logit scores, which is
# why this filter runs on BM25 score, before any optional reranking).
RELEVANCE_THRESHOLD = 0.5

_cache = {}


def load_classifiers():
    if "intent" not in _cache:
        _cache["intent"] = joblib.load(CLASSIFIER_DIR / "intent_clf.joblib")
        _cache["framework"] = joblib.load(CLASSIFIER_DIR / "framework_clf.joblib")
    return _cache["intent"], _cache["framework"]


def framework_scoped_retrieve(bm25, chunks, query, predicted_framework, top_k, pool_size=50):
    """Retrieve top_k chunks (each annotated with its bm25_score), preferring
    the predicted framework's own documents. Falls back to the plain global
    ranking when that framework doesn't have enough hits in the pool (or the
    prediction is "multi") -- a hard filter would be too risky given the
    classifier's macro-F1 is 0.805, not 1.0; a wrong prediction shouldn't be
    able to hide the right answer entirely."""
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:pool_size]

    def _with_score(idx):
        c = dict(chunks[idx])
        c["bm25_score"] = float(scores[idx])
        return c

    if predicted_framework == "multi":
        return [_with_score(i) for i in ranked[:top_k]], False

    scoped = [i for i in ranked if chunks[i]["framework"] == predicted_framework]
    if len(scoped) >= top_k:
        return [_with_score(i) for i in scoped[:top_k]], True

    return [_with_score(i) for i in ranked[:top_k]], False


def filter_relevant(hits, threshold=RELEVANCE_THRESHOLD):
    """Drop hits whose bm25_score isn't within `threshold` of the best score
    in this pool, instead of always keeping a fixed count -- this is the
    direct fix for "1st chunk half-detail, 2nd best, 3rd irrelevant": the
    irrelevant one gets dropped rather than padded in just to fill top_k.
    Always keeps at least the top hit."""
    if not hits:
        return hits
    best = max(h.get("bm25_score", 0.0) for h in hits)
    if best <= 0:
        return hits[:1]
    kept = [h for h in hits if h.get("bm25_score", 0.0) / best >= threshold]
    return kept or hits[:1]


def format_citations(hits):
    return [
        {"chunk_id": c["chunk_id"], "framework": c["framework"],
         "title": c["title"], "section": c["section"], "source_url": c["source_url"]}
        for c in hits
    ]


def build_verbatim(query, hits):
    """Synthesize the exact regulatory text across every (already relevance-
    filtered) hit -- an expert gets the complete, precise picture instead of
    one narrow single-chunk extract."""
    spans = extract_multi_span(hits, query)
    matched_spans = [s for s in spans if s["matched"]] or spans[:1]
    return "\n\n".join(
        f"Per {s['framework']} — {s['title']}, § {s['section']}: {s['span']}"
        for s in matched_spans
    )


def build_polished_answer(query, hits):
    """The default answer format: relevance-filtered, multi-citation verbatim
    text rewritten as ONE clean answer (src/polish.py, flan-t5-large),
    faithfulness-checked against that verbatim text. No separate framing
    sentence or "simplified" sub-layer -- just one polished answer, falling
    back to the verbatim text itself if the paraphrase doesn't check out."""
    verbatim = build_verbatim(query, hits)

    from src.polish import polish  # deferred: only loads the ~3GB model if actually used
    answer, faithfulness, used_polish = polish(verbatim, query)

    return {
        "answer": answer,
        "verbatim": verbatim,
        "used_polish": used_polish,
        "faithfulness_score": faithfulness,
        "citations": format_citations(hits),
    }


def handle_query(query, generator="polished", top_k=3, use_reranker=False):
    """Runs the full pipeline. generator:
      "polished" (default) -- relevance-filtered, multi-citation verbatim
                  synthesis rewritten as one clean answer (flan-t5-large).
      "baseline" -- answer.py's single-chunk extractive approach.
      "rag"      -- generate_rag.py's flan-t5-base synthesis (Phase 3 tier).

    use_reranker: optional cross-encoder reordering of the candidate pool
    (src/rerank.py) before the relevance filter. Default OFF: the formal
    148-question evaluation showed it's a wash -- MRR improves slightly
    (0.732 -> 0.749) but Recall@5/10 actually got WORSE (-4 points each),
    likely because the reranker model is trained on general web search
    passages, not regulatory/legal text. Left available, not default.
    """
    intent_clf, framework_clf = load_classifiers()
    predicted_intent = intent_clf.predict([query])[0]
    predicted_framework = framework_clf.predict([query])[0]

    flagged, safety_prob = check_advice_seeking(query)

    bm25, chunks = load_index()
    pool_k = RERANK_POOL_SIZE if use_reranker else max(top_k, RERANK_POOL_SIZE)
    hits, used_framework_scope = framework_scoped_retrieve(
        bm25, chunks, query, predicted_framework, pool_k)

    if use_reranker and hits:
        from src.rerank import rerank  # deferred: only loads the reranker model if used
        hits = rerank(query, hits, top_k=pool_k)

    # top_k is a ceiling, not a fixed target: the relevance filter may keep
    # fewer than top_k (that's the point -- drop a genuinely irrelevant
    # candidate rather than pad it in), but never more than the user asked for.
    hits = filter_relevant(hits)[:top_k]

    result = {
        "query": query,
        "predicted_intent": predicted_intent,
        "predicted_framework": predicted_framework,
        "used_framework_scoped_retrieval": used_framework_scope,
        "used_reranker": use_reranker,
        "n_relevant_chunks": len(hits),
        "safety_flagged": flagged,
        "safety_confidence": safety_prob,
        "disclaimer": DISCLAIMER if flagged else None,
        "generator": generator,
    }

    if not hits:
        result["answer"] = "No matching regulatory text found."
        result["citations"] = []
        return result

    if generator == "baseline":
        primary = hits[0]
        result["answer"] = extract_best_span(primary["text"], query)
        result["citations"] = format_citations(hits)
    elif generator == "rag":
        from src.generate_rag import generate  # deferred: only loads the ~1GB model if actually used
        gen_result = generate(query, hits=hits)
        result["answer"] = gen_result["answer"]
        result["faithfulness_score"] = gen_result.get("faithfulness_score")
        result["citations"] = gen_result["sources"]
    elif generator == "polished":
        polished = build_polished_answer(query, hits)
        result["answer"] = polished["answer"]
        result["used_polish"] = polished["used_polish"]
        result["faithfulness_score"] = polished["faithfulness_score"]
        result["citations"] = polished["citations"]
    else:
        raise ValueError(f"unknown generator {generator!r}, expected 'polished', 'baseline', or 'rag'")

    return result


def main():
    ap = argparse.ArgumentParser(description="Phase 4 integrated chatbot pipeline")
    ap.add_argument("--query", required=True)
    ap.add_argument("--generator", choices=["polished", "baseline", "rag"], default="polished")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--use-reranker", action="store_true",
                     help="enable the cross-encoder reranker (off by default -- see handle_query docstring)")
    args = ap.parse_args()

    result = handle_query(args.query, generator=args.generator, top_k=args.top_k,
                           use_reranker=args.use_reranker)

    print(f"\nPredicted intent: {result['predicted_intent']}   "
          f"framework: {result['predicted_framework']}"
          f"{' (scoped retrieval used)' if result['used_framework_scoped_retrieval'] else ' (global fallback)'}"
          f"   relevant chunks kept: {result.get('n_relevant_chunks', '?')}")

    if result["disclaimer"]:
        print(f"\n{result['disclaimer']}\n(safety classifier confidence: {result['safety_confidence']:.2f})")

    print("\nAnswer:")
    print(" ", result["answer"])

    if result.get("faithfulness_score") is not None:
        note = ""
        if "used_polish" in result:
            note = ", used polished rewrite" if result["used_polish"] else \
                ", showing verbatim (paraphrase failed faithfulness check)"
        print(f"\n(faithfulness score: {result['faithfulness_score']:.2f}{note})")

    if result["citations"]:
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  - {c['framework']} {c['title']} § {c['section']} ({c['chunk_id']})")
            print(f"    {c['source_url']}")


if __name__ == "__main__":
    main()
