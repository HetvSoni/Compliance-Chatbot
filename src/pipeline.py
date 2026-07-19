#!/usr/bin/env python
"""
pipeline.py — Phase 4 deliverable (chatbot integration)

Wires every Phase 3 component into one query -> answer flow:

  query -> intent/framework classifier
        -> framework-scoped retrieval (falls back to global if the
           predicted framework doesn't have enough/strong hits)
        -> generator (baseline extractive, or intermediate RAG)
        -> citation formatter
        -> safety-flag check

This is the single entry point the Streamlit UI (app.py) and any future
API endpoint should call -- nothing else should re-implement this wiring.

Usage
-----
  python pipeline.py --query "..." --generator baseline|rag
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib

from src.answer import extract_best_span, load_index
from src.build_index import _tokenize
from src.safety import DISCLAIMER, check_advice_seeking

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = ROOT / "models" / "baseline_classifier"

_cache = {}


def load_classifiers():
    if "intent" not in _cache:
        _cache["intent"] = joblib.load(CLASSIFIER_DIR / "intent_clf.joblib")
        _cache["framework"] = joblib.load(CLASSIFIER_DIR / "framework_clf.joblib")
    return _cache["intent"], _cache["framework"]


def framework_scoped_retrieve(bm25, chunks, query, predicted_framework, top_k, pool_size=50):
    """Retrieve top_k chunks, preferring the predicted framework's own
    documents. Falls back to the plain global ranking when that framework
    doesn't have enough hits in the pool (or the prediction is "multi") --
    a hard filter would be too risky given the classifier's macro-F1 is
    0.805, not 1.0; a wrong prediction shouldn't be able to hide the right
    answer entirely."""
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:pool_size]

    if predicted_framework == "multi":
        return [chunks[i] for i in ranked[:top_k]], False

    scoped = [i for i in ranked if chunks[i]["framework"] == predicted_framework]
    if len(scoped) >= top_k:
        return [chunks[i] for i in scoped[:top_k]], True

    return [chunks[i] for i in ranked[:top_k]], False


def format_citations(hits):
    return [
        {"chunk_id": c["chunk_id"], "framework": c["framework"],
         "title": c["title"], "section": c["section"], "source_url": c["source_url"]}
        for c in hits
    ]


def handle_query(query, generator="baseline", top_k=3):
    """Runs the full pipeline. generator: "baseline" (answer.py's extractive
    approach) or "rag" (generate_rag.py's flan-t5-base synthesis)."""
    intent_clf, framework_clf = load_classifiers()
    predicted_intent = intent_clf.predict([query])[0]
    predicted_framework = framework_clf.predict([query])[0]

    flagged, safety_prob = check_advice_seeking(query)

    bm25, chunks = load_index()
    hits, used_framework_scope = framework_scoped_retrieve(
        bm25, chunks, query, predicted_framework, top_k)

    result = {
        "query": query,
        "predicted_intent": predicted_intent,
        "predicted_framework": predicted_framework,
        "used_framework_scoped_retrieval": used_framework_scope,
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
    else:
        raise ValueError(f"unknown generator {generator!r}, expected 'baseline' or 'rag'")

    return result


def main():
    ap = argparse.ArgumentParser(description="Phase 4 integrated chatbot pipeline")
    ap.add_argument("--query", required=True)
    ap.add_argument("--generator", choices=["baseline", "rag"], default="baseline")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    result = handle_query(args.query, generator=args.generator, top_k=args.top_k)

    print(f"\nPredicted intent: {result['predicted_intent']}   "
          f"framework: {result['predicted_framework']}"
          f"{' (scoped retrieval used)' if result['used_framework_scoped_retrieval'] else ' (global fallback)'}")

    if result["disclaimer"]:
        print(f"\n{result['disclaimer']}\n(safety classifier confidence: {result['safety_confidence']:.2f})")

    print("\nAnswer:")
    print(" ", result["answer"])

    if result.get("faithfulness_score") is not None:
        print(f"\n(faithfulness score: {result['faithfulness_score']:.2f})")

    if result["citations"]:
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  - {c['framework']} {c['title']} § {c['section']} ({c['chunk_id']})")
            print(f"    {c['source_url']}")


if __name__ == "__main__":
    main()
