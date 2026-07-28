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

from src.answer import extract_best_span, extract_multi_span, load_index
from src.build_index import _tokenize
from src.safety import DISCLAIMER, check_advice_seeking

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = ROOT / "models" / "baseline_classifier"

# Layer 1 of the "layered" answer format: a short, low-risk (templated, not
# generated) framing sentence so a beginner has context before the precise
# regulatory text. Intent labels come from the Phase 2 classifier's label set.
INTENT_FRAMING = {
    "GMP requirement": "This is a Good Manufacturing Practice (GMP) requirement under {framework} regulations.",
    "definition": "This asks for a regulatory definition under {framework}.",
    "sterile manufacturing": "This is about sterile/aseptic manufacturing requirements under {framework}.",
    "process validation": "This is about process validation requirements under {framework}.",
    "environmental monitoring": "This is about environmental and contamination-monitoring requirements under {framework}.",
    "pharmacopoeia spec": "This asks about a pharmacopoeia specification or test method.",
}
DEFAULT_FRAMING = "This is a regulatory compliance question under {framework} regulations."

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


def build_layered_answer(query, hits, predicted_intent, predicted_framework, include_simplified=True):
    """Three-layer answer format, built to serve both a beginner and an
    expert from the same response instead of picking one audience:

      1. Framing   -- one templated (not generated) sentence of context.
                       Zero hallucination risk since it's a lookup, not text
                       the model wrote.
      2. Verbatim  -- the exact regulatory text, synthesized across EVERY
                       retrieved citation (not just the top-1 chunk), so an
                       expert gets the complete, precise picture rather than
                       one narrow extract.
      3. Simplified (optional) -- a flan-t5 paraphrase of layer 2 in plain
                       English, explicitly labeled as AI-simplified and
                       faithfulness-checked against layer 2 itself (not the
                       raw retrieved context) so it can't introduce a claim
                       that isn't already verified in layer 2.
    """
    framing = INTENT_FRAMING.get(predicted_intent, DEFAULT_FRAMING).format(framework=predicted_framework)

    spans = extract_multi_span(hits, query)
    matched_spans = [s for s in spans if s["matched"]] or spans[:1]
    verbatim_parts = [
        f"Per {s['framework']} — {s['title']}, § {s['section']}: {s['span']}"
        for s in matched_spans
    ]
    verbatim = "\n\n".join(verbatim_parts)

    result = {
        "framing": framing,
        "verbatim": verbatim,
        "citations": format_citations(hits),
        "simplified": None,
        "simplified_faithfulness": None,
    }

    if include_simplified:
        from src.generate_rag import simplify  # deferred: only loads the model if used
        simplified, score = simplify(verbatim, query)
        result["simplified"] = simplified
        result["simplified_faithfulness"] = score

    return result


def handle_query(query, generator="baseline", top_k=3):
    """Runs the full pipeline. generator: "baseline" (answer.py's extractive
    approach), "rag" (generate_rag.py's flan-t5-base synthesis), or
    "layered" (framing + multi-citation verbatim + optional plain-English
    simplification -- see build_layered_answer)."""
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
    elif generator == "layered":
        layered = build_layered_answer(query, hits, predicted_intent, predicted_framework)
        result["framing"] = layered["framing"]
        result["answer"] = layered["verbatim"]
        result["simplified"] = layered["simplified"]
        result["simplified_faithfulness"] = layered["simplified_faithfulness"]
        result["citations"] = layered["citations"]
    else:
        raise ValueError(f"unknown generator {generator!r}, expected 'baseline', 'rag', or 'layered'")

    return result


def main():
    ap = argparse.ArgumentParser(description="Phase 4 integrated chatbot pipeline")
    ap.add_argument("--query", required=True)
    ap.add_argument("--generator", choices=["baseline", "rag", "layered"], default="baseline")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    result = handle_query(args.query, generator=args.generator, top_k=args.top_k)

    print(f"\nPredicted intent: {result['predicted_intent']}   "
          f"framework: {result['predicted_framework']}"
          f"{' (scoped retrieval used)' if result['used_framework_scoped_retrieval'] else ' (global fallback)'}")

    if result["disclaimer"]:
        print(f"\n{result['disclaimer']}\n(safety classifier confidence: {result['safety_confidence']:.2f})")

    if result.get("framing"):
        print(f"\n{result['framing']}")

    print("\nAnswer:")
    print(" ", result["answer"])

    if result.get("faithfulness_score") is not None:
        print(f"\n(faithfulness score: {result['faithfulness_score']:.2f})")

    if result.get("simplified"):
        print(f"\nIn plain English (AI-simplified, faithfulness={result['simplified_faithfulness']:.2f}):")
        print(" ", result["simplified"])
    elif "simplified" in result and result["simplified"] is None:
        print("\n(Plain-English simplification withheld -- it didn't check out "
              "against the verbatim answer above.)")

    if result["citations"]:
        print("\nCitations:")
        for c in result["citations"]:
            print(f"  - {c['framework']} {c['title']} § {c['section']} ({c['chunk_id']})")
            print(f"    {c['source_url']}")


if __name__ == "__main__":
    main()
