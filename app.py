#!/usr/bin/env python
"""
app.py — Phase 4 deliverable (chat interface)

Streamlit UI over src/pipeline.py's handle_query(): a chat window where each
user question runs through intent/framework classification, framework-
scoped BM25 retrieval, the chosen generator, and the safety-flag check, with
the answer's citations shown alongside it.

Run with:
  streamlit run app.py
(NOT `python app.py` -- Streamlit apps must be launched through the
`streamlit` command so its script-rerun server wrapper is in effect.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.pipeline import handle_query

st.set_page_config(page_title="Pharma Regulatory Compliance Chatbot", page_icon="💊")

st.title("Regulatory Compliance Q&A")
st.caption(
    "RAG chatbot over FDA / EU / ICH / USP regulatory text (Team 14). "
    "Ask a factual regulatory question -- this system retrieves and "
    "summarizes real regulatory text, it does not predict outcomes or give advice."
)

with st.sidebar:
    st.header("Settings")
    generator = st.radio(
        "Answer generator",
        options=["layered", "baseline", "rag"],
        index=0,
        format_func=lambda g: {
            "layered": "Layered (recommended: framing + full-detail citations + plain-English)",
            "baseline": "Baseline (single citation, extractive, zero hallucination risk)",
            "rag": "Intermediate (flan-t5-base RAG only, faithfulness-checked)",
        }[g],
        help="Layered: best for most users -- combines a plain framing sentence, the "
             "precise regulatory text synthesized across ALL retrieved citations (not "
             "just one), and an optional AI-simplified explanation. Baseline: narrower, "
             "single-citation extract. RAG: pure model synthesis, no verbatim layer. "
             "First query using layered/rag is slow (~1GB model download+load).",
    )
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=8, value=3)
    st.divider()
    st.caption(
        "Known limitations (see eval/phase3_summary.json): retrieval MRR is "
        "0.732 -- the right section isn't always ranked #1, especially for "
        "'definition'-style questions. The safety classifier is tuned for "
        "recall (0.90) over precision, so it will occasionally flag a "
        "genuinely factual question."
    )

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

query = st.chat_input("Ask a regulatory compliance question...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            result = handle_query(query, generator=generator, top_k=top_k)

        parts = []
        parts.append(
            f"*Predicted intent: **{result['predicted_intent']}** · "
            f"framework: **{result['predicted_framework']}** "
            f"({'framework-scoped' if result['used_framework_scoped_retrieval'] else 'global fallback'})*"
        )

        if result["disclaimer"]:
            parts.append(f"\n> ⚠️ {result['disclaimer']} (confidence: {result['safety_confidence']:.2f})\n")

        if result.get("framing"):
            parts.append(f"\n*{result['framing']}*")

        parts.append(f"\n**Answer:** {result['answer']}")

        if result.get("faithfulness_score") is not None:
            parts.append(f"\n*(faithfulness score: {result['faithfulness_score']:.2f})*")

        if result.get("simplified"):
            parts.append(
                f"\n\n**In plain English** (AI-simplified, faithfulness="
                f"{result['simplified_faithfulness']:.2f}): {result['simplified']}"
            )
        elif "simplified" in result and result["simplified"] is None:
            parts.append("\n\n*(Plain-English simplification withheld -- it didn't "
                          "check out against the verbatim answer above.)*")

        if result["citations"]:
            parts.append("\n\n**Citations:**")
            for c in result["citations"]:
                parts.append(f"\n- {c['framework']} — {c['title']}, § {c['section']}  \n  {c['source_url']}")
        else:
            parts.append("\n\n*(No citations -- no strongly matching regulatory text was found for this question.)*")

        content = "".join(parts)
        st.markdown(content)
        st.session_state.history.append({"role": "assistant", "content": content})
