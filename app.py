#!/usr/bin/env python
"""
app.py — Phase 4 deliverable (chat interface)

Streamlit UI over src/pipeline.py's handle_query(): a chat window where each
user question runs through intent/framework classification, framework-
scoped + relevance-filtered BM25 retrieval, the chosen generator, and the
safety-flag check, with the answer's citations shown alongside it.

Default generator is "polished": relevance-filtered, multi-citation
verbatim regulatory text rewritten as ONE clean answer (flan-t5-large) --
no separate framing/simplified sub-layers, a single readable response.
"baseline" and "rag" stay available for the Phase 3 tier comparison.

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

st.set_page_config(page_title="Pharma Regulatory Compliance Chatbot", page_icon="💊", layout="centered")

st.markdown("""
<style>
.stChatMessage { padding: 0.75rem 1rem; }
.rc-badge {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem;
}
.rc-badge-framework { background: #2b6cb0; color: white; }
.rc-badge-intent { background: #4a5568; color: white; }
.rc-warn {
    background: rgba(237, 137, 54, 0.12); border-left: 3px solid #ed8936;
    padding: 0.6rem 0.9rem; border-radius: 6px; margin: 0.6rem 0;
}
.rc-citation {
    border: 1px solid rgba(128,128,128,0.25); border-radius: 8px;
    padding: 0.6rem 0.9rem; margin-top: 0.4rem;
}
.rc-citation a { text-decoration: none; }
</style>
""", unsafe_allow_html=True)

st.title("💊 Regulatory Compliance Q&A")
st.caption(
    "RAG chatbot over FDA / EU / ICH / USP regulatory text (Team 14). "
    "Ask a factual regulatory question — this system retrieves and "
    "summarizes real regulatory text; it does not predict outcomes or give advice."
)

with st.sidebar:
    st.header("⚙️ Settings")
    generator = st.radio(
        "Answer style",
        options=["polished", "baseline", "rag"],
        index=0,
        format_func=lambda g: {
            "polished": "Polished (recommended)",
            "baseline": "Baseline (raw extract)",
            "rag": "RAG only (flan-t5-base)",
        }[g],
        help="Polished: relevance-filtered regulatory text, synthesized across "
             "every genuinely relevant citation, rewritten as one clean answer "
             "(flan-t5-large). Baseline: single raw citation, zero hallucination "
             "risk. RAG: Phase 3's flan-t5-base tier, for comparison. First "
             "'Polished'/'RAG' query is slow (model download+load).",
    )
    top_k = st.slider("Max chunks to consider", min_value=1, max_value=8, value=3,
                       help="Upper bound only -- the relevance filter may keep fewer "
                            "if some candidates don't meet the relevance threshold.")
    use_reranker = st.checkbox(
        "Use neural reranker", value=False,
        help="Re-orders candidates with a cross-encoder before the relevance filter. "
             "Off by default: the formal evaluation showed it's a wash (MRR improves "
             "slightly, but Recall@5/10 got worse) -- likely a domain mismatch, since "
             "the reranker is trained on general web search, not regulatory text.",
    )
    st.divider()
    st.caption(
        "**Known limitations** — retrieval MRR is 0.732 without reranking; "
        "'definition'-style questions are the weakest spot. The safety "
        "classifier favors recall (0.90) over precision, so it will "
        "occasionally flag a genuinely factual question. Full detail in "
        "`docs/ERROR_ANALYSIS.md`."
    )

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"], unsafe_allow_html=True)

query = st.chat_input("Ask a regulatory compliance question...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            result = handle_query(query, generator=generator, top_k=top_k, use_reranker=use_reranker)

        badges = (
            f'<span class="rc-badge rc-badge-intent">{result["predicted_intent"]}</span>'
            f'<span class="rc-badge rc-badge-framework">{result["predicted_framework"]}</span>'
        )
        st.markdown(badges, unsafe_allow_html=True)

        if result["disclaimer"]:
            st.markdown(
                f'<div class="rc-warn">⚠️ {result["disclaimer"]}<br>'
                f'<small>(safety classifier confidence: {result["safety_confidence"]:.2f})</small></div>',
                unsafe_allow_html=True,
            )

        st.markdown(result["answer"])

        caveats = []
        if result.get("faithfulness_score") is not None and "used_polish" in result and not result["used_polish"]:
            caveats.append("the rewritten version didn't check out against the source text, "
                            "so the verbatim regulatory text is shown above instead")
        if caveats:
            st.caption("ℹ️ " + "; ".join(caveats) + ".")

        if result["citations"]:
            st.markdown("**Sources**")
            for c in result["citations"]:
                st.markdown(
                    f'<div class="rc-citation">📄 <b>{c["framework"]}</b> — {c["title"]}, § {c["section"]}'
                    f'<br><a href="{c["source_url"]}" target="_blank">{c["source_url"]}</a></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No citations — no strongly matching regulatory text was found for this question.")

        # Rebuild the same content as plain markdown for session history (chat_message
        # re-renders from this string on rerun, so it needs the same HTML embedded).
        history_parts = [badges]
        if result["disclaimer"]:
            history_parts.append(
                f'<div class="rc-warn">⚠️ {result["disclaimer"]}<br>'
                f'<small>(safety classifier confidence: {result["safety_confidence"]:.2f})</small></div>'
            )
        history_parts.append(f"\n\n{result['answer']}")
        if result["citations"]:
            history_parts.append("\n\n**Sources**")
            for c in result["citations"]:
                history_parts.append(
                    f'<div class="rc-citation">📄 <b>{c["framework"]}</b> — {c["title"]}, § {c["section"]}'
                    f'<br><a href="{c["source_url"]}" target="_blank">{c["source_url"]}</a></div>'
                )
        st.session_state.history.append({"role": "assistant", "content": "".join(history_parts)})
