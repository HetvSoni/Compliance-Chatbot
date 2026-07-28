#!/usr/bin/env python
"""
generate_rag.py — Phase 3 deliverable (intermediate-tier RAG generation)

Intermediate tier for answer generation: retrieves top-k chunks via BM25,
then feeds them + the question to a small instruction-tuned seq2seq model
(flan-t5-base) which synthesizes a real answer in its own words, grounded
in the retrieved context, instructed to say so when the context doesn't
answer the question.

This exists alongside — not instead of — the baseline extractive generator
in answer.py: the phase plan wants a baseline-vs-intermediate comparison,
so both stay runnable. Baseline (answer.py) is a pure lexical extract with
zero hallucination risk; this tier can phrase things more naturally and
handle paraphrased questions better, at the cost of the small (but
nonzero) hallucination risk any generative model carries.

Usage
-----
  python generate_rag.py --query "What is a Contamination Control Strategy?"
  python generate_rag.py --query "..." --top-k 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.answer import load_index, retrieve
from src.build_index import _tokenize
from src.safety import DISCLAIMER, check_advice_seeking

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_TOKENS = 512
MAX_CHUNK_CHARS = 500  # per-source excerpt length, keeps the prompt within budget
FAITHFULNESS_THRESHOLD = 0.45  # min fraction of the answer's content words that must

NO_ANSWER_PHRASE = "The provided regulatory excerpts do not answer this question."
UNFAITHFUL_PHRASE = ("The model's answer did not check out against the retrieved text "
                      "(faithfulness check failed) -- not shown, to avoid presenting an "
                      "ungrounded answer as fact. See the sources below yourself instead.")

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. Do not use any
outside knowledge. If the context does not contain the answer, respond
with exactly: "{no_answer}"

{context}

Question: {question}
Answer:"""


_model_cache = {}


def load_generator():
    if "model" not in _model_cache:
        _model_cache["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model_cache["model"] = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return _model_cache["tokenizer"], _model_cache["model"]


def build_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        excerpt = c["text"][:MAX_CHUNK_CHARS]
        blocks.append(f"[Source {i}: {c['framework']} — {c['title']}, § {c['section']}]\n{excerpt}")
    return "\n\n".join(blocks)


def _content_words(text):
    return {t for t in _tokenize(text) if len(t) > 2}


def faithfulness_score(answer_text, context_text):
    """What fraction of the answer's distinct content words actually appear
    somewhere in the retrieved context. Catches genuine hallucination (the
    model answering from its own pretrained knowledge instead of the
    provided text, e.g. "rainy or cloudy" for a weather question that has
    no business being answered from a pharma regulatory corpus at all).

    Does NOT catch the model confidently parroting an irrelevant-but-real
    retrieved chunk back -- that's a retrieval-precision problem (the same
    one behind MRR=0.732), not a faithfulness problem, since those words
    genuinely did come from the context."""
    answer_words = _content_words(answer_text)
    if not answer_words:
        return 1.0
    context_words = _content_words(context_text)
    return len(answer_words & context_words) / len(answer_words)


SIMPLIFY_THRESHOLD = 0.35  # looser than the main faithfulness check (0.45): a
                           # good paraphrase is SUPPOSED to use different words
                           # than the source -- this only guards against the
                           # model adding facts that weren't in the verbatim text

SIMPLIFY_PROMPT = """Explain the following regulatory text in plain, simple English
for someone new to the pharmaceutical industry. Keep it to 2-3 short sentences.
Do not add any facts that are not in the text below.

Regulatory text:
{verbatim}

Question asked: {question}
Simple explanation:"""


def simplify(verbatim_text, question, max_new_tokens=100):
    """Generate a beginner-friendly paraphrase of an ALREADY-VERIFIED verbatim
    answer (not the raw retrieved context) -- faithfulness is checked against
    that verbatim text, so the check only asks "did the model invent a new
    claim", not "did it use different words" (paraphrasing is the whole
    point). Returns (simplified_text_or_None, faithfulness_score)."""
    tokenizer, model = load_generator()
    prompt = SIMPLIFY_PROMPT.format(verbatim=verbatim_text[:1500], question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    score = faithfulness_score(generated, verbatim_text)
    if score < SIMPLIFY_THRESHOLD:
        return None, score
    return generated, score


def generate(query, top_k=4, max_new_tokens=150, hits=None):
    """hits: optionally pass already-retrieved chunks (e.g. from the
    pipeline's framework-scoped retrieval) instead of doing a fresh
    global BM25 lookup here."""
    if hits is None:
        bm25, all_chunks = load_index()
        hits = retrieve(bm25, all_chunks, query, top_k)
    if not hits:
        return {"answer": NO_ANSWER_PHRASE, "sources": []}

    tokenizer, model = load_generator()
    context = build_context(hits)
    prompt = PROMPT_TEMPLATE.format(no_answer=NO_ANSWER_PHRASE, context=context, question=query)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    self_reported_no_answer = NO_ANSWER_PHRASE.lower() in generated.lower()
    faithfulness = None if self_reported_no_answer else faithfulness_score(generated, context)
    unfaithful = faithfulness is not None and faithfulness < FAITHFULNESS_THRESHOLD

    if self_reported_no_answer:
        final_answer, grounded = NO_ANSWER_PHRASE, False
    elif unfaithful:
        final_answer, grounded = UNFAITHFUL_PHRASE, False
    else:
        final_answer, grounded = generated, True

    return {
        "answer": final_answer,
        "raw_generated": generated,
        "faithfulness_score": faithfulness,
        "grounded": grounded,
        "sources": [
            {"chunk_id": c["chunk_id"], "framework": c["framework"],
             "title": c["title"], "section": c["section"], "source_url": c["source_url"]}
            for c in hits
        ] if grounded else [],
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 3 intermediate-tier RAG generator")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=4, help="how many chunks to feed as context")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    args = ap.parse_args()

    flagged, prob = check_advice_seeking(args.query)
    if flagged:
        print(f"\n{DISCLAIMER}\n(safety classifier confidence: {prob:.2f})")

    result = generate(args.query, top_k=args.top_k, max_new_tokens=args.max_new_tokens)
    print("\nAnswer:")
    print(" ", result["answer"])
    if result["faithfulness_score"] is not None:
        print(f"\n(faithfulness score: {result['faithfulness_score']:.2f}"
              f" -- fraction of the answer's content words found in the retrieved context)")
    if result["sources"]:
        print("\nGrounded in:")
        for s in result["sources"]:
            print(f"  - {s['framework']} {s['title']} § {s['section']} ({s['chunk_id']})")
            print(f"    {s['source_url']}")
    else:
        print("\n(No sources cited — model reported the context doesn't answer this question.)")


if __name__ == "__main__":
    main()
