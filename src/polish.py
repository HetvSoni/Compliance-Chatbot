#!/usr/bin/env python
"""
polish.py — Phase 4 addition: single-answer polished paraphrase

Takes the verbatim, multi-citation synthesis (src/answer.py's
extract_multi_span, already filtered to relevant chunks) and rewrites it as
ONE clean, well-organized answer instead of showing raw regulatory text
directly or splitting the response into multiple visible "layers".

Uses flan-t5-large (~780M params, ~3GB download) -- bigger than the
flan-t5-base used for the Phase 3 "rag" tier comparison (src/generate_rag.py),
kept as a SEPARATE model/cache so those already-reported, reproducible
metrics are untouched. Chosen over a hosted GPT API deliberately: no cost,
no API key, stays a locally-run model for this course project's own
pipeline (see conversation/project notes on that tradeoff).

Faithfulness-checked against the verbatim text it was built from (not the
raw retrieved context) -- if the paraphrase doesn't check out (dropped or
invented a detail), the original verbatim text is shown instead of a risky
rewrite, never silently.

Usage
-----
  python polish.py --verbatim "..." --query "..."
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.build_index import _tokenize

MODEL_NAME = "google/flan-t5-large"
MAX_INPUT_TOKENS = 512
FAITHFULNESS_THRESHOLD = 0.35  # a good paraphrase uses different words than the
                                # source by design -- this only guards against the
                                # model adding facts that weren't in the verbatim text

POLISH_PROMPT = """Rewrite the following regulatory text as one clear, well-organized
answer to the question. Keep every specific requirement, number, and
condition from the text -- do not omit or invent any detail. Use plain,
professional language a newcomer to the industry could follow, but keep it
precise enough for an expert to trust.

Regulatory text:
{verbatim}

Question: {question}
Answer:"""

_model_cache = {}


def load_polisher():
    if "model" not in _model_cache:
        _model_cache["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model_cache["model"] = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return _model_cache["tokenizer"], _model_cache["model"]


def _content_words(text):
    return {t for t in _tokenize(text) if len(t) > 2}


def faithfulness_score(answer_text, source_text):
    answer_words = _content_words(answer_text)
    if not answer_words:
        return 1.0
    source_words = _content_words(source_text)
    return len(answer_words & source_words) / len(answer_words)


def polish(verbatim_text, question, max_new_tokens=220):
    """Returns (answer_text, faithfulness_score, used_polish). used_polish
    =False means the paraphrase failed its faithfulness check and the
    ORIGINAL verbatim text is returned unchanged -- never silently drop
    content that didn't check out."""
    tokenizer, model = load_polisher()
    prompt = POLISH_PROMPT.format(verbatim=verbatim_text[:2000], question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    score = faithfulness_score(generated, verbatim_text)
    if not generated or score < FAITHFULNESS_THRESHOLD:
        return verbatim_text, score, False
    return generated, score, True


def main():
    ap = argparse.ArgumentParser(description="Phase 4 single-answer polish (flan-t5-large)")
    ap.add_argument("--verbatim", required=True, help="verbatim synthesized text to polish")
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    answer, score, used = polish(args.verbatim, args.query)
    print(f"\nused_polish={used}  faithfulness={score:.2f}\n")
    print(answer)


if __name__ == "__main__":
    main()
