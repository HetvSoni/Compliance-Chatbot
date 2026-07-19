# Error Analysis

Phase 4 deliverable. Consolidates every failure mode found and characterized
during Phase 3/4 development — each one was reproduced with a concrete
example while building the system, not identified after the fact.

---

## 1. Retrieval: definition-style questions are a systematic weak spot

Of the 8 questions BM25 misses entirely within top-20 (`eval/retrieval_metrics.json`),
6 are "definition" intent (e.g. *"How does ICH Q8 define a Critical Quality
Attribute?"*). Root cause: a definitional question shares generic words
("define," "what is") with many chunks that merely *use* the term
operationally, rather than the one chunk that actually defines it. BM25 has
no notion of "this passage is *the* definition" versus "this passage just
mentions the word a lot."

**Consequence:** Recall@5/10/20 (87.8% / 93.2% / 94.6%) looks strong in
aggregate but is not evenly distributed — definition questions specifically
underperform.

## 2. Retrieval: MRR (0.732) means "usually top-3, not always top-1"

The correct chunk is in the top-20 candidate pool almost always, but not
reliably ranked #1. Verified directly: for the literal gold question *"How
does 21 CFR 210.3 define a 'batch'?"*, the correct chunk (`cfr_21_210::0002`)
sits at rank 7 of 20 — present, but not what a naive top-1-only system would
surface.

**Consequence:** any answer generator that trusts only the #1 BM25 hit will
sometimes present the wrong (but real) section as if it were authoritative.
Both `answer.py` and `generate_rag.py` show runner-up chunks explicitly for
this reason, rather than hiding the uncertainty.

## 3. Corpus composition skews retrieval toward enforcement records

~17,760 of 19,485 chunks (91%) are single-line FDA enforcement/recall
records, all of which mention generic phrases like "21 CFR Part XXX" and
"product." For queries whose informative content is a specific regulation
citation plus common words, an unrelated recall record can outscore the
actual regulatory text purely on term frequency. Reproduced directly: the
query *"How does 21 CFR 210.3 define a batch of drug product?"* returned an
unrelated ophthalmic-drops recall record as the #1 hit.

**Consequence:** documented, not fixed — a real fix needs either corpus
rebalancing (down-weighting or separately indexing the enforcement layer)
or a trained relevance model, both out of scope here.

## 4. Baseline generator (`answer.py`): brittle to paraphrasing

Two earlier cross-chunk reranking attempts were tried and discarded (see
the module's docstring) before landing on the current design: trust BM25's
own chunk-level ranking, only re-rank sentences *within* the one chunk it
already selected. This avoids introducing new bugs, but doesn't fix
retrieval misses (#2 above) — if BM25's #1 chunk is wrong, the baseline
generator confidently extracts text from the wrong section.

## 5. RAG generator (`generate_rag.py`, flan-t5-base): two distinct failure modes

- **Genuine hallucination** (model answers from its own pretrained
  knowledge, ignoring the retrieved context entirely) — reproduced with an
  out-of-domain query ("what is the weather today" → "rainy or cloudy",
  a word that appears nowhere in a pharma regulatory corpus). **This is
  what the faithfulness check catches** (content-word overlap between the
  answer and the retrieved context; threshold 0.45). Verified: the weather
  example scores 0.00 and is correctly suppressed.
- **Confident parroting of irrelevant-but-real retrieved text** — when
  retrieval itself surfaces the wrong chunk (failure mode #2/#3), the model
  faithfully summarizes that wrong chunk. The faithfulness check reports
  1.00 (the words *are* in the context) and does not catch this, because it
  isn't a faithfulness problem — it's a retrieval-precision problem.
  **This is an open, unsolved gap.** Fixing it needs either a trained
  relevance/reranking model or a much larger, better instruction-following
  LLM (a 250M-parameter model is simply not reliable at "notice the context
  doesn't answer this and say so").

## 6. Safety classifier: tuned for recall, not precision

Threshold chosen to guarantee recall ≥ 0.90 on the advice-seeking class
(achieved exactly: 0.900, with precision 1.00 on the held-out test split).
Tuning for recall means occasional false positives are the expected
trade-off. Reproduced directly: *"What is a Contamination Control Strategy
and what is it meant to establish?"* — a purely factual definitional
question — scored 0.50, just over the 0.493 threshold, and got flagged
with the advice-seeking disclaimer. This is an accepted cost: missing a
genuine advice-seeking question (false negative) is worse than an
occasional unnecessary disclaimer (false positive).

## 7. Cross-framework comparison: relative-score threshold is empirically tuned, not principled

`compare_frameworks.py` decides whether a framework has "real coverage" for
a query by comparing its best score to the query's own global-best score
(relative threshold = 0.5). This was calibrated by testing exactly two
queries (one where USP genuinely has no coverage, one where it's the best
match) — a reasonable spot-check, not a rigorous calibration. It may not
generalize perfectly to other queries/frameworks.

## 8. No ROUGE-L / BERTScore for generation quality

The phase plan's metrics-tracking item calls for these, but they require
reference *answer text* to score against. `data/eval/gold_passages.csv` has
gold *citations* (which chunk answers a question), never gold answer
prose — that annotation pass was never done. Rather than fabricate
reference answers after the fact, this is left as an explicit, stated gap
(see `eval/phase3_summary.json`), not silently omitted.

---

## Summary table

| # | Failure mode | Caught by the system? | Fix would require |
|---|---|---|---|
| 1 | Definition-question retrieval miss | No | Dense retrieval / better query expansion |
| 2 | Correct chunk not ranked #1 | Partially (runner-ups shown) | Cross-encoder reranker |
| 3 | Enforcement-record corpus skew | No | Corpus rebalancing or trained relevance model |
| 4 | Baseline generator wrong-section extraction | No (inherits #2) | Same as #2 |
| 5a | RAG genuine hallucination | **Yes** (faithfulness check) | — already handled |
| 5b | RAG parroting wrong-but-real content | No | Reranker or bigger LLM |
| 6 | Safety classifier false positive | Accepted trade-off | Larger labeled dataset, precision-recall re-tuning |
| 7 | Cross-framework threshold generalization | Unknown | Broader calibration set |
| 8 | No ROUGE-L/BERTScore | N/A (documented gap) | Author reference answers |
