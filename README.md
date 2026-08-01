Regulatory Compliance Q&A Chatbot

A RAG-based chatbot over FDA / EU / ICH / USP pharmaceutical regulatory text —
21 CFR parts 210/211/600/601/606/610, FDA sterile-processing and process-
validation guidance, EU GMP Annex 1 (2022), ICH Q7/Q8/Q9/Q10, USP `<232>`/`<233>`,
plus an openFDA enforcement/recall layer and two NLP benchmark datasets
(BillSum, EURLEX-57K).

Retrieval is **BM25-only** — dense retrieval/FAISS was evaluated and
deliberately dropped from scope in favor of shipping the full baseline
pipeline (see `docs/ERROR_ANALYSIS.md` for why, and what that trades off).

## Repository structure

```
pharma-chatbot/
├── app.py                       ← Phase 4: Streamlit chat UI (entry point)
├── requirements.txt
├── docs/
│   ├── PHASE_PLAN.md            ← the build roadmap this repo follows
│   ├── ERROR_ANALYSIS.md        ← every known failure mode, reproduced + explained
│   ├── MANUAL_COLLECTION.md     ← which sources needed manual (non-scripted) collection
│   └── document_manifest.csv    ← Phase 1: tracked record of every source
├── src/
│   ├── sources.py                ← THE source registry (single source of truth)
│   ├── manifest.py, http_util.py, state.py   ← Phase 1 plumbing
│   ├── ecfr.py, openfda.py, direct.py,
│   │   warning_letters.py, benchmarks.py     ← Phase 1 per-source downloaders
│   ├── download_sources.py, extract_text.py  ← Phase 1 deliverables
│   ├── chunk.py, build_index.py              ← Phase 2: chunking + BM25 index
│   ├── review_questions.py, split_questions.py ← Phase 2: annotation review + train/test split
│   ├── train_classifier.py                   ← Phase 3: intent/framework classifier (baseline)
│   ├── eval_retrieval.py                      ← Phase 3: Recall@K / MRR / nDCG
│   ├── answer.py                              ← Phase 3: baseline extractive generator
│   ├── generate_rag.py                        ← Phase 3: intermediate RAG generator (flan-t5-base)
│   ├── train_safety_classifier.py, safety.py ← Phase 3: advice-seeking flagger
│   ├── compare_frameworks.py                  ← Phase 3 advanced tier: cross-framework comparison
│   ├── summarize_metrics.py                   ← Phase 3: consolidated metrics report
│   ├── rerank.py                              ← Phase 4: cross-encoder reranker (off by default, see below)
│   ├── polish.py                              ← Phase 4: single-answer paraphrase (flan-t5-large)
│   └── pipeline.py                            ← Phase 4: wires everything together (default generator: "polished")
├── data/{raw,text,processed}/     raw = originals · text = extractions · processed = chunks.jsonl
├── data/eval/                      questions_draft.csv → gold_passages.csv (reviewed) → splits
├── models/{baseline_classifier,bm25,safety_classifier}/
├── eval/                           retrieval_metrics.json, phase3_summary.json, human_eval.csv
└── state/watermarks.json           incremental-refresh state
```

## Setup

```bash
python -m venv venv && venv\Scripts\activate    # (or source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
export OPENFDA_API_KEY="your_free_key"          # https://open.fda.gov/apis/authentication/ (optional, raises rate limit)
```

All commands below assume you're running from the repo root with the venv
active, invoking scripts as modules (`python -m src.<name>`) so their
internal `from src import ...` imports resolve regardless of working
directory.

## Run order (all four phases, start to finish)

```bash
# Phase 1 — acquire + extract every source
python -m src.download_sources
python -m src.extract_text

# Phase 2 — chunk, index, build the labeled question/gold-passage set
python -m src.chunk
python -m src.build_index
python -m src.review_questions        # interactive: approve/edit/reject drafted questions
python -m src.split_questions         # 80/20 classifier split + gold_passages.csv

# Phase 3 — train/evaluate every model tier
python -m src.train_classifier        # intent + framework baseline classifiers
python -m src.eval_retrieval          # Recall@K / MRR / nDCG vs. the gold set
python -m src.train_safety_classifier # advice-seeking flagger, recall-tuned
python -m src.summarize_metrics       # consolidates everything above into one report

# Phase 4 — run the integrated chatbot
streamlit run app.py                  # interactive UI (default generator: polished)
# or, from the CLI directly:
python -m src.pipeline --query "What is a Contamination Control Strategy?"                 # polished (default)
python -m src.pipeline --query "..." --generator baseline                                  # raw single-citation extract
python -m src.pipeline --query "..." --generator rag                                       # Phase 3 flan-t5-base tier
python -m src.pipeline --query "..." --use-reranker                                        # opt-in cross-encoder (see caveat below)
python -m src.compare_frameworks --query "How many APS runs validate an aseptic process?"
python -m src.run_human_eval          # generates eval/human_eval.csv for manual scoring

# Optional: quantify the reranker's effect on retrieval quality
python -m src.eval_retrieval --rerank # writes eval/retrieval_metrics_reranked.json
```

## Current results (see `eval/phase3_summary.json` for the machine-readable version)

| Component | Result |
|---|---|
| Intent classifier (TF-IDF + LogReg) | macro-F1 0.726 (118 train examples) |
| Framework classifier (TF-IDF + LogReg) | macro-F1 0.805 |
| Retrieval (BM25, 148-question gold set) | Recall@5 87.8%, Recall@10 93.2%, MRR 0.732 |
| Retrieval + cross-encoder reranker | MRR 0.749 (▲) but Recall@5 83.8%, Recall@10 89.2% (▼ both) — mixed, **off by default** |
| Relevance filter (`pipeline.filter_relevant`) | Drops chunks below 50% of the query's own top BM25 score instead of forcing a fixed count |
| Safety classifier (advice-seeking flagger) | Recall 0.900 (target met), precision 1.00 |
| Baseline generator | Single-citation extract, zero hallucination risk |
| Intermediate generator (flan-t5-base, Phase 3 "rag" tier) | Faithfulness-checked RAG synthesis |
| **Polished generator (default, flan-t5-large)** | Relevance-filtered multi-citation synthesis rewritten as one answer, faithfulness-checked against its own verbatim source |

**Read `docs/ERROR_ANALYSIS.md` before trusting any of the above at face
value** — it documents exactly where and why each component fails,
reproduced with concrete examples, not just the passing-case numbers.

## Known limitations (see `docs/ERROR_ANALYSIS.md` for full detail)

- No dense retrieval/FAISS — BM25 (+ optional reranker) was a deliberate scope decision.
- The cross-encoder reranker is a genuine mixed bag, not a clean win — MRR
  improves but Recall@5/10 gets worse, likely a domain mismatch (trained on
  general web search, not regulatory/legal text). Off by default; opt in with
  `--use-reranker` / the UI checkbox if you want to compare it yourself.
- Definition-style questions are retrieval's weakest spot (vocabulary
  mismatch between how questions are phrased and how source text defines terms).
- The relevance filter and faithfulness checks catch invented content and
  padding-in of low-relevance chunks, but not a confident summary of a
  *wrong-but-real* retrieved chunk if BM25's whole candidate pool missed —
  that's a retrieval-recall gap, not something a downstream filter/check can fix.
- No ROUGE-L/BERTScore — those need reference answer text, which was never
  authored (the gold set has citations, not gold answer prose).
- Small training sets throughout (118 classifier examples, 50 safety
  examples) — indicative baselines, not production-scale.
- The "polished" generator's flan-t5-large download (~3GB) is a one-time
  cost on first use; chosen over a hosted GPT API deliberately, to keep this
  a locally-run, no-cost, no-API-key part of the project's own pipeline.
