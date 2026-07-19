# Regulatory Compliance Q&A Chatbot — Build Roadmap
### Team 14 — RAG-based NLP Chatbot for Biopharmaceutical Regulatory Compliance

This roadmap turns the proposal into 4 buildable phases, each with concrete steps and deliverables. RAG (retrieval-augmented generation) is the backbone of the whole system, since the proposal explicitly requires grounded, citation-backed answers.

---

## Phase 1 — Data Collection & Infrastructure Setup

**Goal:** Get every source document and benchmark dataset onto disk in a clean, versioned, reusable form.

1. **Project scaffolding** — repo structure (`/data/raw`, `/data/processed`, `/src`, `/models`, `/eval`), virtual env, `requirements.txt` (transformers, sentence-transformers, faiss-cpu, rank-bm25, pandas, requests, beautifulsoup4/pdfplumber).
2. **Collect the primary regulatory knowledge base**:
   - FDA: 21 CFR Part 210 & 211 (eCFR), Aseptic Processing Guidance (2004), Process Validation Guidance (2011), 21 CFR Parts 600–610.
   - FDA DailyMed SPLs (sample subset for classification examples).
   - EU: GMP Annex 1 (2022).
   - ISO: 14644-1, 14644-2, 13485 (note: some ISO standards are paywalled — confirm access/summarized-source strategy).
   - Swissmedic GMP guidelines.
   - USP General Chapters <1>, <85>, <797>, <1116>, <232>/<233>.
   - ICH Q7, Q8, Q9, Q10.
3. **Collect NLP benchmark datasets**: BioASQ, EURLEX-57K, BillSum (via HuggingFace `datasets`).
4. **Build a document registry** — a manifest (CSV/JSON) tracking: source, title, framework, retrieval date, URL, file path, license/usage note.
5. **Raw storage format** — save originals as-is (PDF/HTML), plus a plain-text extraction pass (pdfplumber/PyMuPDF for PDFs, BeautifulSoup for HTML), so you never re-scrape.

**Deliverable:** a populated `/data/raw` + `/data/text` directory, a document manifest, and a working scraping/download pipeline you can re-run.

---

## Phase 2 — Data Mapping & Knowledge Base Construction

**Goal:** Turn raw text into a structured, chunked, labeled corpus ready for retrieval and classification.

1. **Section-level chunking** — split each document by its native section/clause numbering (not fixed token windows), preserving hierarchy.
2. **Metadata tagging per chunk** — framework, document title, section number, source URL, doc date (this is what makes citations possible later).
3. **Build the annotated question dataset** — write/collect representative regulatory questions, each labeled with: intent (GMP requirement, sterile manufacturing, process validation, environmental monitoring, pharmacopoeia spec, definition) and framework (FDA/ISO/EU/Swissmedic/USP/multi).
4. **Gold passage set** — for 200–300 held-out questions, manually link the correct supporting chunk(s) — this is required for Recall@K/MRR/nDCG evaluation later.
5. **Splits** — 80/20 train/test for classification; separate held-out gold set for retrieval/QA.
6. **Indexing prep** — BM25 index (rank-bm25 or Elasticsearch) over chunks; embedding index scaffold (FAISS) ready for Phase 3 vectors.

**Deliverable:** a clean chunked+labeled corpus, a manifest-linked metadata store, a BM25 index, and a labeled question/gold-passage dataset.

---

## Phase 3 — Model Development & Training

**Goal:** Build classification, retrieval, and generation components at baseline → intermediate → advanced tiers, per the proposal.

1. **Intent/framework classification**
   - Baseline: TF-IDF + Logistic Regression.
   - Intermediate: fine-tune PubMedBERT on the labeled question set.
   - Additional: single-layer CNN classifier with corpus-trained embeddings (for course alignment).
2. **Retrieval**
   - Baseline: BM25 over chunks (from Phase 2).
   - Intermediate: dense retrieval — all-MiniLM-L6-v2, then a PubMedBERT-based sentence encoder; FAISS ANN index.
   - Evaluation: Recall@K, MRR, nDCG against the gold passage set.
3. **Answer generation (RAG)**
   - Concatenate top-k retrieved chunks + query → seq2seq generator, constrained to grounded answers with section/URL citation appended.
   - Summarization: BART or T5 fine-tuned on BillSum.
   - Faithfulness check: verify each generated claim traces to a retrieved span (source-support score).
4. **Advanced tier**
   - Cross-framework comparison module (multi-framework retrieval → structured comparison table).
   - Reranker before generation.
5. **Safety classifier** — flag advice-seeking questions (e.g., inspection outcome predictions) for disclaimers; target recall ≥90%.
6. **Metrics tracking** — log macro-F1, Recall@K/MRR/nDCG, ROUGE-L, BERTScore across all tiers for the final comparison table in your report.

**Deliverable:** trained baseline/intermediate/advanced models for each of the three components, with logged evaluation metrics.

---

## Phase 4 — Chatbot Integration, Evaluation & Delivery

**Goal:** Wire everything into a working chatbot, validate it, and package it for submission/demo.

1. **Pipeline integration** — query → intent classifier → framework-scoped retrieval → reranker (if advanced) → generator → citation formatter → safety-flag check.
2. **Interface** — a simple chat UI (Streamlit/Gradio is fastest) or API endpoint, showing the answer plus its cited section/source.
3. **Human evaluation** — team review of 50 representative questions for accuracy, citation correctness, clarity.
4. **Safety evaluation** — run the held-out safety set, confirm ≥90% recall on flaggable questions.
5. **Error analysis** — where retrieval fails (OOV terms, ambiguous questions), where generation hallucinates despite grounding.
6. **Final report/write-up** — results tables (baseline vs intermediate vs advanced), expected-vs-actual metrics comparison, limitations, future work.
7. **Packaging** — repo cleanup, README, reproducibility instructions, demo script/recording.

**Deliverable:** a working demo chatbot, evaluation results, and the final report/presentation.

---

## Suggested order of attack
Phase 1 → Phase 2 → Phase 3 (baseline first, all three components, before moving to intermediate/advanced) → Phase 4. Within Phase 3, get one end-to-end baseline RAG loop working (BM25 + TF-IDF classifier + simple generator) before investing in PubMedBERT/FAISS/advanced tiers — it de-risks integration problems early.
