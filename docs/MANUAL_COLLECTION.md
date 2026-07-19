# Manual Data Collection Guide

`download_sources.py` only works for sources that are (a) open-access and (b) a **direct file URL**.
Several of your sources fail one of those two conditions. Here's what needs manual handling, and why.

---

## 1. Landing pages, not direct files (script downloads the wrong thing)

These URLs point to a webpage that *links to* the real file — the script just saves that
webpage's HTML, not the document itself.

| Source | What to do |
|---|---|
| **Swissmedic GMP Guidelines** | Visit https://www.swissmedic.ch/swissmedic/en/home/authorisation/manufacturing-licences-gmp/gmp-guidelines.html — the page lists multiple guideline PDFs. Open each linked document and save it to `data/raw/swissmedic/`. |
| **FDA DailyMed SPLs** | Visit https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm — this offers a **bulk ZIP download** of structured product labels (large, updated periodically). Download the ZIP manually, unzip into `data/raw/fda/dailymed/`. You likely only need a sample subset (e.g. a few hundred SPLs), not the full corpus, for classification examples. |
| **ICH Quality Guidelines** *(you already got this working — just confirming why)* | https://www.ich.org/page/quality-guidelines is itself a landing page; the actual Q7/Q8/Q9/Q10 PDFs are linked from it. Since these downloaded for you, you likely followed the individual links already — good. Double check Q7, Q8, Q9, and Q10 all downloaded (4 separate PDFs), not just one. |

## 2. Paywalled / subscription-required (script correctly skips these)

You cannot download these with a script — they require a paid account or institutional access.

| Source | What to do |
|---|---|
| **ISO 14644-1, 14644-2, ISO 13485** | Check whether your university library has an ANSI Webstore or Techstreet subscription — many universities do, even if the campus library site doesn't advertise it directly. Ask a librarian. If no access, you can still describe/cite the *particle count limits* — those specific numeric limits from ISO 14644-1 are reproduced in EU Annex 1 §4.27 (Table 1), which you already have, so you're not fully blocked for the RAG corpus, just missing the primary ISO source text itself. |
| **USP General Chapters <1>, <85>, <797>, <1116>, <232>/<233>** | Same issue — usp.org/USP-NF/General-Chapters requires a paid USP-NF subscription. Check university library access first. If genuinely unavailable, note this as a **limitation** in your proposal/report rather than substituting unofficial paraphrases, since exact compendial wording matters for a compliance chatbot. |

## 3. Datasets — need the `datasets` library, not `requests`

These aren't single downloadable files; they're structured datasets served through APIs/libraries.

```bash
pip install datasets
```

```python
from datasets import load_dataset

# EURLEX-57K (via LexGLUE)
eurlex = load_dataset("lex_glue", "eurlex")

# BillSum
billsum = load_dataset("FiscalNote/billsum")
```

Save these locally with `.save_to_disk("../data/raw/benchmarks/eurlex")` etc. so you're not
re-downloading every run.

**BioASQ** is different again — it requires **free registration** at http://www.bioasq.org
before you can access task data (no direct download link exists). Register, then download
the specific task's training/test files from your BioASQ account.

## 4. If EU Annex 1 also failed to download

The PDF URL (`https://health.ec.europa.eu/system/files/2022-08/20220825_gmp-an1_en_0.pdf`) is a
direct file link and should work with `requests`. If it failed, it's likely the server rejected
the generic `requests` User-Agent or timed out on the 58-page PDF. Try increasing `timeout=60` in
`download_sources.py`, or download it manually via a browser and place it in `data/raw/eu/`.

---

## Summary of what's genuinely achievable right now

- ✅ **Scriptable today**: FDA (21 CFR 210/211/600-610, guidance PDFs), EU Annex 1, ICH Q7-Q10
- 🖱️ **Manual, but free**: Swissmedic (click through), DailyMed (bulk ZIP), BioASQ (register first)
- 📚 **Needs institutional access**: ISO 14644-1/2, ISO 13485, USP General Chapters
- 🐍 **Needs `datasets` library, not `requests`**: EURLEX-57K, BillSum

This is a reasonable corpus even without ISO/USP text directly — FDA + EU Annex 1 alone give you
two full regulatory frameworks (the core of your proposal), and ICH gives cross-cutting quality
guidelines. If ISO/USP access doesn't come through, scope the "multi-framework" comparison feature
to FDA/EU/ICH/Swissmedic and note ISO/USP as a stated limitation rather than blocking the whole project.
