"""
THE source registry — the single source of truth for Phase 1.

Every regulatory document, benchmark dataset, and enforcement/product source the
project uses is declared here once. Both `download_sources.py` and the manifest
builder read from this list, so adding a source in one place wires up its
download, its text extraction, and its manifest row automatically.

Fields
------
id           unique short slug (also the raw-file stem)
framework    FDA / EU / ICH / Swissmedic / ISO / USP / Benchmark / ...
title        human-readable document title
category     primary_kb | enforcement | product_ref | benchmark
type         regulation | guidance | structured | standard | pharmacopoeia | dataset
access       open | paywalled | registration
method       how to fetch: ecfr_api | openfda_api | direct_pdf | direct_html |
             scrape | hf_dataset | manual
url          canonical source URL (for the manifest + direct downloads)
extra        method-specific params (dict)

Sources come from two places, deliberately merged:
  • the Phase-1 table in the team's build plan (primary KB + benchmarks), and
  • the enforcement / product-reference layers discussed with the assistant
    (openFDA recalls, warning letters, Orange Book, Purple Book).
The `category` field keeps those layers separable downstream.
"""

ECFR_TITLE = 21

SOURCES = [
    # ------------------------------------------------------------------ #
    # PRIMARY KNOWLEDGE BASE — FDA regulations (eCFR API, open)
    # ------------------------------------------------------------------ #
    {"id": "cfr_21_210", "framework": "FDA", "title": "21 CFR 210 — cGMP in Manufacturing, Processing, Packing, or Holding of Drugs",
     "category": "primary_kb", "type": "regulation", "access": "open", "method": "ecfr_api",
     "url": "https://www.ecfr.gov/current/title-21/part-210", "extra": {"part": "210"}},
    {"id": "cfr_21_211", "framework": "FDA", "title": "21 CFR 211 — cGMP for Finished Pharmaceuticals",
     "category": "primary_kb", "type": "regulation", "access": "open", "method": "ecfr_api",
     "url": "https://www.ecfr.gov/current/title-21/part-211", "extra": {"part": "211"}},
    {"id": "cfr_21_600", "framework": "FDA", "title": "21 CFR 600 — Biological Products: General",
     "category": "primary_kb", "type": "regulation", "access": "open", "method": "ecfr_api",
     "url": "https://www.ecfr.gov/current/title-21/part-600", "extra": {"part": "600"}},
    {"id": "cfr_21_601", "framework": "FDA", "title": "21 CFR 601 — Licensing (Biologics)",
     "category": "primary_kb", "type": "regulation", "access": "open", "method": "ecfr_api",
     "url": "https://www.ecfr.gov/current/title-21/part-601", "extra": {"part": "601"}},
    {"id": "cfr_21_606", "framework": "FDA", "title": "21 CFR 606 — cGMP for Blood and Blood Components",
     "category": "primary_kb", "type": "regulation", "access": "open", "method": "ecfr_api",
     "url": "https://www.ecfr.gov/current/title-21/part-606", "extra": {"part": "606"}},
    {"id": "cfr_21_610", "framework": "FDA", "title": "21 CFR 610 — General Biological Products Standards",
     "category": "primary_kb", "type": "regulation", "access": "open", "method": "ecfr_api",
     "url": "https://www.ecfr.gov/current/title-21/part-610", "extra": {"part": "610"}},

    # ------------------------------------------------------------------ #
    # PRIMARY KNOWLEDGE BASE — FDA guidance PDFs (direct, open; verified URLs)
    # ------------------------------------------------------------------ #
    {"id": "fda_aseptic_2004", "framework": "FDA", "title": "Guidance: Sterile Drug Products Produced by Aseptic Processing (2004)",
     "category": "primary_kb", "type": "guidance", "access": "open", "method": "direct_pdf",
     "url": "https://www.fda.gov/media/71026/download", "extra": {}},
    {"id": "fda_process_val_2011", "framework": "FDA", "title": "Guidance: Process Validation — General Principles and Practices (2011)",
     "category": "primary_kb", "type": "guidance", "access": "open", "method": "direct_pdf",
     "url": "https://www.fda.gov/media/71021/download", "extra": {}},

    # DailyMed SPLs — full bulk is multi-GB; we pull a labelled sample via openFDA.
    {"id": "dailymed_spl_sample", "framework": "FDA", "title": "DailyMed Structured Product Labels (sample subset)",
     "category": "primary_kb", "type": "structured", "access": "open", "method": "openfda_api",
     "url": "https://api.fda.gov/drug/label.json",
     "extra": {"endpoint": "/drug/label.json", "sample_limit": 500}},

    # ------------------------------------------------------------------ #
    # PRIMARY KNOWLEDGE BASE — EU / ICH / Swissmedic
    # ------------------------------------------------------------------ #
    {"id": "eu_gmp_annex1_2022", "framework": "EU", "title": "EU GMP Annex 1 (2022) — Manufacture of Sterile Medicinal Products",
     "category": "primary_kb", "type": "guidance", "access": "open", "method": "direct_pdf",
     "url": "https://health.ec.europa.eu/document/download/e05af55b-38e9-42bf-8495-194bbf0b9262_en", "extra": {}},

    # ICH quality guideline PDFs live on database.ich.org. The filename pattern is
    # usually stable, but CONFIRM each URL once (see notes) — the downloader
    # degrades to 'partial' and keeps the page URL if a file 404s.
    {"id": "ich_q7", "framework": "ICH", "title": "ICH Q7 — Good Manufacturing Practice for APIs",
     "category": "primary_kb", "type": "guidance", "access": "open", "method": "direct_pdf",
     "url": "https://database.ich.org/sites/default/files/Q7%20Guideline.pdf",
     "extra": {"landing": "https://www.ich.org/page/quality-guidelines"}},
    {"id": "ich_q8", "framework": "ICH", "title": "ICH Q8(R2) — Pharmaceutical Development",
     "category": "primary_kb", "type": "guidance", "access": "open", "method": "direct_pdf",
     "url": "https://database.ich.org/sites/default/files/Q8%28R2%29%20Guideline.pdf",
     "extra": {"landing": "https://www.ich.org/page/quality-guidelines"}},
    {"id": "ich_q9", "framework": "ICH", "title": "ICH Q9(R1) — Quality Risk Management",
     "category": "primary_kb", "type": "guidance", "access": "open", "method": "direct_pdf",
     "url": "https://database.ich.org/sites/default/files/ICH_Q9(R1)_Guideline_Step4_2025_0115_0.pdf",
     "extra": {"landing": "https://www.ich.org/page/quality-guidelines"}},
    {"id": "ich_q10", "framework": "ICH", "title": "ICH Q10 — Pharmaceutical Quality System",
     "category": "primary_kb", "type": "guidance", "access": "open", "method": "direct_pdf",
     "url": "https://database.ich.org/sites/default/files/Q10%20Guideline.pdf",
     "extra": {"landing": "https://www.ich.org/page/quality-guidelines"}},

    {"id": "swissmedic_gmp", "framework": "Swissmedic", "title": "Swissmedic GMP Guidelines & Annex 1 Implementation Guidance",
     "category": "primary_kb", "type": "guidance", "access": "open", "method": "direct_html",
     "url": "https://www.swissmedic.ch/swissmedic/en/home/authorisation/manufacturing-licences-gmp/gmp-guidelines.html", "extra": {}},

    # ------------------------------------------------------------------ #
    # PRIMARY KNOWLEDGE BASE — paywalled (documented, NOT downloaded)
    # ------------------------------------------------------------------ #
    {"id": "iso_14644_1", "framework": "ISO", "title": "ISO 14644-1 — Cleanrooms: Classification of Air Cleanliness",
     "category": "primary_kb", "type": "standard", "access": "paywalled", "method": "manual",
     "url": "https://www.iso.org/standard/53394.html", "extra": {"note": "Buy via ISO webstore or institutional library. Use classification tables + cite by clause; do not ingest full text."}},
    {"id": "iso_14644_2", "framework": "ISO", "title": "ISO 14644-2 — Cleanrooms: Monitoring",
     "category": "primary_kb", "type": "standard", "access": "paywalled", "method": "manual",
     "url": "https://www.iso.org/standard/53392.html", "extra": {"note": "Paywalled. Cite by clause only."}},
    {"id": "iso_13485", "framework": "ISO", "title": "ISO 13485 — Medical Devices QMS",
     "category": "primary_kb", "type": "standard", "access": "paywalled", "method": "manual",
     "url": "https://www.iso.org/standard/59752.html", "extra": {"note": "Paywalled. Cite by clause only."}},
    {"id": "usp_chapters", "framework": "USP", "title": "USP General Chapters <1>, <85>, <797>, <1116>",
     "category": "primary_kb", "type": "pharmacopoeia", "access": "paywalled", "method": "manual",
     "url": "https://www.usp.org/USP-NF/General-Chapters", "extra": {"note": "Requires USP-NF subscription. Reference by chapter number; do not ingest full text."}},
    {"id": "usp_232", "framework": "USP", "title": "USP General Chapter <232> Elemental Impurities—Limits",
     "category": "primary_kb", "type": "pharmacopoeia", "access": "paywalled", "method": "manual",
     "url": "https://www.usp.org/USP-NF/General-Chapters",
     "extra": {"note": "Manually obtained via institutional USP-NF access; PDF placed directly in data/raw/."}},
    {"id": "usp_233", "framework": "USP", "title": "USP General Chapter <233> Elemental Impurities—Procedures",
     "category": "primary_kb", "type": "pharmacopoeia", "access": "paywalled", "method": "manual",
     "url": "https://www.usp.org/USP-NF/General-Chapters",
     "extra": {"note": "Manually obtained via institutional USP-NF access; PDF placed directly in data/raw/."}},

    # ------------------------------------------------------------------ #
    # ENFORCEMENT LAYER (chat additions) — case-study examples, NOT primary answers
    # ------------------------------------------------------------------ #
    {"id": "openfda_drug_enforcement", "framework": "FDA", "title": "openFDA Drug Enforcement / Recall Reports",
     "category": "enforcement", "type": "structured", "access": "open", "method": "openfda_api",
     "url": "https://api.fda.gov/drug/enforcement.json",
     "extra": {"endpoint": "/drug/enforcement.json", "date_field": "report_date",
               "product_filter": 'product_type:"drugs"'}},
    {"id": "fda_warning_letters", "framework": "FDA", "title": "FDA Warning Letters (drug/biologic cGMP)",
     "category": "enforcement", "type": "structured", "access": "open", "method": "scrape",
     "url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/warning-letters",
     "extra": {"note": "No official text API; scrape the JSON feed behind the datatable. Selectors need confirming against the live page."}},

    # ------------------------------------------------------------------ #
    # PRODUCT / BIOLOGICS REFERENCE (chat additions)
    # ------------------------------------------------------------------ #
    {"id": "openfda_orange_book", "framework": "FDA", "title": "FDA Orange Book / Drugs@FDA (approvals, therapeutic equivalence)",
     "category": "product_ref", "type": "structured", "access": "open", "method": "openfda_api",
     "url": "https://api.fda.gov/drug/drugsfda.json",
     "extra": {"endpoint": "/drug/drugsfda.json", "sample_limit": 1000}},
    {"id": "fda_purple_book", "framework": "FDA", "title": "FDA Purple Book — Licensed Biological Products & Biosimilars",
     "category": "product_ref", "type": "structured", "access": "open", "method": "manual",
     "url": "https://purplebooksearch.fda.gov/",
     "extra": {"note": "Downloadable database export on purplebooksearch.fda.gov. Confirm the current export file URL, then switch method to direct_pdf/direct_html or drop the CSV into data/raw manually."}},

    # ------------------------------------------------------------------ #
    # BENCHMARK DATASETS
    # ------------------------------------------------------------------ #
    {"id": "billsum", "framework": "Benchmark", "title": "BillSum — US legislation summarization",
     "category": "benchmark", "type": "dataset", "access": "open", "method": "hf_dataset",
     "url": "https://huggingface.co/datasets/FiscalNote/billsum", "extra": {"hf_id": "FiscalNote/billsum"}},
    {"id": "eurlex57k", "framework": "Benchmark", "title": "EURLEX-57K — multi-label legal classification",
     "category": "benchmark", "type": "dataset", "access": "open", "method": "hf_dataset",
     "url": "https://huggingface.co/datasets/coastalcph/lex_glue",
     "extra": {"hf_id": "coastalcph/lex_glue", "hf_config": "eurlex"}},
    {"id": "bioasq", "framework": "Benchmark", "title": "BioASQ — Biomedical Semantic Indexing & QA",
     "category": "benchmark", "type": "dataset", "access": "registration", "method": "manual",
     "url": "http://www.bioasq.org",
     "extra": {"note": "Free but requires registration BEFORE download. Register in Phase 1; place files in data/raw/bioasq/."}},
]


def by_id(source_id):
    for s in SOURCES:
        if s["id"] == source_id:
            return s
    raise KeyError(source_id)
