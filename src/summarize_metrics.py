#!/usr/bin/env python
"""
summarize_metrics.py — Phase 3 deliverable (metrics tracking)

Consolidates every metric produced across Phase 3 (classifier macro-F1,
retrieval Recall@K/MRR/nDCG, safety-classifier recall/precision) into one
file for the report's tier-comparison table.

Deliberately NOT included: ROUGE-L / BERTScore for the generators. Those
compare a generated answer against a REFERENCE answer, and we never
authored reference answers -- data/eval/gold_passages.csv has gold
*citations* (which chunk answers a question), not gold *answer text*.
Computing ROUGE-L/BERTScore here would mean fabricating reference answers
after the fact, which isn't reliable ground truth. This gap is called out
explicitly rather than silently omitted, and stays open as future work if
those reference answers get written.

Usage
-----
  python summarize_metrics.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "eval" / "phase3_summary.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main():
    classifier = _load(ROOT / "models" / "baseline_classifier" / "metrics.json")
    retrieval = _load(ROOT / "eval" / "retrieval_metrics.json")
    safety = _load(ROOT / "models" / "safety_classifier" / "metrics.json")

    summary = {
        "classification": {
            "n_train": classifier["n_train"], "n_test": classifier["n_test"],
            "intent_macro_f1": classifier["intent"]["macro_f1"],
            "framework_macro_f1": classifier["framework"]["macro_f1"],
            "caveat": "small dataset (118 train examples) -- indicative baseline, not production-grade",
        },
        "retrieval": {
            "n_questions": retrieval["n_questions"],
            "mrr": retrieval["mrr"],
            "recall_at_k": retrieval["recall_at_k"],
            "ndcg_at_k": retrieval["ndcg_at_k"],
            "caveat": "definition-intent questions are a documented weak spot (6 of 8 misses)",
        },
        "safety_classifier": {
            "target_recall": safety["target_recall"],
            "achieved_recall": safety["recall_at_threshold"],
            "achieved_precision": safety["precision_at_threshold"],
            "chosen_threshold": safety["chosen_threshold"],
            "default_threshold_recall_for_comparison": safety["default_threshold_recall"],
        },
        "generation": {
            "baseline_tier": "answer.py -- extractive, zero hallucination risk, brittle to phrasing",
            "intermediate_tier": "generate_rag.py -- flan-t5-base + faithfulness check "
                                 "(catches invented content, not retrieval-irrelevant-but-real content)",
            "rouge_l": None,
            "bertscore": None,
            "why_missing": "no reference answer text exists to score against -- gold_passages.csv "
                            "has gold CITATIONS, not gold ANSWER TEXT. Would need that annotation "
                            "pass first; not fabricated here.",
        },
        "advanced_tier": {
            "cross_framework_comparison": "compare_frameworks.py -- per-framework top match with "
                                           "relative-score coverage check (threshold=0.5, empirically "
                                           "calibrated, see module docstring)",
            "reranker": "not built -- judged not worth the added complexity relative to Phase 4 "
                        "integration work; documented as an open option, not silently dropped",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
