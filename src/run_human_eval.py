#!/usr/bin/env python
"""
run_human_eval.py — Phase 4 deliverable (human evaluation harness)

The phase plan calls for "team review of 50 representative questions for
accuracy, citation correctness, clarity." That review has to be done by an
actual person -- this script does the other half: sample 50 questions
stratified across intent/framework (so "representative" is more than a
word), run BOTH generators (baseline + rag) through the real pipeline, and
write everything to a CSV with empty score columns ready for a human to
fill in. It does not grade itself.

Usage
-----
  python run_human_eval.py                 # sample 50, run both generators
  python run_human_eval.py --n 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.pipeline import handle_query

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "eval" / "gold_passages.csv"
OUT_PATH = ROOT / "eval" / "human_eval.csv"

SEED = 42


def stratified_sample(df, n):
    """Proportional sample across intent, so 'representative' actually
    reflects the real intent distribution rather than picking randomly."""
    frac = n / len(df)
    parts = [
        group.sample(max(1, round(len(group) * frac)), random_state=SEED)
        for _, group in df.groupby("intent")
    ]
    sampled = pd.concat(parts)
    return sampled.head(n).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Phase 4 human evaluation harness")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    gold = pd.read_csv(GOLD_PATH, dtype=str, keep_default_na=False)
    sample = stratified_sample(gold, args.n)
    print(f"sampled {len(sample)} questions across intents: "
          f"{sample['intent'].value_counts().to_dict()}")

    rows = []
    for i, row in sample.iterrows():
        print(f"[{i+1}/{len(sample)}] {row['question_id']}", end="\r")
        for gen in ("baseline", "rag"):
            result = handle_query(row["question"], generator=gen, top_k=3)
            citation_str = "; ".join(
                f"{c['framework']} {c['title']} § {c['section']}" for c in result["citations"]
            )
            rows.append({
                "question_id": row["question_id"],
                "question": row["question"],
                "gold_chunk_ids": row["gold_chunk_ids"],
                "generator": gen,
                "predicted_intent": result["predicted_intent"],
                "predicted_framework": result["predicted_framework"],
                "answer": result["answer"],
                "citations": citation_str,
                "safety_flagged": result["safety_flagged"],
                # empty columns for the human reviewer to fill in:
                "accuracy_1to5": "",
                "citation_correct_yn": "",
                "clarity_1to5": "",
                "reviewer_notes": "",
            })

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\n\nwrote {len(out_df)} rows ({len(sample)} questions x 2 generators) -> {OUT_PATH}")
    print("Fill in accuracy_1to5 / citation_correct_yn / clarity_1to5 / reviewer_notes by hand.")


if __name__ == "__main__":
    main()
