#!/usr/bin/env python
"""
split_questions.py — Phase 2 deliverable (train/test split)

Splits the approved question set into an 80/20 train/test split for the
Phase 3 intent/framework classifier, stratified by intent so small classes
(e.g. pharmacopoeia spec, 10 rows) aren't wiped out of either side.

The full approved set is also kept intact as data/eval/gold_passages.csv —
per the phase plan, retrieval evaluation (Recall@K/MRR/nDCG) uses the whole
held-out gold set, not a train/test split; splitting it would just shrink
the Recall@K denominator for no benefit.

Usage
-----
  python split_questions.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "eval" / "questions_draft.csv"
GOLD_PATH = ROOT / "data" / "eval" / "gold_passages.csv"
TRAIN_PATH = ROOT / "data" / "eval" / "classifier_train.csv"
TEST_PATH = ROOT / "data" / "eval" / "classifier_test.csv"

SEED = 42


def main():
    df = pd.read_csv(QUESTIONS_PATH, dtype=str, keep_default_na=False)
    approved = df[df["status"] == "approved"].reset_index(drop=True)
    print(f"{len(approved)} approved rows (of {len(df)} total)")

    # Full approved set stays intact as the retrieval gold-passage set.
    approved.to_csv(GOLD_PATH, index=False)
    print(f"gold passage set (unsplit) -> {GOLD_PATH} ({len(approved)} rows)")

    # 80/20 stratified split by intent for the Phase 3 classifier.
    train, test = train_test_split(
        approved, test_size=0.2, random_state=SEED, stratify=approved["intent"]
    )
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"classifier train -> {TRAIN_PATH} ({len(train)} rows)")
    print(f"classifier test  -> {TEST_PATH} ({len(test)} rows)")

    print("\ntrain intent distribution:")
    print(train["intent"].value_counts())
    print("\ntest intent distribution:")
    print(test["intent"].value_counts())


if __name__ == "__main__":
    main()
