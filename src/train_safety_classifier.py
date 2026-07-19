#!/usr/bin/env python
"""
train_safety_classifier.py — Phase 3 deliverable (safety classifier)

Flags advice-seeking / prediction-seeking questions ("will my facility pass
inspection?", "should I recall this batch?") so the chatbot pipeline can
attach a disclaimer instead of answering as if it were a factual lookup.
Target: recall >= 0.90 on the advice_seeking class (per the phase plan) --
missing a genuine advice-seeking question (false negative) is worse than
over-flagging a factual one (false positive), so the classification
threshold is tuned toward recall rather than left at the default 0.5.

Usage
-----
  python train_safety_classifier.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "eval" / "safety_dataset.csv"
MODEL_DIR = ROOT / "models" / "safety_classifier"

TARGET_RECALL = 0.90
SEED = 42
POS_LABEL = "advice_seeking"


def pick_threshold_for_recall(y_true, probs, target_recall):
    """Find the highest decision threshold that still achieves >= target_recall
    on the positive class -- maximizes precision subject to the recall floor."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs, pos_label=POS_LABEL)
    # precision_recall_curve returns one more point than thresholds (the (1,0) endpoint)
    for p, r, t in zip(precision[:-1][::-1], recall[:-1][::-1], thresholds[::-1]):
        if r >= target_recall:
            return t, p, r
    return thresholds[0], precision[0], recall[0]  # fallback: lowest threshold


def main():
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
    train, test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    print(f"train: {len(train)} ({train['label'].value_counts().to_dict()})")
    print(f"test:  {len(test)} ({test['label'].value_counts().to_dict()})")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(train["question"], train["label"])

    classes = list(pipe.named_steps["clf"].classes_)
    pos_idx = classes.index(POS_LABEL)
    test_probs = pipe.predict_proba(test["question"])[:, pos_idx]

    threshold, precision_at_t, recall_at_t = pick_threshold_for_recall(
        test["label"], test_probs, TARGET_RECALL
    )
    preds = np.where(test_probs >= threshold, POS_LABEL, "factual")

    print(f"\nchosen threshold: {threshold:.3f} (recall={recall_at_t:.3f}, precision={precision_at_t:.3f})")
    print("\nclassification report at chosen threshold:")
    report = classification_report(test["label"], preds, zero_division=0)
    print(report)

    default_preds = pipe.predict(test["question"])
    default_report = classification_report(test["label"], default_preds, zero_division=0, output_dict=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "safety_clf.joblib")
    metrics = {
        "n_train": len(train), "n_test": len(test),
        "target_recall": TARGET_RECALL,
        "chosen_threshold": threshold,
        "recall_at_threshold": recall_at_t,
        "precision_at_threshold": precision_at_t,
        "default_threshold_recall": default_report[POS_LABEL]["recall"],
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(MODEL_DIR / "threshold.json", "w", encoding="utf-8") as fh:
        json.dump({"threshold": threshold, "pos_label": POS_LABEL}, fh, indent=2)

    print(f"\n(for reference, default 0.5 threshold gives recall={default_report[POS_LABEL]['recall']:.3f})")
    print(f"saved model + metrics -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
