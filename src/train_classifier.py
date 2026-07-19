#!/usr/bin/env python
"""
train_classifier.py — Phase 3 deliverable (baseline intent/framework classifier)

Baseline tier: TF-IDF + Logistic Regression, trained separately for intent and
framework labels on the Phase 2 question set (classifier_train.csv /
classifier_test.csv). Reports macro-F1 (the metric the phase plan tracks
across tiers) plus a full per-class report, and saves both fitted pipelines
to models/baseline_classifier/ for reuse in the chatbot pipeline.

Usage
-----
  python train_classifier.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "eval" / "classifier_train.csv"
TEST_PATH = ROOT / "data" / "eval" / "classifier_test.csv"
MODEL_DIR = ROOT / "models" / "baseline_classifier"

TARGETS = ["intent", "framework"]


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def main():
    train = pd.read_csv(TRAIN_PATH, dtype=str, keep_default_na=False)
    test = pd.read_csv(TEST_PATH, dtype=str, keep_default_na=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {"n_train": len(train), "n_test": len(test)}
    for target in TARGETS:
        pipe = build_pipeline()
        pipe.fit(train["question"], train[target])
        preds = pipe.predict(test["question"])

        macro_f1 = f1_score(test[target], preds, average="macro", zero_division=0)
        report = classification_report(test[target], preds, zero_division=0)

        print(f"\n=== {target} classifier (baseline: TF-IDF + LogReg) ===")
        print(f"macro-F1: {macro_f1:.3f}")
        print(report)

        joblib.dump(pipe, MODEL_DIR / f"{target}_clf.joblib")
        metrics[target] = {"macro_f1": macro_f1}

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nsaved models + metrics -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
