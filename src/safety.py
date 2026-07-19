"""
safety.py — Phase 3 helper: safety classifier lookup

Loads the trained safety classifier (train_safety_classifier.py) and exposes
one call, check_advice_seeking(question), that answer.py and generate_rag.py
both use to decide whether to attach a disclaimer before returning a result.
"""
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "safety_classifier"

DISCLAIMER = (
    "[Disclaimer: this looks like a question asking for a prediction, recommendation, "
    "or judgment call (e.g. inspection outcomes, whether to recall/report something) "
    "rather than a factual regulatory lookup. This system only retrieves and summarizes "
    "regulatory text -- it cannot assess your specific situation or predict regulatory "
    "outcomes. Consult qualified regulatory/quality personnel for that judgment.]"
)

_cache = {}


def _load():
    if "pipe" not in _cache:
        _cache["pipe"] = joblib.load(MODEL_DIR / "safety_clf.joblib")
        _cache["threshold"] = json.loads((MODEL_DIR / "threshold.json").read_text())["threshold"]
    return _cache["pipe"], _cache["threshold"]


def check_advice_seeking(question):
    """Returns (flagged: bool, probability: float)."""
    pipe, threshold = _load()
    classes = list(pipe.named_steps["clf"].classes_)
    pos_idx = classes.index("advice_seeking")
    prob = pipe.predict_proba([question])[0][pos_idx]
    return bool(prob >= threshold), float(prob)
