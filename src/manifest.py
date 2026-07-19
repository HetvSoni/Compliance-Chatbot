"""
Document manifest — the Phase 1 tracked record of every source.

Columns: id, framework, title, category, type, access, method, source_url,
status, raw_path, text_path, retrieval_date, notes.

`init()` seeds the CSV from the registry (before any download) with sensible
default statuses; `update_row()` writes back results as each source is fetched
or extracted. The manifest is the human-readable deliverable; machine watermarks
live separately in state/watermarks.json.
"""
from datetime import date
from pathlib import Path

import pandas as pd

from src.sources import SOURCES

MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "document_manifest.csv"

COLUMNS = ["id", "framework", "title", "category", "type", "access", "method",
           "source_url", "status", "raw_path", "text_path", "retrieval_date", "notes"]

# Default pre-download status by access type.
_DEFAULT_STATUS = {"open": "pending", "paywalled": "blocked", "registration": "pending"}


def _seed_row(s):
    note = s["extra"].get("note", "")
    return {
        "id": s["id"], "framework": s["framework"], "title": s["title"],
        "category": s["category"], "type": s["type"], "access": s["access"],
        "method": s["method"], "source_url": s["url"],
        "status": _DEFAULT_STATUS.get(s["access"], "pending"),
        "raw_path": "", "text_path": "", "retrieval_date": "", "notes": note,
    }


def init(overwrite=False):
    """Create the manifest from the registry if missing (or overwrite)."""
    if MANIFEST.exists() and not overwrite:
        return load()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([_seed_row(s) for s in SOURCES], columns=COLUMNS)
    df.to_csv(MANIFEST, index=False)
    return df


def load():
    return pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)


def save(df):
    df.to_csv(MANIFEST, index=False)


def update_row(df, source_id, **fields):
    if "status" in fields and fields["status"] in ("collected", "partial"):
        fields.setdefault("retrieval_date", date.today().isoformat())
    mask = df["id"] == source_id
    for k, v in fields.items():
        df.loc[mask, k] = "" if v is None else str(v)
    return df


def summary(df):
    return df.groupby("status")["id"].count().to_dict()
