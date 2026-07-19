"""
Incremental-refresh state (watermarks).

Phase 1's deliverable is a *reusable* download script — so re-running it should
not re-download everything. We store a small watermark per source:
  • eCFR parts   -> latest_amended_on   (skip if unchanged)
  • openFDA      -> dataset last_updated (skip if unchanged)
  • warning ltrs -> set of seen URLs     (fetch only new)
  • direct files -> ETag / Last-Modified (skip if unchanged)

`download_sources.py --refresh` consults these; a plain run ignores them and
(re)downloads. Kept separate from the manifest so the manifest stays a clean,
human-readable record while this file holds machine state.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parents[1] / "state" / "watermarks.json"


def load():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_FILE)


def get(state, source_id):
    return state.get(source_id, {})


def set(state, source_id, value):
    value = dict(value)
    value["last_run"] = datetime.now(timezone.utc).isoformat()
    state[source_id] = value
    return state
