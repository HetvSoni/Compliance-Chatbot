"""Shared HTTP helpers: pooled session, polite pacing, retry/backoff, streamed
file download. Every downloader uses this so rate-limit handling lives in one place.
"""
import time
import logging
from pathlib import Path

import requests

log = logging.getLogger("http")

PACING_SECONDS = 0.3
TIMEOUT = 90

_session = requests.Session()
_session.headers.update({"User-Agent": "Team14-RegChatbot/1.0 (NLP course project)"})
_last = 0.0


def _pace():
    global _last
    wait = PACING_SECONDS - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.time()


def get(url, params=None, max_retries=5, stream=False):
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        _pace()
        try:
            resp = _session.get(url, params=params, timeout=TIMEOUT, stream=stream)
        except requests.RequestException as exc:
            log.warning("network error (%s/%s): %s", attempt, max_retries, exc)
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue
        if resp.status_code == 404:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            log.warning("HTTP %s (%s/%s) backoff %.0fs", resp.status_code, attempt, max_retries, backoff)
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"exhausted retries for {url}")


def download_file(url, dest: Path, params=None):
    """Stream a (possibly large) file to disk. Returns the byte count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = get(url, params=params, stream=True)
    if resp.status_code == 404:
        raise FileNotFoundError(f"404 for {url}")
    total = 0
    with open(dest, "wb") as fh:
        for block in resp.iter_content(chunk_size=1 << 16):
            if block:
                fh.write(block); total += len(block)
    return total
