"""openFDA downloader — writes records to raw JSONL.

Handles the three openFDA sources in the registry:
  • drug/label      (DailyMed SPL sample)  -> sample_limit records
  • drug/enforcement (recalls)             -> incremental via report_date
  • drug/drugsfda   (Orange Book/approvals) -> sample_limit records

Paging beyond ~26k hits uses the search_after cursor in the Link header.
API key (env OPENFDA_API_KEY) raises the daily cap to 120k.
"""
import os
import json
import logging

from src.http_util import get

log = logging.getLogger("openfda")
BASE = "https://api.fda.gov"
API_KEY = os.getenv("OPENFDA_API_KEY", "L0ToHHo5cM3mV7IcSaVTG0XqtB63Rfa65UzODTRU")


def _params(base=None):
    p = dict(base or {})
    if API_KEY:
        p["api_key"] = API_KEY
    return p


def dataset_last_updated(endpoint):
    resp = get(BASE + endpoint, params=_params({"limit": 1}))
    if resp.status_code == 404:
        return None
    return resp.json().get("meta", {}).get("last_updated")


def _iter(endpoint, search=None, sort=None, page_size=1000, cap=None):
    """Yield records, following search_after pages, up to optional `cap`."""
    params = _params({"limit": page_size})
    if search:
        params["search"] = search
    if sort:
        params["sort"] = sort

    next_url, seen = None, 0
    while True:
        resp = get(next_url) if next_url else get(BASE + endpoint, params=params)
        if resp.status_code == 404:
            return
        for rec in resp.json().get("results", []):
            yield rec
            seen += 1
            if cap and seen >= cap:
                return
        link = resp.headers.get("Link", "")
        nxt = _next_link(link)
        if not nxt:
            return
        next_url = nxt


def _next_link(header):
    for part in header.split(","):
        if 'rel="next"' in part:
            a, b = part.find("<"), part.find(">")
            if 0 <= a < b:
                return part[a + 1:b]
    return None


def download(source, raw_dir, watermark=None):
    extra = source["extra"]
    endpoint = extra["endpoint"]
    date_field = extra.get("date_field")
    product_filter = extra.get("product_filter")
    cap = extra.get("sample_limit")

    latest = dataset_last_updated(endpoint)
    if watermark and latest and watermark.get("dataset_last_updated") == latest \
       and not date_field:
        return {"status": "collected", "skipped": True,
                "watermark": {"dataset_last_updated": latest}}

    since = watermark.get("last_date") if (watermark and date_field) else None
    clauses = []
    if product_filter:
        clauses.append(product_filter)
    if date_field and since:
        clauses.append(f"{date_field}:[{since}+TO+30000101]")
    search = "+AND+".join(clauses) if clauses else None
    sort = f"{date_field}:asc" if date_field else None

    dest = raw_dir / f"{source['id']}.jsonl"
    mode = "a" if (date_field and since) else "w"    # append deltas, else fresh
    n, max_date = 0, since
    with open(dest, mode, encoding="utf-8") as fh:
        for rec in _iter(endpoint, search=search, sort=sort, cap=cap):
            if date_field and rec.get(date_field):
                max_date = max(max_date or "", rec[date_field])
            rec["_source"] = source["id"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    status = "collected" if n else "pending"
    if cap and n >= cap:
        status = "partial"          # sample subset, not the full dataset
    return {"status": status, "raw_path": str(dest), "records": n,
            "watermark": {"dataset_last_updated": latest, "last_date": max_date}}
