"""Warning-letters downloader (scraper).

FDA offers no official API for warning-letter *text*. The index page renders its
table from a JSON feed; the stable approach is to hit that feed directly rather
than parse rendered HTML. This module keeps a set of seen letter URLs so a
refresh fetches only new letters. Selectors below are stubs to confirm against
the live page/feed.
"""
import json
import logging

from src.http_util import get

log = logging.getLogger("warning_letters")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


def _list_links(index_url, seen):
    resp = get(index_url)
    soup = BeautifulSoup(resp.text, "lxml")
    links = []
    for a in soup.select("a[href*='/warning-letters/']"):
        href = a.get("href", "")
        if href.startswith("/"):
            href = "https://www.fda.gov" + href
        if href and href not in seen and href not in links:
            links.append(href)
    return links


def _fetch_letter(url):
    resp = get(url)
    soup = BeautifulSoup(resp.text, "lxml")
    body = soup.select_one("article, main, div.inline-block") or soup
    return {"source_url": url,
            "text": " ".join(body.get_text(" ", strip=True).split())}


def download(source, raw_dir, watermark=None):
    if BeautifulSoup is None:
        return {"status": "pending", "raw_path": "",
                "note": "pip install beautifulsoup4 lxml to scrape warning letters"}

    seen = set((watermark or {}).get("seen_urls", []))
    try:
        new_links = _list_links(source["url"], seen)
    except Exception as exc:
        return {"status": "pending", "raw_path": "", "note": f"scrape failed: {exc}"}

    dest = raw_dir / f"{source['id']}.jsonl"
    added = 0
    with open(dest, "a", encoding="utf-8") as fh:
        for link in new_links:
            try:
                rec = _fetch_letter(link)
            except Exception as exc:
                log.warning("skip %s: %s", link, exc)
                continue
            rec["_source"] = source["id"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.add(link); added += 1

    status = "collected" if seen else "pending"
    return {"status": status, "raw_path": str(dest), "records": added,
            "watermark": {"seen_urls": sorted(seen)}}
