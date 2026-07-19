"""Direct-file downloader for PDFs and HTML pages (FDA guidance, EU Annex 1,
ICH guidelines, Swissmedic). Uses ETag / Last-Modified for refresh skipping and
falls back gracefully to 'partial' when a file URL 404s (e.g. an ICH filename
that has shifted)."""
import logging

from src.http_util import get, download_file, _session

log = logging.getLogger("direct")


def _head_signature(url):
    """Return (etag, last_modified) without downloading the body."""
    try:
        resp = _session.head(url, timeout=30, allow_redirects=True)
        return resp.headers.get("ETag", ""), resp.headers.get("Last-Modified", "")
    except Exception:
        return "", ""


def download(source, raw_dir, watermark=None, kind="pdf"):
    url = source["url"]
    ext = "pdf" if kind == "pdf" else "html"
    dest = raw_dir / f"{source['id']}.{ext}"

    etag, lastmod = _head_signature(url)
    sig = etag or lastmod
    if watermark and sig and watermark.get("signature") == sig and dest.exists():
        return {"status": "collected", "skipped": True, "watermark": {"signature": sig}}

    try:
        if kind == "pdf":
            n = download_file(url, dest)
        else:
            resp = get(url)
            if resp.status_code == 404:
                raise FileNotFoundError(f"404 for {url}")
            dest.write_text(resp.text, encoding="utf-8")
            n = len(resp.text)
    except FileNotFoundError:
        # URL shifted (common for ICH). Keep the landing page in the manifest note.
        landing = source["extra"].get("landing", url)
        return {"status": "partial", "raw_path": "",
                "note": f"file 404 — confirm URL; landing: {landing}"}

    return {"status": "collected", "raw_path": str(dest), "bytes": n,
            "watermark": {"signature": sig}}
