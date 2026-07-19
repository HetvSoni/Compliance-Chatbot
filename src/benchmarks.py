"""Benchmark dataset downloader.

EURLEX-57K and BillSum load through the HuggingFace `datasets` library and are
cached under data/raw/<id>/. BioASQ requires registration, so it is documented,
not auto-downloaded.
"""
import logging

log = logging.getLogger("benchmarks")


def download(source, raw_dir, watermark=None):
    extra = source["extra"]

    if source["method"] == "manual":       # BioASQ
        return {"status": "pending", "raw_path": "",
                "note": "registration required — register at bioasq.org, "
                        "place files in data/raw/bioasq/"}

    try:
        from datasets import load_dataset
    except ImportError:
        return {"status": "pending", "raw_path": "",
                "note": "pip install datasets to fetch HuggingFace benchmarks"}

    cache_dir = raw_dir / source["id"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"cache_dir": str(cache_dir)}
    if extra.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True

    args = [extra["hf_id"]]
    if extra.get("hf_config"):
        args.append(extra["hf_config"])

    try:
        ds = load_dataset(*args, **kwargs)
    except Exception as exc:
        return {"status": "pending", "raw_path": "", "note": f"HF load failed: {exc}"}

    splits = {name: len(split) for name, split in ds.items()}
    return {"status": "collected", "raw_path": str(cache_dir), "splits": splits}
