
import time
import requests
from pathlib import Path
from loguru import logger
from configs.settings import DATA_RAW, DOCUMENT_SOURCES

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
    "Accept": "application/pdf,*/*",
}

def download_all_documents(force=False):
    available = {}
    for doc_id, meta in DOCUMENT_SOURCES.items():
        dest = DATA_RAW / f"{doc_id}.pdf"
        if dest.exists() and not force:
            logger.info(f"Already exists: {doc_id}")
            available[doc_id] = str(dest)
            continue
        try:
            logger.info(f"Downloading {doc_id}...")
            r = requests.get(meta["url"], headers=HEADERS, timeout=30, stream=True)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(8192):
                    fh.write(chunk)
            if dest.stat().st_size < 1000:
                logger.warning(f"{doc_id}: file too small, likely failed")
                dest.unlink()
            else:
                logger.success(f"Downloaded {doc_id} ({dest.stat().st_size//1024}KB)")
                available[doc_id] = str(dest)
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Failed {doc_id}: {e}")
    return available

def verify_downloads():
    status = {}
    for doc_id in DOCUMENT_SOURCES:
        path = DATA_RAW / f"{doc_id}.pdf"
        if not path.exists():
            status[doc_id] = "missing"
        else:
            with open(path, "rb") as f:
                magic = f.read(4)
            status[doc_id] = "ok" if magic == b"%PDF" else "corrupt"
    return status
