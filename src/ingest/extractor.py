
import re
import json
from pathlib import Path
import fitz  # PyMuPDF
from loguru import logger
from configs.settings import DATA_RAW, DATA_PROCESSED, DOCUMENT_SOURCES, CHUNK_SIZE

ARTICLE_RE = re.compile(
    r"^(Article\s+\d[\d\.]*|ARTICLE\s+\d|Chapter\s+[IVXLC\d]+|"
    r"CHAPTER\s+[IVXLC\d]+|Rule\s+\d[\d\.]*|Section\s+\d[\d\.]*)",
    re.MULTILINE
)

def extract_pdf_text(pdf_path):
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages

def _fixed_size_chunks(pages, doc_meta, chunk_size=CHUNK_SIZE):
    full_text = " ".join(p["text"] for p in pages)
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        segment = " ".join(words[i:i+chunk_size])
        chunks.append({
            "chunk_id": f"{doc_meta['doc_id']}_{i//chunk_size:04d}",
            "doc_id": doc_meta["doc_id"],
            "regulator": doc_meta["regulator"],
            "category": doc_meta["category"],
            "year": doc_meta["year"],
            "text": segment,
            "article_number": None,
            "page_num": None,
            "citation": f"{doc_meta['regulator']} ({doc_meta['year']}), chunk {i//chunk_size}",
            "cross_refs": [],
            "defined_terms": [],
        })
    return chunks

def _split_at_article_boundaries(pages, doc_meta):
    full_text = ""
    for p in pages:
        full_text += p["text"] + "\n"
    
    splits = list(ARTICLE_RE.finditer(full_text))
    if len(splits) < 3:
        return _fixed_size_chunks(pages, doc_meta)
    
    chunks = []
    for idx, match in enumerate(splits):
        start = match.start()
        end = splits[idx+1].start() if idx+1 < len(splits) else len(full_text)
        segment = full_text[start:end].strip()
        if len(segment) < 50:
            continue
        article_num = match.group(0).strip()
        cross_refs = re.findall(r"Article\s+\d[\d\.]*|Rule\s+\d[\d\.]*", segment)
        chunks.append({
            "chunk_id": f"{doc_meta['doc_id']}_{idx:04d}",
            "doc_id": doc_meta["doc_id"],
            "regulator": doc_meta["regulator"],
            "category": doc_meta["category"],
            "year": doc_meta["year"],
            "text": segment[:1500],
            "article_number": article_num,
            "page_num": None,
            "citation": f"{doc_meta['regulator']} ({doc_meta['year']}), {article_num}",
            "cross_refs": list(set(cross_refs)),
            "defined_terms": [],
        })
    return chunks

def process_all_documents(available_docs):
    all_chunks = []
    for doc_id, path in available_docs.items():
        meta = DOCUMENT_SOURCES.get(doc_id, {})
        meta["doc_id"] = doc_id
        logger.info(f"Processing {doc_id}...")
        try:
            pages = extract_pdf_text(path)
            doc_type = meta.get("doc_type", "")
            if doc_type == "rulebook":
                chunks = _split_at_article_boundaries(pages, meta)
            else:
                chunks = _fixed_size_chunks(pages, meta)
            all_chunks.extend(chunks)
            logger.success(f"{doc_id}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed {doc_id}: {e}")
    
    out = DATA_PROCESSED / "all_chunks.json"
    with open(out, "w") as f:
        json.dump(all_chunks, f)
    logger.success(f"Total: {len(all_chunks)} chunks saved")
    return all_chunks
