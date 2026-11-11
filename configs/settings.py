
from pathlib import Path
import os

BASE_DIR = Path("/content/gcc-regulatory-rag")
DATA_RAW = BASE_DIR / "data/raw"
DATA_PROCESSED = BASE_DIR / "data/processed"
DATA_GRAPHS = BASE_DIR / "data/graphs"
OUTPUTS = BASE_DIR / "outputs"

for d in [DATA_RAW, DATA_PROCESSED, DATA_GRAPHS, OUTPUTS]:
    d.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
HYBRID_ALPHA = 0.6

DOCUMENT_SOURCES = {
    "cbuae_fsr_2024": {
        "url": "https://www.centralbank.ae/media/4iznakgl/cbuae-issues-2024-financial-stability-report-en.pdf",
        "regulator": "CBUAE", "category": "prudential", "year": 2024,
        "jurisdiction": "UAE_Federal", "doc_type": "stability_report"
    },
    "imf_uae_2024": {
        "url": "https://www.imf.org/-/media/Files/Publications/CR/2024/English/1areea2024001-print-pdf.ashx",
        "regulator": "IMF", "category": "macro", "year": 2024,
        "jurisdiction": "International", "doc_type": "article_iv"
    },
    "dfsa_aml_2024": {
        "url": "https://www.dfsa.ae/documents/rulebooks/anti-money-laundering-rulebook",
        "regulator": "DFSA", "category": "aml", "year": 2024,
        "jurisdiction": "DIFC", "doc_type": "rulebook"
    },
    "adgm_fsra_gen": {
        "url": "https://en.adgm.thomsonreuters.com/sites/default/files/net_file_store/FSRA_GEN_14032019.pdf",
        "regulator": "ADGM", "category": "licensing", "year": 2024,
        "jurisdiction": "ADGM", "doc_type": "rulebook"
    },
}
