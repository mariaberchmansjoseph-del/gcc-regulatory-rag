
import json
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI
from loguru import logger
from configs.settings import DATA_PROCESSED, OPENAI_API_KEY, EMBEDDING_MODEL, HYBRID_ALPHA

client = OpenAI(api_key=OPENAI_API_KEY)

class HybridRetriever:
    def __init__(self):
        self.chunks = []
        self.faiss_index = None
        self.bm25 = None

    def _embed_batch(self, texts, batch_size=100):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            embeddings.extend([e.embedding for e in resp.data])
        return np.array(embeddings, dtype="float32")

    def build_index(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self._embed_batch(texts)
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        faiss.write_index(self.faiss_index, str(DATA_PROCESSED / "faiss_index.bin"))
        with open(DATA_PROCESSED / "bm25_index.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(DATA_PROCESSED / "chunk_metadata.json", "w") as f:
            json.dump(self.chunks, f)
        logger.success("Index saved")

    def load_index(self):
        self.faiss_index = faiss.read_index(str(DATA_PROCESSED / "faiss_index.bin"))
        with open(DATA_PROCESSED / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(DATA_PROCESSED / "chunk_metadata.json") as f:
            self.chunks = json.load(f)
        logger.success(f"Loaded index with {len(self.chunks)} chunks")

    def retrieve(self, query, top_k=10, filter_regulator=None):
        q_emb = self._embed_batch([query])
        faiss.normalize_L2(q_emb)
        scores, indices = self.faiss_index.search(q_emb, top_k * 2)
        dense = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
        bm25_scores = self.bm25.get_scores(query.lower().split())
        sparse = sorted(enumerate(bm25_scores), key=lambda x: -x[1])[:top_k * 2]
        dense_ranks = {i: r for r, (i, _) in enumerate(dense)}
        sparse_ranks = {i: r for r, (i, _) in enumerate(sparse)}
        all_ids = set(dense_ranks) | set(sparse_ranks)
        rrf = {}
        for i in all_ids:
            rrf[i] = (
                HYBRID_ALPHA / (60 + dense_ranks.get(i, 999)) +
                (1 - HYBRID_ALPHA) / (60 + sparse_ranks.get(i, 999))
            )
        ranked = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx].copy()
            chunk["score"] = score
            if filter_regulator and chunk.get("regulator") != filter_regulator:
                continue
            results.append(chunk)
        return results
