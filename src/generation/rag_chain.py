
from openai import OpenAI
from loguru import logger
from configs.settings import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from src.retrieval.indexer import HybridRetriever
from src.graph.builder import RegulatoryGraphBuilder

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a GCC regulatory intelligence assistant.
Answer ONLY from the provided context. Cite sources as [Source N].
If the context does not contain the answer, say so clearly.
End every answer with a confidence level: [Confidence: HIGH/MEDIUM/LOW]"""

class GCCRegulatoryRAG:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.graph = RegulatoryGraphBuilder()
        self.loaded = False

    def load(self):
        self.retriever.load_index()
        try:
            self.graph.load()
            logger.success("Graph loaded")
        except Exception:
            logger.warning("No graph found — running without GraphRAG")
        self.loaded = True

    def _build_context(self, chunks):
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[Source {i}] {c['citation']}\n{c['text']}")
        return "\n\n".join(parts)

    def _augment_with_graph(self, chunks):
        extra_ids = set()
        for c in chunks[:3]:
            refs = self.graph.find_cross_references(c["chunk_id"], depth=1)
            extra_ids.update(refs[:2])
        extra = []
        chunk_map = {c["chunk_id"]: c for c in self.retriever.chunks}
        for cid in extra_ids:
            if cid in chunk_map:
                ch = chunk_map[cid].copy()
                ch["graph_augmented"] = True
                extra.append(ch)
        return extra[:3]

    def query(self, question, filter_regulator=None):
        chunks = self.retriever.retrieve(question, top_k=5, filter_regulator=filter_regulator)
        graph_chunks = self._augment_with_graph(chunks)
        all_chunks = chunks + graph_chunks
        context = self._build_context(all_chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
        )
        answer = resp.choices[0].message.content
        confidence = "HIGH" if "HIGH" in answer else "MEDIUM" if "MEDIUM" in answer else "LOW"
        sources = [{"rank": i+1, "citation": c["citation"], "regulator": c.get("regulator",""),
                    "graph_augmented": c.get("graph_augmented", False)} for i, c in enumerate(all_chunks)]
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "query_type": "GENERAL",
            "confidence": confidence,
            "graph_augmented": len(graph_chunks) > 0,
            "context_chunks": len(all_chunks),
        }
