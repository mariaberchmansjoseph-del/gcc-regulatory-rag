import pickle
import networkx as nx
from pathlib import Path
from loguru import logger

# Use hardcoded path to avoid import issues
INDEX_DIR = Path("/content/gcc-regulatory-rag/data/index")
GRAPH_PATH = INDEX_DIR / "knowledge_graph.pkl"

REGULATORY_ENTITIES = [
    "CBUAE", "DFSA", "SCA", "ADGM", "FSRA", "IMF",
    "Basel III", "Basel IV", "FATF", "AML", "CFT",
    "CET1", "Tier-1", "CAR", "LCR", "NSFR", "D-SIB",
    "capital adequacy", "liquidity", "anti-money laundering",
    "know your customer", "KYC", "sanctions", "correspondent banking",
]

class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def build_from_chunks(self, chunks):
        logger.info(f"Building knowledge graph from {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            node_id = f"chunk_{i}"
            self.graph.add_node(node_id,
                type="chunk",
                text=chunk["text"][:200],
                source=chunk.get("source", ""),
                regulator=chunk.get("regulator", ""),
                article=chunk.get("article_number", ""),
            )

        for i, chunk in enumerate(chunks):
            text_lower = chunk["text"].lower()
            for entity in REGULATORY_ENTITIES:
                if entity.lower() in text_lower:
                    ent_node = f"entity_{entity.replace(' ', '_')}"
                    if not self.graph.has_node(ent_node):
                        self.graph.add_node(ent_node, type="entity", name=entity)
                    self.graph.add_edge(f"chunk_{i}", ent_node, weight=1)

        regulator_chunks = {}
        for i, chunk in enumerate(chunks):
            reg = chunk.get("regulator", "unknown")
            regulator_chunks.setdefault(reg, []).append(i)

        for reg, idxs in regulator_chunks.items():
            for a in range(len(idxs)):
                for b in range(a+1, min(a+4, len(idxs))):
                    self.graph.add_edge(f"chunk_{idxs[a]}", f"chunk_{idxs[b]}",
                                        weight=0.5, relation="same_regulator")

        logger.success(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def save(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)
        logger.success(f"Graph saved to {GRAPH_PATH}")

    def load(self):
        with open(GRAPH_PATH, "rb") as f:
            self.graph = pickle.load(f)
        return self.graph

    def get_related_chunks(self, chunk_indices, hops=1):
        related = set(chunk_indices)
        for idx in chunk_indices:
            node = f"chunk_{idx}"
            if node not in self.graph:
                continue
            neighbors = nx.single_source_shortest_path_length(
                self.graph, node, cutoff=hops)
            for n in neighbors:
                if n.startswith("chunk_"):
                    related.add(int(n.split("_")[1]))
        return list(related)
