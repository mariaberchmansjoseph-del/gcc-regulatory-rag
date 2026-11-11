
import json
import networkx as nx
from loguru import logger
from configs.settings import DATA_GRAPHS

class RegulatoryGraphBuilder:
    def __init__(self):
        self.G = nx.DiGraph()

    def build_from_chunks(self, chunks):
        for chunk in chunks:
            cid = chunk["chunk_id"]
            self.G.add_node(cid, **{k: v for k, v in chunk.items() if k != "text"})
            for ref in chunk.get("cross_refs", []):
                self.G.add_edge(cid, ref, type="REFERENCES")
        concept_map = {}
        for chunk in chunks:
            for term in chunk.get("defined_terms", []):
                concept_map.setdefault(term, []).append(chunk["chunk_id"])
        for term, ids in concept_map.items():
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    self.G.add_edge(ids[i], ids[j], type="SHARES_CONCEPT", concept=term)
        logger.success(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def find_cross_references(self, chunk_id, depth=2):
        if chunk_id not in self.G:
            return []
        related = set()
        frontier = {chunk_id}
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                next_frontier.update(self.G.successors(n))
                next_frontier.update(self.G.predecessors(n))
            related.update(next_frontier)
            frontier = next_frontier
        return list(related - {chunk_id})

    def save(self):
        data = nx.node_link_data(self.G)
        with open(DATA_GRAPHS / "regulatory_graph.json", "w") as f:
            json.dump(data, f)

    def load(self):
        with open(DATA_GRAPHS / "regulatory_graph.json") as f:
            data = json.load(f)
        self.G = nx.node_link_graph(data)

    def get_document_summary(self):
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
        }
