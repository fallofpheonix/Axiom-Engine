from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from uuid import uuid4


@dataclass
class ConceptNode:
    """Knowledge graph concept node."""

    name: str
    embedding: tuple[float, ...]
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: float = field(default_factory=time)


@dataclass
class RelationEdge:
    """Knowledge graph relation edge."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: float = field(default_factory=time)


class KnowledgeGraph:
    """In-memory knowledge graph with deterministic query interfaces."""

    def __init__(self):
        self.nodes: dict[str, ConceptNode] = {}
        self.edges: dict[str, RelationEdge] = {}
        self.name_index: dict[str, str] = {}

    def add_node(self, node: ConceptNode) -> str:
        if node.name in self.name_index:
            existing = self.name_index[node.name]
            self.nodes[existing].metadata.update(node.metadata)
            return existing
        self.nodes[node.id] = node
        self.name_index[node.name] = node.id
        return node.id

    def add_edge(self, edge: RelationEdge) -> str:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise KeyError("edge endpoints must exist in graph")
        self.edges[edge.id] = edge
        return edge.id

    def get_node(self, node_id_or_name: str) -> ConceptNode:
        node_id = self.name_index.get(node_id_or_name, node_id_or_name)
        return self.nodes[node_id]

    def neighbors(self, node_id_or_name: str, relation: str | None = None) -> list[ConceptNode]:
        node_id = self.name_index.get(node_id_or_name, node_id_or_name)
        result: list[ConceptNode] = []
        for edge in self.edges.values():
            if edge.source == node_id and (relation is None or edge.relation == relation):
                result.append(self.nodes[edge.target])
        return result

    def query_edges(self, relation: str | None = None) -> list[RelationEdge]:
        return [edge for edge in self.edges.values() if relation is None or edge.relation == relation]

    def to_dict(self) -> dict:
        return {
            "nodes": {node_id: node.__dict__ for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.__dict__ for edge_id, edge in self.edges.items()},
        }
