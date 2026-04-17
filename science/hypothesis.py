from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from science.knowledge_graph import KnowledgeGraph


@dataclass
class Hypothesis:
    """Testable relation proposal."""

    subject: str
    relation: str
    object: str
    confidence: float = 0.5
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))


class HypothesisGenerator:
    """Generates hypotheses from missing graph relations."""

    def __init__(self, default_relation: str = "influences"):
        self.default_relation = default_relation

    def generate(self, graph: KnowledgeGraph, limit: int = 1) -> list[Hypothesis]:
        nodes = list(graph.nodes.values())
        existing = {(edge.source, edge.target, edge.relation) for edge in graph.edges.values()}
        hypotheses: list[Hypothesis] = []
        for source in nodes:
            for target in nodes:
                if source.id == target.id:
                    continue
                key = (source.id, target.id, self.default_relation)
                if key in existing:
                    continue
                hypotheses.append(
                    Hypothesis(
                        subject=source.id,
                        relation=self.default_relation,
                        object=target.id,
                        confidence=0.5,
                    )
                )
                if len(hypotheses) >= limit:
                    return hypotheses
        return hypotheses
