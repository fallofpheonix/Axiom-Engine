from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from time import time
from uuid import uuid4

from science.hypothesis import Hypothesis
from science.knowledge_graph import KnowledgeGraph


@dataclass
class ExperimentPlan:
    """Executable experiment plan derived from a hypothesis."""

    hypothesis_id: str
    subject: str
    object: str
    intervention: dict
    expected_relation: str
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ExperimentResult:
    """Experiment result with enough metadata for analysis and audit."""

    plan_id: str
    hypothesis_id: str
    effect_size: float
    novelty: float
    success: bool
    metrics: dict
    created_at: float = field(default_factory=time)


class ExperimentPlanner:
    """Plans simulation experiments from hypotheses."""

    def plan(self, hypothesis: Hypothesis, graph: KnowledgeGraph) -> ExperimentPlan:
        subject = graph.get_node(hypothesis.subject)
        target = graph.get_node(hypothesis.object)
        return ExperimentPlan(
            hypothesis_id=hypothesis.id,
            subject=subject.id,
            object=target.id,
            intervention={"delta": self._distance(subject.embedding, target.embedding)},
            expected_relation=hypothesis.relation,
        )

    @staticmethod
    def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        width = min(len(a), len(b))
        if width == 0:
            return 0.0
        return sqrt(sum((a[i] - b[i]) ** 2 for i in range(width)))


class SimulationExperimentRunner:
    """Deterministic simulation executor for experiment plans."""

    def run(self, plan: ExperimentPlan) -> ExperimentResult:
        delta = float(plan.intervention.get("delta", 0.0))
        effect = 1.0 / (1.0 + delta)
        novelty = min(1.0, delta)
        return ExperimentResult(
            plan_id=plan.id,
            hypothesis_id=plan.hypothesis_id,
            effect_size=effect,
            novelty=novelty,
            success=effect > 0.05,
            metrics={"delta": delta, "effect": effect, "novelty": novelty},
        )
