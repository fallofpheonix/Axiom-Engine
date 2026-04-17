from __future__ import annotations

from dataclasses import dataclass

from science.analyzer import ExperimentAnalyzer
from science.experiment import ExperimentPlanner, SimulationExperimentRunner
from science.hypothesis import Hypothesis, HypothesisGenerator
from science.knowledge_graph import ConceptNode, KnowledgeGraph, RelationEdge
from infrastructure.queue import Task, TaskQueue
from infrastructure.storage import JsonlStorage


@dataclass
class OrchestratorState:
    """Runtime counters for the research loop."""

    iterations: int = 0
    accepted: int = 0
    rejected: int = 0


class Orchestrator:
    """Main autonomous research loop."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        hypothesis_generator: HypothesisGenerator,
        planner: ExperimentPlanner,
        runner: SimulationExperimentRunner,
        analyzer: ExperimentAnalyzer,
        queue: TaskQueue,
        storage: JsonlStorage,
    ):
        self.graph = graph
        self.hypothesis_generator = hypothesis_generator
        self.planner = planner
        self.runner = runner
        self.analyzer = analyzer
        self.queue = queue
        self.storage = storage
        self.state = OrchestratorState()
        self._hypotheses: dict[str, Hypothesis] = {}

    def seed_concepts(self, concepts: list[ConceptNode]) -> None:
        for concept in concepts:
            self.graph.add_node(concept)

    def run_once(self) -> dict:
        hypotheses = self.hypothesis_generator.generate(self.graph, limit=1)
        if not hypotheses:
            return {"status": "no_hypothesis"}
        hypothesis = hypotheses[0]
        self._hypotheses[hypothesis.id] = hypothesis
        plan = self.planner.plan(hypothesis, self.graph)
        task = Task(kind="experiment", payload={"plan": plan})
        queued = self.queue.put(task)
        if not queued:
            return {"status": "queue_full"}
        task = self.queue.get(timeout=0.0)
        if task is None:
            return {"status": "queue_empty"}
        result = self.runner.run(task.payload["plan"])
        decision = self.analyzer.analyze(result)
        if decision.accepted:
            self.graph.add_edge(
                RelationEdge(
                    source=hypothesis.subject,
                    target=hypothesis.object,
                    relation=hypothesis.relation,
                    weight=decision.score,
                    metadata={"hypothesis_id": hypothesis.id, "plan_id": plan.id},
                )
            )
            self.state.accepted += 1
        else:
            self.state.rejected += 1
        self.state.iterations += 1
        self.queue.task_done()
        record = {
            "iteration": self.state.iterations,
            "hypothesis": hypothesis.__dict__,
            "plan": plan.__dict__,
            "result": result.__dict__,
            "decision": decision.__dict__,
        }
        self.storage.append(record)
        return record

    def run(self, iterations: int) -> list[dict]:
        return [self.run_once() for _ in range(iterations)]


def build_default_orchestrator(storage_path: str = "output/experiments.jsonl") -> Orchestrator:
    graph = KnowledgeGraph()
    return Orchestrator(
        graph=graph,
        hypothesis_generator=HypothesisGenerator(),
        planner=ExperimentPlanner(),
        runner=SimulationExperimentRunner(),
        analyzer=ExperimentAnalyzer(),
        queue=TaskQueue(max_size=128),
        storage=JsonlStorage(storage_path),
    )
