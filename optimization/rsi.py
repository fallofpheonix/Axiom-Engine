from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from optimization.meta_learning import EvolutionStrategy, Genome


@dataclass(frozen=True)
class ImprovementResult:
    """Validated self-improvement proposal result."""

    accepted: bool
    previous: Genome
    candidate: Genome
    margin: float


class SelfImprovementLoop:
    """Bounded proposal -> validation -> adoption loop."""

    def __init__(self, strategy: EvolutionStrategy, evaluator: Callable[[Genome], float], min_margin: float = 0.02):
        self.strategy = strategy
        self.evaluator = evaluator
        self.min_margin = min_margin

    def step(self, genome: Genome) -> ImprovementResult:
        previous_score = self.evaluator(genome)
        candidate = self.strategy.mutate(genome)
        candidate_score = self.evaluator(candidate)
        accepted = candidate_score > previous_score * (1.0 + self.min_margin)
        candidate = Genome(
            hidden_dim=candidate.hidden_dim,
            learning_rate=candidate.learning_rate,
            entropy_weight=candidate.entropy_weight,
            score=candidate_score,
        )
        previous = Genome(
            hidden_dim=genome.hidden_dim,
            learning_rate=genome.learning_rate,
            entropy_weight=genome.entropy_weight,
            score=previous_score,
        )
        return ImprovementResult(
            accepted=accepted,
            previous=previous,
            candidate=candidate if accepted else previous,
            margin=candidate_score - previous_score,
        )
