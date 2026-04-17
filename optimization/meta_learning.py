from __future__ import annotations

from dataclasses import dataclass, replace
from random import random


@dataclass(frozen=True)
class Genome:
    """Mutable system design parameters for evolution."""

    hidden_dim: int = 256
    learning_rate: float = 1e-3
    entropy_weight: float = 0.01
    score: float = 0.0


class EvolutionStrategy:
    """Simple selection and mutation strategy for system genomes."""

    def __init__(self, mutation_scale: float = 0.1):
        self.mutation_scale = mutation_scale

    def mutate(self, genome: Genome) -> Genome:
        direction = 1 if random() > 0.5 else -1
        hidden = max(32, genome.hidden_dim + direction * 16)
        lr = max(1e-6, genome.learning_rate * (1.0 + self.mutation_scale * direction))
        entropy = max(0.0, genome.entropy_weight * (1.0 + self.mutation_scale * direction))
        return replace(genome, hidden_dim=hidden, learning_rate=lr, entropy_weight=entropy, score=0.0)

    def select(self, genomes: list[Genome], keep: int) -> list[Genome]:
        return sorted(genomes, key=lambda item: item.score, reverse=True)[:keep]

    def evolve(self, genomes: list[Genome]) -> list[Genome]:
        keep = max(1, len(genomes) // 2)
        parents = self.select(genomes, keep)
        children = [self.mutate(parent) for parent in parents]
        return parents + children
