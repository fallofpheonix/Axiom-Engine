from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AxiomConfig:
    """Top-level runtime configuration."""

    seed: int = 7
    iterations: int = 3
    storage_path: str = "output/experiments.jsonl"
    obs_dim: int = 16
    action_dim: int = 4
    num_agents: int = 2
