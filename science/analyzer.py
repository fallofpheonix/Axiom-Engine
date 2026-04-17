from __future__ import annotations

from dataclasses import dataclass

from science.experiment import ExperimentResult


@dataclass(frozen=True)
class AnalysisDecision:
    """Decision for accepting or rejecting a hypothesis result."""

    accepted: bool
    score: float
    reason: str


class ExperimentAnalyzer:
    """Filters experiment results by effect size and novelty."""

    def __init__(self, min_effect_size: float = 0.05, min_novelty: float = 0.0):
        self.min_effect_size = min_effect_size
        self.min_novelty = min_novelty

    def analyze(self, result: ExperimentResult) -> AnalysisDecision:
        if result.effect_size < self.min_effect_size:
            return AnalysisDecision(False, result.effect_size, "effect_size_below_threshold")
        if result.novelty < self.min_novelty:
            return AnalysisDecision(False, result.novelty, "novelty_below_threshold")
        score = result.effect_size * (1.0 + result.novelty)
        return AnalysisDecision(True, score, "accepted")
