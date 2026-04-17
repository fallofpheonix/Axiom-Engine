from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


FactValue = float
FactSet = dict[str, FactValue]


@dataclass(frozen=True)
class Rule:
    """Soft logic rule over scalar facts."""

    name: str
    condition: Callable[[FactSet], FactValue]
    conclusion: Callable[[FactValue], FactSet]


class RuleEngine:
    """Differentiable-style soft rule engine using scalar truth values."""

    def __init__(self, rules: list[Rule] | None = None):
        self.rules = list(rules or [])

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def infer(self, facts: FactSet) -> FactSet:
        inferred = dict(facts)
        for rule in self.rules:
            strength = self._clamp(rule.condition(inferred))
            for key, value in rule.conclusion(strength).items():
                inferred[key] = max(inferred.get(key, 0.0), self._clamp(value))
        return inferred

    @staticmethod
    def and_(a: FactValue, b: FactValue) -> FactValue:
        return a * b

    @staticmethod
    def or_(a: FactValue, b: FactValue) -> FactValue:
        return a + b - a * b

    @staticmethod
    def not_(a: FactValue) -> FactValue:
        return 1.0 - a

    @staticmethod
    def _clamp(value: FactValue) -> FactValue:
        return max(0.0, min(1.0, float(value)))
