from __future__ import annotations


class Metrics:
    """Named scalar metric accumulator."""

    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add(self, name: str, value: float) -> None:
        self.values.setdefault(name, []).append(float(value))

    def mean(self, name: str) -> float:
        values = self.values.get(name, [])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def snapshot(self) -> dict[str, float]:
        return {name: self.mean(name) for name in sorted(self.values)}
