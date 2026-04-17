from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.ir.constraint_dsl import Eq


@dataclass
class Transform:
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Constraint:
    kind: Literal["sit_on", "above", "align", "offset", "distance", "look_at"]
    target: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    priority: int = 0
    mode: Literal["hard", "soft"] = "soft"
    weight: float = 1.0
    axis: Literal["x", "y", "z"] | None = None


@dataclass
class IRNode:
    id: str
    kind: str
    geometry: str
    params: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    symbolic: bool = True
    resolved: bool = False
    fixed: bool = False
    symbolic_transform: Transform | list[Constraint] | None = None
    resolved_transform: Transform | None = None
    bounds: dict | None = None
    locked_axes: list[bool] = field(default_factory=lambda: [False, False, False])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EquationConstraint:
    eq: "Eq"
    mode: Literal["hard", "soft"] = "soft"
    weight: float = 1.0
    priority: int = 0
    owner: str | None = None
    source: str = ""


@dataclass
class SceneIR:
    nodes: dict[str, IRNode] = field(default_factory=dict)
    equations: list[EquationConstraint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()}}


@dataclass(frozen=True)
class ExecOp:
    op: str
    target: str = ""
    params: dict = field(default_factory=dict)
    deps: tuple[str, ...] = ()

    @property
    def type(self) -> str:
        return self.op

    @property
    def args(self) -> dict:
        return self.params

    def to_dict(self) -> dict:
        return asdict(self)
