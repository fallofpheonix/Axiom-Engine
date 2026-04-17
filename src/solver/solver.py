from __future__ import annotations

import math

import numpy as np

from src.ir.constraint_dsl import Var, eval_expr, jacobian, residual
from src.ir.graph import IRGraph
from src.ir.ir import EquationConstraint, SceneIR, Transform
from src.utils.errors import CompileError


class Solver:
    """Resolve symbolic scene constraints into concrete transforms.

    The vertical slice intentionally uses only NumPy. Heavy ML packages are not
    required for compiling a deterministic Blender scene.
    """

    MAX_ITER = 40
    EPSILON = 1e-9
    LM_DAMPING = 1e-3

    def __init__(self, ir: SceneIR):
        self.ir = ir
        self.graph = IRGraph(ir)
        self.graph.build()
        self.order = self.graph.topo_order()

    def solve(self) -> SceneIR:
        self._initialize()
        self._solve_hard_constraints()
        self._solve_soft_constraints()
        self._compute_all_bounds()

        for node_id in self.order:
            node = self.ir.nodes[node_id]
            node.symbolic = False
            node.resolved = True

        return self.ir

    def _initialize(self) -> None:
        for node in self.ir.nodes.values():
            node.locked_axes = [node.fixed, node.fixed, node.fixed]
            if isinstance(node.symbolic_transform, Transform):
                node.resolved_transform = Transform(location=node.symbolic_transform.location)
            else:
                node.resolved_transform = Transform(location=(0.0, 0.0, 0.0))

    def _solve_hard_constraints(self) -> None:
        by_owner: dict[str, list[EquationConstraint]] = {}
        for equation in self.ir.equations:
            if equation.mode == "hard" and equation.owner is not None:
                by_owner.setdefault(equation.owner, []).append(equation)

        for node_id in self.order:
            node = self.ir.nodes[node_id]
            equations = sorted(
                by_owner.get(node_id, []), key=lambda item: item.priority, reverse=True
            )
            for equation in equations:
                self._apply_hard_equation(node_id, equation)
            node.bounds = self._compute_bounds(node)

    def _apply_hard_equation(self, node_id: str, equation: EquationConstraint) -> None:
        node = self.ir.nodes[node_id]
        if node.resolved_transform is None:
            raise CompileError("node not initialized", node_id)
        if not isinstance(equation.eq.left, Var):
            raise CompileError("hard equation left-hand side must be a variable", node_id)
        if equation.eq.left.obj != node_id:
            raise CompileError("hard equation owner mismatch", node_id)

        axis = equation.eq.left.axis
        values = self._current_values_map()
        try:
            value = float(eval_expr(equation.eq.right, values))
        except KeyError as exc:
            raise CompileError("target not resolved", node_id) from exc

        location = list(node.resolved_transform.location)
        if node.locked_axes[axis] and not math.isclose(location[axis], value, abs_tol=1e-8):
            raise CompileError(f"conflicting hard constraints on {node_id}", node_id)

        location[axis] = value
        node.locked_axes[axis] = True
        node.resolved_transform = Transform(location=tuple(location))

    def _solve_soft_constraints(self) -> None:
        equations = [
            equation
            for equation in self.ir.equations
            if equation.mode == "soft" and equation.weight > 0.0
        ]
        if not equations:
            return

        var_index = self._build_var_index()
        if not var_index:
            return

        x = self._build_x(var_index)
        damping = self.LM_DAMPING
        previous_loss = float("inf")

        for _ in range(self.MAX_ITER):
            values = self._values_from_x(var_index, x)
            r, j = self._build_system(equations, values, var_index)
            loss = float(r @ r)
            if abs(previous_loss - loss) < self.EPSILON:
                break

            step = self._lm_step(r, j, damping)
            candidate = x + step
            candidate_values = self._values_from_x(var_index, candidate)
            candidate_r, _candidate_j = self._build_system(
                equations, candidate_values, var_index
            )
            candidate_loss = float(candidate_r @ candidate_r)

            if candidate_loss <= loss:
                x = candidate
                previous_loss = candidate_loss
                damping = max(damping * 0.3, 1e-9)
            else:
                previous_loss = loss
                damping *= 10.0

        self._apply_x(var_index, x)

    def _build_var_index(self) -> dict[tuple[str, int], int]:
        var_index: dict[tuple[str, int], int] = {}
        for node_id in self.order:
            node = self.ir.nodes[node_id]
            for axis in range(3):
                if not node.locked_axes[axis]:
                    var_index[(node_id, axis)] = len(var_index)
        return var_index

    def _build_x(self, var_index: dict[tuple[str, int], int]) -> np.ndarray:
        x = np.zeros(len(var_index), dtype=float)
        for key, idx in var_index.items():
            node_id, axis = key
            node = self.ir.nodes[node_id]
            if node.resolved_transform is None:
                raise CompileError("node not initialized", node_id)
            x[idx] = float(node.resolved_transform.location[axis])
        return x

    def _apply_x(self, var_index: dict[tuple[str, int], int], x: np.ndarray) -> None:
        for node_id in self.order:
            node = self.ir.nodes[node_id]
            if node.resolved_transform is None:
                raise CompileError("node not initialized", node_id)
            location = list(node.resolved_transform.location)
            for axis in range(3):
                idx = var_index.get((node_id, axis))
                if idx is not None:
                    location[axis] = float(x[idx])
            node.resolved_transform = Transform(location=tuple(location))

    def _values_from_x(
        self, var_index: dict[tuple[str, int], int], x: np.ndarray
    ) -> dict[tuple[str, int], float]:
        values = self._current_values_map()
        for key, idx in var_index.items():
            values[key] = float(x[idx])
        return values

    def _build_system(
        self,
        equations: list[EquationConstraint],
        values: dict[tuple[str, int], float],
        var_index: dict[tuple[str, int], int],
    ) -> tuple[np.ndarray, np.ndarray]:
        r = np.zeros(len(equations), dtype=float)
        j = np.zeros((len(equations), len(var_index)), dtype=float)

        for row, equation in enumerate(equations):
            weight = math.sqrt(float(equation.weight))
            r[row] = weight * residual(equation.eq, values)
            gradients = jacobian(equation.eq, values)
            for key, value in gradients.items():
                col = var_index.get(key)
                if col is not None:
                    j[row, col] = weight * value

        return r, j

    @staticmethod
    def _lm_step(r: np.ndarray, j: np.ndarray, damping: float) -> np.ndarray:
        del damping
        if j.size == 0:
            return np.zeros(0, dtype=float)
        return np.linalg.lstsq(j, -r, rcond=None)[0]

    @staticmethod
    def _damped_lm_step(r: np.ndarray, j: np.ndarray, damping: float) -> np.ndarray:
        lhs = j.T @ j + damping * np.eye(j.shape[1])
        rhs = -(j.T @ r)
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    def _current_values_map(self) -> dict[tuple[str, int], float]:
        values: dict[tuple[str, int], float] = {}
        for node_id in self.order:
            node = self.ir.nodes[node_id]
            if node.resolved_transform is None:
                raise CompileError("node not initialized", node_id)
            for axis in range(3):
                values[(node_id, axis)] = float(node.resolved_transform.location[axis])
        return values

    def _compute_all_bounds(self) -> None:
        for node_id in self.order:
            node = self.ir.nodes[node_id]
            node.bounds = self._compute_bounds(node)

    @staticmethod
    def _default_size_for(geometry: str) -> tuple[float, float, float]:
        if geometry == "plane":
            return (1.0, 1.0, 0.0)
        if geometry == "cube":
            return (1.0, 1.0, 1.0)
        raise ValueError(f"unsupported geometry '{geometry}'")

    @classmethod
    def _size_for(cls, node) -> tuple[float, float, float]:
        size = node.params.get("size") if node.params else None
        if size is None:
            return cls._default_size_for(node.geometry)
        return tuple(size)

    def _compute_bounds(self, node) -> dict:
        if node.resolved_transform is None:
            raise ValueError(f"node '{node.id}' has no resolved transform")

        sx, sy, sz = self._size_for(node)
        x, y, z = node.resolved_transform.location
        return {
            "min": [x - sx / 2.0, y - sy / 2.0, z - sz / 2.0],
            "max": [x + sx / 2.0, y + sy / 2.0, z + sz / 2.0],
        }
