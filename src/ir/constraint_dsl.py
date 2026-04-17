from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

AxisKey = tuple[str, int]


@dataclass(frozen=True)
class Var:
    obj: str
    axis: int


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class Add:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Func:
    name: str
    args: tuple[str, ...]


Expr = Union[Var, Const, Add, Func]


@dataclass(frozen=True)
class Eq:
    left: Expr
    right: Expr


_AXIS_MAP = {"x": 0, "y": 1, "z": 2}


def parse_var(token: str) -> Var:
    text = token.strip()
    try:
        obj, axis_name = text.split(".")
    except ValueError as exc:
        raise ValueError(f"invalid variable token '{token}'") from exc

    axis_name = axis_name.strip().lower()
    if axis_name not in _AXIS_MAP:
        raise ValueError(f"invalid axis in '{token}'")
    return Var(obj=obj.strip(), axis=_AXIS_MAP[axis_name])


def _split_top_level_plus(expr: str) -> tuple[str, str] | None:
    depth = 0
    for idx, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "+" and depth == 0:
            return expr[:idx], expr[idx + 1 :]
    return None


def _parse_function(expr: str) -> Func | None:
    text = expr.strip()
    if not text.endswith(")"):
        return None

    open_idx = text.find("(")
    if open_idx <= 0:
        return None

    name = text[:open_idx].strip().lower()
    inside = text[open_idx + 1 : -1]
    args = tuple(arg.strip() for arg in inside.split(",") if arg.strip())

    if name == "distance":
        if len(args) != 2:
            raise ValueError("distance() expects exactly 2 object IDs")
        return Func(name="distance", args=args)

    raise ValueError(f"unsupported function '{name}'")


def parse_expr(expr: str) -> Expr:
    text = expr.strip()
    if not text:
        raise ValueError("empty expression")

    split = _split_top_level_plus(text)
    if split is not None:
        left_text, right_text = split
        return Add(left=parse_expr(left_text), right=parse_expr(right_text))

    func = _parse_function(text)
    if func is not None:
        return func

    if "." in text:
        return parse_var(text)

    return Const(float(text))


def parse_constraint(line: str) -> Eq:
    text = line.strip()
    if not text:
        raise ValueError("empty constraint line")
    if "=" not in text:
        raise ValueError("constraint must contain '='")

    left_text, right_text = text.split("=", 1)
    left = parse_expr(left_text)
    right = parse_expr(right_text)
    return Eq(left=left, right=right)


def eval_expr(expr: Expr, values: dict[AxisKey, float]) -> float:
    if isinstance(expr, Const):
        return expr.value

    if isinstance(expr, Var):
        key = (expr.obj, expr.axis)
        if key not in values:
            raise KeyError(f"missing variable value for {expr.obj}.{expr.axis}")
        return values[key]

    if isinstance(expr, Add):
        return eval_expr(expr.left, values) + eval_expr(expr.right, values)

    if isinstance(expr, Func):
        if expr.name == "distance":
            a, b = expr.args
            ax = values[(a, 0)]
            ay = values[(a, 1)]
            az = values[(a, 2)]
            bx = values[(b, 0)]
            by = values[(b, 1)]
            bz = values[(b, 2)]
            dx = ax - bx
            dy = ay - by
            dz = az - bz
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        raise ValueError(f"unsupported function '{expr.name}'")

    raise TypeError(f"unsupported expression node '{type(expr)}'")


def eval_constraint_autodiff(eq: Eq, values: dict[AxisKey, float]) -> tuple[float, dict[AxisKey, float]]:
    value = eval_expr(eq.left, values) - eval_expr(eq.right, values)
    return value, jacobian(eq, values)


def grad_expr(expr: Expr, values: dict[AxisKey, float], eps: float = 1e-8) -> dict[AxisKey, float]:
    wrapped = Eq(left=expr, right=Const(0.0))
    return jacobian(wrapped, values, eps=eps)


def residual(eq: Eq, values: dict[AxisKey, float]) -> float:
    return eval_expr(eq.left, values) - eval_expr(eq.right, values)


def jacobian(eq: Eq, values: dict[AxisKey, float], eps: float = 1e-8) -> dict[AxisKey, float]:
    base = residual(eq, values)
    grad: dict[AxisKey, float] = {}

    for key in sorted(values):
        plus = dict(values)
        minus = dict(values)
        plus[key] += eps
        minus[key] -= eps
        deriv = (residual(eq, plus) - residual(eq, minus)) / (2.0 * eps)
        if abs(deriv) > 1e-8:
            grad[key] = deriv

    del base
    return grad


def refs_in_expr(expr: Expr) -> set[str]:
    if isinstance(expr, Const):
        return set()
    if isinstance(expr, Var):
        return {expr.obj}
    if isinstance(expr, Add):
        return refs_in_expr(expr.left) | refs_in_expr(expr.right)
    if isinstance(expr, Func):
        return set(expr.args)
    raise TypeError(f"unsupported expression node '{type(expr)}'")


def refs_in_eq(eq: Eq) -> set[str]:
    return refs_in_expr(eq.left) | refs_in_expr(eq.right)
