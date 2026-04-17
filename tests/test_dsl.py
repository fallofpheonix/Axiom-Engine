from __future__ import annotations

from src.compiler.compiler import Compiler
from src.ir.constraint_dsl import jacobian, parse_constraint, residual
from src.schema.scene_spec import SceneSpec
from src.solver.solver import Solver


def test_dsl_linear_equation_solves() -> None:
    scene = SceneSpec.model_validate(
        {
            "objects": [
                {
                    "id": "wall",
                    "type": "mesh",
                    "geometry": "cube",
                    "location": [5.0, 0.0, 0.0],
                },
                {
                    "id": "cube",
                    "type": "mesh",
                    "geometry": "cube",
                    "equations": ["cube.x = wall.x + 2"],
                },
            ]
        }
    )

    resolved = Solver(Compiler(scene).compile()).solve()
    assert abs(resolved.nodes["cube"].resolved_transform.location[0] - 7.0) < 1e-6


def test_dsl_distance_constraint_converges() -> None:
    scene = SceneSpec.model_validate(
        {
            "objects": [
                {
                    "id": "origin",
                    "type": "mesh",
                    "geometry": "cube",
                    "location": [0.0, 0.0, 0.0],
                },
                {
                    "id": "probe",
                    "type": "mesh",
                    "geometry": "cube",
                    "equations": [
                        {"expression": "distance(probe, origin) = 5", "weight": 10.0},
                        {"expression": "probe.x = 5", "weight": 1.0},
                        {"expression": "probe.y = 0", "weight": 1.0},
                        {"expression": "probe.z = 0", "weight": 1.0},
                    ],
                },
            ]
        }
    )

    resolved = Solver(Compiler(scene).compile()).solve()
    px, py, pz = resolved.nodes["probe"].resolved_transform.location
    dist = (px * px + py * py + pz * pz) ** 0.5

    assert abs(dist - 5.0) < 1e-5
    assert abs(px - 5.0) < 1e-5
    assert abs(py) < 1e-5
    assert abs(pz) < 1e-5


def test_distance_jacobian_matches_finite_difference() -> None:
    eq = parse_constraint("distance(a, b) = 5")
    values = {
        ("a", 0): 0.0,
        ("a", 1): 0.0,
        ("a", 2): 0.0,
        ("b", 0): 3.0,
        ("b", 1): 4.0,
        ("b", 2): 0.0,
    }

    row = jacobian(eq, values)
    eps = 1e-6

    values_plus = dict(values)
    values_minus = dict(values)
    values_plus[("b", 0)] += eps
    values_minus[("b", 0)] -= eps

    numeric = (residual(eq, values_plus) - residual(eq, values_minus)) / (2 * eps)
    analytic = row[("b", 0)]

    assert abs(analytic - numeric) < 1e-6


def test_composed_expression_jacobian_matches_finite_difference() -> None:
    eq = parse_constraint("distance(a, b) + a.x = 7")
    values = {
        ("a", 0): 1.0,
        ("a", 1): 2.0,
        ("a", 2): 3.0,
        ("b", 0): -2.0,
        ("b", 1): 0.5,
        ("b", 2): 1.5,
    }

    row = jacobian(eq, values)
    eps = 1e-6

    values_plus = dict(values)
    values_minus = dict(values)
    values_plus[("a", 1)] += eps
    values_minus[("a", 1)] -= eps

    numeric = (residual(eq, values_plus) - residual(eq, values_minus)) / (2 * eps)
    analytic = row[("a", 1)]

    assert abs(analytic - numeric) < 1e-6


def test_distance_zero_is_stable_for_autodiff() -> None:
    eq = parse_constraint("distance(a, b) = 0")
    values = {
        ("a", 0): 1.0,
        ("a", 1): 1.0,
        ("a", 2): 1.0,
        ("b", 0): 1.0,
        ("b", 1): 1.0,
        ("b", 2): 1.0,
    }

    row = jacobian(eq, values)
    value = residual(eq, values)

    assert abs(value) < 1e-12
    assert row == {}
