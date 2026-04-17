from __future__ import annotations

from src.compiler.compiler import Compiler
from src.schema.scene_spec import ObjectSpec, SceneSpec
from src.solver.solver import Solver
from src.utils.errors import CompileError


def test_conflicting_soft_constraints_compromise() -> None:
    scene = SceneSpec.model_validate(
        {
            "objects": [
                {"id": "plane", "type": "mesh", "geometry": "plane"},
                {
                    "id": "cube",
                    "type": "mesh",
                    "geometry": "cube",
                    "constraints": [
                        {"type": "sit_on", "target": "plane"},
                        {"type": "above", "target": "plane"},
                    ],
                },
            ]
        }
    )

    ir = Compiler(scene).compile()
    resolved = Solver(ir).solve()
    z = resolved.nodes["cube"].resolved_transform.location[2]
    assert abs(z - 1.0) < 1e-6


def test_multiple_constraints_same_result_do_not_raise() -> None:
    scene = SceneSpec.model_validate(
        {
            "objects": [
                {"id": "plane", "type": "mesh", "geometry": "plane"},
                {
                    "id": "cube",
                    "type": "mesh",
                    "geometry": "cube",
                    "constraints": [
                        {"type": "sit_on", "target": "plane", "priority": 1},
                        {"type": "sit_on", "target": "plane", "priority": 2},
                    ],
                },
            ]
        }
    )

    ir = Compiler(scene).compile()
    resolved = Solver(ir).solve()
    assert resolved.nodes["cube"].resolved_transform is not None
    assert abs(resolved.nodes["cube"].resolved_transform.location[2] - 0.5) < 1e-6


def test_weighted_soft_constraints_bias_solution() -> None:
    scene = SceneSpec.model_validate(
        {
            "objects": [
                {
                    "id": "left",
                    "type": "mesh",
                    "geometry": "cube",
                    "location": [0.0, 0.0, 0.0],
                },
                {
                    "id": "right",
                    "type": "mesh",
                    "geometry": "cube",
                    "location": [10.0, 0.0, 0.0],
                },
                {
                    "id": "cube",
                    "type": "mesh",
                    "geometry": "cube",
                    "constraints": [
                        {
                            "type": "align",
                            "target": "left",
                            "axis": "x",
                            "mode": "soft",
                            "weight": 1,
                        },
                        {
                            "type": "align",
                            "target": "right",
                            "axis": "x",
                            "mode": "soft",
                            "weight": 9,
                        },
                    ],
                },
            ]
        }
    )

    resolved = Solver(Compiler(scene).compile()).solve()
    x = resolved.nodes["cube"].resolved_transform.location[0]
    assert abs(x - 9.0) < 1e-6


def test_hard_constraints_lock_axes_against_soft_updates() -> None:
    scene = SceneSpec.model_validate(
        {
            "objects": [
                {
                    "id": "table",
                    "type": "mesh",
                    "geometry": "cube",
                    "size": [2.0, 2.0, 1.0],
                    "location": [0.0, 0.0, 0.0],
                },
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
                    "constraints": [
                        {"type": "sit_on", "target": "table", "mode": "hard"},
                        {
                            "type": "align",
                            "target": "wall",
                            "axis": "x",
                            "mode": "soft",
                            "weight": 1,
                        },
                    ],
                },
            ]
        }
    )

    resolved = Solver(Compiler(scene).compile()).solve()
    tx, ty, tz = resolved.nodes["cube"].resolved_transform.location
    assert abs(tx - 5.0) < 1e-6
    assert abs(ty - 0.0) < 1e-6
    assert abs(tz - 1.0) < 1e-6


def test_conflicting_hard_constraints_raise() -> None:
    scene = SceneSpec.model_validate(
        {
            "objects": [
                {
                    "id": "left",
                    "type": "mesh",
                    "geometry": "cube",
                    "location": [0.0, 0.0, 0.0],
                },
                {
                    "id": "right",
                    "type": "mesh",
                    "geometry": "cube",
                    "location": [10.0, 0.0, 0.0],
                },
                {
                    "id": "cube",
                    "type": "mesh",
                    "geometry": "cube",
                    "constraints": [
                        {"type": "align", "target": "left", "axis": "x", "mode": "hard"},
                        {"type": "align", "target": "right", "axis": "x", "mode": "hard"},
                    ],
                },
            ]
        }
    )

    try:
        Solver(Compiler(scene).compile()).solve()
    except CompileError as exc:
        assert "conflicting hard constraints" in exc.message
    else:
        raise AssertionError("expected CompileError")
