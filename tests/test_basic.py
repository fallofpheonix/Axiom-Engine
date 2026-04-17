from __future__ import annotations

from src.compiler.compiler import Compiler
from src.planner.planner import Planner
from src.solver.solver import Solver
from src.utils.loader import load_scene
from src.utils.errors import CompileError


def test_load_and_solve() -> None:
    scene = load_scene("examples/simple.yaml")
    assert len(scene.objects) == 2

    ir = Compiler(scene).compile()
    assert ir.nodes["cube"].dependencies == ["plane"]
    assert ir.nodes["cube"].resolved_transform is None
    assert ir.nodes["cube"].constraints[0].priority == 1

    resolved = Solver(ir).solve()
    assert resolved.nodes["cube"].resolved_transform is not None
    assert abs(resolved.nodes["cube"].resolved_transform.location[2] - 0.5) < 1e-6
    assert resolved.nodes["cube"].bounds is not None
    assert abs(resolved.nodes["cube"].bounds["max"][2] - 1.0) < 1e-6

    plan = Planner(resolved).build_plan()
    assert plan[0].type == "reset"


def test_invalid_reference_raises_compile_error() -> None:
    from src.schema.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec

    spec = SceneSpec.model_construct(
        objects=[
            ObjectSpec.model_construct(
                id="cube",
                type="mesh",
                geometry="cube",
                constraints=[ConstraintSpec.model_construct(type="sit_on", target="missing")],
            )
        ]
    )
    try:
        Compiler(spec).compile()
    except CompileError as exc:
        assert "invalid reference" in exc.message
    else:
        raise AssertionError("expected CompileError")
