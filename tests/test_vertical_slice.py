from __future__ import annotations

from pathlib import Path

from src.compiler.compiler import Compiler
from src.planner.planner import Planner
from src.solver.solver import Solver
from src.utils.loader import load_scene


def test_vertical_slice_cube_sits_on_plane() -> None:
    scene = load_scene(Path("examples/simple.yaml"))
    symbolic_ir = Compiler(scene).compile()

    cube = symbolic_ir.nodes["cube"]
    assert cube.dependencies == ["plane"]
    assert cube.resolved_transform is None

    resolved_ir = Solver(symbolic_ir).solve()
    cube = resolved_ir.nodes["cube"]
    assert cube.resolved_transform is not None
    assert abs(cube.resolved_transform.location[2] - 0.5) < 1e-6

    exec_plan = Planner(resolved_ir).build_plan()
    assert exec_plan[0].type == "reset"
    assert any(op.type == "create" and op.args["id"] == "cube" for op in exec_plan)
