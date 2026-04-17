from __future__ import annotations

from src.compiler.compiler import Compiler
from src.schema.scene_spec import SceneSpec
from src.solver.solver import Solver
from src.utils.errors import CompileError


def test_cycle_detection_raises_compile_error() -> None:
    scene = SceneSpec.model_validate(
        {
            "objects": [
                {
                    "id": "A",
                    "type": "mesh",
                    "geometry": "cube",
                    "constraints": [{"type": "sit_on", "target": "B"}],
                },
                {
                    "id": "B",
                    "type": "mesh",
                    "geometry": "cube",
                    "constraints": [{"type": "sit_on", "target": "A"}],
                },
            ]
        }
    )

    ir = Compiler(scene).compile()
    try:
        Solver(ir).solve()
    except CompileError as exc:
        assert "cycle detected" in exc.message
    else:
        raise AssertionError("expected CompileError")
