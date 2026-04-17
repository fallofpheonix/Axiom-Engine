from __future__ import annotations

import json
import sys

from src.codegen.bpy_codegen import generate_script
from src.compiler.compiler import Compiler
from src.planner.planner import Planner
from src.solver.solver import Solver
from src.utils.loader import load_scene


def main() -> int:
    scene_path = sys.argv[1] if len(sys.argv) == 2 else "examples/simple.yaml"
    scene = load_scene(scene_path)
    symbolic_ir = Compiler(scene).compile()
    symbolic_payload = symbolic_ir.to_dict()
    resolved_ir = Solver(symbolic_ir).solve()
    exec_plan = Planner(resolved_ir).build_plan()
    bpy_script = generate_script(exec_plan)

    payload = {
        "scene": scene.model_dump(mode="python"),
        "symbolic_ir": symbolic_payload,
        "resolved_ir": resolved_ir.to_dict(),
        "execution_plan": [op.to_dict() for op in exec_plan],
        "bpy_script": bpy_script,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
