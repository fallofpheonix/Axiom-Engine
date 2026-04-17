from __future__ import annotations

from pathlib import Path

from src.compiler.compiler import Compiler
from src.ir.graph import IRGraph
from src.utils.loader import load_scene


def test_topological_order() -> None:
    spec = load_scene(Path("examples/simple.yaml"))
    ir = Compiler(spec).compile()

    order = IRGraph(ir).topo_order()

    assert "plane" in order
    assert "cube" in order
    assert order.index("plane") < order.index("cube")
