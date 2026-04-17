from __future__ import annotations

from src.ir.ir import SceneIR
from src.utils.errors import CompileError


class IRGraph:
    def __init__(self, ir: SceneIR):
        self.ir = ir
        self.graph: dict[str, set[str]] = {}

    def build(self) -> dict[str, set[str]]:
        self.graph = {node_id: set(node.dependencies) for node_id, node in self.ir.nodes.items()}
        self._topological_sort()
        return self.graph

    def topo_order(self) -> list[str]:
        if not self.graph:
            self.build()
        return self._topological_sort()

    def _topological_sort(self) -> list[str]:
        permanent: set[str] = set()
        temporary: set[str] = set()
        order: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in permanent:
                return
            if node_id in temporary:
                raise CompileError("cycle detected in dependency graph", node_id)
            if node_id not in self.ir.nodes:
                raise CompileError(f"missing dependency: {node_id}", node_id)

            temporary.add(node_id)
            for dep in sorted(self.graph.get(node_id, ())):
                visit(dep)
            temporary.remove(node_id)
            permanent.add(node_id)
            order.append(node_id)

        for node_id in sorted(self.graph):
            visit(node_id)

        return order
