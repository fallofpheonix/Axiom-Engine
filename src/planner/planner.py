from __future__ import annotations

from src.ir.graph import IRGraph
from src.ir.ir import ExecOp, SceneIR


class Planner:
    def __init__(self, ir: SceneIR):
        self.ir = ir
        self.graph = IRGraph(ir)
        self.graph.build()

    def build_plan(self) -> list[ExecOp]:
        plan: list[ExecOp] = [ExecOp(op="reset")]

        for node_id in self.graph.topo_order():
            node = self.ir.nodes[node_id]
            plan.append(
                ExecOp(
                    op="create",
                    target=node.id,
                    params={"id": node.id, "geometry": node.geometry, "params": node.params},
                    deps=tuple(node.dependencies),
                )
            )
            plan.append(
                ExecOp(
                    op="transform",
                    target=node.id,
                    params={
                        "location": node.resolved_transform.location if node.resolved_transform else None,
                    },
                    deps=(node.id,),
                )
            )

        plan.append(ExecOp(op="render"))
        return plan
