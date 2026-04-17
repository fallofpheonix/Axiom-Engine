from __future__ import annotations

from src.ir.ir import ExecOp


def generate_script(plan: list[ExecOp]) -> str:
    lines = [
        "import bpy",
        "bpy.ops.wm.read_factory_settings(use_empty=True)",
    ]

    for op in plan:
        if op.op == "reset":
            continue
        if op.op == "create":
            geometry = op.params["geometry"]
            if geometry == "cube":
                lines.append("bpy.ops.mesh.primitive_cube_add()")
            elif geometry == "plane":
                lines.append("bpy.ops.mesh.primitive_plane_add()")
            else:
                lines.append(f"# unsupported geometry: {geometry}")
        elif op.op == "transform":
            x, y, z = op.params["location"]
            lines.append(f"bpy.context.object.location = ({x}, {y}, {z})")
        elif op.op == "render":
            lines.append("bpy.ops.render.render(write_still=True)")
        else:
            lines.append(f"# unsupported op: {op.op}")

    return "\n".join(lines)
