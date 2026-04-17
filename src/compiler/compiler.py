from __future__ import annotations

from src.ir.constraint_dsl import Eq, Var, Add, Const, parse_constraint, refs_in_eq
from src.ir.ir import Constraint, EquationConstraint, IRNode, SceneIR, Transform
from src.schema.scene_spec import EquationConstraintSpec, SceneSpec
from src.utils.errors import CompileError


class Compiler:
    def __init__(self, spec: SceneSpec):
        self.spec = spec

    def validate_refs(self) -> None:
        ids = {obj.id for obj in self.spec.objects}
        for obj in self.spec.objects:
            for constraint in obj.constraints:
                if constraint.target not in ids:
                    raise CompileError(f"invalid reference: {constraint.target}", obj.id)

    def compile(self) -> SceneIR:
        self.validate_refs()
        ir = SceneIR()
        object_by_id = {obj.id: obj for obj in self.spec.objects}

        for obj in self.spec.objects:
            parsed_equations: list[tuple[EquationConstraintSpec, Eq]] = []
            equation_refs: set[str] = set()
            for entry in obj.equations:
                spec = self._normalize_equation_entry(entry)
                eq = self._parse_equation(spec.expression, obj.id)
                parsed_equations.append((spec, eq))
                equation_refs.update(refs_in_eq(eq))

            constraints = [
                Constraint(
                    kind=constraint.kind,
                    target=constraint.target,
                    offset=constraint.offset,
                    priority=constraint.priority,
                    mode=constraint.mode,
                    weight=constraint.weight,
                    axis=constraint.axis,
                )
                for constraint in obj.constraints
            ]
            node = IRNode(
                id=obj.id,
                kind=obj.kind,
                geometry=obj.geometry,
                params={"size": obj.size},
                dependencies=sorted(
                    {constraint.target for constraint in constraints}
                    | {ref for ref in equation_refs if ref != obj.id}
                ),
                constraints=constraints,
                fixed=(obj.location is not None or (not constraints and not parsed_equations)),
            )

            if obj.location is not None:
                node.symbolic_transform = Transform(location=obj.location)
            elif not constraints and not parsed_equations:
                node.symbolic_transform = Transform(location=(0.0, 0.0, 0.0))
            else:
                node.symbolic_transform = constraints

            ir.nodes[obj.id] = node

            for constraint in constraints:
                ir.equations.append(
                    self._equation_from_structured_constraint(
                        node_id=obj.id,
                        constraint=constraint,
                        object_by_id=object_by_id,
                    )
                )

            for spec, eq in parsed_equations:
                ir.equations.append(
                    EquationConstraint(
                        eq=eq,
                        mode=spec.mode,
                        weight=spec.weight,
                        priority=spec.priority,
                        owner=self._owner_for_eq(eq),
                        source=f"{obj.id}:equation",
                    )
                )

        return ir

    @staticmethod
    def _normalize_equation_entry(entry: EquationConstraintSpec | str) -> EquationConstraintSpec:
        if isinstance(entry, str):
            return EquationConstraintSpec(expression=entry)
        return entry

    @staticmethod
    def _owner_for_eq(eq: Eq) -> str | None:
        if isinstance(eq.left, Var):
            return eq.left.obj
        return None

    def _parse_equation(self, expression: str, owner_id: str) -> Eq:
        try:
            eq = parse_constraint(expression)
        except ValueError as exc:
            raise CompileError(f"invalid equation '{expression}': {exc}", owner_id) from exc

        known = {obj.id for obj in self.spec.objects}
        for ref in refs_in_eq(eq):
            if ref not in known:
                raise CompileError(f"invalid reference in equation: {ref}", owner_id)

        return eq

    def _equation_from_structured_constraint(
        self,
        node_id: str,
        constraint: Constraint,
        object_by_id,
    ) -> EquationConstraint:
        if constraint.kind == "align":
            axis = constraint.axis or "x"
            axis_map = {"x": 0, "y": 1, "z": 2}
            axis_idx = axis_map[axis]
            eq = Eq(
                left=Var(node_id, axis_idx),
                right=Add(Var(constraint.target, axis_idx), Const(constraint.offset[axis_idx])),
            )
        elif constraint.kind == "sit_on":
            node_half = self._half_height(object_by_id[node_id])
            target_half = self._half_height(object_by_id[constraint.target])
            delta = target_half + node_half + constraint.offset[2]
            eq = Eq(
                left=Var(node_id, 2),
                right=Add(Var(constraint.target, 2), Const(delta)),
            )
        elif constraint.kind == "above":
            node_half = self._half_height(object_by_id[node_id])
            target_half = self._half_height(object_by_id[constraint.target])
            delta = target_half + node_half + 1.0 + constraint.offset[2]
            eq = Eq(
                left=Var(node_id, 2),
                right=Add(Var(constraint.target, 2), Const(delta)),
            )
        else:
            raise CompileError(f"unknown constraint {constraint.kind}", node_id)

        return EquationConstraint(
            eq=eq,
            mode=constraint.mode,
            weight=constraint.weight,
            priority=constraint.priority,
            owner=node_id,
            source=f"{node_id}:{constraint.kind}",
        )

    @staticmethod
    def _default_size_for(geometry: str) -> tuple[float, float, float]:
        if geometry == "plane":
            return (1.0, 1.0, 0.0)
        if geometry == "cube":
            return (1.0, 1.0, 1.0)
        raise CompileError(f"unsupported geometry '{geometry}'")

    @classmethod
    def _half_height(cls, obj) -> float:
        size = obj.size if obj.size is not None else cls._default_size_for(obj.geometry)
        return float(size[2]) / 2.0
