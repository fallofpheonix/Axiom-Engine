from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


class ConstraintSpec(BaseModel):
    kind: Literal["sit_on", "above", "align", "offset", "distance", "look_at"] = Field(
        validation_alias=AliasChoices("kind", "type")
    )
    target: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    priority: int = 0
    mode: Literal["hard", "soft"] = "soft"
    weight: float = 1.0
    axis: Literal["x", "y", "z"] | None = None


class EquationConstraintSpec(BaseModel):
    expression: str
    mode: Literal["hard", "soft"] = "soft"
    weight: float = 1.0
    priority: int = 0


class ObjectSpec(BaseModel):
    id: str
    kind: Literal["mesh"] = Field(validation_alias=AliasChoices("kind", "type"))
    geometry: Literal["cube", "plane"]
    size: tuple[float, float, float] | None = None
    location: tuple[float, float, float] | None = None
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    equations: list[EquationConstraintSpec | str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_kind(cls, data):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("type") == "mesh" and normalized.get("kind") in {"cube", "plane"}:
            normalized.pop("kind")
        return normalized


class SceneSpec(BaseModel):
    objects: list[ObjectSpec]

    @model_validator(mode="after")
    def validate_ids_and_refs(self) -> "SceneSpec":
        object_ids = [obj.id for obj in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("object IDs must be globally unique")

        known = set(object_ids)
        for obj in self.objects:
            for constraint in obj.constraints:
                if constraint.target not in known:
                    raise ValueError(
                        f"constraint target '{constraint.target}' for '{obj.id}' does not exist"
                    )
                if constraint.target == obj.id:
                    raise ValueError(f"self-referential constraint is invalid for '{obj.id}'")
        return self
