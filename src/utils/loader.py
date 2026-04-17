from __future__ import annotations

from pathlib import Path

import yaml

from src.schema.scene_spec import SceneSpec


def load_scene(path: str | Path) -> SceneSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SceneSpec.model_validate(data)
