from __future__ import annotations

import subprocess
from os import environ
from pathlib import Path
from shutil import which


def run_blender(script_text: str, output_dir: str = "output") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    script_path = out_dir / "script.py"
    script_path.write_text(script_text, encoding="utf-8")

    blender_bin = environ.get("BLENDER_BIN") or which("blender")
    if not blender_bin:
        raise FileNotFoundError(
            "blender executable not found; set BLENDER_BIN or add blender to PATH"
        )

    cmd = [blender_bin, "--background", "--python", str(script_path)]
    subprocess.run(cmd, check=False)
