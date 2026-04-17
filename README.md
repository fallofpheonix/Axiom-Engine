# Axiom Engine

Axiom Engine is a deterministic scene compiler core for declarative 3D scene generation.

Current scope:

```text
SceneSpec YAML
-> validation
-> symbolic IR
-> dependency graph
-> constraint solve
-> resolved IR
-> execution plan
-> Blender bpy script generation
```

This repository intentionally starts with the compiler vertical slice. It does not yet include a Blender runtime, UI, neural world model, RenderGraph, or autonomous research platform.

## Supported Vertical Slice

- primitive objects: `cube`, `plane`
- object references by ID
- dependency graph and cycle rejection
- constraints: `sit_on`, `above`, `align`
- custom equations, including `distance(a, b) = value`
- hard constraints that lock axes
- weighted soft constraints solved by NumPy least squares
- execution plan generation
- Blender Python script generation

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python main.py examples/simple.yaml
```

The command prints:

- original validated scene
- symbolic IR
- resolved IR
- execution plan
- generated `bpy` script

## Test

```bash
.venv/bin/python -m pytest -q
```

Expected status:

```text
15 passed
```

## Example

```yaml
version: 1
objects:
  - id: plane
    type: mesh
    geometry: plane
    size: [4.0, 4.0, 0.0]
    location: [0.0, 0.0, 0.0]

  - id: cube
    type: mesh
    geometry: cube
    size: [1.0, 1.0, 1.0]
    constraints:
      - type: sit_on
        target: plane
```

Resolved result:

```text
cube.location.z = 0.5
```

## Project Layout

```text
src/schema/      SceneSpec validation
src/ir/          IR, dependency graph, constraint DSL
src/compiler/    SceneSpec -> symbolic IR
src/solver/      symbolic IR -> resolved IR
src/planner/     resolved IR -> execution plan
src/codegen/     execution plan -> bpy script
tests/           compiler/solver regression tests
examples/        minimal input scenes
```

## Non-Goals For Current Phase

- no UI
- no ML dependency
- no RenderGraph
- no distributed runtime
- no Blender process execution by default

The next implementation phase should wire the generated `bpy` script into a safe headless Blender runner.
