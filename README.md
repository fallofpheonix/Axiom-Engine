# Axiom Engine

Axiom Engine is a modular autonomous research system combining:

- Dreamer-style RSSM world model
- latent actor-critic agents
- PettingZoo-style multi-agent coordination
- symbolic communication and soft rule reasoning
- knowledge graph based scientific loop
- Ray-style queue and worker infrastructure
- meta-learning and bounded self-improvement primitives

## Directory Structure

```text
core/
  world_model.py        RSSM, Dreamer-style wrapper, imagination rollout
  agent.py              latent actor, critic, centralized critic
  multi_agent.py        multi-agent shared-environment coordinator
  language.py           symbolic communication channel
  reasoning.py          soft rule engine

science/
  knowledge_graph.py    concept nodes and relation edges
  hypothesis.py         hypothesis generation
  experiment.py         experiment planning and simulation execution
  analyzer.py           result analysis and acceptance

infrastructure/
  orchestrator.py       autonomous research loop
  worker.py             long-running task worker
  queue.py              bounded task queue
  storage.py            JSONL persistence

optimization/
  meta_learning.py      genome and evolution strategy
  rsi.py                proposal-validation self-improvement loop

utils/
  config.py             runtime config
  metrics.py            metric accumulator
  logging.py            logging setup

src/
  Existing Blender scene compiler vertical slice.
```

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run Axiom Engine

```bash
.venv/bin/python main.py --iterations 3
```

Output includes:

- iterations executed
- accepted/rejected experiment count
- resulting knowledge graph
- experiment records

## Run Tests

```bash
.venv/bin/python -m pytest -q
```

Current expected result:

```text
15 passed
```

## Current Status

Implemented:

- RSSM world model with prior/posterior latent dynamics
- Dreamer-style imagination rollout interface
- actor-critic agent operating in latent space
- multi-agent coordinator using a PettingZoo-style protocol
- symbolic speaker/listener module
- soft rule engine
- knowledge graph add/query
- hypothesis -> experiment -> analyze -> graph update loop
- bounded task queue and worker abstraction
- JSONL persistence
- meta-learning genome and RSI validation loop
- Blender compiler vertical slice remains intact under `src/`

Reference repos are cloned outside this repository at:

```text
/Users/fallofpheonix/Project/axiom-reference-repos
```

They are reference components, not vendored source.
