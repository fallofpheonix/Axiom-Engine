from __future__ import annotations

import argparse
import json
import random

import torch

from core.agent import AgentConfig, LatentAgent
from core.multi_agent import MultiAgentSystem
from core.world_model import DreamerWorldModel, WorldModelConfig
from infrastructure.orchestrator import build_default_orchestrator
from science.knowledge_graph import ConceptNode
from utils.config import AxiomConfig
from utils.logging import configure_logging, get_logger


def build_system(config: AxiomConfig) -> dict:
    """Build core Axiom Engine components."""

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    world_config = WorldModelConfig(obs_dim=config.obs_dim, action_dim=config.action_dim)
    world_model = DreamerWorldModel(world_config)
    agent_config = AgentConfig(latent_dim=world_model.feat_dim, action_dim=config.action_dim)
    agents = {
        f"agent_{idx}": LatentAgent(agent_config)
        for idx in range(config.num_agents)
    }
    multi_agent = MultiAgentSystem(agents)
    orchestrator = build_default_orchestrator(config.storage_path)
    return {
        "world_model": world_model,
        "multi_agent": multi_agent,
        "orchestrator": orchestrator,
    }


def seed_graph(orchestrator) -> None:
    """Seed minimal concepts for the science loop."""

    orchestrator.seed_concepts(
        [
            ConceptNode("constraint_solver", (1.0, 0.0, 0.0), {"kind": "system"}),
            ConceptNode("world_model", (0.0, 1.0, 0.0), {"kind": "model"}),
            ConceptNode("experiment_loop", (0.0, 0.0, 1.0), {"kind": "process"}),
        ]
    )


def run(config: AxiomConfig) -> dict:
    """Run bounded autonomous research iterations."""

    components = build_system(config)
    orchestrator = components["orchestrator"]
    seed_graph(orchestrator)
    records = orchestrator.run(config.iterations)
    return {
        "iterations": orchestrator.state.iterations,
        "accepted": orchestrator.state.accepted,
        "rejected": orchestrator.state.rejected,
        "graph": orchestrator.graph.to_dict(),
        "records": records,
    }


def parse_args() -> AxiomConfig:
    parser = argparse.ArgumentParser(description="Run Axiom Engine")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--storage-path", default="output/experiments.jsonl")
    args = parser.parse_args()
    return AxiomConfig(iterations=args.iterations, storage_path=args.storage_path)


def main() -> int:
    configure_logging()
    log = get_logger("axiom")
    config = parse_args()
    result = run(config)
    log.info("completed iterations=%s accepted=%s", result["iterations"], result["accepted"])
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
