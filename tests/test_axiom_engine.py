from __future__ import annotations

import torch

from core.agent import AgentConfig, LatentAgent
from core.language import LanguageConfig, SymbolicCommunication
from core.reasoning import Rule, RuleEngine
from core.world_model import DreamerWorldModel, WorldModelConfig
from infrastructure.orchestrator import build_default_orchestrator
from science.knowledge_graph import ConceptNode


def test_world_model_observe_and_imagine() -> None:
    config = WorldModelConfig(obs_dim=8, action_dim=3, deter_dim=16, stoch_dim=4, hidden_dim=32)
    model = DreamerWorldModel(config)
    obs = torch.zeros(2, 5, 8)
    actions = torch.zeros(2, 5, 3)
    posts, priors = model.observe(obs, actions)
    assert len(posts) == 5
    assert len(priors) == 5
    agent = LatentAgent(AgentConfig(latent_dim=model.feat_dim, action_dim=3, hidden_dim=32))
    imagined = model.imagination_rollout(posts[-1], agent.act, horizon=3)
    assert len(imagined) == 3
    assert imagined[-1].feat.shape[-1] == model.feat_dim


def test_symbolic_language_and_reasoning() -> None:
    language = SymbolicCommunication(LanguageConfig(latent_dim=12, vocab_size=8, message_length=3))
    latent = torch.zeros(4, 12)
    message, context = language(latent, hard=True)
    assert message.shape == (4, 3, 8)
    assert context.shape == (4, 12)

    engine = RuleEngine(
        [
            Rule(
                name="approaching",
                condition=lambda facts: RuleEngine.and_(facts["near"], facts["moving"]),
                conclusion=lambda strength: {"approaching": strength},
            )
        ]
    )
    inferred = engine.infer({"near": 1.0, "moving": 0.5})
    assert inferred["approaching"] == 0.5


def test_science_orchestrator_updates_graph(tmp_path) -> None:
    orchestrator = build_default_orchestrator(str(tmp_path / "experiments.jsonl"))
    orchestrator.seed_concepts(
        [
            ConceptNode("a", (1.0, 0.0)),
            ConceptNode("b", (0.0, 1.0)),
        ]
    )
    record = orchestrator.run_once()
    assert record["decision"]["accepted"] is True
    assert orchestrator.state.iterations == 1
    assert len(orchestrator.graph.edges) == 1
