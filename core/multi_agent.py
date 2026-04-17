from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from core.agent import CentralCritic, LatentAgent


class ParallelMultiAgentEnv(Protocol):
    """PettingZoo-style parallel environment protocol."""

    possible_agents: list[str]

    def reset(self, seed: int | None = None) -> tuple[dict[str, Tensor], dict]:
        ...

    def step(
        self, actions: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, float], dict[str, bool], dict[str, bool], dict]:
        ...


@dataclass(frozen=True)
class MultiAgentStep:
    """Single shared-environment transition."""

    observations: dict[str, Tensor]
    actions: dict[str, Tensor]
    rewards: dict[str, float]
    terminations: dict[str, bool]
    truncations: dict[str, bool]
    infos: dict


class MultiAgentSystem:
    """Coordinates multiple latent agents in a shared environment."""

    def __init__(
        self,
        agents: dict[str, LatentAgent],
        centralized_critic: CentralCritic | None = None,
    ):
        self.agents = agents
        self.centralized_critic = centralized_critic

    def act(self, latents: dict[str, Tensor], deterministic: bool = False) -> dict[str, Tensor]:
        missing = set(self.agents) - set(latents)
        if missing:
            raise KeyError(f"missing latents for agents: {sorted(missing)}")
        return {
            agent_id: agent.act(latents[agent_id], deterministic=deterministic)
            for agent_id, agent in self.agents.items()
        }

    def rollout(self, env: ParallelMultiAgentEnv, horizon: int, deterministic: bool = False) -> list[MultiAgentStep]:
        observations, info = env.reset()
        del info
        trajectory: list[MultiAgentStep] = []
        for _ in range(horizon):
            actions = self.act(observations, deterministic=deterministic)
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            trajectory.append(
                MultiAgentStep(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    terminations=terms,
                    truncations=truncs,
                    infos=infos,
                )
            )
            observations = next_obs
            if all(terms.get(agent_id, False) or truncs.get(agent_id, False) for agent_id in self.agents):
                break
        return trajectory

    def joint_value(self, latents: dict[str, Tensor], actions: dict[str, Tensor]) -> Tensor:
        if self.centralized_critic is None:
            raise RuntimeError("centralized critic is not configured")
        joint_latent = torch.cat([latents[agent_id] for agent_id in sorted(self.agents)], dim=-1)
        joint_action = torch.cat([actions[agent_id] for agent_id in sorted(self.agents)], dim=-1)
        return self.centralized_critic(joint_latent, joint_action)
