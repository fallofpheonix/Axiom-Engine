from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AgentConfig:
    """Latent actor-critic configuration."""

    latent_dim: int = 160
    action_dim: int = 4
    hidden_dim: int = 256
    min_std: float = 0.05


class Actor(nn.Module):
    """Continuous latent-space policy."""

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.backbone = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.std = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, latent: Tensor, deterministic: bool = False) -> Tensor:
        h = self.backbone(latent)
        mean = torch.tanh(self.mean(h))
        if deterministic:
            return mean
        std = F.softplus(self.std(h)) + self.config.min_std
        action = mean + torch.randn_like(mean) * std
        return torch.clamp(action, -1.0, 1.0)

    def log_prob(self, latent: Tensor, action: Tensor) -> Tensor:
        h = self.backbone(latent)
        mean = torch.tanh(self.mean(h))
        std = F.softplus(self.std(h)) + self.config.min_std
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)


class Critic(nn.Module):
    """Latent value estimator."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent).squeeze(-1)


class CentralCritic(nn.Module):
    """Centralized critic for multi-agent training."""

    def __init__(self, joint_latent_dim: int, joint_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_latent_dim + joint_action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_latent: Tensor, joint_action: Tensor) -> Tensor:
        return self.net(torch.cat([joint_latent, joint_action], dim=-1)).squeeze(-1)


class LatentAgent(nn.Module):
    """Actor-critic agent operating entirely in latent space."""

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.actor = Actor(config)
        self.critic = Critic(config.latent_dim, config.hidden_dim)

    def act(self, latent: Tensor, deterministic: bool = False) -> Tensor:
        return self.actor(latent, deterministic=deterministic)

    def value(self, latent: Tensor) -> Tensor:
        return self.critic(latent)

    def actor_loss(self, imagined_latents: list[Tensor], discount: float = 0.99) -> Tensor:
        loss = torch.zeros((), device=imagined_latents[0].device)
        for t, latent in enumerate(imagined_latents):
            loss = loss - (discount**t) * self.value(latent).mean()
        return loss / max(len(imagined_latents), 1)

    def critic_loss(self, latents: Tensor, returns: Tensor) -> Tensor:
        return F.mse_loss(self.value(latents), returns.detach())


def lambda_returns(rewards: Tensor, values: Tensor, discount: float = 0.99, lambda_: float = 0.95) -> Tensor:
    """Compute TD-lambda returns for imagined trajectories."""

    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have identical shape")
    returns = torch.zeros_like(rewards)
    next_value = values[:, -1]
    for t in reversed(range(rewards.shape[1])):
        next_value = rewards[:, t] + discount * ((1.0 - lambda_) * values[:, t] + lambda_ * next_value)
        returns[:, t] = next_value
    return returns
