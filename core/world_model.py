from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WorldModelConfig:
    """Configuration for a Dreamer-style recurrent state-space model."""

    obs_dim: int = 16
    action_dim: int = 4
    deter_dim: int = 128
    stoch_dim: int = 32
    hidden_dim: int = 256
    min_std: float = 0.1


@dataclass
class RSSMState:
    """Deterministic and stochastic latent state."""

    deter: Tensor
    stoch: Tensor
    mean: Tensor
    std: Tensor

    @property
    def feat(self) -> Tensor:
        return torch.cat([self.deter, self.stoch], dim=-1)


class Encoder(nn.Module):
    """Observation encoder."""

    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)


class Decoder(nn.Module):
    """Latent feature decoder."""

    def __init__(self, feat_dim: int, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, feat: Tensor) -> Tensor:
        return self.net(feat)


class RSSM(nn.Module):
    """Dreamer-style recurrent state-space model."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.recurrent = nn.GRUCell(config.stoch_dim + config.action_dim, config.deter_dim)
        self.prior = nn.Sequential(
            nn.Linear(config.deter_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2 * config.stoch_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(config.deter_dim + config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2 * config.stoch_dim),
        )

    def initial(self, batch_size: int, device: torch.device | str) -> RSSMState:
        device = torch.device(device)
        deter = torch.zeros(batch_size, self.config.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.config.stoch_dim, device=device)
        mean = torch.zeros_like(stoch)
        std = torch.ones_like(stoch)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def img_step(self, prev: RSSMState, action: Tensor) -> RSSMState:
        x = torch.cat([prev.stoch, action], dim=-1)
        deter = self.recurrent(x, prev.deter)
        mean, std = self._stats(self.prior(deter))
        stoch = self._sample(mean, std)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def obs_step(self, prev: RSSMState, action: Tensor, embed: Tensor) -> tuple[RSSMState, RSSMState]:
        prior = self.img_step(prev, action)
        mean, std = self._stats(self.posterior(torch.cat([prior.deter, embed], dim=-1)))
        stoch = self._sample(mean, std)
        post = RSSMState(deter=prior.deter, stoch=stoch, mean=mean, std=std)
        return post, prior

    def observe(self, embeds: Tensor, actions: Tensor) -> tuple[list[RSSMState], list[RSSMState]]:
        if embeds.ndim != 3 or actions.ndim != 3:
            raise ValueError("embeds and actions must be [batch, time, dim]")
        batch, steps, _ = embeds.shape
        state = self.initial(batch, embeds.device)
        posts: list[RSSMState] = []
        priors: list[RSSMState] = []
        for t in range(steps):
            state, prior = self.obs_step(state, actions[:, t], embeds[:, t])
            posts.append(state)
            priors.append(prior)
        return posts, priors

    def imagine(self, state: RSSMState, policy, horizon: int) -> list[RSSMState]:
        states: list[RSSMState] = []
        current = state
        for _ in range(horizon):
            action = policy(current.feat)
            current = self.img_step(current, action)
            states.append(current)
        return states

    def _stats(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        mean, raw_std = torch.chunk(tensor, 2, dim=-1)
        std = F.softplus(raw_std) + self.config.min_std
        return mean, std

    @staticmethod
    def _sample(mean: Tensor, std: Tensor) -> Tensor:
        return mean + torch.randn_like(std) * std


class DreamerWorldModel(nn.Module):
    """World model wrapper with encoder, RSSM, decoder, reward, and continuation heads."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.obs_dim, config.hidden_dim)
        self.rssm = RSSM(config)
        feat_dim = config.deter_dim + config.stoch_dim
        self.decoder = Decoder(feat_dim, config.obs_dim, config.hidden_dim)
        self.reward = nn.Sequential(nn.Linear(feat_dim, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, 1))
        self.continue_head = nn.Sequential(
            nn.Linear(feat_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    @property
    def feat_dim(self) -> int:
        return self.config.deter_dim + self.config.stoch_dim

    def observe(self, observations: Tensor, actions: Tensor) -> tuple[list[RSSMState], list[RSSMState]]:
        embeds = self.encoder(observations)
        return self.rssm.observe(embeds, actions)

    def reconstruct(self, state: RSSMState) -> Tensor:
        return self.decoder(state.feat)

    def predict_reward(self, state: RSSMState) -> Tensor:
        return self.reward(state.feat)

    def predict_continue(self, state: RSSMState) -> Tensor:
        return torch.sigmoid(self.continue_head(state.feat))

    def imagination_rollout(self, start: RSSMState, policy, horizon: int) -> list[RSSMState]:
        return self.rssm.imagine(start, policy, horizon)

    def training_loss(self, observations: Tensor, actions: Tensor, rewards: Tensor) -> dict[str, Tensor]:
        posts, priors = self.observe(observations, actions)
        recon_loss = torch.zeros((), device=observations.device)
        reward_loss = torch.zeros((), device=observations.device)
        kl_loss = torch.zeros((), device=observations.device)
        for t, (post, prior) in enumerate(zip(posts, priors)):
            recon_loss = recon_loss + F.mse_loss(self.reconstruct(post), observations[:, t])
            reward_loss = reward_loss + F.mse_loss(self.predict_reward(post).squeeze(-1), rewards[:, t])
            kl_loss = kl_loss + self.kl(post, prior).mean()
        steps = max(len(posts), 1)
        total = (recon_loss + reward_loss + kl_loss) / steps
        return {
            "loss": total,
            "reconstruction": recon_loss / steps,
            "reward": reward_loss / steps,
            "kl": kl_loss / steps,
        }

    @staticmethod
    def kl(post: RSSMState, prior: RSSMState) -> Tensor:
        var_q = post.std.square()
        var_p = prior.std.square()
        return 0.5 * (
            (var_q + (post.mean - prior.mean).square()) / var_p
            + 2.0 * (prior.std.log() - post.std.log())
            - 1.0
        ).sum(dim=-1)
