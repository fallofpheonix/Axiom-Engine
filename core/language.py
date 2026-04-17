from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LanguageConfig:
    """Discrete symbolic communication configuration."""

    latent_dim: int = 160
    vocab_size: int = 64
    message_length: int = 8
    temperature: float = 1.0


class Speaker(nn.Module):
    """Maps latent state to a differentiable discrete message."""

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config
        self.net = nn.Linear(config.latent_dim, config.vocab_size * config.message_length)

    def forward(self, latent: Tensor, hard: bool = False) -> Tensor:
        logits = self.net(latent).view(-1, self.config.message_length, self.config.vocab_size)
        return F.gumbel_softmax(logits, tau=self.config.temperature, hard=hard, dim=-1)


class Listener(nn.Module):
    """Maps discrete messages back into latent context."""

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.vocab_size * config.message_length, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

    def forward(self, message: Tensor) -> Tensor:
        return self.net(message.reshape(message.shape[0], -1))


class SymbolicCommunication(nn.Module):
    """Speaker-listener communication channel."""

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.speaker = Speaker(config)
        self.listener = Listener(config)

    def forward(self, latent: Tensor, hard: bool = False) -> tuple[Tensor, Tensor]:
        message = self.speaker(latent, hard=hard)
        context = self.listener(message)
        return message, context
