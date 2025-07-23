import torch
import torch.nn as nn
from perceiver_pytorch import PerceiverIO

from .policy_api import PolicyAPI

class PerceiverMoEPolicy(nn.Module, PolicyAPI):
    """Perceiver IO with a simple Mixture-of-Experts decoder."""

    def __init__(self, input_dim: int, latent_dim: int = 512, num_experts: int = 4, depth: int = 6, num_latents: int = 256):
        super().__init__()
        self.perceiver = PerceiverIO(
            depth=depth,
            dim=input_dim,
            queries_dim=input_dim,
            logits_dim=latent_dim,
            num_latents=num_latents,
            latent_dim=latent_dim,
        )
        self.experts = nn.ModuleList([nn.Linear(latent_dim, input_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(latent_dim, num_experts)

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        latent = self.perceiver(x, queries=x)
        weights = torch.softmax(self.gate(latent), dim=-1)
        expert_outputs = torch.stack([expert(latent) for expert in self.experts], dim=-1)
        out = (expert_outputs * weights.unsqueeze(-2)).sum(dim=-1)
        return out

    def step(self, weights: torch.Tensor, t: int) -> torch.Tensor:  # type: ignore[override]
        return self.forward(weights, t)

