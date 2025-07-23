import torch
import torch.nn as nn
from perceiver_pytorch import PerceiverIO

from .policy_api import PolicyAPI

class PerceiverIOPolicy(nn.Module, PolicyAPI):
    """Weight-denoising policy using the Perceiver IO architecture."""

    def __init__(self, input_dim: int, latent_dim: int = 512, depth: int = 6, num_latents: int = 256):
        super().__init__()
        self.perceiver = PerceiverIO(
            depth=depth,
            dim=input_dim,
            queries_dim=input_dim,
            logits_dim=input_dim,
            num_latents=num_latents,
            latent_dim=latent_dim,
        )

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        queries = x
        return self.perceiver(x, queries=queries)

    def step(self, weights: torch.Tensor, t: int) -> torch.Tensor:  # type: ignore[override]
        return self.forward(weights, t)

