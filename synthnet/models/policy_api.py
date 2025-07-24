import torch
from torch import nn
from typing import Dict, Tuple


class PolicyAPI(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, weights: Dict[str, torch.Tensor], t: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError
