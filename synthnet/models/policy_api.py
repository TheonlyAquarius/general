from abc import ABC, abstractmethod
from typing import Any

class PolicyAPI(ABC):
    """Abstract interface for SynthNet policies."""

    @abstractmethod
    def step(self, weights: Any, t: int) -> Any:
        """Perform a single diffusion step on ``weights`` at timestep ``t``."""
        pass
