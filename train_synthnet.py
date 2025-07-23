import hydra
from omegaconf import DictConfig
import torch

from synthnet.models.perceiver_moe_policy import PerceiverMoEPolicy


@hydra.main(config_path="synthnet/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    policy_cfg = cfg.model
    policy = PerceiverMoEPolicy(
        latent_dim=policy_cfg.latent_dim,
        num_experts=policy_cfg.num_experts,
    )

    print("Initialized SynthNet policy with config:")
    print(policy_cfg)
    # Placeholder for sequence-aware GRPO loop
    for iter in range(cfg.rl.max_steps):
        dummy_weights = torch.zeros(1, policy_cfg.latent_dim)
        policy.step(dummy_weights, iter)
        if (iter + 1) % cfg.rl.max_steps == 0:
            print(f"Completed step {iter+1}")

if __name__ == "__main__":
    main()
