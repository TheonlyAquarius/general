import hydra
from omegaconf import DictConfig
import torch

from synthnet.models.perceiver_moe_policy import PerceiverMoEPolicy

@hydra.main(config_path="synthnet/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    policy_cfg = cfg.model
    policy = PerceiverMoEPolicy(
        input_dim=policy_cfg.input_dim,
        latent_dim=policy_cfg.latent_dim,
        num_experts=policy_cfg.num_experts,
        num_latents=policy_cfg.num_latents,
        depth=policy_cfg.depth,
    )

    print("Initialized SynthNet policy with config:")
    print(policy_cfg)
    for iter in range(cfg.rl.max_steps):
        dummy = torch.zeros(1, 4, policy_cfg.input_dim)
        policy.step(dummy, iter)
        if (iter + 1) % cfg.rl.max_steps == 0:
            print(f"Completed step {iter+1}")

if __name__ == "__main__":
    main()
