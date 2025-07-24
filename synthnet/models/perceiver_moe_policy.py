import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math
from .policy_api import PolicyAPI


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding




class MoELayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 4),
                nn.GELU(),
                nn.Linear(input_dim * 4, output_dim)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)
        top_k_gates, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)

        router_probs = F.softmax(gate_logits, dim=-1)
        expert_load = router_probs.mean(dim=0)
        expert_prob = router_probs.mean(dim=0)
        aux_loss = (expert_load * expert_prob).sum() * self.num_experts

        batch_size, seq_len, dim = x.shape
        flat_x = x.reshape(-1, dim)
        flat_indices = top_k_indices.reshape(-1)
        
        output = torch.zeros_like(flat_x)
        
        for i in range(self.num_experts):
            mask = (flat_indices == i)
            if mask.any():
                 expert_input = flat_x[mask]
                 expert_output = self.experts[i](expert_input)
                 output[mask] = expert_output

        return output.reshape(batch_size, seq_len, -1), aux_loss


class PerceiverMoEPolicy(PolicyAPI):
    def __init__(self, latent_dim: int, num_latents: int, num_layers: int, num_heads: int, num_experts: int):
        super().__init__()
        self.latent_dim = latent_dim
        self._structure_initialized = False

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.time_embed = TimestepEmbedding(latent_dim)

        self.cross_attention_encoder = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)

        self.latent_transformer = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True),
                nn.LayerNorm(latent_dim),
                MoELayer(input_dim=latent_dim, output_dim=latent_dim, num_experts=num_experts)
            ) for _ in range(num_layers)
        ])

        self.cross_attention_decoder = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)

    def _initialize_structure(self, weights_dict: Dict[str, torch.Tensor]):
        self._param_shapes = {k: v.shape for k, v in weights_dict.items()}
        self._param_names = list(weights_dict.keys())
        self._param_sizes = [v.numel() for v in weights_dict.values()]
        self._total_params = sum(self._param_sizes)
        
        self.input_proj = nn.Linear(self._total_params, self.latent_dim)
        self.output_proj_mu = nn.Linear(self.latent_dim, self._total_params)
        
        self._structure_initialized = True

    def _flatten_weights(self, weights_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([weights_dict[k].flatten(1) for k in self._param_names], dim=1)

    def _unflatten_weights(self, flat_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        weights_dict = {}
        start_idx = 0
        for i, name in enumerate(self._param_names):
            size = self._param_sizes[i]
            shape = self._param_shapes[name]
            weights_dict[name] = flat_tensor[:, start_idx:start_idx + size].view(-1, *shape)
            start_idx += size
        return weights_dict

    def forward(self, weights: Dict[str, torch.Tensor], t: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if not self._structure_initialized:
            with torch.no_grad():
                self._initialize_structure(weights)

        batch_size = next(iter(weights.values())).shape[0]
        device = next(iter(weights.values())).device

        x_flat = self._flatten_weights(weights)
        
        x_proj = self.input_proj(x_flat).unsqueeze(1)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        
        t_emb = self.time_embed(t).unsqueeze(1)
        latents = latents + t_emb

        latents, _ = self.cross_attention_encoder(query=latents, key=x_proj, value=x_proj)
        
        total_aux_loss = 0
        for layer in self.latent_transformer:
            norm1, attn, norm2, moe = layer[0], layer[1], layer[2], layer[3]
            latents_norm = norm1(latents)
            attn_output, _ = attn(query=latents_norm, key=latents_norm, value=latents_norm)
            latents = latents + attn_output
            
            latents_norm = norm2(latents)
            moe_output, aux_loss = moe(latents_norm)
            latents = latents + moe_output
            total_aux_loss += aux_loss

        output, _ = self.cross_attention_decoder(query=x_proj, key=latents, value=latents)
        
        mu_flat = self.output_proj_mu(output.squeeze(1))
        
        mu_dict = self._unflatten_weights(mu_flat)
        
        return mu_dict, total_aux_loss

    def step(self, weights: Dict[str, torch.Tensor], t: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self.forward(weights, t)
