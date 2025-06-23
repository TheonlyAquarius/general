import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # Normalize to start at 1
    # Original paper uses  alphas_cumprod = alphas_cumprod / alphas_cumprod[0] * (1 - s + s * torch.cos(torch.pi * s) ** 2)
    # but this can lead to values > 1 if s is small.
    # For stability, we ensure it starts at 1 and then derive betas.
    # Let's stick to the simpler normalization ensuring it starts at 1.
    betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999) # Clip to avoid numerical issues

def get_index_from_list(vals, t, x_shape):
    """
    Returns a tensor of values from the list `vals` based on the indices
    provided in `t` while considering the shape of the output `x_shape`.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu()) # .cpu() might be needed if t is on GPU and vals on CPU
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class SimpleWeightDiffusion(nn.Module):
    def __init__(self, latent_dim, timesteps, noise_schedule='linear'): # Removed diffusion_steps, not standard
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps

        if noise_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif noise_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unsupported noise schedule: {noise_schedule}")

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # Clip variance to prevent NaN issues if 1. - self.alphas_cumprod is too small
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)


        # Denoising model: a simple MLP
        # The input to denoise_fn will be the noisy data x_t and the timestep t
        # Timestep t is usually embedded.
        time_embed_dim = latent_dim # Choose an embedding dimension for time
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim), # Simple linear projection for timestep
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        self.denoise_fn = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, 512), # Input: latent_dim (data) + time_embed_dim
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim) # Output: predicted noise (same dim as data)
        )

        # Ensure all buffers are registered and moved to device with the model
        self.register_buffer('betas_buf', self.betas)
        self.register_buffer('alphas_buf', self.alphas)
        self.register_buffer('alphas_cumprod_buf', self.alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev_buf', self.alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod_buf', self.sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod_buf', self.sqrt_one_minus_alphas_cumprod)
        self.register_buffer('posterior_variance_buf', self.posterior_variance)


    def _extract(self, vals, t, x_shape):
        # Make sure vals is on the same device as t
        return get_index_from_list(vals.to(t.device), t, x_shape)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Diffuse the data (x_start) for a given number of diffusion steps (t).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod_buf, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod_buf, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        """
        Loss for training the denoising model.
        Predicts the noise added to x_start at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Timestep embedding
        # Normalize t to be in [0, 1] for the MLP, then pass through time_mlp
        # t is 0-indexed, timesteps is the total count
        time_input = (t.float() / self.timesteps).unsqueeze(-1) # Shape: (batch_size, 1)
        time_embedded = self.time_mlp(time_input) # Shape: (batch_size, time_embed_dim)

        denoise_input = torch.cat([x_noisy, time_embedded], dim=-1)
        predicted_noise = self.denoise_fn(denoise_input)

        return F.mse_loss(noise, predicted_noise)

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        Sample x_{t-1} from x_t using the denoising model.
        """
        betas_t = self._extract(self.betas_buf, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod_buf, t, x_t.shape)
        # sqrt_recip_alphas_t is sqrt(1/alpha_t)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self._extract(self.alphas_buf, t, x_t.shape))


        # Timestep embedding for denoising model
        time_input = (t.float() / self.timesteps).unsqueeze(-1)
        time_embedded = self.time_mlp(time_input)
        denoise_input = torch.cat([x_t, time_embedded], dim=-1)

        # Equation 11 in DDPM paper:
        # model_mean = sqrt_recip_alphas_t * (x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)
        predicted_noise = self.denoise_fn(denoise_input)
        model_mean = sqrt_recip_alphas_t * (
            x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise
        )

        if t[0].item() == 0: # Check if all elements in t are 0. t is a tensor.
            return model_mean # No noise added at the last step
        else:
            posterior_variance_t = self._extract(self.posterior_variance_buf, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape):
        """
        Generate new samples by iterating the reverse diffusion process.
        shape: (batch_size, latent_dim)
        """
        device = next(self.parameters()).device # Get device model is on

        # Start from pure noise x_T
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
            # imgs.append(img.cpu().numpy()) # Optional: store intermediate steps
        return img # Return the final denoised sample x_0

if __name__ == '__main__':
    # Example Usage
    latent_dim = 64 # Example, should match extracted weights
    timesteps = 100
    model = SimpleWeightDiffusion(latent_dim, timesteps)

    # Test q_sample
    x0 = torch.randn(10, latent_dim) # Batch of 10, latent_dim features
    t_sample = torch.randint(0, timesteps, (10,), dtype=torch.long)
    x_noisy_sample = model.q_sample(x0, t_sample)
    print(f"x_noisy_sample shape: {x_noisy_sample.shape}")

    # Test p_losses
    loss = model.p_losses(x0, t_sample)
    print(f"Loss: {loss.item()}")

    # Test sampling
    generated_samples = model.sample((5, latent_dim)) # Generate 5 samples
    print(f"Generated samples shape: {generated_samples.shape}")

    # Test with cosine schedule
    model_cosine = SimpleWeightDiffusion(latent_dim, timesteps, noise_schedule='cosine')
    loss_cosine = model_cosine.p_losses(x0, t_sample)
    print(f"Loss (cosine schedule): {loss_cosine.item()}")
    generated_samples_cosine = model_cosine.sample((5, latent_dim))
    print(f"Generated samples shape (cosine): {generated_samples_cosine.shape}")

    # Check if buffers are on the correct device if model is moved
    if torch.cuda.is_available():
        print("Moving model to CUDA")
        model.cuda()
        x0_cuda = x0.cuda()
        t_sample_cuda = t_sample.cuda()
        loss_cuda = model.p_losses(x0_cuda, t_sample_cuda)
        print(f"Loss on CUDA: {loss_cuda.item()}")
        generated_samples_cuda = model.sample((5, latent_dim)) # Shape will be (5, latent_dim)
        print(f"Generated samples shape on CUDA: {generated_samples_cuda.shape}")
        # Check a buffer
        print(f"betas_buf device: {model.betas_buf.device}")
    else:
        print("CUDA not available, tests run on CPU.")
