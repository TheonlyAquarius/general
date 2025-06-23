import torch
from torch.utils.data import TensorDataset, DataLoader
import yaml
from models.weight_diffusion_model import SimpleWeightDiffusion
from tqdm import tqdm
import os
import numpy as np # For checking latent_dim consistency

def main():
    # Load configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs/mnist_config.yaml')

    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    config = full_config['weight_diffusion']
    extraction_config = full_config['weight_extraction']

    # Paths
    extracted_weights_relative_path = extraction_config['output_path']
    diffusion_model_output_relative_path = config['output_path']

    extracted_weights_abs_path = os.path.join(project_root, extracted_weights_relative_path)
    diffusion_model_output_abs_path = os.path.join(project_root, diffusion_model_output_relative_path)

    # Ensure output directory for diffusion model exists
    diffusion_output_dir = os.path.dirname(diffusion_model_output_abs_path)
    if not os.path.exists(diffusion_output_dir):
        os.makedirs(diffusion_output_dir)
        print(f"Created directory: {diffusion_output_dir}")

    # Diffusion model parameters from config
    latent_dim = config['latent_dim']
    timesteps = config['timesteps']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']
    noise_schedule = config.get('noise_schedule', 'linear') # Default to linear if not specified

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the extracted weights
    try:
        # The extracted weights are a single tensor. We need to treat each set of weights
        # (if multiple were extracted and concatenated, which is not the current setup but for future)
        # or the single set of weights as a data point.
        # Currently, `extracted_weights.pth` stores ONE concatenated tensor of all weights from ONE model.
        # For diffusion, we ideally want a dataset of *many* such weight vectors.
        # In this PoC, we'll train the diffusion model on this single weight vector by repeating it,
        # or by loading multiple if extract_weights.py were adapted to save them individually from multiple trained MNIST models.

        all_weights_tensor = torch.load(extracted_weights_abs_path).float()
        print(f"Loaded extracted weights from: {extracted_weights_abs_path}, shape: {all_weights_tensor.shape}")

        # Verify latent_dim consistency
        if all_weights_tensor.numel() != latent_dim:
            print(f"Warning: Latent dimension in config ({latent_dim}) does not match "
                  f"the number of elements in loaded weights ({all_weights_tensor.numel()}).")
            print("Please re-run `scripts/extract_weights.py` if the MNIST model changed, or check config.")
            # Option: Update latent_dim here, or rely on extract_weights.py to have set it correctly.
            # For now, we will proceed with the config's latent_dim but this could lead to issues.
            # It's better if extract_weights.py is the source of truth for latent_dim.
            # Assuming extract_weights.py correctly set it.

        # Reshape the single tensor to be (1, latent_dim) to act as a single data point.
        # If you had N sets of weights, it would be (N, latent_dim).
        if all_weights_tensor.dim() == 1:
            all_weights_tensor = all_weights_tensor.unsqueeze(0)

        # To make a "dataset" for this PoC, we can repeat this single data point.
        # This is not ideal for learning a true distribution but demonstrates the training loop.
        num_repeats = 1000 # Arbitrary number of repeats to form a larger dataset
        repeated_weights = all_weights_tensor.repeat(num_repeats, 1).to(device)

        print(f"Using repeated single weight vector for training (shape: {repeated_weights.shape})")
        dataset = TensorDataset(repeated_weights)

    except FileNotFoundError:
        print(f"Error: Extracted weights not found at {extracted_weights_abs_path}. "
              "Please run `scripts/extract_weights.py` first.")
        return
    except Exception as e:
        print(f"Error loading or processing weights: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize the diffusion model
    # The SimpleWeightDiffusion model expects latent_dim to be the feature size of one data point.
    model = SimpleWeightDiffusion(
        latent_dim=latent_dim, # Should be all_weights_tensor.shape[1] after unsqueeze
        timesteps=timesteps,
        noise_schedule=noise_schedule
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting Weight Diffusion model training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for batch_idx, (batch_weights,) in enumerate(pbar): # batch_weights is (batch_size, latent_dim)
            batch_weights = batch_weights.to(device) # Ensure data is on the correct device
            optimizer.zero_grad()

            # t is a tensor of random timesteps for each item in the batch
            t = torch.randint(0, timesteps, (batch_weights.shape[0],), device=device).long()

            loss = model.p_losses(batch_weights, t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), diffusion_model_output_abs_path)
    print(f"Weight Diffusion Model saved to: {diffusion_model_output_abs_path}")

if __name__ == '__main__':
    main()
