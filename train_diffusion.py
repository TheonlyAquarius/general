import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np

from diffusion_model import SimpleWeightSpaceDiffusion, flatten_state_dict, get_target_model_flat_dim
from target_cnn import TargetCNN # To get a reference state_dict for unflattening

class WeightTrajectoryDataset(Dataset):
    def __init__(self, trajectory_dir, reference_state_dict, max_trajectory_len=None):
        """
        Dataset for loading pairs of (current_weights, next_weights) from a saved trajectory.
        The "forward" process of the diffusion model learns to reverse the optimization.
        So, if optimization goes W_t -> W_{t+1} (less loss),
        the diffusion's "noising" learns W_{t+1} -> W_t.
        The diffusion model itself will be trained to predict W_{t+1} given W_t and t.

        Args:
            trajectory_dir (str): Directory containing saved .pth weight files.
            reference_state_dict (dict): A sample state_dict from the target model,
                                         used to know the structure for unflattening if needed,
                                         and to determine flat_dim.
            max_trajectory_len (int, optional): If set, limits the number of trajectory steps used.
        """
        self.trajectory_dir = trajectory_dir
        self.reference_state_dict = reference_state_dict
        self.flat_dim = get_target_model_flat_dim(reference_state_dict)

        weight_files = sorted(
            glob.glob(os.path.join(trajectory_dir, "weights_epoch_*.pth")),
            key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else -1
        )
        # Ensure 'weights_epoch_0.pth' is first if it exists, then numeric epochs.
        # 'weights_epoch_final.pth' is usually the same as the last numeric epoch, handle as needed.
        # For this setup, we assume numeric epoch files are the primary sequence.

        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {trajectory_dir}. Run train_target_model.py first.")

        print(f"Found {len(weight_files)} weight files.")

        self.trajectory_weights = []
        for wf_path in weight_files:
            try:
                # Load weights to CPU to avoid GPU memory issues if dataset is large
                state_dict = torch.load(wf_path, map_location=torch.device('cpu'))
                flat_weights = flatten_state_dict(state_dict)
                if flat_weights.shape[0] != self.flat_dim:
                    print(f"Warning: Mismatch in flat_dim for {wf_path}. Expected {self.flat_dim}, got {flat_weights.shape[0]}. Skipping.")
                    continue
                self.trajectory_weights.append(flat_weights)
            except Exception as e:
                print(f"Error loading or processing {wf_path}: {e}")

        if not self.trajectory_weights:
            raise ValueError("No valid trajectory weights could be loaded.")

        self.num_total_steps = len(self.trajectory_weights)
        print(f"Successfully loaded {self.num_total_steps} weight states for the trajectory.")

        # Create pairs for (current_state, target_next_state)
        # The diffusion model learns to go from W_t (current, "noisier") to W_{t+1} (target, "less noisy")
        self.data_pairs = []
        for i in range(self.num_total_steps - 1):
            # current_weights_t = W_i (earlier in training, considered "noisier" in reverse)
            # target_weights_t_plus_1 = W_{i+1} (later in training, "less noisy")
            current_weights_t = self.trajectory_weights[i]
            target_weights_t_plus_1 = self.trajectory_weights[i+1]
            # The timestep 't' corresponds to the index 'i' of current_weights_t
            self.data_pairs.append((current_weights_t, target_weights_t_plus_1, float(i)))

        if max_trajectory_len and max_trajectory_len < len(self.data_pairs):
            self.data_pairs = self.data_pairs[:max_trajectory_len]
            print(f"Using a partial trajectory of length {len(self.data_pairs)}")


    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        current_w, target_next_w, t = self.data_pairs[idx]
        # Timestep t is the index of the current_w in the original trajectory
        # It can be normalized or scaled if needed by the diffusion model's time embedding
        return current_w, target_next_w, torch.tensor([t])


def train_diffusion_model(
    trajectory_dir,
    target_model_ref_for_dims, # An instance of TargetCNN or its state_dict
    epochs=100,
    lr=0.0001,
    batch_size=32,
    time_emb_dim=64,
    hidden_dim_diff_model=512,
    save_path="trained_diffusion_model.pth",
    max_traj_len_for_training=None
):
    print("Starting Diffusion Model training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if isinstance(target_model_ref_for_dims, nn.Module):
        reference_state_dict = target_model_ref_for_dims.state_dict()
    else: # assuming it's already a state_dict
        reference_state_dict = target_model_ref_for_dims

    target_flat_dim = get_target_model_flat_dim(reference_state_dict)
    print(f"Target model flattened dimension: {target_flat_dim}")

    # Initialize Diffusion Model
    diffusion_model = SimpleWeightSpaceDiffusion(
        target_model_flat_dim=target_flat_dim,
        time_emb_dim=time_emb_dim,
        hidden_dim=hidden_dim_diff_model
    ).to(device)
    print("Diffusion model initialized.")

    # Prepare Dataset and DataLoader
    print("Loading weight trajectory dataset...")
    dataset = WeightTrajectoryDataset(trajectory_dir, reference_state_dict, max_trajectory_len=max_traj_len_for_training)
    if len(dataset) == 0:
        print("Dataset is empty. Cannot train. Ensure 'trajectory_weights_cnn' has valid data.")
        return
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset loaded with {len(dataset)} pairs.")

    # Loss and Optimizer
    criterion = nn.MSELoss() # Predicts the next state directly
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
    print("Criterion and Optimizer initialized.")

    # Training Loop
    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        diffusion_model.train()
        total_loss = 0
        for current_weights_flat, target_next_weights_flat, timesteps_t in dataloader:
            current_weights_flat = current_weights_flat.to(device)
            target_next_weights_flat = target_next_weights_flat.to(device)
            timesteps_t = timesteps_t.to(device) # Shape [batch_size, 1]

            optimizer.zero_grad()

            # The diffusion model predicts W_{t+1} given W_t and t
            predicted_next_weights_flat = diffusion_model(current_weights_flat, timesteps_t)

            loss = criterion(predicted_next_weights_flat, target_next_weights_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}], Average Loss: {avg_loss:.6f}")

    # Save the trained model
    torch.save(diffusion_model.state_dict(), save_path)
    print(f"Trained diffusion model saved to {save_path}")


if __name__ == '__main__':
    # --- Configuration ---
    # Directory where train_target_model.py saved the CNN's weight trajectory
    CNN_TRAJECTORY_DIR = "trajectory_weights_cnn"

    # Number of epochs to train the diffusion model
    DIFFUSION_EPOCHS = 50 # Adjust as needed, more epochs generally better
    LEARNING_RATE_DIFFUSION = 0.0005
    BATCH_SIZE_DIFFUSION = 16 # Depends on memory and trajectory length

    # Diffusion model architecture parameters
    TIME_EMB_DIM_DIFF = 128
    HIDDEN_DIM_DIFF = 1024 # MLP hidden layer size

    # Path to save the trained diffusion model
    SAVED_DIFFUSION_MODEL_PATH = "diffusion_optimizer.pth"

    # Optional: Limit the number of trajectory steps used for training (e.g., for faster iteration)
    # Set to None to use the full trajectory.
    MAX_TRAJECTORY_LENGTH_FOR_TRAINING = None
    # Example: 10 would mean using only the first 10 transitions (W0->W1, W1->W2, ..., W9->W10)

    # --- End Configuration ---

    # We need a reference model or state_dict to know the dimensions and structure
    # Initialize a dummy TargetCNN for this purpose.
    # Ensure this matches the architecture used in train_target_model.py
    print("Initializing reference TargetCNN for dimensions...")
    reference_cnn = TargetCNN()
    # If trajectory_weights_cnn was generated with a different model, this must match.

    # Check if trajectory directory exists
    if not os.path.exists(CNN_TRAJECTORY_DIR) or not os.listdir(CNN_TRAJECTORY_DIR):
        print(f"Error: Trajectory directory '{CNN_TRAJECTORY_DIR}' is empty or does not exist.")
        print("Please run 'train_target_model.py' first to generate the weight trajectory.")
    else:
        train_diffusion_model(
            trajectory_dir=CNN_TRAJECTORY_DIR,
            target_model_ref_for_dims=reference_cnn.state_dict(), # Pass the state_dict
            epochs=DIFFUSION_EPOCHS,
            lr=LEARNING_RATE_DIFFUSION,
            batch_size=BATCH_SIZE_DIFFUSION,
            time_emb_dim=TIME_EMB_DIM_DIFF,
            hidden_dim_diff_model=HIDDEN_DIM_DIFF,
            save_path=SAVED_DIFFUSION_MODEL_PATH,
            max_traj_len_for_training=MAX_TRAJECTORY_LENGTH_FOR_TRAINING
        )
