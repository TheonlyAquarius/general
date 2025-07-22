import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import glob
from diffusion_model import SimpleWeightSpaceDiffusion, flatten_state_dict
from target_cnn import TargetCNN

# --- Configuration ---
TRAJECTORY_DIR = "trajectory_weights_cnn"
DIFFUSION_MODEL_SAVE_PATH = "diffusion_optimizer.pth"
EPOCHS = 50
LEARNING_RATE = 0.0001
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class WeightTrajectoryDataset(Dataset):
    """
    Dataset to load pairs of consecutive weights from a directory.
    Each item is (current_weights, next_weights, timestep).
    """
    def __init__(self, trajectory_dir, ref_state_dict):
        self.file_paths = sorted(
            glob.glob(os.path.join(trajectory_dir, "weights_epoch_*.pth")),
            key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
        )
        self.ref_state_dict = ref_state_dict

    def __len__(self):
        return len(self.file_paths) - 1

    def __getitem__(self, idx):
        # Load current and next state dicts
        current_state_path = self.file_paths[idx]
        next_state_path = self.file_paths[idx + 1]

        current_state_dict = torch.load(current_state_path, map_location=DEVICE)
        next_state_dict = torch.load(next_state_path, map_location=DEVICE)

        # Flatten them into vectors
        current_weights_flat = flatten_state_dict(current_state_dict)
        next_weights_flat = flatten_state_dict(next_state_dict)

        # The timestep 't' is the current epoch index
        timestep = torch.tensor([idx], dtype=torch.float32).to(DEVICE)

        return current_weights_flat, next_weights_flat, timestep

# --- Main Training Function ---
def train_diffusion_model(trajectory_dir, target_model_ref_for_dims, epochs, lr, batch_size, save_path):
    """
    Trains the diffusion model on the saved weight trajectory.
    """
    print(f"Using device: {DEVICE}")

    # Prepare dataset and dataloader
    dataset = WeightTrajectoryDataset(trajectory_dir, target_model_ref_for_dims)
    if len(dataset) == 0:
        print(f"Error: No weight files found in '{trajectory_dir}'.")
        print("Please run 'train_target_model.py' first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded a trajectory of {len(dataset)} steps from '{trajectory_dir}'.")

    # Get the dimension of the flattened weights from the reference model
    flat_dim = sum(p.numel() for p in target_model_ref_for_dims.values())

    # Instantiate the diffusion model
    diffusion_model = SimpleWeightSpaceDiffusion(
        target_model_flat_dim=flat_dim,
        time_emb_dim=128,
        hidden_dim=1024
    ).to(DEVICE)

    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    print("Starting training of the diffusion model...")

    # Training loop
    for epoch in range(1, epochs + 1):
        diffusion_model.train()
        total_loss = 0
        for current_w, next_w, t in dataloader:
            optimizer.zero_grad()
            predicted_w = diffusion_model(current_w, t)
            loss = criterion(predicted_w, next_w)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Average MSE Loss: {avg_loss:.6f}")

    # Save the trained model
    torch.save(diffusion_model.state_dict(), save_path)
    print(f"\nTraining finished. Diffusion model saved to '{save_path}'.")

if __name__ == "__main__":
    # Create a reference model instance to get its structure
    ref_model = TargetCNN()
    train_diffusion_model(
        trajectory_dir=TRAJECTORY_DIR,
        target_model_ref_for_dims=ref_model.state_dict(),
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        save_path=DIFFUSION_MODEL_SAVE_PATH
    )
