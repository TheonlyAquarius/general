import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from target_cnn import TargetCNN
from utils.paths import ensure_directory, resolve_path

# --- Configuration ---
DEFAULT_TRAJECTORY_DIR = "trajectory_weights_cnn"
EPOCHS = 7
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Main Execution ---
def train_and_capture_trajectory(trajectory_dir: Path) -> None:
    """
    Trains a TargetCNN on MNIST and saves its state_dict after each epoch
    to create an optimization trajectory.
    """
    print(f"Using device: {DEVICE}")

    # Data loading and transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, optimizer, and loss function
    model = TargetCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    print("Starting training of TargetCNN...")

    # Save initial random weights (Epoch 0)
    initial_save_path = trajectory_dir / "weights_epoch_0.pth"
    torch.save(model.state_dict(), initial_save_path)
    print(f"Saved initial random weights to {initial_save_path}")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

        # Save model weights after each epoch
        epoch_save_path = trajectory_dir / f"weights_epoch_{epoch}.pth"
        torch.save(model.state_dict(), epoch_save_path)
        print(f"Saved weights for epoch {epoch} to {epoch_save_path}")

    print("\nFinished training and capturing trajectory.")
    print(f"All weight files are saved in '{trajectory_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=str, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--output-base-dir", type=str, default=None)
    args = parser.parse_args()
    trajectory_dir = resolve_path(args.trajectory_dir, base_dir=args.output_base_dir)
    if not trajectory_dir.exists():
        ensure_directory(trajectory_dir)
        print(f"Created directory: {trajectory_dir}")
    else:
        ensure_directory(trajectory_dir)
    train_and_capture_trajectory(trajectory_dir)
