import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from diffusion_model import SimpleWeightSpaceDiffusion, flatten_state_dict, unflatten_to_state_dict, get_target_model_flat_dim
from target_cnn import TargetCNN
from utils.paths import ensure_parent_directory, resolve_path

# --- Configuration ---
DEFAULT_DIFFUSION_MODEL_PATH = "diffusion_optimizer.pth"
DEFAULT_TRAJECTORY_DIR = "trajectory_weights_cnn"
DEFAULT_PLOT_PATH = "diffusion_evaluation_plot.png"
NUM_GENERATION_STEPS = 6
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Function ---
def evaluate_model(model, test_loader, device):
    """Evaluates a model's accuracy and loss on the test set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss

# --- Main Execution ---
def run_evaluation(
    diffusion_model_path: Path,
    trajectory_dir: Path,
    plot_path: Path,
    num_generation_steps: int,
    batch_size: int
) -> None:
    print("Starting evaluation of diffusion-generated trajectory...")
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Initialization ---
    # Create a reference TargetCNN to get dimensions and structure
    ref_cnn = TargetCNN()
    ref_state_dict = ref_cnn.state_dict()
    flat_dim = get_target_model_flat_dim(ref_state_dict)
    print(f"Target CNN: flat dimension = {flat_dim}")

    # Load the trained diffusion model
    diffusion_model = SimpleWeightSpaceDiffusion(
        target_model_flat_dim=flat_dim,
        time_emb_dim=128,
        hidden_dim=1024
    ).to(DEVICE)
    diffusion_model.load_state_dict(torch.load(diffusion_model_path, map_location=DEVICE))
    diffusion_model.eval()
    print(f"Diffusion model loaded from {diffusion_model_path}")

    # --- Trajectory Generation ---
    print(f"Will generate a trajectory of {num_generation_steps} steps.")
    generated_weights_sequence = []

    # 1. Create a NEW TargetCNN with random weights
    generated_cnn = TargetCNN().to(DEVICE)
    current_weights_flat = flatten_state_dict(generated_cnn.state_dict())
    generated_weights_sequence.append(current_weights_flat)

    # 2. Iteratively apply the diffusion model
    print(f"Generating trajectory with diffusion model for {num_generation_steps} steps...")
    with torch.no_grad():
        for t in range(num_generation_steps):
            timestep = torch.tensor([[t]], dtype=torch.float32).to(DEVICE)
            # Predict the next set of weights
            next_weights_flat = diffusion_model(current_weights_flat.unsqueeze(0), timestep).squeeze(0)
            generated_weights_sequence.append(next_weights_flat)
            current_weights_flat = next_weights_flat
            print(f"Generated step {t+1}/{num_generation_steps}")

    # --- Performance Evaluation ---
    print("\nEvaluating performance along the generated trajectory...")
    gen_accuracies, gen_losses = [], []
    for i, weights_flat in enumerate(generated_weights_sequence):
        # Load the generated weights into the CNN
        new_state_dict = unflatten_to_state_dict(weights_flat, ref_state_dict)
        generated_cnn.load_state_dict(new_state_dict)
        accuracy, avg_loss = evaluate_model(generated_cnn, test_loader, DEVICE)
        gen_accuracies.append(accuracy)
        gen_losses.append(avg_loss)
        if i == 0:
            print(f"Step {i} (Initial Random Weights): Accuracy = {accuracy:.2f}%, Avg Loss = {avg_loss:.4f}")
        else:
            print(f"Generated Step {i}/{num_generation_steps}: Accuracy = {accuracy:.2f}%, Avg Loss = {avg_loss:.4f}")

    # --- Compare with Original Trajectory ---
    print("\nEvaluating performance along the original CNN training trajectory...")
    orig_accuracies, orig_losses = [], []
    original_cnn = TargetCNN().to(DEVICE)
    for epoch in range(num_generation_steps + 1):
        weight_file = trajectory_dir / f"weights_epoch_{epoch}.pth"
        if weight_file.exists():
            original_cnn.load_state_dict(torch.load(weight_file, map_location=DEVICE))
            accuracy, avg_loss = evaluate_model(original_cnn, test_loader, DEVICE)
            orig_accuracies.append(accuracy)
            orig_losses.append(avg_loss)
            print(f"Original CNN Epoch {epoch}: Accuracy = {accuracy:.2f}%, Avg Loss = {avg_loss:.4f}")
        else:
            print(f"Warning: Original weight file {weight_file} not found. Skipping.")


    # --- Plotting Results ---
    print("\nPlotting results...")
    plt.figure(figsize=(12, 5))
    plt.suptitle("Comparison of Original Training vs. Diffusion-Generated Performance", fontsize=16)

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(orig_accuracies, 'bo-', label='Original CNN Training')
    plt.plot(gen_accuracies, 'ro-', label='Diffusion-Generated CNN')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch / Generation Step")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(orig_losses, 'bo-', label='Original CNN Training')
    plt.plot(gen_losses, 'ro-', label='Diffusion-Generated CNN')
    plt.title("Average Model Loss")
    plt.xlabel("Epoch / Generation Step")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    print("\nEvaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-model-path", type=str, default=DEFAULT_DIFFUSION_MODEL_PATH)
    parser.add_argument("--trajectory-dir", type=str, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--plot-path", type=str, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--output-base-dir", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=NUM_GENERATION_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    base_dir = args.output_base_dir
    diffusion_model_path = resolve_path(args.diffusion_model_path, base_dir=base_dir)
    trajectory_dir = resolve_path(args.trajectory_dir, base_dir=base_dir)
    plot_path = ensure_parent_directory(args.plot_path, base_dir=base_dir)
    if not diffusion_model_path.exists():
        print(f"Error: Diffusion model not found at '{diffusion_model_path}'.")
    elif not trajectory_dir.is_dir() or not any(trajectory_dir.iterdir()):
        print(f"Error: Trajectory directory '{trajectory_dir}' is empty or does not exist.")
    else:
        run_evaluation(
            diffusion_model_path=diffusion_model_path,
            trajectory_dir=trajectory_dir,
            plot_path=plot_path,
            num_generation_steps=args.num_steps,
            batch_size=args.batch_size
        )
