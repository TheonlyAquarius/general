import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from target_cnn import TargetCNN
from diffusion_model import SimpleWeightSpaceDiffusion, flatten_state_dict, unflatten_to_state_dict, get_target_model_flat_dim
from utils.paths import ensure_parent_directory, resolve_path

def evaluate_model_performance(model, test_loader, device, criterion):
    """Evaluates the model on the test set and returns accuracy and loss."""
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0) # Sum batch loss
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

    avg_loss = test_loss / total_samples
    accuracy = 100. * correct / total_samples
    return accuracy, avg_loss

def generate_trajectory_with_diffusion(
    diffusion_model,
    initial_weights_flat, # Starting point for generation (e.g., random weights)
    num_steps, # Number of "denoising" steps to perform, corresponds to trajectory length
    target_model_reference_state_dict, # For unflattening
    device
):
    """
    Generates a sequence of weight states using the diffusion model's reverse process.
    """
    print(f"Generating trajectory with diffusion model for {num_steps} steps...")
    generated_weights_sequence_flat = []

    current_weights_flat = initial_weights_flat.to(device)
    diffusion_model.to(device)
    diffusion_model.eval()

    for t_idx in range(num_steps):
        # The timestep 't' for the diffusion model should correspond to how it was trained.
        # If trained with t = 0, 1, ..., N-1 for N states,
        # then t_idx here represents the current position in the *original* trajectory.
        # The model was trained: model(W_t, t) -> W_{t+1}
        timestep_tensor = torch.tensor([[float(t_idx)]], device=device) # Shape [1, 1]

        with torch.no_grad():
            # Predict the next state (more optimized state)
            predicted_next_weights_flat = diffusion_model(current_weights_flat.unsqueeze(0), timestep_tensor)
            predicted_next_weights_flat = predicted_next_weights_flat.squeeze(0)

        generated_weights_sequence_flat.append(predicted_next_weights_flat.cpu())
        current_weights_flat = predicted_next_weights_flat # Update for the next step

        if (t_idx + 1) % 10 == 0 or (t_idx + 1) == num_steps:
            print(f"  Generated step {t_idx+1}/{num_steps}")

    return generated_weights_sequence_flat


def evaluate_diffusion_generated_trajectory(
    diffusion_model_path: Path,
    target_cnn_reference: TargetCNN,
    trajectory_dir_original_cnn: Path,
    plot_path: Optional[Path],
    batch_size_eval: int = 256,
    plot_results: bool = True
) -> None:
    print("Starting evaluation of diffusion-generated trajectory...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load reference TargetCNN and its properties ---
    target_cnn_reference.to(device)
    reference_state_dict = target_cnn_reference.state_dict()
    target_flat_dim = get_target_model_flat_dim(reference_state_dict)
    print(f"Target CNN: flat dimension = {target_flat_dim}")

    # --- 2. Load the trained Diffusion Model ---
    # Infer diffusion model parameters (need to match how it was trained)
    # This is a simplification; ideally, these params would be saved with the model.
    # Assuming common parameters used in train_diffusion.py
    # These must match the parameters used to train the diffusion_optimizer.pth
    # If they don't match, the loaded state_dict will cause errors.
    time_emb_dim_diff = 128 # Must match the one used for training
    hidden_dim_diff = 1024  # Must match the one used for training

    diffusion_model = SimpleWeightSpaceDiffusion(
        target_model_flat_dim=target_flat_dim,
        time_emb_dim=time_emb_dim_diff, # Ensure this matches the trained model
        hidden_dim=hidden_dim_diff      # Ensure this matches the trained model
    )
    try:
        diffusion_model.load_state_dict(torch.load(diffusion_model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading diffusion model state_dict: {e}")
        print("This often happens if the model architecture parameters (flat_dim, time_emb_dim, hidden_dim) do not match the saved model.")
        print(f"Parameters used for loading: flat_dim={target_flat_dim}, time_emb_dim={time_emb_dim_diff}, hidden_dim={hidden_dim_diff}")
        return
    diffusion_model.eval()
    print(f"Diffusion model loaded from {diffusion_model_path}")

    # --- 3. Prepare Initial Weights and Trajectory Length ---
    initial_cnn_weights_path = trajectory_dir_original_cnn / "weights_epoch_0.pth"
    if not initial_cnn_weights_path.exists():
        print(f"Initial weights file not found: {initial_cnn_weights_path}")
        print("Cannot proceed. Ensure 'train_target_model.py' was run and weights_epoch_0.pth exists.")
        return

    initial_cnn_state_dict = torch.load(initial_cnn_weights_path, map_location=torch.device('cpu'))
    initial_weights_flat = flatten_state_dict(initial_cnn_state_dict).to(device)

    # Determine the number of steps from the original trajectory
    def sort_key(path: Path) -> int:
        name = path.stem.split("_")[-1]
        return int(name) if name.isdigit() else -1

    original_weight_files = sorted(trajectory_dir_original_cnn.glob("weights_epoch_*.pth"), key=sort_key)
    # Number of steps for generation should match the number of transitions the diffusion model learned
    # If N states (W_0, ..., W_{N-1}), there are N-1 transitions.
    # The timesteps t used during training were 0, 1, ..., N-2.
    # So, we need to generate N-1 steps.
    num_generation_steps = len(original_weight_files) - 1
    if num_generation_steps <= 0:
        print(f"Not enough weight files in {trajectory_dir_original_cnn} to determine generation steps.")
        return
    print(f"Will generate a trajectory of {num_generation_steps} steps.")


    # --- 4. Generate Trajectory using Diffusion Model ---
    generated_weights_flat_sequence = generate_trajectory_with_diffusion(
        diffusion_model,
        initial_weights_flat,
        num_generation_steps,
        reference_state_dict, # For unflattening structure
        device
    )

    # --- 5. Evaluate Performance along the Generated Trajectory ---
    print("\nEvaluating performance along the generated trajectory...")
    # Load MNIST test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='sum') # Sum losses for batch, then average manually

    eval_model = TargetCNN().to(device) # Create a new model instance for evaluation

    accuracies_generated = []
    losses_generated = []

    # Evaluate initial random state (corresponds to step 0 of diffusion generation / epoch 0 of CNN)
    eval_model.load_state_dict(unflatten_to_state_dict(initial_weights_flat.cpu(), reference_state_dict))
    acc, loss = evaluate_model_performance(eval_model, test_loader, device, criterion)
    accuracies_generated.append(acc)
    losses_generated.append(loss)
    print(f"Step 0 (Initial Random Weights): Accuracy = {acc:.2f}%, Avg Loss = {loss:.4f}")

    # Evaluate each generated state
    for i, flat_weights in enumerate(generated_weights_flat_sequence):
        generated_state_dict = unflatten_to_state_dict(flat_weights, reference_state_dict)
        eval_model.load_state_dict(generated_state_dict)
        acc, loss = evaluate_model_performance(eval_model, test_loader, device, criterion)
        accuracies_generated.append(acc)
        losses_generated.append(loss)
        print(f"Generated Step {i+1}/{num_generation_steps}: Accuracy = {acc:.2f}%, Avg Loss = {loss:.4f}")

    # --- 6. (Optional) Load and Evaluate Original Trajectory for Comparison ---
    accuracies_original = []
    losses_original = []
    if trajectory_dir_original_cnn.exists():
        print("\nEvaluating performance along the original CNN training trajectory...")
        for i in range(num_generation_steps + 1): # Evaluate N states (0 to N-1)
            weight_file = trajectory_dir_original_cnn / f"weights_epoch_{i}.pth"
            if weight_file.exists():
                state_dict_orig = torch.load(weight_file, map_location=device)
                eval_model.load_state_dict(state_dict_orig)
                acc, loss = evaluate_model_performance(eval_model, test_loader, device, criterion)
                accuracies_original.append(acc)
                losses_original.append(loss)
                print(f"Original CNN Epoch {i}: Accuracy = {acc:.2f}%, Avg Loss = {loss:.4f}")
            else:
                print(f"Warning: Original weight file {weight_file} not found. Skipping.")
                # Add NaN or break if strict comparison is needed for all points
                accuracies_original.append(np.nan)
                losses_original.append(np.nan)


    # --- 7. Plot Results if enabled ---
    if plot_results and plot_path is not None:
        print("\nPlotting results...")
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(accuracies_generated, label="Diffusion Generated Trajectory", marker='o')
        if accuracies_original:
             # Ensure x-axis matches if lengths are different due to missing files
            plt.plot(range(len(accuracies_original)), accuracies_original, label="Original CNN Trajectory", marker='x')
        plt.xlabel("Optimization Step / Epoch")
        plt.ylabel("Test Accuracy (%)")
        plt.title("Accuracy: Diffusion vs Original CNN")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(losses_generated, label="Diffusion Generated Trajectory", marker='o')
        if losses_original:
            plt.plot(range(len(losses_original)), losses_original, label="Original CNN Trajectory", marker='x')
        plt.xlabel("Optimization Step / Epoch")
        plt.ylabel("Average Test Loss")
        plt.title("Loss: Diffusion vs Original CNN")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        # plt.show() # This might not work in all environments; saving is safer.

    print("Evaluation finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-model-path", type=str, default="diffusion_optimizer.pth")
    parser.add_argument("--trajectory-dir", type=str, default="trajectory_weights_cnn")
    parser.add_argument("--plot-path", type=str, default="diffusion_evaluation_plot.png")
    parser.add_argument("--output-base-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    base_dir = args.output_base_dir
    diffusion_model_path = resolve_path(args.diffusion_model_path, base_dir=base_dir)
    trajectory_dir = resolve_path(args.trajectory_dir, base_dir=base_dir)
    plot_path = None if args.no_plot else ensure_parent_directory(args.plot_path, base_dir=base_dir)
    if not diffusion_model_path.exists():
        print(f"Error: Trained diffusion model not found at '{diffusion_model_path}'.")
        print("Please run 'train_diffusion.py' first.")
    elif not trajectory_dir.is_dir() or not any(trajectory_dir.iterdir()):
        print(f"Error: CNN trajectory directory '{trajectory_dir}' is empty or does not exist.")
        print("Please run 'train_target_model.py' first.")
    else:
        reference_cnn_instance = TargetCNN()
        evaluate_diffusion_generated_trajectory(
            diffusion_model_path=diffusion_model_path,
            target_cnn_reference=reference_cnn_instance,
            trajectory_dir_original_cnn=trajectory_dir,
            plot_path=plot_path,
            batch_size_eval=args.batch_size,
            plot_results=not args.no_plot
        )
