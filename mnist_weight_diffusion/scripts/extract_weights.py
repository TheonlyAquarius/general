import torch
import yaml
import os # Added for path handling

def main():
    # Load configuration
    with open('configs/mnist_config.yaml', 'r') as f:
        config_full = yaml.safe_load(f)
    config = config_full['weight_extraction']
    mnist_config = config_full['mnist_classifier'] # Needed to load the model structure

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # mnist_weight_diffusion

    model_path_relative = config['model_path']
    output_path_relative = config['output_path']

    model_path_absolute = os.path.join(project_root, model_path_relative)
    output_path_absolute = os.path.join(project_root, output_path_relative)

    # Ensure output directory for extracted weights exists
    output_dir = os.path.dirname(output_path_absolute)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load the model state dict
    try:
        model_state_dict = torch.load(model_path_absolute, map_location=torch.device('cpu')) # Load to CPU
        print(f"Successfully loaded model from: {model_path_absolute}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path_absolute}. Please train the MNIST classifier first by running scripts/train_mnist.py")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract all weight parameters (excluding biases for simplicity in this POC)
    # We need to know the model structure to correctly interpret the state_dict,
    # but for this script, we'll just iterate through it.
    weights = []
    total_elements = 0
    print("Extracting weights (parameters named 'weight'):")
    for name, param in model_state_dict.items():
        if 'weight' in name: # A common convention for weight tensors
            print(f"  - Found weight tensor: {name} with shape {param.shape} and {param.numel()} elements")
            weights.append(param.flatten())
            total_elements += param.numel()
        elif 'bias' in name:
            print(f"  - Skipping bias tensor: {name} with shape {param.shape}")
        else:
            print(f"  - Skipping other parameter: {name} with shape {param.shape}")


    if not weights:
        print("No weight tensors found in the model. Check the model structure and naming convention.")
        # Update config to reflect the actual latent_dim if no weights are found or if it's different.
        # This is tricky because the diffusion model config is static.
        # For now, we'll just print a warning if total_elements is 0.
        if total_elements == 0:
            print("Warning: No elements found in weights. The latent_dim in weight_diffusion config might be incorrect.")
        return

    # Concatenate all weights into a single tensor
    all_weights_tensor = torch.cat(weights)

    # Save the extracted weights
    torch.save(all_weights_tensor, output_path_absolute)
    print(f"Extracted weights saved to: {output_path_absolute} (Shape: {all_weights_tensor.shape}, Total Elements: {all_weights_tensor.numel()})")

    # Update the latent_dim in the diffusion config if it's different
    # This is an important step for the diffusion model.
    # We'll read, update, and write back the YAML.

    config_file_path = os.path.join(project_root, 'configs/mnist_config.yaml')
    with open(config_file_path, 'r') as f:
        full_config_for_update = yaml.safe_load(f)

    current_latent_dim = all_weights_tensor.numel()
    if full_config_for_update['weight_diffusion']['latent_dim'] != current_latent_dim:
        print(f"Updating latent_dim in mnist_config.yaml from {full_config_for_update['weight_diffusion']['latent_dim']} to {current_latent_dim}")
        full_config_for_update['weight_diffusion']['latent_dim'] = current_latent_dim
        with open(config_file_path, 'w') as f:
            yaml.dump(full_config_for_update, f, sort_keys=False)
        print("mnist_config.yaml updated successfully.")
    else:
        print(f"latent_dim in mnist_config.yaml ({full_config_for_update['weight_diffusion']['latent_dim']}) matches extracted weights ({current_latent_dim}). No update needed.")


if __name__ == '__main__':
    main()
