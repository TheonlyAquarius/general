import torch
import yaml
from models.weight_diffusion_model import SimpleWeightDiffusion
from models.mnist_classifier import get_mnist_model # To get original model structure
import os
import numpy as np

def main():
    # Load configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs/mnist_config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    diffusion_config = config['weight_diffusion']
    mnist_config = config['mnist_classifier'] # For model structure
    generation_config = config['generation']
    extraction_config = config['weight_extraction'] # To get original weight shapes if needed

    # Paths
    diffusion_model_relative_path = diffusion_config['output_path']
    diffusion_model_abs_path = os.path.join(project_root, diffusion_model_relative_path)

    output_dir_relative = generation_config['output_dir']
    output_dir_abs = os.path.join(project_root, output_dir_relative)
    os.makedirs(output_dir_abs, exist_ok=True)

    # Parameters
    latent_dim = diffusion_config['latent_dim'] # Should be correct due to extract_weights.py
    timesteps = diffusion_config['timesteps']
    noise_schedule = diffusion_config.get('noise_schedule', 'linear')
    num_samples = generation_config['num_samples']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained diffusion model
    diffusion_model = SimpleWeightDiffusion(
        latent_dim=latent_dim,
        timesteps=timesteps,
        noise_schedule=noise_schedule
    ).to(device)

    try:
        diffusion_model.load_state_dict(torch.load(diffusion_model_abs_path, map_location=device))
        diffusion_model.eval() # Set to evaluation mode
        print(f"Successfully loaded diffusion model from: {diffusion_model_abs_path}")
    except FileNotFoundError:
        print(f"Error: Diffusion model not found at {diffusion_model_abs_path}. "
              "Please run `scripts/train_diffusion.py` first.")
        return
    except Exception as e:
        print(f"Error loading diffusion model: {e}")
        return

    # To reshape generated weights, we need the structure of the original MNIST model's weights.
    # One way: Instantiate an MNIST model and get its state_dict structure.
    # Another way: Load the original extracted_weights.pth if it stored metadata (it currently doesn't).
    # Let's use the MNIST model instantiation.

    temp_mnist_model = get_mnist_model(mnist_config) # No need to load weights, just for structure
    original_state_dict_template = temp_mnist_model.state_dict()

    # Filter to get only 'weight' parameters and their shapes, in order.
    # This order must match how weights were extracted in `extract_weights.py`.
    weight_param_info = [] # List of tuples (name, shape, num_elements)
    current_total_elements = 0
    for name, param in original_state_dict_template.items():
        if 'weight' in name:
            numel = param.numel()
            weight_param_info.append({'name': name, 'shape': param.shape, 'numel': numel})
            current_total_elements += numel

    if current_total_elements != latent_dim:
        print(f"CRITICAL ERROR: The total number of elements from the MNIST model structure ({current_total_elements}) "
              f"does not match the diffusion model's latent dimension ({latent_dim}). "
              "This means generated weights cannot be correctly reshaped.")
        print("Ensure `extract_weights.py` ran correctly and updated `latent_dim` in the config, "
              "and that `mnist_config` matches the model whose weights were extracted.")
        return

    print(f"Generating {num_samples} weight samples using the diffusion model...")
    with torch.no_grad():
        # The sample() method in SimpleWeightDiffusion expects shape (batch_size, latent_dim)
        generated_flat_weights_batch = diffusion_model.sample(shape=(num_samples, latent_dim))

    print(f"Generated {num_samples} flat weight vectors (shape: {generated_flat_weights_batch.shape})")

    # "Evaluation": Reshape generated flat weights back into model state_dicts
    # and save them. A more advanced evaluation would load these into an MNIST
    # model and test its performance on the MNIST test set.

    for i in range(num_samples):
        print(f"\nProcessing generated sample {i+1}/{num_samples}:")
        flat_weights_sample = generated_flat_weights_batch[i].cpu() # Move to CPU for manipulation

        generated_state_dict = {}
        current_idx = 0
        all_weights_reshaped = True

        for info in weight_param_info:
            name, shape, numel = info['name'], info['shape'], info['numel']
            if current_idx + numel > flat_weights_sample.numel():
                print(f"  Error: Not enough elements in generated flat weights to reconstruct '{name}'. "
                      f"Required: {numel}, Available: {flat_weights_sample.numel() - current_idx}")
                all_weights_reshaped = False
                break

            try:
                weight_tensor = flat_weights_sample[current_idx : current_idx + numel].reshape(shape)
                generated_state_dict[name] = weight_tensor
                print(f"  Reshaped '{name}' with shape {shape}")
                current_idx += numel
            except Exception as e:
                print(f"  Error reshaping '{name}' with shape {shape} from slice [{current_idx}:{current_idx+numel}]: {e}")
                all_weights_reshaped = False
                break

        if all_weights_reshaped and current_idx == latent_dim:
            # Add biases from the template model if you want a full state_dict.
            # For this PoC, we only care about weights.
            # If you wanted to load biases, you'd copy them from original_state_dict_template:
            # for name, param in original_state_dict_template.items():
            #     if 'bias' in name:
            #         generated_state_dict[name] = param.clone()

            output_file_path = os.path.join(output_dir_abs, f"generated_weights_sample_{i+1}.pth")
            torch.save(generated_state_dict, output_file_path)
            print(f"  Successfully reshaped all weights. Saved to: {output_file_path}")

            # Further evaluation idea:
            # new_mnist_model = get_mnist_model(mnist_config)
            # # Create a full state dict if biases are needed
            # full_gen_state_dict = new_mnist_model.state_dict()
            # full_gen_state_dict.update(generated_state_dict) # Overwrite weights
            # new_mnist_model.load_state_dict(full_gen_state_dict)
            # new_mnist_model.to(device)
            # new_mnist_model.eval()
            # # ... then run evaluation on MNIST test set ...
            # print("  (Placeholder for evaluating the performance of this generated model)")

        elif not all_weights_reshaped:
            print(f"  Failed to reshape all weights for sample {i+1}.")
        elif current_idx != latent_dim:
            print(f"  Mismatch in total elements processed ({current_idx}) vs latent_dim ({latent_dim}) for sample {i+1}.")


    print(f"\nFinished generating and attempting to reshape weights. Check the '{output_dir_abs}' directory.")

if __name__ == '__main__':
    main()
