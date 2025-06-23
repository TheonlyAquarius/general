# MNIST Weight Diffusion Proof of Concept

This project is a proof of concept for generating neural network weights for an MNIST classifier using a diffusion model. The idea is to train a standard MNIST classifier, extract its weights, train a diffusion model on these weights (or a dataset of such weights), and then use the trained diffusion model to generate new sets of weights.

## Project Structure

```
mnist_weight_diffusion/
├── configs/
│   └── mnist_config.yaml       # Configuration for models, training, and paths
├── models/
│   ├── mnist_classifier.py     # Defines CNN and MLP models for MNIST classification
│   └── weight_diffusion_model.py # Defines the SimpleWeightDiffusion model
├── scripts/
│   ├── train_mnist.py          # Trains an MNIST classifier
│   ├── extract_weights.py      # Extracts weights from a trained MNIST classifier
│   ├── train_diffusion.py      # Trains the weight diffusion model on extracted weights
│   └── generate_and_evaluate.py # Generates new weights and provides basic reshaping
├── outputs/                    # Default directory for saved models, weights, and results
│   ├── mnist_models/           # Trained MNIST classifier models
│   ├── diffusion_models/       # Trained weight diffusion models
│   ├── extracted_weights.pth   # Default name for the single tensor of extracted weights
│   ├── generated_weights/      # Directory for generated weight samples
│   └── evaluation_results/     # (Placeholder for more detailed evaluation results)
├── data/                       # MNIST dataset will be downloaded here by train_mnist.py
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

1.  **Clone the repository (if applicable) or ensure all files are in place.**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Proof of Concept

The scripts should be run from the `mnist_weight_diffusion` root directory.

1.  **Train the MNIST Classifier:**
    This script will train an MNIST classifier (CNN by default, configurable in `configs/mnist_config.yaml`) and save its state dictionary. The MNIST dataset will be downloaded to `./data/` if not present.
    ```bash
    python scripts/train_mnist.py
    ```
    *Output:* Saves the trained model to `outputs/mnist_models/mnist_classifier.pth`.

2.  **Extract Weights from the Trained Model:**
    This script loads the trained MNIST classifier, extracts its 'weight' parameters, flattens and concatenates them into a single tensor. It then updates the `latent_dim` in `configs/mnist_config.yaml` to match the total number of extracted weight elements.
    ```bash
    python scripts/extract_weights.py
    ```
    *Output:* Saves the flat weight tensor to `outputs/extracted_weights.pth` and updates `configs/mnist_config.yaml`.

3.  **Train the Weight Diffusion Model:**
    This script trains the `SimpleWeightDiffusion` model. In this PoC, it trains on the single extracted weight vector by repeating it to form a dataset (this is a simplification for demonstration).
    ```bash
    python scripts/train_diffusion.py
    ```
    *Output:* Saves the trained diffusion model to `outputs/diffusion_models/weight_diffusion.pth`.

4.  **Generate New Weights (and Basic Evaluation):**
    This script uses the trained diffusion model to generate new flat weight vectors. It then attempts to reshape these flat vectors back into the structure of the original MNIST model's weights and saves them.
    ```bash
    python scripts/generate_and_evaluate.py
    ```
    *Output:* Saves generated weight state dictionaries (e.g., `generated_weights_sample_1.pth`) to `outputs/generated_weights/`.

## Configuration

All major parameters, paths, and model types can be configured in `configs/mnist_config.yaml`. This includes:
- MNIST classifier architecture (CNN/MLP) and hyperparameters.
- Weight diffusion model parameters (latent dimension, timesteps, noise schedule).
- Number of samples to generate.
- Output paths for models and generated data.

The `latent_dim` for the `weight_diffusion` model is automatically updated by `scripts/extract_weights.py` to match the size of the weights extracted from the MNIST model.

## Further Development & Evaluation

This is a basic proof of concept. More advanced steps could include:
-   **Training multiple MNIST classifiers** with different initializations or architectures to create a diverse dataset of weight vectors for training the diffusion model. `extract_weights.py` and `train_diffusion.py` would need to be adapted to handle multiple weight files.
-   **More sophisticated diffusion models:** Using more complex denoising networks (e.g., U-Nets if weights had spatial structure, though less applicable for flat vectors).
-   **Rigorous evaluation:**
    -   Loading the generated weights into a new MNIST classifier instance.
    -   Evaluating its performance on the MNIST test set (accuracy, loss).
    -   Analyzing the properties of generated weights (e.g., distribution, similarity to original weights).
-   **Conditional generation:** Training the diffusion model to generate weights conditioned on some property (e.g., desired accuracy or layer type).

## Notes
- The project assumes a PyTorch environment.
- If CUDA is available, it will be used for training and generation; otherwise, it defaults to CPU.
- The current diffusion training uses a single weight vector repeated. For learning a meaningful distribution of weights, a dataset of diverse weight vectors would be necessary.
```
