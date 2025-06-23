import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
from models.mnist_classifier import get_mnist_model
from tqdm import tqdm
import os # Added for path creation

def main():
    # Load configuration
    with open('configs/mnist_config.yaml', 'r') as f:
        config_full = yaml.safe_load(f)
    config = config_full['mnist_classifier']

    # Ensure output directory exists
    output_dir = os.path.dirname(config['output_path'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")


    model = get_mnist_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Data loading and transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
    ])

    # Adjusted data path to be relative to the script execution or a common root
    # Assuming data will be downloaded to a 'data' folder in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # mnist_weight_diffusion
    data_path = os.path.join(project_root, 'data')


    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    print(f"Starting MNIST classifier training for {config['epochs']} epochs on {device}...")
    for epoch in range(config['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        total_loss = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['epochs']} - Average Loss: {avg_loss:.4f}")


    # Construct the full path for saving the model, relative to project root
    model_save_path = os.path.join(project_root, config['output_path'])
    torch.save(model.state_dict(), model_save_path)
    print(f"MNIST Classifier saved to: {model_save_path}")

if __name__ == '__main__':
    main()
