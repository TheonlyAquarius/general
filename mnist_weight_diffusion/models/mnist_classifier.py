import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1, cnn_channels=[16, 32], kernel_size=3, padding=1, pool_kernel=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, cnn_channels[0], kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(pool_kernel)
        self.conv2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(pool_kernel)
        # Adjusted calculation for flattened features assuming 28x28 input
        # After two pooling layers with kernel_size 2, the 28x28 image becomes 7x7
        self.fc = nn.Linear(cnn_channels[1] * (28 // pool_kernel // pool_kernel) * (28 // pool_kernel // pool_kernel), num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class SimpleMLP(nn.Module):
    def __init__(self, num_classes=10, input_dim=784, hidden_dim=128):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(-1, 784) # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_mnist_model(config):
    model_type = config['model_type']
    num_classes = config['num_classes']
    if model_type == 'cnn':
        return SimpleCNN(num_classes=num_classes,
                         input_channels=config['input_channels'],
                         cnn_channels=config['cnn_channels'],
                         kernel_size=config['kernel_size'],
                         padding=config['padding'],
                         pool_kernel=config['pool_kernel'])
    elif model_type == 'mlp':
        return SimpleMLP(num_classes=num_classes,
                         input_dim=28*28, # MNIST images are 28x28
                         hidden_dim=config['mlp_hidden_dim'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == '__main__':
    # Example usage
    cnn_config = {
        'model_type': 'cnn',
        'num_classes': 10,
        'input_channels': 1,
        'cnn_channels': [16, 32],
        'kernel_size': 3,
        'padding': 1,
        'pool_kernel': 2
    }
    cnn_model = get_mnist_model(cnn_config)
    print("CNN Model:", cnn_model)

    mlp_config = {
        'model_type': 'mlp',
        'num_classes': 10,
        'mlp_hidden_dim': 128
    }
    mlp_model = get_mnist_model(mlp_config)
    print("MLP Model:", mlp_model)
