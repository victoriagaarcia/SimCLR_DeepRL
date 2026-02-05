# Apply linear probing to evaluate the quality of learned representations.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import datasets, transforms
from datetime import datetime

from models.linear_classifier import LinearClassifier
from models.resnet_simclr import ResNetSimCLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_encoder(checkpoint_path: str, arch: str, out_dim: int = 128) -> nn.Module:
    """Load the pre-trained encoder from a checkpoint, without the projection head."""
    # Load the full SimCLR model (encoder + projection head)
    full_model = ResNetSimCLR(base_model=arch, out_dim=out_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    full_model.load_state_dict(checkpoint['state_dict'])

    # Extract the encoder (backbone) part
    # ResNetSimCLR has an attribute 'backbone', which is the ResNet
    # The projection head is stored in self.backbone.fc
    encoder = full_model.backbone

    # Extract the input dimension of the projection head
    # Consider if the projection head is a Sequential model
    if isinstance(encoder.fc, nn.Sequential):
        in_dim = encoder.fc[-1].in_features
    else:
        in_dim = encoder.fc.in_features # In case it's a single Linear layer

    # Remove the projection head by replacing it with identity
    encoder.fc = nn.Identity()

    # Move encoder to device and set to eval mode
    encoder = encoder.to(device)
    encoder.eval()

    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder, in_dim

def get_loaders(batch_size: int, data_dir: str = './datasets', num_workers : int = 4):
    """Create DataLoaders for CIFAR-10 dataset."""
    # Standard CIFAR-10 normalization for evaluation
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def train_linear_probe(encoder: nn.Module, classifier: nn.Module,
                       train_loader: DataLoader, in_dim: int, batch_size: int,
                       num_epochs: int, learning_rate: float, weight_decay: float, save_dir: str):
    """Train a linear classifier on top of frozen encoder features."""

    # Create writer for TensorBoard logging
    log_dir = os.path.join(save_dir, f"logs_lp_{num_epochs}_{batch_size}_{learning_rate}_{weight_decay}")
    writer = SummaryWriter(log_dir=log_dir)
    # writer = SummaryWriter(log_dir=save_dir)
        
    # Freeze encoder params (no gradients)
    for p in encoder.parameters():
        p.requires_grad = False

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Starting linear probe training...")
    print("Latent dimension:", in_dim)

    n_iter = 0
    # Training loop with tqdm progress bar
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        encoder.train() # Parameters are frozen, but set to train mode for BN compatibility
        classifier.train()
        
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Extract features using the frozen encoder
            with torch.no_grad():
                features = encoder(images)
            # print("Features shape:", features.shape)

            # Forward pass through the linear classifier
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            writer.add_scalar('Train/Loss', loss.item(), n_iter)
            n_iter += 1

            # Use test_loader as validation?!

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/Acc', train_accuracy, n_iter)
        writer.add_scalar('Train/Loss_avg', avg_loss, n_iter) 
    
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Save the trained linear classifier
    os.makedirs(save_dir, exist_ok=True)
    classifier_path = os.path.join(save_dir, f'linear_classifier_{num_epochs}_{batch_size}_{learning_rate}_{weight_decay}.pth')
    torch.save(classifier.state_dict(), classifier_path)
    
def evaluate_linear_probe(encoder: nn.Module, classifier: nn.Module, test_loader: DataLoader):
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            # Extract features using the frozen encoder
            features = encoder(images)

            # Forward pass through the linear classifier
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100. * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")


def main():
    # Configuration
    # Path to pre-trained SimCLR checkpoint
    checkpoint_path = 'runs/Feb02_18_31_27/checkpoint_final_0300.pth.tar'  
    # Backbone architecture
    arch = 'resnet50'  

    # Output dimension of the projection head during pre-training
    out_dim = 128
    # Training hyperparameters
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    data_dir = './datasets'
    save_dir = './linear_probing/' + datetime.now().strftime('%b%d_%H_%M_%S')

    # Load pre-trained encoder
    encoder, in_dim = get_encoder(checkpoint_path, arch, out_dim)

    # Create DataLoaders
    train_loader, test_loader = get_loaders(batch_size, data_dir)

    # Initialize linear classifier
    classifier = LinearClassifier(in_dim=in_dim, num_classes=10).to(device)

    # Train linear probe
    train_linear_probe(encoder, classifier, train_loader, in_dim, batch_size,
                       num_epochs, learning_rate, weight_decay, save_dir)

    # Evaluate linear probe
    evaluate_linear_probe(encoder, classifier, test_loader)

if __name__ == '__main__':
    main()