import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import datasets, transforms, models
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_encoder(checkpoint_path: str, arch: str, num_classes: int = 10) -> nn.Module:
    """
    Carga el encoder siguiendo el enfoque de limpieza de prefijos y 
    sustitución de la capa fc por una lineal de clasificación.
    """
    # 1. Cargar modelo base de torchvision (sin pesos preentrenados de ImageNet)
    if arch == 'resnet18':
        model = models.resnet18(weights=None)
    elif arch == 'resnet50':
        model = models.resnet50(weights=None)
    else:
        raise ValueError("Arquitectura no soportada en este script")

    # 2. Cargar checkpoint y limpiar el state_dict (como en la imagen)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # Eliminar prefijo 'backbone.' y descartar la antigua 'fc' (projection head)
        if k.startswith('backbone.') and not k.startswith('backbone.fc'):
            new_key = k[len('backbone.'):]
            new_state_dict[new_key] = v

    # 3. Cargar pesos en el modelo base (strict=False para ignorar la fc que no cargamos)
    model.load_state_dict(new_state_dict, strict=False)

    # 4. Sustituir la projection head por la capa lineal final
    dim_mlp = model.fc.in_features
    model.fc = nn.Linear(dim_mlp, num_classes)

    # 5. Congelar el backbone y asegurar que la nueva fc tenga gradientes
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device)

def get_loaders(batch_size: int, data_dir: str = './datasets', num_workers: int = 4):
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

def train_linear_probe(model: nn.Module, train_loader: DataLoader, 
                       num_epochs: int, learning_rate: float, weight_decay: float, save_dir: str):
    
    log_dir = os.path.join(save_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    
    criterion = nn.CrossEntropyLoss()
    # Importante: solo optimizamos los parámetros de model.fc
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Iniciando entrenamiento. Capa lineal: {model.fc}")

    n_iter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass (el modelo ya incluye backbone + fc)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            writer.add_scalar('Train/Loss_Step', loss.item(), n_iter)
            n_iter += 1

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        writer.add_scalar('Train/Acc_Epoch', train_acc, epoch)
        writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Acc: {train_acc:.2f}%")

    writer.close()

def evaluate_linear_probe(model: nn.Module, test_loader: DataLoader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Final Test Accuracy: {100. * correct / total:.2f}%")

def main():
    # --- Configuración ---
    checkpoint_path = 'runs/Feb02_18_31_27/checkpoint_final_0300.pth.tar'  
    arch = 'resnet50'  
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-5
    
    save_dir = './linear_probing/' + datetime.now().strftime('%b%d_%H_%M_%S') + "_v2"

    # 1. Obtener el modelo completo (Backbone preentrenado + FC nueva)
    model = get_encoder(checkpoint_path, arch, num_classes=10)

    # 2. Cargar datos
    train_loader, test_loader = get_loaders(batch_size)

    # 3. Entrenar (el modelo ya es self-contained)
    train_linear_probe(model, train_loader, num_epochs, learning_rate, weight_decay, save_dir)

    # 4. Evaluar
    evaluate_linear_probe(model, test_loader)

if __name__ == '__main__':
    main()