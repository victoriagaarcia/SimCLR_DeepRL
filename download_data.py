from torchvision.datasets import CIFAR10
from torchvision import transforms

# Se descargará y creará automáticamente la carpeta 'datasets'
CIFAR10(
    root="datasets",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

CIFAR10(
    root="datasets",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
# Verify installation
if __name__ == "__main__":
    import os

    train_data_path = os.path.join("datasets", "cifar-10-batches-py")
    test_data_path = os.path.join("datasets", "cifar-10-batches-py")

    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        print("CIFAR-10 dataset downloaded successfully!")
    else:
        print("Failed to download CIFAR-10 dataset.")
        
