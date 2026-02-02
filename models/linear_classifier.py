import torch.nn as nn

class LinearClassifier(nn.Module):
    """Simple linear classifier for linear evaluation (linear probing)."""
    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
