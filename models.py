"""Model architectures UNCHANGED from the original project.

Only the constructor arguments are parameterized so the same networks can be
reused across datasets:
  * SimpleCNN  : identical layer stack, classifier width = num_classes.
                 Requires 32x32 RGB input (all RGB datasets are resized to 32).
  * SimpleMLP  : identical layer stack, used for MNIST (28x28 -> 784).
"""

import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(dataset, num_classes=10, image_size=32, in_channels=3):
    """Backward compatible: get_model("CIFAR10") still returns SimpleCNN()."""
    if str(dataset).upper() == "MNIST":
        return SimpleMLP(input_dim=in_channels * image_size * image_size,
                         num_classes=num_classes)
    return SimpleCNN(num_classes=num_classes, in_channels=in_channels)