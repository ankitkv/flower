"""A simple ConvNet model."""

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Implement a CNN model for CIFAR-{10,100}.

    Parameters
    ----------
    hidden_dims : List[int]
        The hidden dimensions for classifier.
    num_classes : int
        The number of classes in the dataset.
    """

    def __init__(self, hidden_dims, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)

        self.fc1 = nn.Linear(64 * 5 * 5, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        """Implement forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
