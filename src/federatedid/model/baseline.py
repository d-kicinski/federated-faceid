import torch
from torch import nn
from torch.nn import Module
from torch.nn.functional import relu


class SimpleCNN(Module):
    def __init__(self, output_dim: int = 10, **kwargs):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.d1 = nn.Linear(4 * 4 * 64, 64)
        self.d2 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(relu(self.conv1(x)))
        x = self.pool2(relu(self.conv2(x)))
        x = relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = relu(self.d1(x))
        x = self.d2(x)
        return x
