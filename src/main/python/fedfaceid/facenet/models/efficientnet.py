from efficientnet_pytorch import EfficientNet
from torch.nn import Module
from torch.nn import functional as F


class EfficientNetNormalized(Module):
    def __init__(self, image_size):
        super().__init__()
        self.model = EfficientNet.from_name('efficientnet-b0')

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, p=2, dim=1)
