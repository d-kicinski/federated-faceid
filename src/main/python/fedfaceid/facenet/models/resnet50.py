import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class Resnet50Triplet(nn.Module):
    def __init__(self, embedding_dimension=128, pretrained=False):
        super(Resnet50Triplet, self).__init__()
        self.model = models.resnet34(pretrained=pretrained, num_classes=embedding_dimension)

    def forward(self, images):
        x = self.model(images)
        return F.normalize(x, p=2, dim=1)
