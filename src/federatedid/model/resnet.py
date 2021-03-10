from typing import Type, List

import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils_resnet import resnet18
from .utils_resnet_fixup import fixup_resnet18


class Resnet18(nn.Module):
    def __init__(self, output_dim=10, pretrained=False, layer_norm: Type[nn.Module] = nn.BatchNorm2d):
        super(Resnet18, self).__init__()

        self.model = resnet18(pretrained=pretrained, norm_layer=layer_norm, num_classes=output_dim)

    def forward(self, images):
        return self.model(images)


class Resnet18Fixup(nn.Module):
    def __init__(self, output_dim: int, **kwargs):
        super(Resnet18Fixup, self).__init__()
        self.model = fixup_resnet18(num_classes=output_dim)

    def forward(self, images: torch.Tensor):
        return self.model(images)


class Resnet18Embedding(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=512, pretrained=False, use_batch_norm=True):
        super(Resnet18Embedding, self).__init__()
        layer_norm: Type[nn.Module]

        if use_batch_norm:
            layer_norm = nn.BatchNorm2d
        else:
            layer_norm = nn.Identity

        self.model = resnet18(pretrained=pretrained, norm_layer=layer_norm)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        last_layer: List[nn.Module] = [
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False)]
        if use_batch_norm:
            last_layer.append(
                nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
            )
        self.model.fc = nn.Sequential(*last_layer)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
