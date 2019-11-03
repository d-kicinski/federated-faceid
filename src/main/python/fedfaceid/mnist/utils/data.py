from typing import *

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10


def split_dataset_iid(dataset: CIFAR10, num_splits: int) -> List[Subset]:
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    indices_per_split = np.array_split(indices, num_splits)
    subsets = [Subset(dataset, indices) for indices in indices_per_split]

    return subsets


def split_dataset_non_iid(dataset: CIFAR10, num_splits: int) -> List[Subset]:
    label2data: List[Subset] = []

    for label_id in range(len(dataset.classes)):
        indices = np.nonzero(np.array(dataset.targets) == label_id)[0]
        label2data.append(Subset(dataset, indices))

    return label2data
