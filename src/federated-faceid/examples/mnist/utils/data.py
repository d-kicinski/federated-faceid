import copy
from typing import *

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10


def split_dataset_iid(dataset: CIFAR10, num_splits: int) -> List[CIFAR10]:
    indices_per_split = np.array_split(np.random.shuffle(np.arange(len(dataset))), num_splits)
    subsets = [Subset(dataset, indices) for indices in indices_per_split]

    return subsets


def split_dataset_non_iid(dataset: CIFAR10) -> List[CIFAR10]:
    label2data: List[CIFAR10] = []

    for label_id in range(len(dataset.classes)):
        indices = np.nonzero(np.array(dataset.targets) == label_id)[0]
        subset = copy.deepcopy(dataset)
        subset.data = subset.data[indices]
        label2data.append(subset)

    return label2data
