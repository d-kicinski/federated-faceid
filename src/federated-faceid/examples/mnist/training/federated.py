from typing import Dict, List

import numpy as np
from torch.nn import Module
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from federated.federated import EdgeDeviceSettings, EdgeDevice, federated_averaging
from utils import data
from utils.settings import Settings


def train_federated(model: Module, dataset: CIFAR10, settings: Settings) -> Module:
    num_users: int = len(dataset.classes)

    settings_edge_device = EdgeDeviceSettings(epochs=settings.num_local_epochs,
                                              batch_size=settings.num_local_batch,
                                              learning_rate=settings.learning_rate,
                                              device=settings.device)

    subsets: List[Subset]
    if settings.non_iid:
        subsets = data.split_dataset_non_iid(dataset)
    else:
        subsets = data.split_dataset_iid(dataset, num_users)

    users = [EdgeDevice(device_id=i, subset=subsets[i], settings=settings_edge_device)
             for i in range(num_users)]

    max_users_in_round = max(int(settings.user_fraction * num_users), 1)

    for i_epoch in range(settings.num_global_epochs):
        local_models: Dict[int, Module] = {}
        local_losses: Dict[int, float] = {}

        users_in_round_ids = np.random.choice(range(num_users),
                                              max_users_in_round,
                                              replace=False)

        for i_user in users_in_round_ids:
            user = users[i_user]
            user.download(model)
            local_loss: float = user.train()
            print(f"User {i_user} done, loss: {local_loss}")

            local_losses[i_user] = local_loss
            local_models[i_user] = user.upload()

        # update global weights
        model = federated_averaging(list(local_models.values()))

        loss_avg = sum(list(local_losses.values())) / len(local_losses)
        print('Round {:3d}, Average loss {:.3f}'.format(i_epoch, loss_avg))

    return model
