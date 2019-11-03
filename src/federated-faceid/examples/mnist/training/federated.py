import dataclasses
from typing import Dict, List

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

from federated.federated import EdgeDeviceSettings, EdgeDevice, federated_averaging
from training.commons import EarlyStopping
from training.evaluation import EvaluationResult, evaluate
from utils import data
from utils.settings import Settings


def train_federated(model: Module, dataset_train: CIFAR10, dataset_validate: CIFAR10,
                    settings: Settings) -> Module:
    dataset_iter_validate = DataLoader(dataset_validate, batch_size=settings.num_global_batch)
    num_users: int = len(dataset_train.classes)

    settings_edge_device = EdgeDeviceSettings(epochs=settings.num_local_epochs,
                                              batch_size=settings.num_local_batch,
                                              learning_rate=settings.learning_rate,
                                              device=settings.device)

    subsets: List[Subset]
    if settings.non_iid:
        subsets = data.split_dataset_non_iid(dataset_train)
    else:
        subsets = data.split_dataset_iid(dataset_train, num_users)

    users = [EdgeDevice(device_id=i, subset=subsets[i], settings=settings_edge_device)
             for i in range(num_users)]

    max_users_in_round = max(int(settings.user_fraction * num_users), 1)

    early_stopping = EarlyStopping(settings.stopping_rounds)
    writer = SummaryWriter(str(settings.save_path.joinpath("tensorboard")))

    global_step = 0
    for i_epoch in range(settings.num_global_epochs):
        model.cuda()
        model.train()

        local_models: Dict[int, Module] = {}
        local_losses: Dict[int, float] = {}

        users_in_round_ids = np.random.choice(range(num_users),
                                              max_users_in_round,
                                              replace=False)

        for i_user in users_in_round_ids:
            user = users[i_user]
            user.download(model)
            local_loss: float = user.train()
            # print(f"User {i_user} done, loss: {local_loss}")

            local_losses[i_user] = local_loss
            local_models[i_user] = user.upload()

        # update global weights
        model = federated_averaging(list(local_models.values()))

        results: EvaluationResult = evaluate(model.cpu(), dataset_iter_validate)
        for key, value in dataclasses.asdict(results).items():
            writer.add_scalar(key, value, global_step=global_step)

        loss_avg = sum(list(local_losses.values())) / len(local_losses)
        writer.add_scalar("train_loss", loss_avg, global_step=global_step)

        print(f"epoch={i_epoch}\t"
              f"train_loss={loss_avg:.3f}\t"
              f"{results}")

        if early_stopping.is_best(results.loss):
            torch.save(model.state_dict(),
                       settings.save_path.joinpath("model.pt"))

        if early_stopping.update(results.loss).should_break:
            print("Early stopping! Loading best model.")
            model.load_state_dict(torch.load(settings.save_path.joinpath("model.pt")))
            break

        global_step += 1

    return model
