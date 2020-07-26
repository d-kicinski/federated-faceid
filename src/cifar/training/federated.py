import dataclasses
from functools import reduce
from typing import Dict, List

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

import federated as fd
from cifar.training.evaluation import EvaluationResult, evaluate
from cifar.utils import data
from cifar.utils.settings import Settings
from common import EarlyStopping


def merge_subsets(s1: Subset, s2: Subset) -> Subset:
    s1.indices += s2.indices
    return s1


def average_results(results: List[fd.TrainingResult]) -> fd.TrainingResult:
    results_avg = fd.TrainingResult(0.0, 0, 0.0)
    for r in results:
        results_avg.loss += r.loss
        results_avg.learning_rate += r.learning_rate
        results_avg.steps += r.steps

    results_avg.loss /= len(results)
    results_avg.learning_rate /= len(results)
    results_avg.steps /= len(results)

    return results_avg


def train_federated(
    model: Module, dataset_train: CIFAR10, dataset_validate: CIFAR10, settings: Settings
) -> Module:
    dataset_iter_validate = DataLoader(
        dataset_validate, batch_size=settings.num_global_batch
    )
    # num_users: int = len(dataset_train.classes)
    num_users = settings.num_users
    subsets_per_user = settings.num_subsets_per_user

    lr = settings.learning_rate * (2 - settings.learning_rate_decay)
    settings_edge_device = fd.EdgeDeviceSettings(
        epochs=settings.num_local_epochs,
        batch_size=settings.num_local_batch,
        learning_rate=lr,
        learning_rate_decay=settings.learning_rate_decay,
        device=settings.device,
    )

    subsets: List[Subset]
    if settings.non_iid:
        subsets = data.split_dataset_non_iid(
            dataset_train, num_users * subsets_per_user
        )
    else:
        subsets = data.split_dataset_iid(dataset_train, num_users)

    users = []
    subsets_indices = list(range(len(subsets)))
    for i in range(num_users):
        indices = np.random.choice(
            subsets_indices, size=subsets_per_user, replace=False
        )

        [subsets_indices.remove(i) for i in indices]
        subset_for_user = reduce(merge_subsets, [subsets[i] for i in indices])

        user = fd.EdgeDevice(
            device_id=i, data_loader=subset_for_user, settings=settings_edge_device
        )
        users.append(user)

    max_users_in_round = max(int(settings.user_fraction * num_users), 1)

    early_stopping = EarlyStopping(settings.stopping_rounds)
    if settings.skip_stopping:
        early_stopping.disable()

    writer = SummaryWriter(
        str(settings.save_path.joinpath("tensorboard").joinpath(settings.id))
    )

    global_step = 0
    for i_epoch in range(settings.num_global_epochs):
        model.train().to(settings.device)

        # local_models: Dict[int, Module] = {}
        local_results: Dict[int, fd.TrainingResult] = {}

        free_users = list(range(num_users))
        while free_users:
            users_in_round_ids = np.random.choice(
                free_users, max_users_in_round, replace=False
            )
            [free_users.remove(i) for i in users_in_round_ids]

            models: List[Module] = []
            for i_user in users_in_round_ids:
                user = users[i_user]
                user.download(model)
                local_results[i_user] = user.train()
                models.append(user.upload())

            # update global weights
            model = fd.federated_averaging(models)

            global_step += 1

        model.eval().cpu()
        results_train: fd.TrainingResult = average_results(list(local_results.values()))
        results_eval: EvaluationResult = evaluate(model, dataset_iter_validate)

        for key, value in dataclasses.asdict(results_eval).items():
            writer.add_scalar(key, value, global_step=global_step)
        writer.add_scalar("train_loss", results_train.loss, global_step=global_step)

        print(
            f"epoch={i_epoch}  "
            f"global_step={global_step}  "
            f"lr={results_train.learning_rate:.4f}  "
            f"train_loss={results_train.loss:.3f}  "
            f"eval_loss={results_eval.loss:.3f}  "
            f"eval_f1={results_eval.f1_score:.3f}  "
            f"eval_acc={results_eval.accuracy:.3f}"
        )

        if early_stopping.is_best(results_eval.loss):
            torch.save(model.state_dict(), settings.save_path.joinpath("model.pt"))

        if early_stopping.update(results_eval.loss).should_break:
            print("Early stopping! Loading best model.")
            model.load_state_dict(torch.load(settings.save_path.joinpath("model.pt")))
            break

    return model
