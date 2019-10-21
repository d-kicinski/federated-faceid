from typing import *

import numpy as np
import torch
from torch import optim, Tensor
from torch.nn import functional, Module
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.baseline import CNNCifar10
from models.federated import EdgeDevice, EdgeDeviceSettings, federated_averaging
from utils import constants, data
from utils.settings import Settings, args_parser


def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        log_probs = net_g(data)
        test_loss += functional.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss


def train():
    # parse args
    settings: Settings = args_parser()
    settings.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(settings.seed)

    # load dataset and split users
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = CIFAR10(constants.PATH_DATASET_CIFAR10,
                            train=True, transform=transform, download=True)

    model: torch.nn.Module = CNNCifar10()
    model.to(settings.device)
    train_federated(model, dataset_train, settings)

    dataset_test = CIFAR10(constants.PATH_DATASET_CIFAR10,
                           train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset_test, batch_size=settings.num_global_batch, shuffle=False)
    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(model.cpu(), test_loader)


def train_federated(model: Module, dataset: CIFAR10, settings: Settings) -> Module:
    num_users: int = len(dataset.classes)

    settings_edge_device = EdgeDeviceSettings(epochs=settings.num_local_epochs,
                                              batch_size=settings.num_local_batch,
                                              learning_rate=settings.learning_rate,
                                              device=settings.device)

    subsets: List[Subset]
    if settings.iid:
        subsets = data.split_dataset_iid(dataset, num_users)
    else:
        subsets = data.split_dataset_non_iid(dataset)

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


def train_server(model: Module, dataset: Dataset, settings: Settings) -> Module:
    dataset_iter = DataLoader(dataset, batch_size=settings.num_global_batch,
                              shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=settings.learning_rate)

    list_loss = []
    model.train()
    for i_epoch in range(settings.num_global_epochs):
        batch_loss = []
        for i_batch, (inputs, target) in enumerate(dataset_iter):
            optimizer.zero_grad()
            inputs = inputs.to(settings.device)
            target = target.to(settings.device)

            output: Tensor = model(inputs)
            loss: Tensor = functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if i_batch % 50 == 0:
                print(f"Train Epoch: {i_epoch}"
                      f"[{i_batch * len(inputs)}/{len(dataset_iter.dataset)} "
                      f"({100.0 * i_batch / len(dataset_iter):.0f}%)]"
                      f"\tLoss: {loss.item():.6f}")
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)
    return model


if __name__ == '__main__':
    train()
