import copy
from dataclasses import dataclass
from typing import *

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler


def federated_averaging(models: List[Module]) -> Module:
    global_model = copy.deepcopy(models[0])
    global_weights = global_model.state_dict()

    local_weights = [m.state_dict() for m in models]

    for k in global_weights.keys():
        for i in range(1, len(local_weights)):
            global_weights[k] += local_weights[i][k]
        global_weights[k] = torch.div(global_weights[k], len(local_weights))

    global_model.load_state_dict(global_weights)
    return global_model


@dataclass
class EdgeDeviceSettings:
    batch_size: int
    epochs: int
    learning_rate: float
    device: str


class EdgeDevice:
    def __init__(self, device_id: int, settings: EdgeDeviceSettings, subset: Subset):
        self.device_id = device_id
        self.setting = settings
        self._loss_func = CrossEntropyLoss()
        self._data_loader = DataLoader(subset.dataset,
                                       sampler=SubsetRandomSampler(subset.indices),
                                       batch_size=self.setting.batch_size)

        self._model: Optional[Module] = None

    def download(self, model: Module):
        self._model = copy.deepcopy(model)

    def upload(self) -> Module:
        if self._model is not None:
            return self._model
        else:
            raise ValueError("Model not found on this device!")

    def train(self, verbose: bool = False) -> float:
        if self._data_loader is None:
            raise ValueError("Dataset not found on this device!")

        self._model.train()
        optimizer = torch.optim.SGD(params=self._model.parameters(),
                                    lr=self.setting.learning_rate,
                                    momentum=0.9)

        epoch_loss = []
        for i_epoch in range(self.setting.epochs):
            batch_loss = []
            for i_batch, (images, labels) in enumerate(self._data_loader):
                images, labels = images.to(self.setting.device), labels.to(self.setting.device)
                self._model.zero_grad()
                log_probs = self._model(images)
                loss = self._loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if verbose and i_batch % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                          .format(i_batch, i_batch * len(images), len(self._data_loader),
                                  100.0 * i_batch / len(self._data_loader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)
