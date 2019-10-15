import copy
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.connection import Connection
from multiprocessing.connection import wait
from typing import *

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset


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


class EdgeDevice(Process):
    def __init__(self, name: str, handle: Connection,
                 settings: EdgeDeviceSettings, dataset: Dataset, model: Optional[Module] = None):
        super().__init__()
        self.name: str = name
        self.handle: Connection = handle

        self.setting: EdgeDeviceSettings = settings
        self._loss_func = CrossEntropyLoss()
        self._data_loader = DataLoader(dataset, batch_size=self.setting.batch_size)

        self._model: Optional[Module] = model
        self._loss: float = -1.0

    def run(self) -> None:
        print(f"[{self.name}] Running device")
        while True:
            print(f"[{self.name}] Waiting ...")
            self.download()
            print(f"[{self.name}] Model is downloaded")
            self._loss: float = 1.0  # self.train()
            self.upload()
            print(f"[{self.name}] Model uploaded")

    def download(self):
        handle: Connection = wait([self.handle])[0]
        model: Module = handle.recv()
        self._model = model

    def upload(self):
        if self._model is None:
            raise ValueError("Model not found on this device!")

        self.handle.send(EdgeDeviceResult(name=self.name,
                                          loss=self._loss,
                                          model=copy.deepcopy(self._model)))
        del self._model

    def train(self, verbose: bool = False) -> float:
        if self._data_loader is None:
            raise ValueError("Dataset not found on this device!")

        self._model.train()
        optimizer = torch.optim.SGD(params=self._model.parameters(),
                                    lr=self.setting.learning_rate,
                                    momentum=0.5)

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


@dataclass
class EdgeDeviceConnection:
    process: EdgeDevice
    handle: Connection


@dataclass
class EdgeDeviceResult:
    name: str
    loss: float
    model: Module
