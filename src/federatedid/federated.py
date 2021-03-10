import copy
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader


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


class ModelAccumulator:
    def __init__(self):
        self.model_counter: int = 0
        self.global_model = None
        self.global_weights = None

    def update(self, model):
        local_weights = model.state_dict()

        if self.global_model is None:
            self.global_model = model
            self.global_weights = local_weights
            self.model_counter += 1
        else:
            for k in self.global_weights.keys():
                self.global_weights[k] += local_weights[k]
            self.model_counter += 1

    def get(self):
        for k in self.global_weights.keys():
            self.global_weights[k] = torch.div(
                self.global_weights[k], self.model_counter
            )

        self.global_model.load_state_dict(self.global_weights)
        return self.global_model

    def reset(self):
        self.global_model = None
        self.global_weights = None
        self.model_counter = 0


@dataclass
class EdgeDeviceSettings:
    batch_size: int
    epochs: int
    learning_rate: float
    learning_rate_decay: float
    device: str


@dataclass
class TrainingResult:
    loss: float
    steps: int
    learning_rate: float


class EdgeDevice:
    def __init__(
        self, device_id: int, settings: EdgeDeviceSettings, data_loader: DataLoader
    ):
        self.device_id = device_id
        self._data_loader = data_loader
        self.setting = copy.deepcopy(settings)
        self._loss_func = CrossEntropyLoss()
        self._model: Optional[Module] = None

    def download(self, model: Module):
        self._model = copy.deepcopy(model)

    def upload(self) -> Module:
        if self._model is not None:
            return copy.deepcopy(self._model)
        else:
            raise ValueError("Model not found on this device!")

    def train(self) -> TrainingResult:
        if self._data_loader is None:
            raise ValueError("Dataset not found on this device!")

        self._model.train()
        self.setting.learning_rate = (
            self.setting.learning_rate * self.setting.learning_rate_decay
        )
        optimizer = torch.optim.SGD(
            params=self._model.parameters(), lr=self.setting.learning_rate
        )
        epoch_loss = []
        local_steps: int = 0
        for _ in range(self.setting.epochs):
            batch_loss = []
            for i_batch, (images, labels) in enumerate(self._data_loader):
                self._model.zero_grad()
                images = images.to(self.setting.device)
                labels = labels.to(self.setting.device)

                logits = self._model(images)
                loss = self._loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                local_steps += 1
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        mean_loss = sum(epoch_loss) / len(epoch_loss)
        return TrainingResult(
            loss=mean_loss, steps=local_steps, learning_rate=self.setting.learning_rate
        )
