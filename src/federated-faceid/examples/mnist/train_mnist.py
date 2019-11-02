from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.nn import functional, Module
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import training
from models.baseline import CNNCifar10
from utils import constants
from utils.settings import Settings, args_parser


@dataclass
class EvaluationResult:
    f1_score: float
    accuracy: float
    loss: float


def evaluate(model: Module, data_loader: DataLoader, verbose: bool = False) -> EvaluationResult:
    model.eval()
    loss: float = 0.0

    labels_golden: List[int] = []
    labels_system: List[int] = []

    for idx, (data, target) in enumerate(data_loader):
        logits: torch.Tensor = model(data)
        loss += functional.cross_entropy(logits, target).item()

        predictions = np.argmax(logits.data, axis=-1)

        labels_golden.extend(target.data.numpy().tolist())
        labels_system.extend(predictions.tolist())

    f1 = f1_score(labels_golden, labels_system, average='macro')
    accuracy = accuracy_score(labels_golden, labels_system)

    if verbose:
        print(classification_report(labels_golden, labels_system))

    return EvaluationResult(f1, accuracy, loss)


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

    if settings.distributed:
        training.train_federated(model, dataset_train, settings)
    else:
        training.train_server(model, dataset_train, settings)

    dataset_test = CIFAR10(constants.PATH_DATASET_CIFAR10,
                           train=False, transform=transform, download=True)

    test_loader = DataLoader(dataset_test, batch_size=settings.num_global_batch, shuffle=False)
    result: EvaluationResult = evaluate(model.cpu(), test_loader, verbose=True)


if __name__ == '__main__':
    train()
