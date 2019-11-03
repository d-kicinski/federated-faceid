import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import training
from models.baseline import CNNCifar10
from training.evaluation import evaluate, EvaluationResult
from utils import constants
from utils.settings import Settings, args_parser


def create_paths():
    shutil.rmtree(str(constants.PATH_OUTPUT_MODEL_SERVER))
    shutil.rmtree(str(constants.PATH_OUTPUT_MODEL_FEDERATED))
    constants.PATH_OUTPUT_MODEL_SERVER.mkdir(exist_ok=True, parents=True)
    constants.PATH_OUTPUT_MODEL_FEDERATED.mkdir(exist_ok=True, parents=True)


def train():
    # parse args
    create_paths()
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

    dataset_test = CIFAR10(constants.PATH_DATASET_CIFAR10,
                           train=False, transform=transform, download=True)

    if settings.distributed:
        training.train_federated(model, dataset_train, settings)
    else:
        training.train_server(model, dataset_train, dataset_test, settings)

    test_loader = DataLoader(dataset_test, batch_size=settings.num_global_batch, shuffle=False)
    result: EvaluationResult = evaluate(model.cpu(), test_loader, verbose=True)


if __name__ == '__main__':
    train()
