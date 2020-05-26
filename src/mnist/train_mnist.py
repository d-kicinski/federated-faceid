import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

import training
from models.baseline import CNNCifar10
from torchsummary import summary
from training.evaluation import EvaluationResult, evaluate
from utils import constants
from utils.settings import Settings, args_parser, create_save_path


def train():
    # parse args
    settings: Settings = args_parser()
    settings.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    settings.save_path = create_save_path(settings)
    torch.manual_seed(settings.seed)

    # load dataset and split users
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset_train = CIFAR10(
        constants.PATH_DATASET_CIFAR10,
        train=True,
        transform=transform_train,
        download=True,
    )

    model: torch.nn.Module = CNNCifar10()
    summary(model, (3, 24, 24))

    model.to(settings.device)

    dataset_test = CIFAR10(
        constants.PATH_DATASET_CIFAR10,
        train=False,
        transform=transform_test,
        download=False,
    )

    if settings.distributed:
        model = training.train(model, dataset_train, dataset_test, settings)
    else:
        model = training.train_server(model, dataset_train, dataset_test, settings)

    test_loader = DataLoader(
        dataset_test, batch_size=settings.num_global_batch, shuffle=False
    )
    result: EvaluationResult = evaluate(model.cpu(), test_loader, verbose=True)
    with settings.save_path.joinpath("evaluation.txt").open("w") as f:
        f.write(str(result))


if __name__ == "__main__":
    train()
