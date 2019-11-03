import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.baseline import CNNCifar10
from training import evaluation
from training.evaluation import EvaluationResult
from utils import constants
from utils.settings import Settings, args_parser, create_save_path


def evaluate():
    # parse args
    settings: Settings = args_parser()
    settings.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    settings.save_path = create_save_path(settings)
    torch.manual_seed(settings.seed)

    # load dataset and split users
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    model: torch.nn.Module = CNNCifar10()
    model.to(settings.device)

    dataset_test = CIFAR10(constants.PATH_DATASET_CIFAR10,
                           train=False, transform=transform, download=True)

    model.load_state_dict(torch.load(settings.save_path.joinpath("model.pt")))

    test_loader = DataLoader(dataset_test, batch_size=settings.num_global_batch, shuffle=False)
    result: EvaluationResult = evaluation.evaluate(model.cpu(), test_loader, verbose=True)

    with settings.save_path.joinpath("evaluation.txt").open("w") as f:
        f.write(str(result))


if __name__ == '__main__':
    evaluate()
