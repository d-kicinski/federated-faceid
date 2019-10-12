import torch
from torch import optim, Tensor
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.baseline import CNNCifar10
from utils import constants
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


if __name__ == '__main__':
    # parse args
    settings: Settings = args_parser()
    settings.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(settings.seed)

    # load dataset and split users
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10(constants.PATH_DATASET_CIFAR10,
                                     transform=transform,
                                     download=True)
    img_size = dataset_train[0][0].shape

    model: torch.nn.Module = CNNCifar10()
    model.cuda()

    # training
    optimizer = optim.SGD(model.parameters(), lr=settings.learning_rate)
    train_loader = DataLoader(dataset_train, batch_size=settings.num_global_batch, shuffle=True)

    list_loss = []
    model.train()
    for i_epoch in range(settings.num_global_epochs):
        batch_loss = []
        for i_batch, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            output: Tensor = model(data)
            loss: Tensor = functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if i_batch % 50 == 0:
                print(f"Train Epoch: {i_epoch}"
                      f"[{i_batch * len(data)}/{len(train_loader.dataset)} "
                      f"({100.0 * i_batch / len(train_loader):.0f}%)]"
                      f"\tLoss: {loss.item():.6f}")
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_test = datasets.CIFAR10(constants.PATH_DATASET_CIFAR10, train=False,
                                    transform=transform)
    test_loader = DataLoader(dataset_test, batch_size=settings.num_global_batch, shuffle=False)

    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(model, test_loader)
