import dataclasses

import torch
from torch import optim, Tensor
from torch.nn import Module, functional
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.commons import EarlyStopping
from training.evaluation import evaluate, EvaluationResult
from utils import constants
from utils.settings import Settings


def train_server(model: Module, dataset_train: Dataset, dataset_validate: Dataset,
                 settings: Settings) -> Module:
    dataset_iter_train = DataLoader(dataset_train, batch_size=settings.num_global_batch,
                                    shuffle=True)
    dataset_iter_validate = DataLoader(dataset_validate, batch_size=settings.num_global_batch)

    optimizer = optim.Adam(model.parameters())
    early_stopping = EarlyStopping(settings.stopping_rounds)
    writer = SummaryWriter(str(constants.PATH_OUTPUT_MODEL_SERVER.joinpath("tensorboard")))

    list_loss = []
    global_step = 0
    for i_epoch in range(settings.num_global_epochs):
        model.train()
        model.cuda()

        batch_loss = []
        for i_batch, (inputs, target) in enumerate(dataset_iter_train):
            optimizer.zero_grad()
            inputs = inputs.to(settings.device)
            target = target.to(settings.device)

            output: Tensor = model(inputs)
            loss: Tensor = functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
            if i_batch % 100 == 0:
                writer.add_scalar("train_loss", sum(batch_loss) / len(batch_loss),
                                  global_step=global_step)
            global_step += 1

        results: EvaluationResult = evaluate(model.cpu(), dataset_iter_validate)
        for key, value in dataclasses.asdict(results).items():
            writer.add_scalar(key, value, global_step=global_step)

        loss_avg = sum(batch_loss) / len(batch_loss)
        list_loss.append(loss_avg)

        print(f"epoch={i_epoch}\t"
              f"train_loss={loss_avg:.3f}\t"
              f"{results}")

        if early_stopping.is_best(results.loss):
            print("Saving best model!")
            torch.save(model.state_dict(),
                       constants.PATH_OUTPUT_MODEL_SERVER.joinpath("model.pt"))

        if early_stopping.update(results.loss).should_break:
            print("Early stopping! Loading best model.")
            model.load_state_dict(
                torch.load(constants.PATH_OUTPUT_MODEL_SERVER.joinpath("model.pt")))
            break

    return model
