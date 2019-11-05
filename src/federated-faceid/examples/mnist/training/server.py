import dataclasses

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from training.commons import EarlyStopping
from training.evaluation import evaluate, EvaluationResult
from utils.settings import Settings


def train_server(model: Module, dataset_train: Dataset, dataset_validate: Dataset,
                 settings: Settings) -> Module:
    dataset_iter_train = DataLoader(dataset_train, batch_size=settings.num_global_batch,
                                    shuffle=True)
    dataset_iter_validate = DataLoader(dataset_validate, batch_size=settings.num_global_batch)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=settings.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                gamma=settings.learning_rate_decay)

    early_stopping = EarlyStopping(settings.stopping_rounds)
    if settings.skip_stopping:
        early_stopping.disable()

    writer = SummaryWriter(str(settings.save_path.joinpath("tensorboard")))

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
            loss: Tensor = criterion(output, target)
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
            torch.save(model.state_dict(),
                       settings.save_path.joinpath("model.pt"))

        if early_stopping.update(results.loss).should_break:
            print("Early stopping! Loading best model.")
            model.load_state_dict(
                torch.load(settings.save_path.joinpath("model.pt")))
            break

        scheduler.step(i_epoch)
    return model
