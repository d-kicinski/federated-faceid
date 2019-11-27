import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Optional, Mapping, Any, List

import torch
from torch.nn import Module
from torch.nn.modules.distance import PairwiseDistance
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from commons import get_train_data_loader, get_validation_data_loader, ModelBuilder, load_checkpoint
from evaluation import EvaluationMetrics, evaluate
from settings import DataSettings, ModelSettings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training FaceNet facial recognition model using Triplet Loss.")

    # Dataset settings
    parser.add_argument("--output_dir", type=lambda p: Path(p),
                        default=DataSettings.output_dir)
    parser.add_argument("--dataset_dir", type=lambda p: Path(p),
                        default=DataSettings.dataset_dir)
    parser.add_argument("--lfw_dir", type=lambda p: Path(p),
                        default=DataSettings.lfw_dir)
    parser.add_argument("--dataset_csv_file", type=lambda p: Path(p),
                        default=DataSettings.dataset_csv_file)
    parser.add_argument("--training_triplets_path", type=lambda p: Path(p),
                        default=DataSettings.training_triplets_path)
    parser.add_argument("--checkpoint_path", type=lambda p: Path(p) if p else None,
                        default=DataSettings.checkpoint_path)

    # Training settings
    parser.add_argument("--lfw_batch_size", default=ModelSettings.lfw_batch_size, type=int)
    parser.add_argument("--lfw_validation_epoch_interval", type=int,
                        default=ModelSettings.lfw_validation_epoch_interval)
    parser.add_argument("--model_architecture", type=str, choices=["resnet18", "resnet34",
                                                                   "resnet50", "resnet101",
                                                                   "inceptionresnetv1",
                                                                   "inceptionresnetv2"],
                        default=ModelSettings.model_architecture)
    parser.add_argument("--epochs", type=int, default=ModelSettings.epochs)
    parser.add_argument("--num_triplets_train", type=int,
                        default=ModelSettings.num_triplets_train)
    parser.add_argument("--batch_size", default=ModelSettings.batch_size, type=int,
                        help="Batch size (default: 64)")
    parser.add_argument("--num_workers", default=ModelSettings.num_workers, type=int,
                        help="Number of workers for data loaders (default: 4)")
    parser.add_argument("--embedding_dim", default=ModelSettings.embedding_dim, type=int,
                        help="Dimension of the embedding vector (default: 128)")
    parser.add_argument("--pretrained_on_imagenet", action="store_true",
                        default=ModelSettings.pretrained_on_imagenet)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adagrad", "rmsprop", "adam"],
                        default=ModelSettings.optimizer)
    parser.add_argument("--learning_rate", type=float, default=ModelSettings.learning_rate)
    parser.add_argument("--triplet_loss_margin", type=float,
                        default=ModelSettings.triplet_loss_margin)
    return parser.parse_args()


class OptimizerBuilder:
    @staticmethod
    def build(model: Module, optimizer: str, learning_rate: float) -> Optimizer:
        # Set optimizers
        if optimizer == "sgd":
            optimizer_model = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer == "adam":
            optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} is unknown!")

        return optimizer_model


@dataclass
class TrainStepResults:
    loss: float
    num_triplets: int


class TrainStep:
    def __init__(self):
        self.embedding_anc = None
        self.embeddings_pos = None
        self.embedding_neg = None

    def __call__(self, *args, **kwargs):
        return self.model_train_step(*args, **kwargs)

    def model_train_step(self,
                         model: Module,
                         optimizer: Optimizer,
                         loss_fn: Module,
                         distance_fn: Module,
                         batch_sample: Any,
                         margin: float) -> Optional[TrainStepResults]:
        image_anc = batch_sample["anc_img"].cuda()
        image_pos = batch_sample["pos_img"].cuda()
        image_neg = batch_sample["neg_img"].cuda()

        # Forward pass - compute embeddings
        embedding_anc = model(image_anc)
        embedding_pos = model(image_pos)
        embedding_neg = model(image_neg)

        # Forward pass - choose hard negatives only for training
        distance_pos = distance_fn(embedding_anc, embedding_pos)
        distance_neg = distance_fn(embedding_anc, embedding_neg)

        distance_difference = (distance_neg - distance_pos) < margin

        hard_triplets = torch.where(distance_difference == 1)
        if len(hard_triplets) == 0:
            return None

        embedding_anc_hard = embedding_anc[hard_triplets]
        embedding_pos_hard = embedding_pos[hard_triplets]
        embedding_neg_hard = embedding_neg[hard_triplets]

        # Calculate triplet loss
        triplet_loss = loss_fn(anchor=embedding_anc_hard,
                               positive=embedding_pos_hard,
                               negative=embedding_neg_hard).cuda()

        # Backward pass
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()

        return TrainStepResults(triplet_loss.item(), len(embedding_anc_hard))


class Tensorboard:
    def __init__(self, log_path: Path):
        self._writer = SummaryWriter(str(log_path))

    def add_dict(self, dictionary: Mapping[str, Any], global_step: int):
        for key, value in dictionary.items():
            self._writer.add_scalar(key, value, global_step=global_step)

    def add_scalar(self, name: str, value: float, global_step: int):
        self._writer.add_scalar(name, value, global_step=global_step)


def train_triplet(settings_data: DataSettings, settings_model: ModelSettings):
    output_dir: Path = settings_data.output_dir
    output_dir_logs = output_dir.joinpath("logs")
    output_dir_plots = output_dir.joinpath("plots")
    output_dir_checkpoints = output_dir.joinpath("checkpoints")
    output_dir_tensorboard = output_dir.joinpath("tensorboard")

    output_dir_logs.mkdir(exist_ok=True, parents=True)
    output_dir_plots.mkdir(exist_ok=True, parents=True)
    output_dir_checkpoints.mkdir(exist_ok=True, parents=True)

    model_architecture = settings_model.model_architecture

    start_epoch: int = 0
    global_step: int = 0

    data_loader_train: DataLoader = get_train_data_loader(settings_model, settings_data)
    data_loader_validate: DataLoader = get_validation_data_loader(settings_model, settings_data)

    model: Module = ModelBuilder.build(settings_model.model_architecture,
                                       settings_model.embedding_dim,
                                       settings_model.pretrained_on_imagenet)

    print("Using {} model architecture.".format(model_architecture))

    # Load model to GPU or multiple GPUs if available
    if torch.cuda.is_available():
        print("Using single-gpu training.")
        model.cuda()

    optimizer: Optimizer = OptimizerBuilder.build(model,
                                                  settings_model.optimizer,
                                                  settings_model.learning_rate)
    if settings_data.checkpoint_path:
        checkpoint = load_checkpoint(output_dir_checkpoints, model, optimizer)
        model = checkpoint.model
        optimizer = checkpoint.optimizer
        start_epoch = checkpoint.epoch
        global_step = checkpoint.global_step

    # Start Training loop
    print(f"Training using triplet loss on {settings_model.num_triplets_train} triplets"
          f" starting for {settings_model.epochs - start_epoch} epoch")

    total_time_start = time.time()
    end_epoch = start_epoch + settings_model.epochs

    l2_distance = PairwiseDistance(2).cuda()
    loss_fn = torch.nn.TripletMarginLoss(margin=settings_model.triplet_loss_margin,
                                         reduction="mean")
    tensorboard = Tensorboard(output_dir_tensorboard)
    train_step = TrainStep()

    for epoch in range(start_epoch, end_epoch):
        epoch_time_start = time.time()

        losses: List[float] = []
        num_valid_triplets: List[int] = []

        # Training pass
        model.train()

        for batch_idx, (batch_sample) in enumerate(tqdm(data_loader_train)):
            result_train = train_step(model=model,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      distance_fn=l2_distance,
                                      batch_sample=batch_sample,
                                      margin=settings_model.triplet_loss_margin)
            if result_train is None:
                continue

            losses.append(result_train.loss)
            num_valid_triplets.append(result_train.num_triplets)

            if batch_idx % 10 == 0:
                tensorboard.add_scalar(name="loss_train",
                                       value=sum(losses[-10:]) / 10,
                                       global_step=global_step)

                tensorboard.add_scalar(name="num_valid_triplets",
                                       value=sum(num_valid_triplets[-10:]) / 10,
                                       global_step=global_step)

            if batch_idx % 1000 == 0:
                model.eval()
                metrics: EvaluationMetrics = evaluate(model, l2_distance, data_loader_validate)
                tensorboard.add_dict(dataclasses.asdict(metrics), global_step)
                model.train()

            global_step += 1

        # Model only trains on hard negative triplets
        loss_epoch_average = sum(losses) / len(losses)

        epoch_time_end = time.time()

        # Print training statistics and add to log
        print(f"epoch {epoch}\t"
              f"avg_loss: {loss_epoch_average:.4f}\t"
              f"time: {(epoch_time_end - epoch_time_start) / 60:.3f} minutes\t"
              f"valid_triplets: {sum(num_valid_triplets)}")

        # Evaluation pass on LFW dataset
        print("Validating on LFW!")
        model.eval()

        figure_name = f"roc_epoch_{epoch}_triplet.png"
        figure_path = output_dir_plots.joinpath(figure_name)
        metrics: EvaluationMetrics = evaluate(model, l2_distance, data_loader_validate, figure_path)
        pprint(dataclasses.asdict(metrics))

        tensorboard.add_dict(dataclasses.asdict(metrics), global_step)

        # Save model checkpoint
        state = {
            "model_architecture": model_architecture,
            "epoch": epoch,
            "global_step": global_step,
            "embedding_dimension": settings_model.embedding_dim,
            "batch_size_training": settings_model.batch_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # "best_distance_threshold": np.mean(best_distances)
        }

        # Save model checkpoint
        checkpoint_name = f"{model_architecture}_{epoch}.pt"
        torch.save(state, output_dir_checkpoints.joinpath(checkpoint_name))

    # Training loop end
    total_time_end = time.time()
    total_time_elapsed = total_time_end - total_time_start
    print(f"Training finished"
          f"*** total time elapsed: {total_time_elapsed / 60:.2f} minutes.")


def main():
    args = parse_args()
    settings_model = ModelSettings(learning_rate=args.learning_rate,
                                   model_architecture=args.model_architecture,
                                   epochs=args.epochs,
                                   num_triplets_train=args.num_triplets_train,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   embedding_dim=args.embedding_dim,
                                   pretrained_on_imagenet=args.pretrained_on_imagenet,
                                   optimizer=args.optimizer,
                                   triplet_loss_margin=args.triplet_loss_margin)

    settings_data = DataSettings(output_dir=args.output_dir,
                                 dataset_dir=args.dataset_dir,
                                 lfw_dir=args.lfw_dir,
                                 dataset_csv_file=args.dataset_csv_file,
                                 training_triplets_path=args.training_triplets_path,
                                 checkpoint_path=args.checkpoint_path)

    train_triplet(settings_data, settings_model)


if __name__ == "__main__":
    main()
