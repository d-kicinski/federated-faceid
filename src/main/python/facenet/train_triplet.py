import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any, List

import numpy as np
import torch
from torch.nn import Module
from torch.nn.modules.distance import PairwiseDistance
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from commons import get_validation_data_loader, ModelBuilder, load_checkpoint
from dataloaders import facemetadataset
from dataloaders.facemetadataset import FaceMetaDataset, PeopleDataset, TripletsDataset, \
    TripletIndexes
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


class Tensorboard:
    def __init__(self, log_path: Path):
        self._writer = SummaryWriter(str(log_path))

    def add_dict(self, dictionary: Mapping[str, Any], global_step: int):
        for key, value in dictionary.items():
            self._writer.add_scalar(key, value, global_step=global_step)

    def add_scalar(self, name: str, value: float, global_step: int):
        self._writer.add_scalar(name, value, global_step=global_step)


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
    steps: int


class TrainEpoch:
    def __init__(self,
                 model: Module,
                 dataset: FaceMetaDataset,
                 dataset_eval_loader: DataLoader,
                 optimizer: Optimizer,
                 tensorboard: Tensorboard,
                 loss_fn: Module,
                 distance_fn: Module,
                 settings_model: ModelSettings,
                 global_step: int = 0):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.distance_fn = distance_fn
        self.settings = settings_model
        self.tensorboard = tensorboard
        self.dataset_eval_loader = dataset_eval_loader

        self.global_step: int = global_step

        self.log_every_step: int = 10
        self.evaluate_every_step: int = 310

    def train_for_epoch(self):
        self.model.train()

        num_batches: int = 0
        while num_batches < self.settings.batches_in_epoch:
            print("Selecting people")
            people_dataset: PeopleDataset = facemetadataset.select_people(
                self.dataset,
                self.settings.people_per_batch,
                self.settings.images_per_person
            )

            print("Calculating embeddings")
            self.model.eval()
            with torch.no_grad():
                embeddings: np.array = self.calculate_embeddings(self.model, people_dataset)

                triplets: List[TripletIndexes] = facemetadataset.select_triplets(
                    embeddings,
                    people_dataset.num_images_per_class,
                    self.settings.people_per_batch,
                    self.settings.triplet_loss_margin
                )

                triplet_dataset = TripletsDataset(triplets, people_dataset)

                self.model.train()
                results: TrainStepResults = self.train_step(triplet_dataset)
                num_batches += results.steps

    def train_step(self, triplet_dataset: TripletsDataset) -> TrainStepResults:
        losses: List[float] = []
        local_step: int = 0

        triplet_loader = DataLoader(triplet_dataset,
                                    batch_size=self.settings.batch_size,
                                    shuffle=True)

        num_batches = int(np.ceil(len(triplet_dataset) / self.settings.batch_size))
        for triplets in tqdm(triplet_loader, total=num_batches):
            # Calculate triplet loss
            triplet_loss = self.loss_fn(anchor=self.model(triplets["anchor"].cuda()),
                                        positive=self.model(triplets["positive"].cuda()),
                                        negative=self.model(triplets["negative"].cuda())).cuda()

            # Backward pass
            self.optimizer.zero_grad()
            triplet_loss.backward()
            self.optimizer.step()

            self.global_step += 1
            local_step += 1
            losses.append(triplet_loss.item())

            if self.global_step % self.log_every_step == 0:
                self.tensorboard.add_scalar(name="loss_train",
                                            value=sum(losses[-self.log_every_step:]) / len(losses),
                                            global_step=self.global_step)
                losses: List[float] = []
            if self.global_step % self.evaluate_every_step == 0:
                print("Validating on LFW!")
                self.model.eval()

                metrics: EvaluationMetrics = evaluate(self.model,
                                                      self.distance_fn,
                                                      self.dataset_eval_loader,
                                                      None)
                self.tensorboard.add_dict(dataclasses.asdict(metrics), self.global_step)
                self.model.train()

        return TrainStepResults(0.0, local_step)

    def calculate_embeddings(self, model: Module, people_dataset: PeopleDataset):
        image_loader = DataLoader(people_dataset,
                                  batch_size=self.settings.batch_size,
                                  shuffle=False)
        num_examples = len(people_dataset)

        embeddings = np.zeros((num_examples, self.settings.embedding_dim))

        start_idx = 0

        for i, image in tqdm(enumerate(image_loader)):
            batch_size = min(num_examples - i * self.settings.batch_size, self.settings.batch_size)
            image = image.cuda()
            embedding = model(image).cpu().detach().numpy()
            embeddings[start_idx: start_idx + batch_size, :] = embedding

            start_idx += self.settings.batch_size

        return embeddings


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

    total_time_start = time.time()
    end_epoch = start_epoch + settings_model.epochs

    face_meta_dataset = FaceMetaDataset(root_dir=settings_data.dataset_dir,
                                        csv_name=settings_data.dataset_csv_file)
    l2_distance = PairwiseDistance(2).cuda()
    loss_fn = torch.nn.TripletMarginLoss(margin=settings_model.triplet_loss_margin,
                                         reduction="mean")
    tensorboard = Tensorboard(output_dir_tensorboard)
    train_step = TrainEpoch(model=model,
                            dataset=face_meta_dataset,
                            dataset_eval_loader=data_loader_validate,
                            distance_fn=l2_distance,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            tensorboard=tensorboard,
                            settings_model=settings_model,
                            global_step=global_step)

    for epoch in range(start_epoch, end_epoch):
        # Training pass
        train_step.model.train()
        train_step.train_for_epoch()
        global_step = train_step.global_step

        # Save model checkpoint
        state = {
            "model_architecture": model_architecture,
            "epoch": epoch,
            "global_step": global_step,
            "embedding_dimension": settings_model.embedding_dim,
            "batch_size_training": settings_model.batch_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        # Save model checkpoint
        checkpoint_name = f"{model_architecture}_{epoch}.pt"
        torch.save(state, output_dir_checkpoints.joinpath(checkpoint_name))


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
