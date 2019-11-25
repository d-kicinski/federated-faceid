import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Mapping, Any, List

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn import Module
from torch.nn.modules.distance import PairwiseDistance
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloaders.LFWDataset import LFWDataset
from dataloaders.triplet_loss_dataloader import TripletFaceDataset
from losses.triplet_loss import TripletLoss
from models.inceptionresnetv2 import InceptionResnetV2Triplet
from models.resnet101 import Resnet101Triplet
from models.resnet18 import Resnet18Triplet
from models.resnet34 import Resnet34Triplet
from models.resnet50 import Resnet50Triplet
from plots import plot_roc_lfw
from validate_on_LFW import evaluate_lfw


@dataclass
class DataSettings:
    output_dir: Path = Path("../../../output_dir_baseline")
    dataset_dir: Path = Path("../../../data/vggface2/train_cropped")
    lfw_dir: Path = Path("../../../data/lfw/data")
    dataset_csv_file: Path = Path("../../../data/vggface2/train_cropped_meta.csv")
    training_triplets_path: Path = Path("../../../data/vggface2/train_triplets_100000.npy")
    checkpoint_path: Optional[Path] = None


@dataclass
class ModelSettings:
    lfw_batch_size: int = 64
    lfw_validation_epoch_interval: int = 1

    model_architecture: str = "resnet34"
    optimizer: str = "adam"

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    embedding_dim: int = 128
    triplet_loss_margin: float = 0.5
    pretrained_on_imagenet: bool = False

    num_triplets_train: int = 100_000
    num_workers: int = 4


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


class ModelBuilder:
    @staticmethod
    def build(model_architecture: str, embedding_dim: int, imagenet_pretrained: bool) -> Module:
        if model_architecture == "resnet18":
            model = Resnet18Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        elif model_architecture == "resnet34":
            model = Resnet34Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        elif model_architecture == "resnet50":
            model = Resnet50Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        elif model_architecture == "resnet101":
            model = Resnet101Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        elif model_architecture == "inceptionresnetv2":
            model = InceptionResnetV2Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        else:
            raise ValueError(f"Architecture {model_architecture} is unknown!")

        return model


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
class CheckpointValue:
    model: Module
    optimizer: Optimizer
    epoch: int
    global_step: int


def load_checkpoint(model: Module, optimizer: Optimizer, checkpoint_path: Path) -> CheckpointValue:
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint {checkpoint_path} doesnt exist!")

    print(f"Loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path))

    # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    print(f"Checkpoint loaded: start epoch from checkpoint = {epoch}")

    return CheckpointValue(model, optimizer, epoch, global_step)


@dataclass
class TrainStepResults:
    loss: float
    num_triplets: int


def model_train_step(model: Module,
                     optimizer: Optimizer,
                     loss_fn: Module,
                     distance_fn: Module,
                     margin: float,
                     batch_sample) -> Optional[TrainStepResults]:
    anc_img = batch_sample["anc_img"].cuda()
    pos_img = batch_sample["pos_img"].cuda()
    neg_img = batch_sample["neg_img"].cuda()

    # Forward pass - compute embeddings
    anc_embedding = model(anc_img)
    pos_embedding = model(pos_img)
    neg_embedding = model(neg_img)

    # Forward pass - choose hard negatives only for training
    pos_dist = distance_fn(anc_embedding, pos_embedding)
    neg_dist = distance_fn(anc_embedding, neg_embedding)

    distance_difference = (neg_dist - pos_dist) < margin
    distance_difference = distance_difference.cpu().numpy().flatten()

    hard_triplets = np.where(distance_difference == 1)
    if len(hard_triplets[0]) == 0:
        return None

    anc_hard_embedding = anc_embedding[hard_triplets].cuda()
    pos_hard_embedding = pos_embedding[hard_triplets].cuda()
    neg_hard_embedding = neg_embedding[hard_triplets].cuda()

    # Calculate triplet loss
    triplet_loss = loss_fn(anchor=anc_hard_embedding,
                           positive=pos_hard_embedding,
                           negative=neg_hard_embedding).cuda()

    # Backward pass
    optimizer.zero_grad()
    triplet_loss.backward()
    optimizer.step()

    return TrainStepResults(triplet_loss.item(), len(anc_hard_embedding))


class Tensorboard:
    def __init__(self, log_path: Path):
        self._writer = SummaryWriter(str(log_path))

    def add_dict(self, dictionary: Mapping[str, Any], global_step: int):
        for key, value in dictionary.items():
            self._writer.add_scalar(key, value, global_step=global_step)

    def add_scalar(self, name: str, value: float, global_step: int):
        self._writer.add_scalar(name, value, global_step=global_step)


def get_train_data_loader(settings_model: ModelSettings,
                          settings_data: DataSettings) -> DataLoader:
    # Define image data pre-processing transforms
    #   ToTensor() normalizes pixel values between [0, 1]
    #   Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) normalizes pixel values between [-1, 1]

    transforms_train = transforms.Compose([
        transforms.RandomCrop(size=160),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    data_loader_train = DataLoader(
        dataset=TripletFaceDataset(root_dir=settings_data.dataset_dir,
                                   csv_name=settings_data.dataset_csv_file,
                                   num_triplets=settings_model.num_triplets_train,
                                   training_triplets_path=settings_data.training_triplets_path,
                                   transform=transforms_train),
        batch_size=settings_model.batch_size,
        num_workers=settings_model.num_workers,
        shuffle=False
    )
    return data_loader_train


def get_validation_data_loader(settings_model: ModelSettings,
                               settings_data: DataSettings) -> DataLoader:
    transforms_lfw = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    data_loader_lfw = DataLoader(
        dataset=LFWDataset(data_path=settings_data.lfw_dir.joinpath("test"),
                           pairs_path=settings_data.lfw_dir.joinpath("LFW_pairs.txt"),
                           transform=transforms_lfw),
        batch_size=settings_model.lfw_batch_size,
        num_workers=settings_model.num_workers,
        shuffle=False
    )
    return data_loader_lfw


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

    model: torch.nn.Module = ModelBuilder.build(settings_model.model_architecture,
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
        checkpoint = load_checkpoint(model, optimizer, output_dir_checkpoints)
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
    loss_fn = TripletLoss(margin=settings_model.triplet_loss_margin)
    tensorboard = Tensorboard(output_dir_tensorboard)

    for epoch in range(start_epoch, end_epoch):
        epoch_time_start = time.time()

        losses: List[float] = []
        num_valid_training_triplets: int = 0

        # Training pass
        model.train()

        for batch_idx, (batch_sample) in enumerate(tqdm(data_loader_train)):
            result_train = model_train_step(model=model,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            distance_fn=l2_distance,
                                            batch_sample=batch_sample,
                                            margin=settings_model.triplet_loss_margin)

            losses.append(result_train.loss)
            num_valid_training_triplets += result_train.num_triplets

            if batch_idx % 1000:
                tensorboard.add_scalar(name="loss_train",
                                       value=sum(losses[-1000:]) / len(losses[-1000:]),
                                       global_step=global_step)
            global_step += 1

            # if batch_idx % 100 == 0:
            #     break

        # Model only trains on hard negative triplets
        loss_epoch_average = sum(losses) / len(losses)

        epoch_time_end = time.time()

        # Print training statistics and add to log
        print(f"epoch {epoch + 1}\t"
              f"avg_loss: {loss_epoch_average:.4f}\t"
              f"time: {(epoch_time_end - epoch_time_start) / 60:.3f} minutes\t"
              f"valid_triplets: {num_valid_training_triplets}")

        # Evaluation pass on LFW dataset
        print("Validating on LFW!")
        model.eval()

        figure_name = f"roc_epoch_{epoch}_triplet.png"
        figure_path = output_dir_plots.joinpath(figure_name)
        metrics: EvaluationMetrics = evaluate(model, l2_distance, data_loader_validate, figure_path)

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


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    tar: float
    far: float
    distance: float


def evaluate(model: Module, distance_fn: Module, data_loader: DataLoader, figure_path: Path) \
        -> EvaluationMetrics:
    with torch.no_grad():
        distances, labels = [], []

        for batch_index, (data_a, data_b, label) in enumerate(tqdm(data_loader)):
            data_a, data_b, label = data_a.cuda(), data_b.cuda(), label.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = distance_fn(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        (true_positive_rate, false_positive_rate,
         precision, recall, accuracy,
         roc_auc, best_distances, tar, far) = evaluate_lfw(distances=distances, labels=labels)

        # Plot ROC curve
        plot_roc_lfw(false_positive_rate=false_positive_rate,
                     true_positive_rate=true_positive_rate,
                     figure_name=str(figure_path))

        # Print statistics and add to log
        return EvaluationMetrics(accuracy=float(np.mean(accuracy)),
                                 precision=float(np.mean(precision)),
                                 recall=float(np.mean(recall)),
                                 roc_auc=roc_auc,
                                 tar=float(np.mean(tar)),
                                 far=float(np.mean(far)),
                                 distance=float(np.mean(best_distances)))


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
