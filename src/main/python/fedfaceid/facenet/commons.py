from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from apex import amp
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import models
from dataloaders.LFWDataset import LFWDataset
from dataloaders.triplet_loss_dataloader import TripletFaceDataset
from settings import ModelSettings, DataSettings


class ModelBuilder:
    @staticmethod
    def build(model_architecture: str, embedding_dim: int, imagenet_pretrained: bool) -> Module:
        if model_architecture == "resnet18":
            model = models.Resnet18Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        elif model_architecture == "resnet34":
            model = models.Resnet34Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        elif model_architecture == "resnet50":
            model = models.Resnet50Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        elif model_architecture == "resnet101":
            model = models.Resnet101Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        elif model_architecture == "inceptionresnetv2":
            model = models.InceptionResnetV2Triplet(
                embedding_dimension=embedding_dim,
                pretrained=imagenet_pretrained
            )
        elif model_architecture == "inceptionresnetv1":
            model = models.InceptionResnetV1Triplet(
                embedding_dimension=embedding_dim)
        elif model_architecture == "inception_resnet_v1_vggface2":
            model = models.InceptionResnetV1(pretrained="vggface2")
        elif model_architecture == "inception_resnet_v1_casia":
            model = models.InceptionResnetV1(pretrained="casia")
        else:
            raise ValueError(f"Architecture {model_architecture} is unknown!")

        return model


@dataclass
class CheckpointValue:
    model: Module
    optimizer: Optimizer
    epoch: int
    global_step: int


def load_checkpoint(checkpoint_path: Path,
                    model: Optional[Module] = None,
                    optimizer: Optional[Optimizer] = None,
                    load_amp: bool = False) -> CheckpointValue:
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint {checkpoint_path} doesnt exist!")

    print(f"Loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path))

    if "model_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint)
        return CheckpointValue(model, optimizer, 0, 0)

    # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if load_amp:
        amp.load_state_dict(checkpoint['amp_state_dict'])

    epoch = checkpoint["epoch"]
    global_step = checkpoint.get("global_step", 0)

    print(f"Checkpoint loaded: start epoch from checkpoint = {epoch}")

    return CheckpointValue(model, optimizer, epoch, global_step)


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
                                   transform=transforms_train),
        batch_size=settings_model.batch_size,
        num_workers=settings_model.num_workers,
        shuffle=True
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
