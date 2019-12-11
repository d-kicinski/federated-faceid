from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataSettings:
    output_dir: Path = Path("../../../output_dir_baseline")
    lfw_dir: Path = Path("../../../data/lfw/data")

    dataset_local_dir: Path = Path("../../../data/lfw/data/train")
    dataset_local_csv_file: Path = Path("../../../data/lfw/data/train_meta.csv")

    dataset_remote_dir: Path = Path("../../../data/nv/nv-gen_cropped")
    dataset_remote_csv_file: Path = Path("../../../data/nv/nv-gen_train_meta.csv")

    checkpoint_path: Optional[Path] = Path("../../../resources/models/resnet50_triplet.pt")


@dataclass
class ModelSettings:
    lfw_batch_size: int = 64
    lfw_validation_epoch_interval: int = 1

    model_architecture: str = "resnet50"

    batch_size: int = 64
    learning_rate: float = 0.001
    embedding_dim: int = 512
    triplet_loss_margin: float = 0.2
    pretrained_on_imagenet: bool = False

    num_workers: int = 4

    people_per_batch: int = 40
    images_per_person: int = 45
    batches_in_epoch: int = 10

    num_local_images_to_use: int = 25  # max images to sample from local dataset
    num_remote_images_to_use: int = sum(range(1, num_local_images_to_use - 1))


@dataclass
class FederatedSettings:
    num_global_epochs: int = 3_000
    num_global_batch: int = 32
    num_local_epochs: int = 1
    num_users: int = -1
    user_fraction: float = 0.1
