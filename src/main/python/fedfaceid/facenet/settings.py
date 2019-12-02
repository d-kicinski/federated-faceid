from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    batch_size: int = 48
    learning_rate: float = 0.001
    embedding_dim: int = 128
    triplet_loss_margin: float = 0.5
    pretrained_on_imagenet: bool = False

    num_triplets_train: int = 100_000
    num_workers: int = 4
