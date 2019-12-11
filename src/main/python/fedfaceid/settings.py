import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fedfaceid import constants


@dataclass
class DataSettings:
    output_dir: Path = Path("../../../output_dir_baseline")
    lfw_dir: Path = Path("../../../data/lfw/data")

    dataset_local_dir: Path = Path("../../../data/vggface2/train_cropped")
    dataset_local_csv_file: Path = Path("../../../data/vggface2/train_cropped_meta.csv")

    dataset_remote_dir: Path = Path("../../../data/vggface2/train_cropped")
    dataset_remote_csv_file: Path = Path("../../../data/vggface2/train_cropped_meta.csv")

    checkpoint_path: Optional[Path] = None


@dataclass
class ModelSettings:
    lfw_batch_size: int = 64
    lfw_validation_epoch_interval: int = 1

    model_architecture: str = "resnet34"
    optimizer: str = "adam"

    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 0.001
    embedding_dim: int = 512
    triplet_loss_margin: float = 0.2
    pretrained_on_imagenet: bool = False

    num_triplets_train: int = 100_000
    num_workers: int = 4

    people_per_batch: int = 40
    images_per_person: int = 45
    batches_in_epoch: int = 1_240

    num_local_images_to_use = 25
    num_remote_images_to_use = sum(range(1, num_local_images_to_use - 1))


@dataclass
class FederatedSettings:
    num_global_epochs: int
    num_global_batch: int
    num_local_epochs: int
    num_local_batch: int
    num_users: int
    user_fraction: float
    num_subsets_per_user: int

    learning_rate: float
    learning_rate_decay: float
    non_iid: bool
    stopping_rounds: int
    skip_stopping: bool
    seed: int

    distributed: bool

    device: str

    id: Optional[str] = None
    save_path: Optional[Path] = None


def args_parser() -> FederatedSettings:
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, required=False)

    parser.add_argument('--skip_stopping', action='store_true', default=constants.SKIP_STOPPING)
    parser.add_argument('--distributed', action='store_true', default=constants.DISTRIBUTED,
                        help='whether use distributed training or not')
    parser.add_argument('--non_iid', action='store_true', default=constants.NON_IID,
                        help='whether i.i.d or not')

    # federated arguments
    parser.add_argument('--num_subsets_per_user', type=int, default=constants.NUM_SUBSETS_PER_USER)
    parser.add_argument('--num_global_epochs', type=int, default=constants.NUM_GLOBAL_EPOCHS)
    parser.add_argument('--num_global_batch', type=int, default=constants.NUM_GLOBAL_BATCH)
    parser.add_argument('--num_users', type=int, default=constants.NUM_USERS,
                        help="number of users: K")
    parser.add_argument('--user_fraction', type=float, default=constants.USER_FRACTION,
                        help="the fraction of clients: C")
    parser.add_argument('--num_local_epochs', type=int, default=constants.NUM_LOCAL_EPOCHS,
                        help="the number of local epochs: E")
    parser.add_argument('--num_local_batch', type=int, default=constants.NUM_LOCAL_BATCH,
                        help="local batch size: B")

    # other arguments
    parser.add_argument('--learning_rate', type=float, default=constants.LEARNING_RATE)
    parser.add_argument('--learning_rate_decay', type=float, default=constants.LEARNING_RATE_DECAY)
    parser.add_argument('--stopping_rounds', type=int, default=constants.STOPPING_ROUNDS,
                        help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=constants.SEED)
    parser.add_argument('--device', type=str, default=constants.DEVICE)
    args = parser.parse_args()
    return FederatedSettings(**args.__dict__)
