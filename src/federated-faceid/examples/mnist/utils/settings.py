import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils import constants


@dataclass
class Settings:
    num_global_epochs: int
    num_global_batch: int
    num_local_epochs: int
    num_local_batch: int
    num_users: int
    user_fraction: float

    learning_rate: float
    learning_rate_decay: float
    non_iid: bool
    stopping_rounds: int
    seed: int

    distributed: bool

    device: str

    save_path: Optional[Path] = None


def create_save_path(settings: Settings) -> Path:
    path = Path("artifacts")
    model_name = "mnist"
    if settings.distributed:
        model_name += "_distributed"
    if settings.non_iid:
        model_name += "_non_iid"
    path = path.joinpath(model_name)

    path.mkdir(exist_ok=True, parents=True)
    if path.joinpath("tensorboard").exists():
        shutil.rmtree(str(path))

    return path


def args_parser() -> Settings:
    parser = argparse.ArgumentParser()

    parser.add_argument('--distributed', action='store_true', default=constants.DISTRIBUTED,
                        help='whether use distributed training or not')
    parser.add_argument('--non_iid', action='store_true', default=constants.NON_IID,
                        help='whether i.i.d or not')

    # federated arguments
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
    return Settings(**args.__dict__)
