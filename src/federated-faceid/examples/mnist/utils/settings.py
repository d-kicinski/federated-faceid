import argparse
from dataclasses import dataclass

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
    iid: bool
    stopping_rounds: int
    seed: int

    device: str


def args_parser() -> Settings:
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--iid', action='store_true', default=constants.IID,
                        help='whether i.i.d or not')

    # other arguments
    parser.add_argument('--learning_rate', type=float, default=constants.LEARNING_RATE,
                        help="learning rate")
    parser.add_argument('--stopping_rounds', type=int, default=constants.STOPPING_ROUNDS,
                        help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=constants.SEED)
    parser.add_argument('--device', type=str, default=constants.DEVICE)
    args = parser.parse_args()
    return Settings(**args.__dict__)
