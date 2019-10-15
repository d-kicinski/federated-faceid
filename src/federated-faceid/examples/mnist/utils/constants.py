NUM_GLOBAL_EPOCHS: int = 3
NUM_GLOBAL_BATCH: int = 64
NUM_LOCAL_EPOCHS: int = 5
NUM_LOCAL_BATCH: int = 10
NUM_USERS: int = 100
USER_FRACTION: float = 0.8

LEARNING_RATE: float = 1e-2
IID: bool = False
STOPPING_ROUNDS: int = 3
SEED: int = 1
DEVICE: str = "cpu"

PATH_DATASET_CIFAR10: str = "data/cifar"

PROCESS_START_METHOD: str = "spawn"

MAX_USERS_IN_ROUND: int = 2
