from pathlib import Path

NUM_GLOBAL_EPOCHS: int = 1000
NUM_GLOBAL_BATCH: int = 32
NUM_LOCAL_EPOCHS: int = 1
NUM_LOCAL_BATCH: int = 64
NUM_USERS: int = 100
USER_FRACTION: float = 1.0

LEARNING_RATE: float = 5e-3
LEARNING_RATE_DECAY: float = 0.99
NON_IID: bool = False
DISTRIBUTED: bool = False
STOPPING_ROUNDS: int = 20
SEED: int = 1
DEVICE: str = "cpu"

PATH_DATASET_CIFAR10: str = "data/cifar"

PATH_OUTPUT_MODEL_SERVER: Path = Path("artifacts/mnist_server")
PATH_OUTPUT_MODEL_FEDERATED: Path = Path("artifacts/mnist_federated")
