from pathlib import Path

NUM_GLOBAL_EPOCHS: int = 100
NUM_GLOBAL_BATCH: int = 64
NUM_LOCAL_EPOCHS: int = 5
NUM_LOCAL_BATCH: int = 128
NUM_USERS: int = 100
USER_FRACTION: float = 0.5

LEARNING_RATE: float = 1e-3
IID: bool = True
DISTRIBUTED: bool = False
STOPPING_ROUNDS: int = 10
SEED: int = 1
DEVICE: str = "cpu"

PATH_DATASET_CIFAR10: str = "data/cifar"

PATH_OUTPUT_MODEL_SERVER: Path = Path("artifacts/mnist_server")
PATH_OUTPUT_MODEL_FEDERATED: Path = Path("artifacts/mnist_federated")
