import argparse
from pathlib import Path
from typing import Union

import torch
from facenet_pytorch import MTCNN, training
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE: int = 64
NUM_WORKERS: int = 6
DATA_DIR: str = "../../../data/vggface2/train"


def preprocess_vggface2(
    data_dir: Union[Path, str],
    output_dir: Union[Path, str],
    num_workers: int,
    batch_size: int,
    image_size: int,
    margin: int,
    use_cpu: bool,
):
    data_dir = str(data_dir)
    output_dir = str(output_dir)

    if use_cpu:
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Running processing on device: {}".format(device))

    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        prewhiten=False,
        device=device,
    )

    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [(p, p.replace(data_dir, output_dir)) for p, _ in dataset.samples]

    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil,
    )

    batch_num = len(loader)
    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print("\rBatch {} of {}".format(i + 1, batch_num), end="")


def parse_args():
    parser = argparse.ArgumentParser(description="Align faces for further training")
    parser.add_argument("--data_dir", type=lambda p: Path(p), default=DATA_DIR)
    parser.add_argument("--output_dir", type=lambda p: Path(p), required=True)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--margin", type=int, required=True)
    parser.add_argument("--cpu", action="store_true", default=False, required=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_vggface2(
        args.data_dir,
        args.output_dir,
        args.num_workers,
        args.batch_size,
        args.image_size,
        args.margin,
        args.cpu,
    )
