import argparse

import pandas as pd

from dataloaders.triplet_loss_dataloader import TripletFaceDataset

NUM_TRIPLETS: int = 100_000


def generate_triplets(data_meta_path: str,
                      num_triplets: int,
                      output_path: str):
    metadata = pd.read_csv(data_meta_path, dtype={'id': object, 'name': object, 'class': int})

    TripletFaceDataset.generate_triplets(metadata, num_triplets, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare triplets for FaceNet triplet "
                                                 "loss training")
    parser.add_argument("--data_meta_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_triplets", type=int, required=False, default=NUM_TRIPLETS)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_triplets(args.data_meta_path, args.num_triplets, args.output_path)
