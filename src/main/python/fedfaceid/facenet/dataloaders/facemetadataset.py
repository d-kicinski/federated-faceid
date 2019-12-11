import csv
from collections import defaultdict, UserDict
from pathlib import Path
from typing import Optional, Callable, List

import PIL
import numpy as np
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class TripletIndexes(UserDict):
    def __init__(self, anchor: int, positive: int, negative: int):
        super().__init__()
        self.anchor: int = anchor
        self.positive: int = positive
        self.negative: int = negative

        self.data = {"anchor": anchor, "positive": positive, "negative": negative}


class Triplet(UserDict):
    def __init__(self, anchor: Tensor, positive: Tensor, negative: Tensor):
        super().__init__()
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

        self.data = {"anchor": anchor, "positive": positive, "negative": negative}


class FaceMetaSamples(UserDict):
    def __init__(self, name: str, image_paths: List[Path]):
        super().__init__()
        self.name: str = name
        self.image_paths: List[Path] = image_paths

        self.data = {"name": name, "image_paths": image_paths}

    def __str__(self):
        return f"{self.name}: {len(self.image_paths)} images"

    def __len__(self):
        return len(self.image_paths)


class PeopleDataset(Dataset):
    def __init__(self,
                 image_paths: List[Path],
                 num_images_per_class: List[int],
                 transform: Optional[Callable] = None):
        self.image_paths: List[Path] = image_paths
        self.num_images_per_class: List[int] = num_images_per_class

        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomCrop(size=160),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform: Optional[Callable] = transform

    def __getitem__(self, idx):
        image: Image = PIL.Image.open(self.image_paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self) -> int:
        return len(self.image_paths)


class TripletsDataset(Dataset):
    def __init__(self, triplets: List[TripletIndexes], people_dataset: PeopleDataset):
        self.triplets = triplets
        self.dataset = people_dataset

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        triplet = self.triplets[idx]

        return Triplet(anchor=self.dataset[triplet.anchor],
                       positive=self.dataset[triplet.positive],
                       negative=self.dataset[triplet.negative]).data


class FaceMetaDataset(Dataset):
    def __init__(self,
                 root_dir: Path,
                 csv_name: Path):
        self.metadata: List[FaceMetaSamples] = FaceMetaDataset.load_metadata(root_dir, csv_name)
        self.root_dir: Path = root_dir

    @staticmethod
    def load_metadata(dataset_path: Path, metadata_path: Path) -> List[FaceMetaSamples]:
        face_samples = defaultdict(list)
        with metadata_path.open("r") as metadata_file:
            reader = csv.DictReader(metadata_file)
            for row in tqdm(reader, desc="Loading metadata"):
                face_samples[row["name"]].append(row["id"])

        face_meta_samples: List[FaceMetaSamples] = []
        for name, filenames in tqdm(face_samples.items(), desc="Processing metadata"):
            image_paths = [
                _add_extension(dataset_path.joinpath(name, filename)) for filename in filenames]
            face_meta_samples.append(FaceMetaSamples(name, image_paths))

        return face_meta_samples

    def __getitem__(self, idx: int):
        return self.metadata[idx]

    def __len__(self) -> int:
        return len(self.metadata)


def _add_extension(path: Path) -> Path:
    if path.with_suffix(".jpg").exists():
        return path.with_suffix(".jpg")
    elif path.with_suffix(".png").exists():
        return path.with_suffix(".png")
    else:
        raise RuntimeError(f"No file {path} with extension png or jpg.")


def select_people(dataset: FaceMetaDataset,
                  people_per_batch: int,
                  images_per_person: int) -> PeopleDataset:
    num_images_to_sample = people_per_batch * images_per_person

    # Sample classes from the dataset
    classes = len(dataset)
    classes_indices = np.arange(classes)
    np.random.shuffle(classes_indices)

    image_paths: List[Path] = []
    num_images_per_class: List[int] = []

    # Sample images from these classes until we have enough
    i = 0
    while len(image_paths) < num_images_to_sample:
        class_index = classes_indices[i]

        num_images_in_class = len(dataset[class_index])
        image_indices = np.arange(num_images_in_class)
        np.random.shuffle(image_indices)
        num_images_from_class = min(num_images_in_class, images_per_person,
                                    num_images_to_sample - len(image_paths))
        idx = image_indices[0: num_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]

        image_paths += image_paths_for_class
        num_images_per_class.append(num_images_from_class)
        i += 1

    return PeopleDataset(image_paths, num_images_per_class)


def select_triplets(embedding: np.array,
                    num_images_per_class: List[int],
                    people_per_batch: int,
                    alpha: float) -> List[TripletIndexes]:
    """ Select the triplets for training"""

    idx_embedding_start: int = 0
    triplets: List[TripletIndexes] = []

    # VGG Face: Choosing good triplets is crucial and should strike a balance between selecting
    # informative (i.e. challenging) examples and swamping training with examples that are too
    # hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling the
    # image n at random, but only between the ones that violate the triplet loss margin. The
    # latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper)
    # than choosing the maximally violating example, as often done in structured output learning.

    for i_person in range(people_per_batch):
        num_images_in_class = int(num_images_per_class[i_person])
        for j_image in range(num_images_in_class):
            idx_anchor = idx_embedding_start + j_image

            # calculate distances of each image to current anchor and mask inter class distances
            distances_neg = np.sum(np.square(embedding[idx_anchor] - embedding), axis=1)
            distances_neg[
            idx_embedding_start: idx_embedding_start + num_images_in_class] = np.NaN

            # For every possible positive  pair.
            for k_image in range(j_image + 1, num_images_in_class):
                idx_pos = idx_embedding_start + k_image

                distance_pos = np.sum(np.square(
                    embedding[idx_anchor] - embedding[idx_pos]
                ))

                # all_neg = np.asarray((distances_neg - distance_pos) < alpha).nonzero()[0]
                all_neg = np.logical_and(distances_neg - distance_pos < alpha,
                                         distance_pos < distances_neg).nonzero()[0]

                num_neg = all_neg.shape[0]

                if num_neg > 0:
                    idx_neg = all_neg[np.random.randint(num_neg)]
                    triplets.append(TripletIndexes(idx_anchor,
                                                   idx_pos,
                                                   idx_neg))

        idx_embedding_start += num_images_in_class
    return triplets
