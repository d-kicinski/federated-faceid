from collections import defaultdict
from pathlib import Path
from typing import Optional, Callable, List, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FaceDateset(Dataset):
    def __init__(self,
                 root_dir: Path,
                 csv_name: Path,
                 num_triplets: int,
                 training_triplets_path: Path,
                 transform: Optional[Callable] = None):

        self.df: pd.DataFrame = pd.read_csv(str(csv_name.resolve()),
                                            dtype={'id': object, 'name': object, 'class': int})
        self.root_dir: Path = root_dir
        self.num_triplets: int = num_triplets
        self.transform: Optional[Callable] = transform
        self.training_triplets = np.load(str(training_triplets_path))

    @staticmethod
    def _make_dictionary_for_face_class(df: pd.DataFrame) -> Dict[str, List[int]]:
        """
          - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
        """
        face_classes = defaultdict(list)
        for idx, label in enumerate(df['class']):
            face_classes[label].append(df.iloc[idx, 0])

        return face_classes

    def __getitem__(self, idx):
        image = _add_extension(self.root_dir.joinpath(str(neg_name), str(neg_id)))

        # Modified to open as PIL image in the first place
        anc_img = Image.open(anc_img)
        pos_img = Image.open(pos_img)
        neg_img = Image.open(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)


def _add_extension(path: Path) -> Path:
    if path.with_suffix(".jpg").exists():
        return path.with_suffix(".jpg")
    elif path.with_suffix(".png").exists():
        return path.with_suffix(".png")
    else:
        raise RuntimeError(f"No file {path} with extension png or jpg.")


class PersonFaces:
    def __init__(self, name: str, image_paths: List[Path]):
        self.name: str = name
        self.image_paths: List[Path] = image_paths

    def __str__(self):
        return f"{self.name}: {len(self.image_paths)} images"

    def __len__(self):
        return len(self.image_paths)


def sample_people(dataset: List[PersonFaces],
                  people_per_batch: int,
                  images_per_person: int):
    num_images_to_sample = people_per_batch * images_per_person

    # Sample classes from the dataset
    classes = len(dataset)
    classes_indices = np.arange(classes)
    np.random.shuffle(classes_indices)

    i = 0

    image_paths: List[Path] = []
    num_per_class = []

    # Sample images from these classes until we have enough
    while len(image_paths) < num_images_to_sample:
        class_index = classes_indices[i]

        images_in_class = len(dataset[class_index])
        image_indices = np.arange(images_in_class)
        np.random.shuffle(image_indices)
        images_from_class = min(images_in_class, images_per_person,
                                nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]

        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_clas
