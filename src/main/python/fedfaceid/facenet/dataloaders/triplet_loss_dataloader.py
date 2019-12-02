from collections import defaultdict
from pathlib import Path
from typing import Optional, Callable, List, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

np.random.seed(0)
torch.manual_seed(0)


class TripletFaceDataset(Dataset):
    def __init__(self,
                 root_dir: Path,
                 csv_name: Path,
                 num_triplets: int,
                 transform: Optional[Callable] = None):

        self.root_dir: Path = root_dir
        self.num_triplets: int = num_triplets
        self.transform: Optional[Callable] = transform

        self.df: pd.DataFrame = pd.read_csv(str(csv_name.resolve()),
                                            dtype={'id': object, 'name': object, 'class': int})
        self.classes = self.df['class'].unique()

        self.face_classes = TripletFaceDataset._make_dictionary_for_face_class(self.df)

    def generate_triplet(self):

        """
        - randomly choose anchor, positive and negative images for triplet loss
        - anchor and positive images in pos_class
        - negative image in neg_class
        - at least, two images needed for anchor and positive images in pos_class
        - negative image should have different class as anchor and positive images by definition
        """

        pos_class = np.random.choice(self.classes)
        neg_class = np.random.choice(self.classes)

        while len(self.face_classes[pos_class]) < 2:
            pos_class = np.random.choice(self.classes)

        while pos_class == neg_class:
            neg_class = np.random.choice(self.classes)

        pos_name = self.df.loc[self.df['class'] == pos_class, 'name'].values[0]
        neg_name = self.df.loc[self.df['class'] == neg_class, 'name'].values[0]

        if len(self.face_classes[pos_class]) == 2:
            ianc, ipos = np.random.choice(2, size=2, replace=False)

        else:
            ianc = np.random.randint(0, len(self.face_classes[pos_class]))
            ipos = np.random.randint(0, len(self.face_classes[pos_class]))

            while ianc == ipos:
                ipos = np.random.randint(0, len(self.face_classes[pos_class]))

        ineg = np.random.randint(0, len(self.face_classes[neg_class]))

        triplet = (self.face_classes[pos_class][ianc],
                   self.face_classes[pos_class][ipos],
                   self.face_classes[neg_class][ineg],
                   pos_class,
                   neg_class,
                   pos_name,
                   neg_name)

        return triplet

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

        (anc_id, pos_id, neg_id,
         pos_class, neg_class,
         pos_name, neg_name) = self.generate_triplet()

        anc_img = _add_extension(self.root_dir.joinpath(str(pos_name), str(anc_id)))
        pos_img = _add_extension(self.root_dir.joinpath(str(pos_name), str(pos_id)))
        neg_img = _add_extension(self.root_dir.joinpath(str(neg_name), str(neg_id)))

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
        return self.num_triplets


def _add_extension(path: Path) -> Path:
    if path.with_suffix(".jpg").exists():
        return path.with_suffix(".jpg")
    elif path.with_suffix(".png").exists():
        return path.with_suffix(".png")
    else:
        raise RuntimeError(f"No file {path} with extension png or jpg.")
