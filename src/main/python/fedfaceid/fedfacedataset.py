from typing import List

import numpy as np
from torch.utils.data import Dataset

from facenet.dataloaders.facemetadataset import PeopleDataset, FaceMetaSamples, \
    TripletIndexes, Triplet


def select_faces(face_samples: FaceMetaSamples,
                 images_per_person: int) -> PeopleDataset:
    num_images_in_class = len(face_samples)
    image_indices = np.arange(num_images_in_class)
    np.random.shuffle(image_indices)
    idx = image_indices[0: min(images_per_person, num_images_in_class)]
    image_paths = [face_samples.image_paths[j] for j in idx]

    return PeopleDataset(image_paths, [num_images_in_class])


def select_triplets(embeddings_local: np.array,
                    embeddings_remote: np.array,
                    alpha: float):
    """ Select the triplets for training"""

    triplets: List[TripletIndexes] = []
    num_embeddings_local: int = embeddings_local.shape[0]

    # VGG Face: Choosing good triplets is crucial and should strike a balance between selecting
    # informative (i.e. challenging) examples and swamping training with examples that are too
    # hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling the
    # image n at random, but only between the ones that violate the triplet loss margin. The
    # latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper)
    # than choosing the maximally violating example, as often done in structured output learning.

    for j_image in range(num_embeddings_local):
        idx_anchor = j_image

        # calculate distances of each image to current anchor and mask inter class distances
        distances_neg = np.linalg.norm(embeddings_local[idx_anchor] - embeddings_remote, axis=1)

        # For every possible positive  pair.
        for k_image in range(j_image + 1, num_embeddings_local):
            idx_pos = k_image
            distance_pos = np.linalg.norm(embeddings_local[idx_anchor] - embeddings_local[idx_pos])

            # all_neg = np.logical_and(distances_neg - distance_pos < alpha,
            #                          distance_pos < distances_neg).nonzero()[0]
            all_neg = np.asarray((distances_neg - distance_pos) < alpha).nonzero()[0]

            num_neg = all_neg.shape[0]

            if num_neg > 0:
                idx_neg = all_neg[np.random.randint(num_neg)]
                triplets.append(TripletIndexes(idx_anchor,
                                               idx_pos,
                                               idx_neg))

    return triplets


class FederatedTripletsDataset(Dataset):
    def __init__(self,
                 triplets: List[TripletIndexes],
                 faces_local: PeopleDataset,
                 faces_remote: PeopleDataset):
        self.triplets = triplets
        self.faces_local = faces_local
        self.faces_remote = faces_remote

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        triplet = self.triplets[idx]

        return Triplet(anchor=self.faces_local[triplet.anchor],
                       positive=self.faces_local[triplet.positive],
                       negative=self.faces_remote[triplet.negative]).data
