import os

import numpy as np
import torchvision.datasets as datasets


class LFWDataset(datasets.ImageFolder):
    def __init__(self, data_path, pairs_path, transform=None):
        super(LFWDataset, self).__init__(str(data_path), transform)

        self.pairs_path = str(pairs_path)
        data_path = str(data_path)

        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_lfw_paths(data_path)

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self, lfw_dir):
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = self.add_extension(
                    os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(
                    os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                issame = True
            elif len(pair) == 4:
                path0 = self.add_extension(
                    os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(
                    os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                issame = False
            if os.path.exists(path0) and os.path.exists(
                    path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def add_extension(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        (path_1, path_2, issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame

    def __len__(self):
        return len(self.validation_images)
