"""Custom dataset for generated samples."""

import os
import numpy as np

from torch.utils.data import Dataset

class SamplesDataset(Dataset):
    """
    Dataset for generated samples
    """

    def __init__(self, folder, transform=None):

        data = []
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            data.append(path)

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        data = {'path': path}

        if self.transform is not None:
            data = self.transform(data)

        return data
