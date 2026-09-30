"""Torch dataset for the UEA classification tasks."""

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesClassification_preprocess(Dataset):
    # format for generic dataset from https://www.timeseriesclassification.com
    def __init__(self, features, labels):
        self.features = features
        _, self.labels = np.unique(labels, return_inverse=True)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        signal = self.features[index]
        label = self.labels[index]
        sample = {'input': signal, 'label': torch.tensor(label, dtype=torch.long)}
        return sample
