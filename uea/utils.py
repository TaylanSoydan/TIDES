import os
import shutil

import numpy as np
import torch
from aeon.datasets import load_classification
from torch.utils.data import DataLoader
from datasets import TimeSeriesClassification_preprocess
from sklearn.model_selection import train_test_split


'''
Note: Some datasets from the Time Series Classification benchmark include some leakage
'''


def remove_duplicates(X, Y):
    # Flatten each sample to a 1D vector so we can find exact duplicates:
    n, c, L = X.shape
    X_flat = X.reshape(n, c * L)

    # np.unique with return_index finds the first index of each unique row:
    _, first_idxs = np.unique(X_flat, axis=0, return_index=True)
    keep = np.sort(first_idxs)             # restore original order

    return X[keep], Y[keep]

def load_uea(name, data_dir):
    """aeon's load_classification, without leaving a half-downloaded dataset behind.

    aeon downloads into <data_dir>/<name>/.  If that fails (e.g. no internet on a
    compute node) the folder it created is removed, so the next call starts a
    clean download instead of finding partial files.
    """
    target = os.path.join(data_dir, name) if data_dir else None
    fresh = target is not None and not os.path.exists(target)
    try:
        return load_classification(name, extract_path=data_dir)
    except BaseException:
        if fresh:
            shutil.rmtree(target, ignore_errors=True)
        raise


def get_dataset_preprocess(config, seed):
    # Create dataset and data loader
    if config.dataset[:4] == "TSC_":
        ds_key = config.dataset[4:]
        data_dir = getattr(config, "data_dir", None)

        X, Y = load_uea(ds_key, data_dir)
        X, Y = remove_duplicates(X, Y)

        X = torch.tensor(np.transpose(X, (0, 2, 1))).float()

        seq_length_original = X.shape[1]
        num_classes = len(np.unique(Y))
        num_features = X.shape[2]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=config.test_size, random_state=seed)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=config.val_size, random_state=seed)
        
        train_dataset = TimeSeriesClassification_preprocess(x_train, y_train)
        val_dataset = TimeSeriesClassification_preprocess(x_val, y_val)
        test_dataset = TimeSeriesClassification_preprocess(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
        num_samples = len(x_train)
    else:
        raise NotImplementedError('Just TSC for now')

    return train_loader, val_loader, test_loader, seq_length_original, num_classes, num_samples, num_features
