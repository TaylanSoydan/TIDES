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

def get_dataset_preprocess(config, seed):
    # Create dataset and data loader
    if config.dataset[:4] == "TSC_":
        ds_key = config.dataset[4:]
        data_dir = getattr(config, "data_dir", None)

        X, Y = load_classification(ds_key, extract_path=data_dir)
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
