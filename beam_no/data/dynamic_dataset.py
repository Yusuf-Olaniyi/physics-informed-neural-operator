"""torch.utils.data.Dataset wrappers for the dynamic forward and inverse
operators, mirroring `data/dataset.py`'s design for the static case.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class DynamicBeamDataset(Dataset):
    """Forward-problem dataset for the dynamic operator.

    X: (num_nodes, num_steps, 3) = [q(x,t), x, t]
    Y: (num_nodes, num_steps, 1) = [w(x,t)]
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, x: np.ndarray, t: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.x, self.t


class InverseDynamicBeamDataset(Dataset):
    """Inverse-problem dataset, derived from a forward `DynamicBeamDataset`.

    Reverses the input/output roles:
        X_inverse = w(x,t)      (x, t concatenated at model-input time)
        Y_inverse = q(x,t)

    No boundary-condition channel is carried, consistent with the forward
    dataset and with the static-problem convention.
    """

    def __init__(self, forward_dataset: DynamicBeamDataset):
        self.forward_dataset = forward_dataset

    def __len__(self):
        return len(self.forward_dataset)

    def __getitem__(self, idx):
        X_forward, Y_forward, x, t = self.forward_dataset[idx]

        q = X_forward[:, :, 0:1]
        deflection = Y_forward

        X_inverse = deflection   # [w(x,t)]
        Y_inverse = q

        return X_inverse, Y_inverse, x, t
