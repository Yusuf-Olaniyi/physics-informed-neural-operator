"""torch.utils.data.Dataset wrappers for the forward and inverse operators."""
import numpy as np
import torch
from torch.utils.data import Dataset


class BeamDataset(Dataset):
    """Forward-problem dataset.

    X: (num_nodes, 2) = [q(x), x]
    Y: (num_nodes, 1) = [w(x)]
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, x: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.x


class InverseBeamDataset(Dataset):
    """Inverse-problem dataset, derived from a forward `BeamDataset`.

    Reverses the input/output roles:
        X_inverse = [w(x), x]
        Y_inverse = q(x)

    No boundary-condition channel is carried over (consistent with the
    forward dataset).
    """

    def __init__(self, forward_dataset: BeamDataset):
        self.forward_dataset = forward_dataset

    def __len__(self):
        return len(self.forward_dataset)

    def __getitem__(self, idx):
        X_forward, Y_forward, x = self.forward_dataset[idx]

        q = X_forward[:, 0:1]
        deflection = Y_forward

        X_inverse = deflection          # [w(x)], x is concatenated at model-input time
        Y_inverse = q

        return X_inverse, Y_inverse, x
