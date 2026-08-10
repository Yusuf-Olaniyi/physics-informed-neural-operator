"""Dataset split, save, and load utilities."""
import numpy as np


def split_dataset(X: np.ndarray, Y: np.ndarray, train_ratio: float = 0.7,
                   val_ratio: float = 0.15, seed: int = None):
    """Split (X, Y) into train/val/test index-disjoint subsets."""
    rng = np.random.default_rng(seed)
    N = len(X)
    indices = rng.permutation(N)

    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    splits = {
        "train": (X[train_idx], Y[train_idx]),
        "val": (X[val_idx], Y[val_idx]),
        "test": (X[test_idx], Y[test_idx]),
    }
    return splits


def save_dataset(filename: str, X: np.ndarray, Y: np.ndarray, x: np.ndarray, **meta):
    np.savez(filename, X=X, Y=Y, x=x, **meta)


def load_dataset(filename: str):
    data = np.load(filename, allow_pickle=True)
    return data["X"], data["Y"], data["x"]
