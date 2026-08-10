"""Dataset split, save, and load utilities for the dynamic (spatiotemporal)
dataset. Mirrors `data/io.py`, extended with the extra `t` coordinate
array and per-sample metadata (spatial/temporal load family).
"""
import json
import numpy as np


def split_dynamic_dataset(X: np.ndarray, Y: np.ndarray, meta: list, train_ratio: float = 0.7,
                           val_ratio: float = 0.15, seed: int = None):
    """Split (X, Y, meta) into train/val/test index-disjoint subsets."""
    rng = np.random.default_rng(seed)
    N = len(X)
    indices = rng.permutation(N)

    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def _take(idx):
        return X[idx], Y[idx], [meta[i] for i in idx]

    return {
        "train": _take(train_idx),
        "val": _take(val_idx),
        "test": _take(test_idx),
    }


def save_dynamic_dataset(filename: str, X: np.ndarray, Y: np.ndarray,
                          x: np.ndarray, t: np.ndarray, meta: list = None, **extra_meta):
    meta_json = json.dumps(meta if meta is not None else [])
    np.savez(filename, X=X, Y=Y, x=x, t=t, meta_json=meta_json, **extra_meta)


def load_dynamic_dataset(filename: str):
    data = np.load(filename, allow_pickle=True)
    meta = json.loads(str(data["meta_json"])) if "meta_json" in data else []
    return data["X"], data["Y"], data["x"], data["t"], meta
