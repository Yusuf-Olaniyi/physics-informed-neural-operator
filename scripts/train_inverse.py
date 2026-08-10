#!/usr/bin/env python3
"""Train the inverse operator G_i: w(x) -> q(x).

Usage:
    python scripts/train_inverse.py --config configs/default.yaml
"""
import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beam_no.data import BeamDataset, InverseBeamDataset, load_dataset
from beam_no.models import BeamFNO
from beam_no.training import train_inverse_fno
from beam_no.evaluation.plots import plot_training_curves
from beam_no.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    data_dir = cfg["dataset"]["output_dir"]
    X_train, Y_train, x = load_dataset(os.path.join(data_dir, "train.npz"))
    X_val, Y_val, _ = load_dataset(os.path.join(data_dir, "val.npz"))

    train_loader = DataLoader(
        InverseBeamDataset(BeamDataset(X_train, Y_train, x)),
        batch_size=cfg["training"]["inverse"]["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        InverseBeamDataset(BeamDataset(X_val, Y_val, x)),
        batch_size=cfg["training"]["inverse"]["batch_size"], shuffle=False,
    )

    model = BeamFNO(**cfg["model"])

    model, history = train_inverse_fno(
        model, train_loader,
        epochs=cfg["training"]["inverse"]["epochs"],
        lr=cfg["training"]["inverse"]["lr"],
        device=device,
        val_loader=val_loader,
        save_dir=cfg["training"]["checkpoint_dir"],
    )

    figures_dir = cfg["evaluation"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    plot_training_curves(history, save_path=os.path.join(figures_dir, "inverse_training_curves.png"))


if __name__ == "__main__":
    main()
