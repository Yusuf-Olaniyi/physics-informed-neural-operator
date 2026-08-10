#!/usr/bin/env python3
"""Train the forward operator G_f: q(x) -> w(x).

Usage:
    python scripts/train_forward.py --config configs/default.yaml
"""
import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beam_no.data import BeamDataset, load_dataset
from beam_no.models import BeamFNO
from beam_no.training import train_pino
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

    train_loader = DataLoader(BeamDataset(X_train, Y_train, x),
                               batch_size=cfg["training"]["forward"]["batch_size"], shuffle=True)
    val_loader = DataLoader(BeamDataset(X_val, Y_val, x),
                             batch_size=cfg["training"]["forward"]["batch_size"], shuffle=False)

    model = BeamFNO(**cfg["model"])

    model, history = train_pino(
        model, train_loader,
        epochs=cfg["training"]["forward"]["epochs"],
        E=cfg["beam"]["E"], I=cfg["beam"]["I"],
        lambda_phys=cfg["training"]["forward"]["lambda_phys"],
        lr=cfg["training"]["forward"]["lr"],
        device=device,
        val_loader=val_loader,
        save_dir=cfg["training"]["checkpoint_dir"],
    )

    figures_dir = cfg["evaluation"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    plot_training_curves(history, save_path=os.path.join(figures_dir, "forward_training_curves.png"))


if __name__ == "__main__":
    main()
