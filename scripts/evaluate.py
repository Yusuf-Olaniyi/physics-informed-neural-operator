#!/usr/bin/env python3
"""Evaluate trained forward and/or inverse operators on the held-out test set.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --direction forward
    python scripts/evaluate.py --config configs/default.yaml --direction inverse
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from beam_no.data import BeamDataset, InverseBeamDataset, load_dataset
from beam_no.models import BeamFNO
from beam_no.evaluation.metrics import evaluate
from beam_no.evaluation.plots import plot_scatter, plot_error_histogram, plot_prediction_vs_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--direction", choices=["forward", "inverse"], default="forward")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = cfg["dataset"]["output_dir"]
    X_test, Y_test, x = load_dataset(os.path.join(data_dir, "test.npz"))
    forward_test = BeamDataset(X_test, Y_test, x)

    if args.direction == "forward":
        dataset = forward_test
        ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "best_forward_model.pth")
        y_label, title_prefix = "Deflection (m)", "Forward"
    else:
        dataset = InverseBeamDataset(forward_test)
        ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "best_inverse_model.pth")
        y_label, title_prefix = "Load q(x)", "Inverse"

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = BeamFNO(**cfg["model"])

    metrics, predictions, targets, coordinates = evaluate(
        model, loader, device=device, checkpoint_path=ckpt
    )

    figures_dir = cfg["evaluation"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    plot_scatter(targets, predictions,
                 save_path=os.path.join(figures_dir, f"{args.direction}_scatter.png"))
    plot_error_histogram(targets, predictions,
                          save_path=os.path.join(figures_dir, f"{args.direction}_error_hist.png"))

    n_plot = min(cfg["evaluation"]["plot_samples"], len(predictions))
    for i in range(n_plot):
        plot_prediction_vs_truth(
            coordinates[i], predictions[i, :, 0], targets[i, :, 0],
            title=f"{title_prefix} — Sample {i}", ylabel=y_label,
            pred_label="FNO prediction", true_label="FEM ground truth",
            save_path=os.path.join(figures_dir, f"{args.direction}_sample_{i}.png"),
        )

    metrics_path = os.path.join(figures_dir, f"{args.direction}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path} and figures to {figures_dir}/")


if __name__ == "__main__":
    main()
