#!/usr/bin/env python3
"""Generate the physics-based (FEM) dataset for the forward operator.

Usage:
    python scripts/generate_dataset.py --config configs/default.yaml
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beam_no.fem import Beam, SupportType
from beam_no.data import DatasetGenerator, split_dataset, save_dataset
from beam_no.utils import set_seed, viz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    beam = Beam(**cfg["beam"])
    support = SupportType.from_str(cfg["support_type"])
    d_cfg = cfg["dataset"]

    print(f"Generating dataset for support = {support.name}, "
          f"num_samples = {d_cfg['num_samples']}, beam = {beam}")

    generator = DatasetGenerator(
        beam, support,
        num_samples=d_cfg["num_samples"],
        amplitude_range=tuple(d_cfg["amplitude_range"]),
        seed=d_cfg["seed"],
    )
    X, Y, x, load_types = generator.generate_dataset()

    splits = split_dataset(X, Y, train_ratio=d_cfg["train_ratio"],
                            val_ratio=d_cfg["val_ratio"], seed=d_cfg["seed"])

    out_dir = d_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for split_name, (Xs, Ys) in splits.items():
        path = os.path.join(out_dir, f"{split_name}.npz")
        save_dataset(path, Xs, Ys, x, support=support.name)
        print(f"  {split_name}: {Xs.shape[0]} samples -> {path}")

    # Illustration figures requested by reviewer: problem schematic + a
    # representative q(x)/w(x) sample from the generated dataset.
    figures_dir = cfg["evaluation"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    viz.plot_beam_problem_schematic(
        support.name.replace("_", " ").title(),
        save_path=os.path.join(figures_dir, "beam_problem_schematic.png"),
    )
    viz.plot_fno_architecture(
        save_path=os.path.join(figures_dir, "fno_architecture.png"),
    )

    from beam_no.evaluation.plots import plot_sample_input_output
    plot_sample_input_output(
        x, X[0, :, 0], Y[0, :, 0],
        save_path=os.path.join(figures_dir, "sample_q_w.png"),
    )
    print(f"Saved schematic figures to {figures_dir}/")


if __name__ == "__main__":
    main()
