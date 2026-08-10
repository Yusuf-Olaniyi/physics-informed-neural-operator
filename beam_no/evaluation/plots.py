"""Plotting utilities for evaluation figures and paper-style illustrations."""
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(history: dict, save_path: str = None):
    fig, axes = plt.subplots(1, 2 if "data_loss" in history else 1, figsize=(12, 4))
    axes = np.atleast_1d(axes)

    axes[0].plot(history["train_loss"], label="Train")
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title("Total Loss")

    if "data_loss" in history:
        axes[1].plot(history["data_loss"], label="Data Loss")
        axes[1].plot(history["physics_loss"], label="Physics Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_yscale("log")
        axes[1].grid(True)
        axes[1].legend()
        axes[1].set_title("Data vs. Physics Loss")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_prediction_vs_truth(x, prediction, target, title="Sample", ylabel="Deflection (m)",
                              pred_label="Prediction", true_label="Ground truth", save_path=None):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, target, label=true_label, linewidth=2)
    plt.plot(x, prediction, "--", label=pred_label, linewidth=2)
    plt.xlabel("Beam Position (m)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_scatter(targets, predictions, save_path=None):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.3, s=5)
    lims = [targets.min(), targets.max()]
    plt.plot(lims, lims, "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction Scatter")
    plt.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_error_histogram(targets, predictions, save_path=None):
    errors = predictions - targets
    fig = plt.figure(figsize=(7, 4))
    plt.hist(errors.flatten(), bins=50)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_sample_input_output(x, q, w, save_path=None):
    """Illustration of one dataset sample: q(x) and w(x) side by side.
    Useful for the 'before evaluation, show one sample' reviewer comment."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(x, q, color="tab:orange")
    axes[0].set_ylabel("q(x)  [load]")
    axes[0].grid(True)

    axes[1].plot(x, w, color="tab:blue")
    axes[1].set_ylabel("w(x)  [deflection]")
    axes[1].set_xlabel("Beam position x (m)")
    axes[1].grid(True)

    fig.suptitle("Representative training sample")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_spatiotemporal_field(x, t, prediction, target, title="Sample",
                               field_label="Deflection (m)", save_path=None):
    """Side-by-side heatmaps of a predicted vs. ground-truth field over the
    (x, t) grid, plus their pointwise difference -- the dynamic-problem
    analogue of `plot_prediction_vs_truth`.

    `prediction`/`target` shape: (num_nodes, num_steps).
    """
    vmax = max(np.abs(target).max(), np.abs(prediction).max())
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    extent = [t.min(), t.max(), x.min(), x.max()]

    im0 = axes[0].imshow(target, aspect="auto", origin="lower", extent=extent,
                          cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground truth")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("x (m)")
    fig.colorbar(im0, ax=axes[0], label=field_label)

    im1 = axes[1].imshow(prediction, aspect="auto", origin="lower", extent=extent,
                          cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("FNO prediction")
    axes[1].set_xlabel("t (s)")
    fig.colorbar(im1, ax=axes[1], label=field_label)

    diff = prediction - target
    dmax = np.abs(diff).max()
    im2 = axes[2].imshow(diff, aspect="auto", origin="lower", extent=extent,
                          cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_title("Difference")
    axes[2].set_xlabel("t (s)")
    fig.colorbar(im2, ax=axes[2], label=f"Error ({field_label})")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_time_slice(t, prediction_slice, target_slice, x_value=None, title="Time history",
                     ylabel="Deflection (m)", save_path=None):
    """1D time history at a fixed spatial location (e.g. midspan) -- the
    natural companion to `plot_spatiotemporal_field` for a quick
    quantitative look at one sensor location."""
    fig = plt.figure(figsize=(8, 4))
    plt.plot(t, target_slice, label="Ground truth", linewidth=2)
    plt.plot(t, prediction_slice, "--", label="FNO prediction", linewidth=2)
    plt.xlabel("t (s)")
    plt.ylabel(ylabel)
    label_suffix = f" at x = {x_value:.2f} m" if x_value is not None else ""
    plt.title(title + label_suffix)
    plt.grid(True)
    plt.legend()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_sample_input_output_dynamic(x, t, q_xt, w_xt, save_path=None):
    """Illustration of one dynamic dataset sample: q(x,t) and w(x,t) as
    heatmaps -- the dynamic-problem analogue of `plot_sample_input_output`."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    extent = [t.min(), t.max(), x.min(), x.max()]

    im0 = axes[0].imshow(q_xt, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title("q(x, t)  [load]")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("x (m)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(w_xt, aspect="auto", origin="lower", extent=extent, cmap="RdBu_r")
    axes[1].set_title("w(x, t)  [deflection]")
    axes[1].set_xlabel("t (s)")
    fig.colorbar(im1, ax=axes[1])

    fig.suptitle("Representative dynamic training sample")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_physics_residual(x, residual, save_path=None):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x, residual)
    plt.xlabel("Beam Position")
    plt.ylabel("Residual")
    plt.title("Physics Residual")
    plt.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
