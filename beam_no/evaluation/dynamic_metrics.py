"""Inference and evaluation for the dynamic forward/inverse operators.
`compute_metrics` (imported from `evaluation.metrics`) is dimension-
agnostic -- it flattens whatever array it's given -- so it's reused as-is;
only `run_inference` needs a dynamic-aware version, since the dynamic
inverse dataset needs both an x and a t channel attached (see
`training/train_inverse_dynamic.py`'s reasoning), unlike the static case
which only needed x.
"""
import numpy as np
import torch

from .metrics import compute_metrics
from beam_no.utils.grid import attach_xt_channels


def run_inference_dynamic(model, dataloader, device="cpu", checkpoint_path=None):
    """Run a model over a dynamic dataloader and collect predictions/
    targets/coordinates. Returns x_coords and t_coords separately, since
    (unlike the static 1D case) they're not the same length."""
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    predictions, targets, x_coords_all, t_coords_all = [], [], [], []

    with torch.no_grad():
        for X_input, Y_target, x_coords, t_coords in dataloader:
            X_input = X_input.to(device)
            x_coords = x_coords.to(device)
            t_coords = t_coords.to(device)

            # InverseDynamicBeamDataset yields X_input with only the
            # signal channel (w(x,t)); attach x/t before calling the
            # model, exactly as train_inverse_fno_dynamic does.
            # DynamicBeamDataset (forward) already carries x, t as
            # channels 1 and 2 of X, so this only fires for the inverse case.
            if X_input.shape[-1] == 1:
                X_input = attach_xt_channels(X_input, x_coords, t_coords)

            pred = model(X_input)
            predictions.append(pred.cpu().numpy())
            targets.append(Y_target.numpy())
            x_coords_all.append(x_coords.cpu().numpy())
            t_coords_all.append(t_coords.cpu().numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(x_coords_all, axis=0),
        np.concatenate(t_coords_all, axis=0),
    )


def evaluate_dynamic(model, dataloader, device="cpu", checkpoint_path=None):
    """Run inference + compute metrics for the dynamic problem. Returns
    metrics dict; call `beam_no.evaluation.plots` separately for figures."""
    predictions, targets, x_coords, t_coords = run_inference_dynamic(
        model, dataloader, device, checkpoint_path
    )
    metrics = compute_metrics(predictions, targets)

    print("=" * 60)
    print("Dynamic Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.6e}" if isinstance(v, float) else f"{k:15s}: {v}")

    return metrics, predictions, targets, x_coords, t_coords
