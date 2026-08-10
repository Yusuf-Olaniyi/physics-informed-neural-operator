"""Quantitative evaluation metrics for the trained operators."""
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_inference(model, dataloader, device="cpu", checkpoint_path=None):
    """Run a model over a dataloader and collect predictions/targets/coords."""
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    predictions, targets, coordinates = [], [], []

    with torch.no_grad():
        for X_input, Y_target, x_coords in dataloader:
            X_input = X_input.to(device)
            x_coords = x_coords.to(device)

            # InverseBeamDataset yields X_input with only the signal
            # channel (w(x)); the x-coordinate channel has to be attached
            # before the model sees it, exactly as train_inverse.py does.
            # BeamDataset (forward) already carries x as channel 1 of X,
            # so this only fires for the inverse case.
            if X_input.shape[-1] == 1:
                X_input = torch.cat([X_input, x_coords.unsqueeze(-1)], dim=-1)

            pred = model(X_input)
            predictions.append(pred.cpu().numpy())
            targets.append(Y_target.numpy())
            coordinates.append(x_coords.numpy())

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(coordinates, axis=0),
    )


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Relative L2, MSE, RMSE, MAE, max error, R^2 over a batch of samples."""
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    r2 = r2_score(targets.flatten(), predictions.flatten())

    rel_errors = [
        np.linalg.norm(predictions[i] - targets[i]) / np.linalg.norm(targets[i])
        for i in range(len(predictions))
    ]
    relative_l2 = float(np.mean(rel_errors))
    max_error = float(np.max(np.abs(predictions - targets)))

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Relative_L2": relative_l2,
        "Max_Error": max_error,
        "R2": float(r2),
    }


def evaluate(model, dataloader, device="cpu", checkpoint_path=None) -> dict:
    """Run inference + compute metrics. Returns metrics dict; call
    `beam_no.evaluation.plots` separately for figures."""
    predictions, targets, coordinates = run_inference(model, dataloader, device, checkpoint_path)
    metrics = compute_metrics(predictions, targets)

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.6e}" if isinstance(v, float) else f"{k:15s}: {v}")

    return metrics, predictions, targets, coordinates
