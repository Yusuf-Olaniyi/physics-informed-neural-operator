"""Physics-informed loss terms based on the Euler-Bernoulli beam equation:

    EI * d^4 w(x) / dx^4 = q(x)

The 4th spatial derivative of the predicted deflection is computed via
automatic differentiation, so `x` must be a leaf tensor with
`requires_grad_(True)` set before the forward pass that produced `w`.
"""
import torch


def fourth_derivative(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute d^4 w / dx^4 via repeated automatic differentiation."""
    dw = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    d2w = torch.autograd.grad(dw, x, grad_outputs=torch.ones_like(dw), create_graph=True)[0]
    d3w = torch.autograd.grad(d2w, x, grad_outputs=torch.ones_like(d2w), create_graph=True)[0]
    d4w = torch.autograd.grad(d3w, x, grad_outputs=torch.ones_like(d3w), create_graph=True)[0]
    return d4w


def compute_physics_residual(prediction: torch.Tensor, q: torch.Tensor,
                              x: torch.Tensor, E: float, I: float) -> torch.Tensor:
    """Residual R = EI * w'''' - q."""
    w4 = fourth_derivative(prediction, x)
    return E * I * w4 - q


def physics_loss(prediction: torch.Tensor, q: torch.Tensor, x: torch.Tensor,
                  E: float, I: float) -> torch.Tensor:
    """Mean squared physics residual: mean((EI * w'''' - q)^2)."""
    residual = compute_physics_residual(prediction, q, x, E, I)
    return torch.mean(residual ** 2)
