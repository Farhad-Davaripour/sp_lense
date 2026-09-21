"""Training-only compact linear controller; prediction requires only base activations."""

import numpy as np


def fit(
    x, y, active, global_x=None, seed=42, input_rank=32, output_rank=8, ridge=1.0, device="cuda"
):
    import torch

    torch.manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    active = torch.as_tensor(active, dtype=torch.bool, device=device)
    global_x = (
        x if global_x is None else torch.as_tensor(global_x, dtype=torch.float32, device=device)
    )
    if not bool(active.any()) or x.ndim != 2 or x.shape != y.shape:
        raise ValueError("Invalid controller training arrays")
    mean = y[active].mean(0)
    if mean.norm() < 1e-10:
        raise ValueError("No nonzero mean target")
    residual = y[active] - mean
    _, _, residual_basis = torch.pca_lowrank(residual, q=output_rank, center=False, niter=3)
    # Explicitly span the mean: zero and constant-mean interventions remain representable.
    basis = torch.linalg.qr(
        torch.cat([mean[:, None] / mean.norm(), residual_basis[:, : output_rank - 1]], 1)
    ).Q
    fit_x = torch.cat([x, global_x], 0)
    x_mean = fit_x.mean(0)
    _, _, input_basis = torch.pca_lowrank(fit_x - x_mean, q=input_rank, center=False, niter=3)
    z = (x - x_mean) @ input_basis
    scale = ((fit_x - x_mean) @ input_basis).std(0, unbiased=False).clamp_min(1e-6)
    z = z / scale
    global_z = ((global_x - x_mean) @ input_basis) / scale
    z = torch.cat([z, global_z, torch.ones((len(z), 1), device=device)], 1)
    coefficients = y @ basis
    penalty = torch.eye(z.shape[1], device=device) * ridge
    penalty[-1, -1] = 0
    zd, yd = z.double(), coefficients.double()
    weights = torch.linalg.solve(zd.T @ zd + penalty.double(), zd.T @ yd).float()
    arrays = {
        "x_mean": x_mean,
        "input_basis": input_basis,
        "scale": scale,
        "basis": basis,
        "weights": weights,
        "mean": mean,
        "train_z": z,
        "train_coefficients": coefficients,
        "ridge_lambda": torch.tensor(ridge, device=device),
    }
    return {name: value.detach().cpu().numpy() for name, value in arrays.items()}


def predict(hidden, arrays, rank):
    """No teacher, case ID, labels, or stored example-specific deltas are accepted."""
    import torch

    def tensor(name):
        return torch.as_tensor(arrays[name], device=hidden.device, dtype=hidden.dtype)

    z = ((hidden - tensor("x_mean")) @ tensor("input_basis")) / tensor("scale")
    z = torch.cat([z, z[-1:].expand_as(z), torch.ones_like(z[..., :1])], -1)
    return (z @ tensor("weights")[:, :rank]) @ tensor("basis")[:, :rank].T


def replay_fit(arrays):
    """Reproduce saved ridge weights from projected TRAIN inputs and targets."""
    z, y = arrays["train_z"].astype(np.float64), arrays["train_coefficients"].astype(np.float64)
    penalty = np.eye(z.shape[1]) * float(arrays["ridge_lambda"])
    penalty[-1, -1] = 0
    return np.linalg.solve(z.T @ z + penalty, z.T @ y)
