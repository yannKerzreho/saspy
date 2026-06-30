"""Shared evaluation utilities for saspy benchmarks."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

VPT_THRESHOLD = 0.4


def autonomous_nrmse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-step NRMSE normalised per channel → (T,).

    NRMSE(t) = sqrt( mean_D( ((pred_t,d - true_t,d) / sigma_d)^2 ) )
    """
    p     = np.asarray(preds,   dtype=np.float64)
    t     = np.asarray(targets, dtype=np.float64)
    sigma = np.maximum(t.std(axis=0), 1e-8)
    return np.sqrt(((p - t) / sigma) ** 2).mean(axis=1)


def compute_vpt(nrmse: np.ndarray, threshold: float = VPT_THRESHOLD) -> int:
    """First step where NRMSE >= threshold (NaN/Inf counts as crossing).

    Returns the full series length when the threshold is never crossed.
    """
    arr = np.asarray(nrmse, dtype=np.float64)
    bad = ~np.isfinite(arr) | (arr >= threshold)
    idx = np.where(bad)[0]
    return int(idx[0]) if len(idx) > 0 else int(len(arr))


def sliced_wasserstein(
    X: np.ndarray,
    Y: np.ndarray,
    n_proj: int = 200,
    seed: int = 0,
) -> float:
    """Sliced Wasserstein Distance between point clouds X and Y (T×D).

    Projects both clouds onto n_proj random unit vectors and averages the
    1-D Wasserstein-1 distance over projections.
    """
    rng    = np.random.default_rng(seed)
    X      = np.asarray(X, dtype=np.float64)
    Y      = np.asarray(Y, dtype=np.float64)
    thetas = rng.standard_normal((n_proj, X.shape[1]))
    thetas /= np.linalg.norm(thetas, axis=1, keepdims=True)
    px = np.sort(X @ thetas.T, axis=0)
    py = np.sort(Y @ thetas.T, axis=0)
    return float(np.abs(px - py).mean())


def save_state(fc) -> np.ndarray:
    """Save reservoir state from a SASForecaster or JaxESNForecaster."""
    if hasattr(fc, "_s_last"):
        return fc._s_last.copy()
    return np.asarray(fc._esn._state, dtype=np.float32).copy()


def restore_state(fc, state: np.ndarray) -> None:
    """Restore a previously saved reservoir state."""
    if hasattr(fc, "_s_last"):
        fc._s_last = state.copy()
    else:
        fc._esn._state = jnp.array(state)
