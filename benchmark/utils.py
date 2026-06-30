"""Shared evaluation utilities for saspy benchmarks."""

from __future__ import annotations

import warnings

import numpy as np
import jax.numpy as jnp

VPT_THRESHOLD = 0.4


def load_dgp(loader, n: int, *, channels=None, bound: float = 0.95, **loader_kwargs) -> np.ndarray:
    """Generate `n` steps from a reservoirpy DGP, min-max normalised to [-bound, bound].

    The reservoir lives on the compact domain [-1, 1]; normalising the DGP to its
    own per-channel range there lets forecasters run with ``scale_input=False``
    (no internal rescaling) — the single place the convention is applied.

    `bound` defaults to 0.95 (not 1.0) so the training range maps to [-0.95, 0.95]:
    this leaves head-room for an autonomous rollout (or test data) to wander a
    little past the train min/max without immediately saturating the [-1, 1] clip.

    Parameters
    ----------
    loader  : a ``reservoirpy.datasets`` attribute name (e.g. "lorenz") or any
              callable ``f(n, **kwargs)`` returning a (T,) or (T, D) array (or a
              ``(t, series)`` tuple, as some reservoirpy generators do).
    n       : number of timesteps to generate.
    channels: optional list of column indices to keep (after generation).
    bound   : target half-range; data is scaled into [-bound, bound].
    **loader_kwargs : forwarded to the loader (None values dropped).

    Returns
    -------
    (n, D) float64 array with every channel scaled into [-bound, bound].
    """
    import reservoirpy.datasets as rpy_datasets
    fn = getattr(rpy_datasets, loader) if isinstance(loader, str) else loader

    kw = {k: v for k, v in loader_kwargs.items() if v is not None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = fn(n, **kw)
    if isinstance(result, tuple):
        result = result[1]
    raw = np.asarray(result, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, None]
    if channels is not None:
        raw = raw[:, channels]

    lo, hi = raw.min(axis=0, keepdims=True), raw.max(axis=0, keepdims=True)
    center = (lo + hi) / 2.0
    half   = np.maximum((hi - lo) / 2.0, 1e-12)
    return bound * (raw - center) / half


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
