"""Ridge regression utilities.

ridge_cv_select   : rolling-window CV → best alpha.
ridge_fit         : fit with a fixed alpha.
ridge_cv          : select + fit in one call.
ridge_cv_grouped  : block-diagonal CV (one alpha per feature group).
ridge_fit_grouped : block-diagonal ridge fit.
"""

from __future__ import annotations

import numpy as np
from itertools import product as _cart_product


# ── alpha grids ───────────────────────────────────────────────────────────────

# 37 log-spaced values from 1e-4 to 1e5 (spacing 0.25 in log10).
# 1e-6 … 1e6, step 0.25 in log10 (53 values).
# Wide enough that real problems should never hit either boundary.
ALPHAS: list[float] = [10 ** x for x in np.arange(-6, 6.01, 0.25)]

# Coarser grid for grouped CV: every 6th value → 9 candidates.
ALPHAS_GROUPED: list[float] = ALPHAS[::6]


# ══════════════════════════════════════════════════════════════════════════════
# Core helpers
# ══════════════════════════════════════════════════════════════════════════════

def ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    penalty_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Ridge regression: (X.T X + α R)⁻¹ X.T y.

    y may be 1-D (T,) → returns weight vector (n,), or
    2-D (T, D) → returns weight matrix (n, D) via multi-RHS solve.

    penalty_mask : (n,) per-feature multipliers for α.
                   None → standard ridge (R = I).

    X and y are cast to float64: float32 Gram matrices develop negative
    eigenvalues for large-amplitude reservoir states, corrupting the solve.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = X.shape[1]
    R = (np.diag(penalty_mask.astype(np.float64))
         if penalty_mask is not None else np.eye(n, dtype=np.float64))
    return np.linalg.solve(X.T @ X + alpha * R, X.T @ y)


def ridge_cv_select(
    X:            np.ndarray,
    y:            np.ndarray,
    n_folds:      int = 5,
    alphas:       list | None = None,
    penalty_mask: np.ndarray | None = None,
) -> float:
    """Rolling-window CV to select the best ridge alpha.

    y may be 1-D (T,) or 2-D (T, D).
    """
    if alphas is None:
        alphas = ALPHAS
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    T, n     = X.shape
    val_size = max(12, T // (n_folds + 1))
    diag_ix  = np.diag_indices(n)

    uniform = penalty_mask is None
    if not uniform:
        pen_diag = np.asarray(penalty_mask, dtype=np.float64)

    fold_cache: list = []
    for fold in range(n_folds):
        cut = T - (n_folds - fold) * val_size
        if cut < n + 5:
            continue
        XTX  = X[:cut].T @ X[:cut]
        XTy  = X[:cut].T @ y[:cut]
        X_val = X[cut: cut + val_size]
        y_val = y[cut: cut + val_size]

        if uniform:
            lam, V = np.linalg.eigh(XTX)
            lam    = np.maximum(lam, 0.0)
            VTXTy  = V.T @ XTy
            XV     = X_val @ V
            fold_cache.append((lam, VTXTy, XV, y_val))
        else:
            xtx_diag = XTX[diag_ix].copy()
            fold_cache.append((XTX, XTy, xtx_diag, X_val, y_val))

    if not fold_cache:
        return alphas[0]

    best, best_mse = alphas[0], np.inf

    if uniform:
        for alpha in alphas:
            total = 0.0
            for lam, VTXTy, XV, y_val in fold_cache:
                # Transpose trick: (lam+α) is (N,); divide from the right so
                # it broadcasts over the N axis for both 1-D and 2-D VTXTy.
                coef  = (VTXTy.T / (lam + alpha)).T
                err   = XV @ coef - y_val
                total += float(np.mean(err ** 2))
            mse = total / len(fold_cache)
            if mse < best_mse:
                best_mse, best = mse, alpha
    else:
        for alpha in alphas:
            total = 0.0
            for XTX, XTy, xtx_diag, X_val, y_val in fold_cache:
                A          = XTX.copy()
                A[diag_ix] = xtx_diag + alpha * pen_diag
                w          = np.linalg.solve(A, XTy)
                err        = X_val @ w - y_val
                total     += float(np.mean(err ** 2))
            mse = total / len(fold_cache)
            if mse < best_mse:
                best_mse, best = mse, alpha

    if best == alphas[-1]:
        import warnings
        warnings.warn(
            f"Ridge alpha={best:.2e} hit upper grid boundary "
            f"[{alphas[0]:.2e}, {alphas[-1]:.2e}]. "
            "Consider extending ALPHAS.", stacklevel=2)
    return best


def ridge_cv(
    X:       np.ndarray,
    y:       np.ndarray,
    n_folds: int = 5,
    alphas:  list | None = None,
) -> np.ndarray:
    """Select alpha via CV then return fitted weights (standard ridge, no mask)."""
    alpha = ridge_cv_select(X, y, n_folds, alphas)
    return ridge_fit(X, y, alpha)


# ══════════════════════════════════════════════════════════════════════════════
# Grouped ridge (block-diagonal penalty, one alpha per feature group)
# ══════════════════════════════════════════════════════════════════════════════

def ridge_cv_grouped(
    S:             np.ndarray,
    Y:             np.ndarray,
    group_sizes:   list[int],
    n_folds:       int,
    alpha_grid_1d: list[float],
) -> tuple:
    """
    Block-diagonal ridge CV: a separate alpha for each feature group.

    The penalty matrix is Λ = diag(α₀·I_{g₀}, α₁·I_{g₁}, …).
    """
    S = np.asarray(S, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    T, n     = S.shape
    val_size = max(12, T // (n_folds + 1))
    combos   = list(_cart_product(alpha_grid_1d, repeat=len(group_sizes)))
    diag_ix  = np.diag_indices(n)

    lam_vecs = [
        np.concatenate([a * np.ones(g) for a, g in zip(combo, group_sizes)])
        for combo in combos
    ]

    sum_mse = np.zeros(len(combos))
    cnt_mse = np.zeros(len(combos), dtype=int)

    for fold in range(n_folds):
        cut = T - (n_folds - fold) * val_size
        if cut < max(5, n // 10):
            continue
        STS      = S[:cut].T @ S[:cut]
        STY      = S[:cut].T @ Y[:cut]
        val_S    = S[cut: cut + val_size]
        val_Y    = Y[cut: cut + val_size]
        sts_diag = STS[diag_ix].copy()

        for ci, lam in enumerate(lam_vecs):
            A            = STS.copy()
            A[diag_ix]   = sts_diag + lam
            w            = np.linalg.solve(A, STY)
            err          = val_S @ w - val_Y
            sum_mse[ci] += float(np.mean(err ** 2))
            cnt_mse[ci] += 1

    valid = cnt_mse > 0
    if not valid.any():
        return combos[0]
    avg = np.where(valid, sum_mse / np.maximum(cnt_mse, 1), np.inf)
    return combos[int(np.argmin(avg))]


def ridge_fit_grouped(
    S:           np.ndarray,
    Y:           np.ndarray,
    group_sizes: list[int],
    alphas:      tuple,
) -> np.ndarray:
    """Block-diagonal ridge: (S.T S + Λ)⁻¹ S.T Y."""
    S = np.asarray(S, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n   = S.shape[1]
    lam = np.concatenate([a * np.ones(g) for a, g in zip(alphas, group_sizes)])
    A   = S.T @ S
    A[np.diag_indices(n)] += lam
    return np.linalg.solve(A, S.T @ Y)
