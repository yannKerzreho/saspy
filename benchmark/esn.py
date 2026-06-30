"""JAX-accelerated ESN with reservoirpy weight initialisation and lax.scan forward pass."""

from __future__ import annotations

import functools
import sys
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
from reservoirpy.nodes import Reservoir as _RpyReservoir

try:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from saspy.ridge import (
        ridge_cv_select as _ridge_cv_select,
        ridge_fit       as _ridge_fit,
        ALPHAS          as _DEFAULT_ALPHAS,
    )
except ImportError:
    _ridge_cv_select = None   # type: ignore
    _ridge_fit       = None   # type: ignore
    _DEFAULT_ALPHAS  = [10 ** x for x in np.arange(-6, 6.01, 0.25)]


@jax.jit
def _esn_scan(W, Win, bias, lr, s0, X):
    """Leaky-integrator ESN scan → (all_states (T, N), last_s (N,))."""
    def step(s, u):
        s_new = (1.0 - lr) * s + lr * jnp.tanh(W @ s + Win @ u + bias)
        return s_new, s_new
    last_s, all_states = jax.lax.scan(step, s0, X)
    return all_states, last_s

class JaxESN:
    """
    ESN with reservoirpy-compatible weight initialisation and JAX lax.scan forward pass.

    Parameters
    ----------
    units  : number of reservoir neurons.
    lr     : leaky rate (default 1.0).
    sr     : spectral radius (default 0.9).
    seed   : RNG seed forwarded to reservoirpy.
    kwargs : extra args forwarded to reservoirpy.nodes.Reservoir.
    """

    def __init__(
        self,
        units: int,
        lr: float = 1.0,
        sr: float = 0.9,
        seed: int = 0,
        **kwargs,
    ) -> None:
        self.units = units
        self.lr    = float(lr)
        self.sr    = float(sr)

        # reservoirpy node — used only for weight initialization
        self._rpy = _RpyReservoir(units, lr=lr, sr=sr, seed=seed, **kwargs)

        self._W:      jnp.ndarray | None = None
        self._Win:    jnp.ndarray | None = None
        self._bias:   jnp.ndarray | None = None
        self._lr_jx:  jnp.ndarray | None = None
        self._state:  jnp.ndarray | None = None  # persists across run() calls

    def _init_weights(self, input_dim: int) -> None:
        dummy = np.zeros((1, input_dim), dtype=np.float64)
        if not self._rpy.initialized:
            self._rpy.initialize(dummy)

        W   = self._rpy.W
        Win = self._rpy.Win
        if hasattr(W,   "toarray"): W   = W.toarray()
        if hasattr(Win, "toarray"): Win = Win.toarray()
        W   = np.asarray(W,   dtype=np.float32)
        Win = np.asarray(Win, dtype=np.float32)

        bias = self._rpy.bias
        if not hasattr(bias, "__len__"):
            bias = np.zeros(self.units, dtype=np.float32)
        else:
            bias = np.asarray(bias, dtype=np.float32).ravel()

        self._W     = jnp.array(W)
        self._Win   = jnp.array(Win)
        self._bias  = jnp.array(bias)
        self._lr_jx = jnp.float32(self.lr)
        self._state = jnp.zeros(self.units, dtype=jnp.float32)

    def reset_state(self) -> None:
        """Reset reservoir state to zeros."""
        self._state = jnp.zeros(self.units, dtype=jnp.float32)

    def run(self, X: np.ndarray) -> np.ndarray:
        """Run on X (T, d) → states (T, N).  State persists across calls."""
        X_f32 = np.asarray(X, dtype=np.float32)
        if self._state is None:
            self._init_weights(X_f32.shape[1])
        X_jx = jnp.array(X_f32)
        all_states, last_s = _esn_scan(
            self._W, self._Win, self._bias, self._lr_jx, self._state, X_jx
        )
        self._state = last_s
        return np.asarray(all_states)


@functools.partial(jax.jit, static_argnames=('n_steps',))
def _esn_autoreg_rollout(W_r, Win, bias, lr, W_out, s0, n_steps: int) -> jnp.ndarray:
    """Advance the reservoir n_steps autoregressively; non-destructive."""
    def body(s, _):
        y_z   = s @ W_out
        u     = jnp.clip(jnp.atleast_1d(y_z), -1.0, 1.0).astype(jnp.float32)
        s_new = (1.0 - lr) * s + lr * jnp.tanh(W_r @ s + Win @ u + bias)
        return s_new, None
    s_final, _ = jax.lax.scan(body, s0, None, length=n_steps)
    return s_final

class JaxESNForecaster:
    """
    Fit/predict wrapper around JaxESN, mirroring SASForecaster's interface.

    mode='direct'  — fit one readout W[h] per horizon; predict(h) = s_last @ W[h].
    mode='autoreg' — fit W[1] only; predict(h) advances h-1 steps autoregressively.

    Parameters
    ----------
    esn        : JaxESN instance.
    washout    : steps discarded before ridge regression.
    mode       : 'direct' | 'autoreg'
    n_cv_folds : rolling-window CV folds for alpha selection.
    alphas     : ridge penalty candidates.
    """

    def __init__(
        self,
        esn:         JaxESN,
        washout:     int  = 50,
        mode:        str  = 'direct',
        n_cv_folds:  int  = 5,
        alphas:      list | None = None,
        clip_output: bool = True,
    ) -> None:
        if mode not in ('direct', 'autoreg'):
            raise ValueError(f"mode must be 'direct' or 'autoreg', got {mode!r}")
        if _ridge_cv_select is None:
            raise ImportError(
                "JaxESNForecaster requires saspy.ridge. "
                "Make sure saspy is on the Python path."
            )
        self._esn        = esn
        self.washout     = washout
        self.mode        = mode
        self.n_cv_folds  = n_cv_folds
        self.alphas      = list(alphas) if alphas is not None else _DEFAULT_ALPHAS
        # No z-scoring: the data is assumed already in the reservoir's [-1, 1]
        # domain (load_dgp normalises).  clip_output clamps predictions there.
        self.clip_output = clip_output

        self._W: dict[int, np.ndarray] = {}

    def fit(self, X_tr: np.ndarray, horizons: list[int]) -> 'JaxESNForecaster':
        """Run the reservoir on X_tr (raw, no z-scoring) and fit ridge readout(s).

        X_tr : (T, D) training data, assumed in [-1, 1].  'autoreg' fits only h=1.
        """
        X_tr = np.asarray(X_tr, dtype=np.float64)
        if X_tr.ndim == 1:
            X_tr = X_tr[:, None]
        T, D = X_tr.shape

        if self._esn._W is None:
            self._esn._state = None
        else:
            self._esn._state = jnp.zeros(self._esn.units, dtype=jnp.float32)
        states = self._esn.run(X_tr[:-1])   # (T-1, N)
        self._esn.run(X_tr[-1:])            # advance to state after the last input

        wo = self.washout
        N  = self._esn.units
        self._W = {}

        fit_horizons = [1] if self.mode == 'autoreg' else horizons
        for h in fit_horizons:
            S = states[wo: T - h].astype(np.float64)
            Y = X_tr  [wo + h: T].astype(np.float64)
            if len(S) < 5:
                self._W[h] = np.zeros((N, D), dtype=np.float32)
                continue
            alpha      = _ridge_cv_select(S, Y, self.n_cv_folds, self.alphas)
            self._W[h] = _ridge_fit(S, Y, alpha).astype(np.float32)

        return self

    def predict(self, h: int) -> np.ndarray:
        """h-step forecast (D,), in the [-1, 1] data scale.  Non-destructive."""
        if not self._W:
            raise RuntimeError("JaxESNForecaster must be fit before predict().")

        s = np.asarray(self._esn._state, dtype=np.float32)

        if self.mode == 'direct':
            if h not in self._W:
                raise KeyError(f"Horizon {h} not trained. Available: {sorted(self._W)}")
            y = s.astype(np.float64) @ self._W[h]
        else:
            if 1 not in self._W:
                raise RuntimeError("autoreg mode requires W[1]; call fit() first.")
            W1     = jnp.array(self._W[1], dtype=jnp.float32)
            s0     = jnp.array(s, dtype=jnp.float32)
            s_prev = _esn_autoreg_rollout(
                self._esn._W, self._esn._Win, self._esn._bias,
                self._esn._lr_jx, W1, s0, h - 1,
            )
            y = (np.asarray(s_prev, dtype=np.float64)
                   @ np.asarray(self._W[1], dtype=np.float64))

        if self.clip_output:
            y = np.clip(y, -1.0, 1.0)
        return y

    def update(self, x: np.ndarray) -> 'JaxESNForecaster':
        """Ingest one new observation ([-1, 1] scale) and advance the reservoir."""
        if not self._W:
            raise RuntimeError("JaxESNForecaster must be fit before update().")
        x_arr = np.asarray(x, dtype=np.float32).ravel()
        self._esn.run(x_arr[None, :])
        return self
