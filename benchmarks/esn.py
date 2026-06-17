"""
JAX-accelerated ESN for fair benchmarking against SAS.

Reuses reservoirpy's weight initialization (Win, W, bias) so that
both models start from the same random matrices for a given seed.
The forward pass runs via lax.scan + JIT — the same execution model
used by SAS — so timing comparisons are apples-to-apples.

Usage
-----
    from esn import JaxESN
    model = JaxESN(units=100, lr=1.0, sr=0.9, seed=42)
    states = model.run(X)   # (T, N) — updates internal state
"""

from __future__ import annotations

import functools
import sys
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
from reservoirpy.nodes import Reservoir as _RpyReservoir

# Ridge utilities — imported from saspy when available.
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


# ─────────────────────────────────────────────────────────────────────────────
# JIT-compiled scan kernel (module-level so the cache is shared across instances)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _esn_scan(
    W:    jnp.ndarray,   # (N, N)
    Win:  jnp.ndarray,   # (N, d)
    bias: jnp.ndarray,   # (N,)
    lr:   jnp.ndarray,   # scalar float32
    s0:   jnp.ndarray,   # (N,)
    X:    jnp.ndarray,   # (T, d)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Leaky-integrator ESN scan:
        x[t] = (1 - lr) * x[t-1] + lr * tanh(W @ x[t-1] + Win @ u[t] + bias)

    Returns
    -------
    all_states : (T, N)  — state after each input step
    last_s     : (N,)    — final state
    """
    def step(s, u):
        s_new = (1.0 - lr) * s + lr * jnp.tanh(W @ s + Win @ u + bias)
        return s_new, s_new

    last_s, all_states = jax.lax.scan(step, s0, X)
    return all_states, last_s


# ─────────────────────────────────────────────────────────────────────────────
# JaxESN
# ─────────────────────────────────────────────────────────────────────────────

class JaxESN:
    """
    ESN with reservoirpy-compatible weight initialization and JAX lax.scan forward pass.

    Parameters
    ----------
    units : int
        Number of reservoir neurons.
    lr : float, default 1.0
        Leaky rate.
    sr : float, default 0.9
        Spectral radius of the recurrent weight matrix.
    seed : int, default 0
        RNG seed forwarded to reservoirpy so weights are identical to an
        equivalent ``reservoirpy.nodes.Reservoir`` with the same seed.
    **kwargs
        Extra keyword arguments forwarded to reservoirpy.nodes.Reservoir
        (e.g. input_connectivity, rc_connectivity, input_scaling).
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

    # ------------------------------------------------------------------
    # Weight extraction
    # ------------------------------------------------------------------

    def _init_weights(self, input_dim: int) -> None:
        """
        Trigger reservoirpy's weight initialization, then copy the resulting
        matrices into JAX arrays.  Only called once (on the first run() call).
        """
        dummy = np.zeros((1, input_dim), dtype=np.float64)
        if not self._rpy.initialized:
            self._rpy.initialize(dummy)

        # reservoirpy stores W/Win as scipy sparse CSR arrays when connectivity < 1.
        # Convert to dense before handing off to JAX.
        W   = self._rpy.W
        Win = self._rpy.Win
        if hasattr(W,   "toarray"): W   = W.toarray()
        if hasattr(Win, "toarray"): Win = Win.toarray()
        W   = np.asarray(W,   dtype=np.float32)   # (N, N)
        Win = np.asarray(Win, dtype=np.float32)   # (N, d)

        bias = self._rpy.bias
        if not hasattr(bias, "__len__"):
            # scalar default (0.0) — represent as a zero vector
            bias = np.zeros(self.units, dtype=np.float32)
        else:
            bias = np.asarray(bias, dtype=np.float32).ravel()  # (N,)

        self._W     = jnp.array(W)
        self._Win   = jnp.array(Win)
        self._bias  = jnp.array(bias)
        self._lr_jx = jnp.float32(self.lr)
        self._state = jnp.zeros(self.units, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset internal state to zeros."""
        self._state = jnp.zeros(self.units, dtype=jnp.float32)

    def run(self, X: np.ndarray) -> np.ndarray:
        """
        Run the ESN on input sequence X via lax.scan + JIT, updating
        internal state so that successive calls continue from where
        the previous one left off (same semantics as reservoirpy).

        Parameters
        ----------
        X : array-like of shape (T, d), float32

        Returns
        -------
        states : np.ndarray of shape (T, N)
            Reservoir state after processing each input step.
        """
        X_f32 = np.asarray(X, dtype=np.float32)

        if self._state is None:
            self._init_weights(X_f32.shape[1])

        X_jx = jnp.array(X_f32)
        all_states, last_s = _esn_scan(
            self._W, self._Win, self._bias, self._lr_jx, self._state, X_jx
        )
        self._state = last_s
        # np.asarray forces JAX to materialise the array, ensuring the
        # computation is complete before run() returns (accurate timing).
        return np.asarray(all_states)


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive rollout kernel (module-level JIT, shared across instances)
# ─────────────────────────────────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=('n_steps',))
def _esn_autoreg_rollout(
    W_r:    jnp.ndarray,  # (N, N)
    Win:    jnp.ndarray,  # (N, d)
    bias:   jnp.ndarray,  # (N,)
    lr:     jnp.ndarray,  # scalar float32
    W_out:  jnp.ndarray,  # (N,) or (N, D)
    s0:     jnp.ndarray,  # (N,)
    n_steps: int,
) -> jnp.ndarray:
    """
    Advance the ESN reservoir n_steps autoregressively.

    At each step the readout W_out maps the current state to a predicted
    (z-scored) output, which is clipped and fed back as the next input.
    Non-destructive: only the final state is returned.

    lax.scan with n_steps=0 returns s0 unchanged — correct for h=1 prediction.
    """
    def body(s, _):
        y_z   = s @ W_out                              # () or (D,)
        u     = jnp.clip(jnp.atleast_1d(y_z), -10.0, 10.0).astype(jnp.float32)
        s_new = (1.0 - lr) * s + lr * jnp.tanh(W_r @ s + Win @ u + bias)
        return s_new, None
    s_final, _ = jax.lax.scan(body, s0, None, length=n_steps)
    return s_final


# ─────────────────────────────────────────────────────────────────────────────
# JaxESNForecaster
# ─────────────────────────────────────────────────────────────────────────────

class JaxESNForecaster:
    """
    Fit/predict wrapper around JaxESN, mirroring SASForecaster's interface.

    Two forecast modes
    ------------------
    direct  (default)
        Fit one ridge readout W[h] per requested horizon.
        predict(h) = s_last @ W[h] — a single linear readout, no reservoir advance.
        This is how ESNs are almost never used in the literature (most papers
        only report the autoreg / closed-loop regime) — but it makes the
        direct comparison with SAS's direct mode possible.

    autoreg
        Fit only W[1] (one-step readout).  predict(h) advances the reservoir
        h-1 steps by feeding its own predictions back, then reads out once.
        Non-destructive: reservoir state is unchanged after predict().

    Parameters
    ----------
    esn        : JaxESN instance (un-run; fit() will reset and run it).
    washout    : steps discarded before ridge regression.
    mode       : 'direct' | 'autoreg'
    n_cv_folds : rolling-window CV folds for ridge alpha selection.
    alphas     : ridge penalty candidates.
    """

    def __init__(
        self,
        esn:        JaxESN,
        washout:    int  = 50,
        mode:       str  = 'direct',
        n_cv_folds: int  = 5,
        alphas:     list | None = None,
    ) -> None:
        if mode not in ('direct', 'autoreg'):
            raise ValueError(f"mode must be 'direct' or 'autoreg', got {mode!r}")
        if _ridge_cv_select is None:
            raise ImportError(
                "JaxESNForecaster requires saspy.ridge. "
                "Make sure saspy is on the Python path."
            )
        self._esn       = esn
        self.washout    = washout
        self.mode       = mode
        self.n_cv_folds = n_cv_folds
        self.alphas     = list(alphas) if alphas is not None else _DEFAULT_ALPHAS

        self._W:     dict[int, np.ndarray] = {}
        self._mu:    np.ndarray | None = None   # (D,) per-channel mean
        self._sigma: np.ndarray | None = None   # (D,) per-channel std

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X_tr: np.ndarray, horizons: list[int]) -> 'JaxESNForecaster':
        """
        Run the reservoir on X_tr and fit ridge readout(s).

        Parameters
        ----------
        X_tr     : (T, D) training data — all D channels, any scale.
                   Column 0 is the nominal forecast target.
                   The forecaster z-scores each channel internally.
        horizons : forecast horizons to fit.
                   In 'autoreg' mode only h=1 is fitted regardless.

        Notes
        -----
        After fit(), the ESN state equals the state after the last training
        step.  Successive update() / predict() calls continue from there.
        """
        X_tr = np.asarray(X_tr, dtype=np.float64)
        if X_tr.ndim == 1:
            X_tr = X_tr[:, None]
        T, D = X_tr.shape

        # z-score each channel independently
        self._mu    = X_tr.mean(axis=0)                    # (D,)
        self._sigma = np.maximum(X_tr.std(axis=0), 1e-8)  # (D,)
        X_z = ((X_tr - self._mu) / self._sigma).astype(np.float32)

        # Run reservoir on X_z[0..T-2]; states[t] = state after X_z[t].
        # If weights aren't initialised yet, set _state=None so run() triggers
        # _init_weights() (which also zeros the state).  Otherwise just zero it.
        if self._esn._W is None:
            self._esn._state = None
        else:
            self._esn._state = jnp.zeros(self._esn.units, dtype=jnp.float32)
        states = self._esn.run(X_z[:-1])   # (T-1, N)

        # Advance one more step so state = state after X_z[T-1]
        self._esn.run(X_z[-1:])

        # Fit readouts:
        #   states[t] predicts X_z[t+h]
        #   valid t: washout <= t <= T-1-h  →  S = states[wo : T-h], Y = X_z[wo+h : T]
        wo = self.washout
        N  = self._esn.units
        self._W = {}

        fit_horizons = [1] if self.mode == 'autoreg' else horizons
        for h in fit_horizons:
            S = states[wo : T - h].astype(np.float64)   # (T-h-wo, N)
            Y = X_z  [wo + h : T ].astype(np.float64)   # (T-h-wo, D)
            if len(S) < 5:
                self._W[h] = np.zeros((N, D), dtype=np.float32)
                continue
            alpha      = _ridge_cv_select(S, Y, self.n_cv_folds, self.alphas)
            self._W[h] = _ridge_fit(S, Y, alpha).astype(np.float32)

        return self

    # ------------------------------------------------------------------
    # Predict / update
    # ------------------------------------------------------------------

    def predict(self, h: int) -> np.ndarray:
        """
        h-step forecast, returned in the original (un-z-scored) scale as (D,).

        Non-destructive: reservoir state is unchanged after this call.

        direct mode  : s_last @ W[h] — single linear readout.
        autoreg mode : advance h-1 steps via lax.scan (non-destructive),
                       then read out once. lax.scan with length=0 (h=1) is
                       a no-op, so predict(1) == s_last @ W[1].
        """
        if not self._W:
            raise RuntimeError("JaxESNForecaster must be fit before predict().")

        s = np.asarray(self._esn._state, dtype=np.float32)

        if self.mode == 'direct':
            if h not in self._W:
                raise KeyError(
                    f"Horizon {h} not trained. Available: {sorted(self._W)}"
                )
            y_z = s.astype(np.float64) @ self._W[h]       # (D,)

        else:  # autoreg
            if 1 not in self._W:
                raise RuntimeError(
                    "autoreg mode requires W[1]; call fit() first."
                )
            W1     = jnp.array(self._W[1], dtype=jnp.float32)  # (N, D)
            s0     = jnp.array(s, dtype=jnp.float32)           # (N,)
            # h-1 autoreg advances; length=0 (h=1) returns s0 unchanged.
            s_prev = _esn_autoreg_rollout(
                self._esn._W, self._esn._Win, self._esn._bias,
                self._esn._lr_jx, W1, s0, h - 1,
            )
            y_z = (np.asarray(s_prev, dtype=np.float64)
                   @ np.asarray(self._W[1], dtype=np.float64))  # (D,)

        return y_z * self._sigma + self._mu                      # (D,) un-z-scored

    def update(self, x: np.ndarray) -> 'JaxESNForecaster':
        """
        Ingest one new observation and advance the reservoir.

        x : float or (D,) array in the original (un-z-scored) scale.
        """
        if self._mu is None:
            raise RuntimeError("JaxESNForecaster must be fit before update().")
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        x_z   = ((x_arr - self._mu) / self._sigma).astype(np.float32)
        self._esn.run(x_z[None, :])
        return self
