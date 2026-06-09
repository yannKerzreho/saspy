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

import numpy as np
import jax
import jax.numpy as jnp
from reservoirpy.nodes import Reservoir as _RpyReservoir


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
