"""Scalar-per-unit diagonal polynomial basis (n_drivers = N).

Each unit i has its own scalar driver z_tilde[i] and independent P/Q weights:

    A_t[i] = Σ_d P_weights[d, i] · z_tilde[i]^d   (clipped to (−1, 1))
    q_t[i] = Σ_d Q_weights[d, i] · z_tilde[i]^d
    s_t[i] = A_t[i] · s_{t-1}[i] + q_t[i]
"""

import jax
import jax.numpy as jnp

from .base import BaseBasis
from .q_init import q_degree_correction


@jax.tree_util.register_pytree_node_class
class DiagonalPoly(BaseBasis):
    """
    Diagonal polynomial basis.

    Parameters
    ----------
    n            : reservoir size N (= n_drivers for this basis).
    p_degree     : degree of z_tilde in the transition polynomial.
    q_degree     : degree of z_tilde in the input-drive polynomial.
    spectral_norm: base eigenvalue range — P_weights[0, i] ∈ (−sn, sn).
    max_input    : clip |z_tilde[i]| to this before polynomial evaluation.
                   Set to None to disable clipping when data lives on a compact
                   attractor and the forecaster handles scaling externally.
    taylor_decay : geometric per-degree shrinkage for Q, ∈ [0, 1].
    """

    def __init__(
        self,
        n:             int,
        p_degree:      int        = 1,
        q_degree:      int        = 1,
        spectral_norm: float      = 0.9,
        max_input:     float|None = 4.0,
        taylor_decay:  float      = 1.0,
        budget_ref:    float|None = None,
    ):
        super().__init__(p_degree, q_degree)
        self._n            = n
        self.spectral_norm = float(spectral_norm)
        self.max_input     = float(max_input) if max_input is not None else None
        self.taylor_decay  = float(taylor_decay)
        self.budget_ref    = float(budget_ref) if budget_ref is not None else None

    # ── dimensions ──────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return self._n

    @property
    def n_drivers(self) -> int:
        return self._n      # each unit is its own driver

    # ── pytree ───────────────────────────────────────────────────────────────

    def tree_flatten(self):
        return (self.P_weights, self.Q_weights), (
            self._n, self.p_degree, self.q_degree,
            self.spectral_norm, self.max_input, self.taylor_decay,
            self.budget_ref,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.P_weights, obj.Q_weights = children
        return obj

    # ── factory ─────────────────────────────────────────────────────────────

    def initialize(self, key) -> "DiagonalPoly":
        N  = self._n
        sn = self.spectral_norm

        n_keys = self.p_degree + self.q_degree + 2
        keys   = jax.random.split(key, n_keys)
        ki     = 0

        p0 = (jax.random.uniform(keys[ki], (N,)) * 2 - 1) * sn
        ki += 1
        p_rows = [p0]

        # Per-degree Volterra budget: ‖P_d‖ · scale_ref^d ≤ headroom / 2^d
        headroom  = jnp.maximum(1.0 - jnp.abs(p0) - 0.01, 0.0)
        budget    = headroom * 0.5
        scale_ref = self._budget_ref()
        for k in range(1, self.p_degree + 1):
            scale = (budget / (2.0 ** k)) / (scale_ref ** k)
            raw   = jax.random.normal(keys[ki], (N,))
            p_rows.append(jnp.clip(raw, -1.0, 1.0) * scale)
            ki += 1
        P = jnp.stack(p_rows, axis=0)

        gamma = jnp.sqrt(1.0 - p0 ** 2)
        dc    = q_degree_correction(self.q_degree, self.taylor_decay)
        Q_raw = jax.random.normal(keys[ki], (self.q_degree + 1, N))
        Q     = Q_raw * dc[:, None] * gamma[None, :]

        obj = DiagonalPoly(N, self.p_degree, self.q_degree,
                           self.spectral_norm, self.max_input, self.taylor_decay,
                           self.budget_ref)
        obj.P_weights, obj.Q_weights = P, Q
        return obj

    # ── per-step evaluators ──────────────────────────────────────────────────

    def eval_p(self, z_tilde_t):
        """z_tilde_t: (N,) → a: (N,) eigenvalues, clipped to (−0.9999, 0.9999)."""
        z = (jnp.clip(z_tilde_t, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde_t)
        powers = jnp.arange(self.p_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[None, :], powers[:, None])          # (p+1, N)
        return jnp.clip(
            jnp.einsum('dn,dn->n', feats, self.P_weights),
            -0.9999, 0.9999,
        )

    def eval_q(self, z_tilde_t):
        """z_tilde_t: (N,) → q: (N,)."""
        z = (jnp.clip(z_tilde_t, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde_t)
        powers = jnp.arange(self.q_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[None, :], powers[:, None])          # (q+1, N)
        return jnp.einsum('dn,dn->n', feats, self.Q_weights)

    # ── batched evaluators (full sequence, no vmap overhead) ─────────────────

    def batch_eval_p(self, z_tilde):
        """z_tilde: (T, N) → (T, N)."""
        z = (jnp.clip(z_tilde, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde)
        powers = jnp.arange(self.p_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(
            z[:, None, :],           # (T, 1, N)
            powers[None, :, None],   # (1, p+1, 1)
        )                            # (T, p+1, N)
        return jnp.clip(
            jnp.einsum('tdn,dn->tn', feats, self.P_weights),
            -0.9999, 0.9999,
        )

    def batch_eval_q(self, z_tilde):
        """z_tilde: (T, N) → (T, N)."""
        z = (jnp.clip(z_tilde, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde)
        powers = jnp.arange(self.q_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[:, None, :], powers[None, :, None])  # (T, q+1, N)
        return jnp.einsum('tdn,dn->tn', feats, self.Q_weights)

    # ── algebraic primitives ─────────────────────────────────────────────────

    def apply(self, a, s):
        """a: (N,), s: (N,) → (N,)  element-wise multiplication."""
        return a * s

    def combine(self, i, j):
        """Element-wise monoid.  i=(a_i, b_i), j=(a_j, b_j)."""
        a_i, b_i = i
        a_j, b_j = j
        return a_j * a_i, a_j * b_i + b_j
