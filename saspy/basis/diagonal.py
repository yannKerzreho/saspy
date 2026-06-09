"""
diagonal.py — Scalar-per-unit diagonal polynomial basis.

Every reservoir unit i gets its own unique scalar driver z_tilde[i] from the
projector.  The recurrence for unit i is:

    s_t[i] = p(z_tilde_t[i])[i] · s_{t-1}[i] + q(z_tilde_t[i])[i]

where p and q are degree polynomials evaluated via a batched einsum:

    feats[d, i] = z_tilde_t[i]^d           shape (p_degree+1, N)
    A_t[i]      = Σ_d P_weights[d, i] · z_tilde_t[i]^d   shape (N,)  clipped to (−1,1)
    q_t[i]      = Σ_d Q_weights[d, i] · z_tilde_t[i]^d   shape (N,)

n_drivers = N  (projector maps d → N; each unit gets its own scalar feature)

Trivial projector (W_in = ones(1, N)) recovers the old "broadcast scalar"
behaviour: z_tilde_t[i] = z_t for every i, reproducing the classic SAS univariate
reservoir exactly.
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
    taylor_decay : geometric per-degree shrinkage for Q, ∈ [0, 1].
    """

    def __init__(
        self,
        n:             int,
        p_degree:      int   = 1,
        q_degree:      int   = 1,
        spectral_norm: float = 0.9,
        max_input:     float = 4.0,
        taylor_decay:  float = 1.0,
    ):
        super().__init__(p_degree, q_degree)
        self._n           = n
        self.spectral_norm = float(spectral_norm)
        self.max_input     = float(max_input)
        self.taylor_decay  = float(taylor_decay)

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

        # ── P_weights: (p_degree+1, N) ────────────────────────────────────
        # Degree-0: base eigenvalues uniformly in (−sn, sn)
        p0 = (jax.random.uniform(keys[ki], (N,)) * 2 - 1) * sn
        ki += 1
        p_rows = [p0]

        # Degrees 1+: Volterra modulation — Taylor-shrunk budget per degree
        headroom = jnp.maximum(1.0 - jnp.abs(p0) - 0.01, 0.0)
        budget   = headroom * 0.5
        for k in range(1, self.p_degree + 1):
            scale = (budget / (2.0 ** k)) / (self.max_input ** k)
            raw   = jax.random.normal(keys[ki], (N,))
            p_rows.append(jnp.clip(raw, -1.0, 1.0) * scale)
            ki += 1
        P = jnp.stack(p_rows, axis=0)          # (p_degree+1, N)

        # ── Q_weights: (q_degree+1, N) ────────────────────────────────────
        gamma = jnp.sqrt(1.0 - p0 ** 2)        # LRU-style input scaling (N,)
        dc    = q_degree_correction(self.q_degree, self.taylor_decay)  # (q+1,)
        Q_raw = jax.random.normal(keys[ki], (self.q_degree + 1, N))
        Q     = Q_raw * dc[:, None] * gamma[None, :]

        obj = DiagonalPoly(N, self.p_degree, self.q_degree,
                           self.spectral_norm, self.max_input, self.taylor_decay)
        obj.P_weights, obj.Q_weights = P, Q
        return obj

    # ── per-step evaluators ──────────────────────────────────────────────────

    def eval_p(self, z_tilde_t):
        """z_tilde_t: (N,) → a: (N,) eigenvalues, clipped to (−0.9999, 0.9999)."""
        z = jnp.clip(z_tilde_t, -self.max_input, self.max_input)
        powers = jnp.arange(self.p_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[None, :], powers[:, None])          # (p+1, N)
        return jnp.clip(
            jnp.einsum('dn,dn->n', feats, self.P_weights),
            -0.9999, 0.9999,
        )

    def eval_q(self, z_tilde_t):
        """z_tilde_t: (N,) → q: (N,)."""
        z = jnp.clip(z_tilde_t, -self.max_input, self.max_input)
        powers = jnp.arange(self.q_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[None, :], powers[:, None])          # (q+1, N)
        return jnp.einsum('dn,dn->n', feats, self.Q_weights)

    # ── batched evaluators (full sequence, no vmap overhead) ─────────────────

    def batch_eval_p(self, z_tilde):
        """z_tilde: (T, N) → (T, N)."""
        z = jnp.clip(z_tilde, -self.max_input, self.max_input)
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
        z = jnp.clip(z_tilde, -self.max_input, self.max_input)
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
