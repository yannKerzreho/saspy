"""
sparse_poly.py — Full sparse polynomial basis (ESN-inspired, Layer 2).

State update for each timestep t:

    phi(z_tilde_t) = [1, z_1^1,...,z_K^1, z_1^2,...,z_K^2,...,z_K^D]  shape (F,)
    A_t            = einsum('f,fij->ij', phi, P_weights)                 shape (N, N)
    q_t            = Q_weights @ phi                                      shape (N,)
    s_t            = A_t @ s_{t-1} + q_t

Features are degree-major: all K drivers at degree 1, then degree 2, …, degree D.
F = 1 + K·D  (degree-0 constant is always phi[0] = 1).

P_weights : (F, N, N)  F sparse N×N matrices, linearly combined by phi.
Q_weights : (N, F)     sparse input-to-state matrix.

Usage
-----
basis = SparsePolyBasis(n=500, n_drivers=d, degree=2)
model = SASModel(
    projector = InputProjector.identity(d),   # identity: z_tilde = z
    basis_p   = basis,
    basis_q   = basis,
)

training_mode
-------------
'parallel'   (default) — batch_eval_p materialises (T, N, N); the standard
             two-level associative scan in engine.py is used.  Requires
             roughly T × N² × 4 bytes of GPU VRAM (≈5 GB for T=5 000, N=500).
'sequential' — the forecaster routes to _stream_scan (jax.lax.scan step-by-step),
             which avoids materialising P_seq.  O(N²) memory instead of O(T·N²).
"""

import jax
import jax.numpy as jnp

from .base   import BaseBasis
from .q_init import q_degree_correction


@jax.tree_util.register_pytree_node_class
class SparsePolyBasis(BaseBasis):
    """
    Full N×N sparse polynomial basis.

    Parameters
    ----------
    n              : reservoir size N.
    n_drivers      : K — input dimension after projection.
                     Use InputProjector.identity(d) so z_tilde = z (no mixing).
    degree         : D — polynomial degree for both P and Q features.
    spectral_norm  : target spectral radius ρ for the base matrix M_0.
    density_P      : Bernoulli density of each M_f in P_weights (default 0.05).
    density_Q      : Bernoulli density of Q_weights (default 0.1).
    max_input      : clip |z_tilde[k]| before polynomial expansion.
    taylor_decay   : geometric per-degree taper for modulation matrices and Q ∈ [0, 1].
    training_mode  : 'parallel' | 'sequential'.
    """

    def __init__(
        self,
        n:             int,
        n_drivers:     int,
        degree:        int        = 2,
        spectral_norm: float      = 0.9,
        density_P:     float      = 0.05,
        density_Q:     float      = 0.1,
        max_input:     float|None = 4.0,
        taylor_decay:  float      = 1.0,
        training_mode: str        = 'parallel',
        budget_ref:    float|None = None,
    ):
        super().__init__(p_degree=degree, q_degree=degree)
        self._n            = n
        self._n_drivers    = n_drivers
        self.degree        = degree
        self.spectral_norm = float(spectral_norm)
        self.density_P     = float(density_P)
        self.density_Q     = float(density_Q)
        self.max_input     = float(max_input) if max_input is not None else None
        self.taylor_decay  = float(taylor_decay)
        self.training_mode = training_mode
        self.budget_ref    = float(budget_ref) if budget_ref is not None else None

    # ── dimensions ──────────────────────────────────────────────────────────

    @property
    def F(self) -> int:
        """Total number of polynomial features F = 1 + K·D."""
        return 1 + self._n_drivers * self.degree

    @property
    def n(self) -> int:
        return self._n

    @property
    def n_drivers(self) -> int:
        return self._n_drivers

    # ── pytree ───────────────────────────────────────────────────────────────

    def tree_flatten(self):
        return (self.P_weights, self.Q_weights), (
            self._n, self._n_drivers, self.degree,
            self.spectral_norm, self.density_P, self.density_Q,
            self.max_input, self.taylor_decay, self.training_mode,
            self.budget_ref,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        n, K, deg, sn, dP, dQ, mi, td, tm, br = aux
        obj = cls(n, K, deg, sn, dP, dQ, mi, td, tm, br)
        obj.P_weights, obj.Q_weights = children
        return obj

    # ── factory ─────────────────────────────────────────────────────────────

    def initialize(self, key) -> "SparsePolyBasis":
        N, K, D = self._n, self._n_drivers, self.degree
        F       = self.F
        sn      = self.spectral_norm
        mi      = self._budget_ref()

        key_P, key_masks_P, key_Q, key_mask_Q = jax.random.split(key, 4)
        keys_P      = jax.random.split(key_P,      F)
        keys_mask_P = jax.random.split(key_masks_P, F)

        # ── P_weights: (F, N, N) ─────────────────────────────────────────

        P_list = []

        # f=0: base autonomous matrix M_0 (phi[0] = 1 always)
        # Hybrid mask: diagonal base guarantees no dead rows, random overlay adds density.
        W = jax.random.normal(keys_P[0], (N, N))
        idx          = jnp.arange(N)
        mask_diag    = jnp.zeros((N, N)).at[idx, idx].set(1.0)
        mask_rand    = (jax.random.uniform(keys_mask_P[0], (N, N)) < self.density_P).astype(jnp.float32)
        W            = W * jnp.clip(mask_diag + mask_rand, 0.0, 1.0)
        rho          = jnp.max(jnp.abs(jnp.linalg.eigvals(W)))
        W            = W * (sn / jnp.maximum(rho, 1e-8))
        P_list.append(W)

        # f=1..F-1: input-modulation matrices M_f (degree-major layout)
        headroom = 1.0 - sn
        for f in range(1, F):
            d_f  = (f - 1) // K + 1    # feature f belongs to degree d_f ∈ {1,…,D}
            W    = jax.random.normal(keys_P[f], (N, N))
            W    = W * (jax.random.uniform(keys_mask_P[f], (N, N)) < self.density_P).astype(jnp.float32)
            fro  = jnp.linalg.norm(W, 'fro')
            # Budget: sum over all features at degree d_f of ||phi_f · M_f||_spec
            # ≤ headroom · taylor_decay^(d-1) / D, split equally over K features.
            # ||phi_f|| ≤ mi^d_f absorbs the input magnitude.
            budget = headroom * (self.taylor_decay ** (d_f - 1)) / (D * K * float(mi ** d_f))
            W = W * (budget / jnp.maximum(fro, 1e-8))
            P_list.append(W)

        P = jnp.stack(P_list, axis=0)   # (F, N, N)

        # ── Q_weights: (N, F) ────────────────────────────────────────────

        gamma = jnp.sqrt(1.0 - sn ** 2)
        dc    = q_degree_correction(D, self.taylor_decay)   # (D+1,), degree 0..D

        # Column 0 → degree 0 (bias), columns 1..K → degree 1, …
        col_degrees = jnp.concatenate([
            jnp.zeros(1, dtype=jnp.int32),
            jnp.repeat(jnp.arange(1, D + 1, dtype=jnp.int32), K),
        ])                                      # (F,)
        col_scale = dc[col_degrees]             # (F,)

        Q_raw = jax.random.normal(key_Q, (N, F))
        # Hybrid mask per row: each reservoir unit sees at least one feature.
        cols_base   = jnp.arange(N) % F
        mask_base_Q = jnp.zeros((N, F)).at[jnp.arange(N), cols_base].set(1.0)
        mask_rand_Q = (jax.random.uniform(key_mask_Q, (N, F)) < self.density_Q).astype(jnp.float32)
        Q = Q_raw * jnp.clip(mask_base_Q + mask_rand_Q, 0.0, 1.0) * col_scale[None, :] * gamma

        obj = SparsePolyBasis(N, K, D, sn, self.density_P, self.density_Q,
                              self.max_input, self.taylor_decay, self.training_mode,
                              self.budget_ref)
        obj.P_weights, obj.Q_weights = P, Q
        return obj

    # ── polynomial feature expansion ─────────────────────────────────────────

    def _poly_features(self, z_tilde_t):
        """z_tilde_t: (K,) → phi: (F,).  Degree-major layout."""
        z = (jnp.clip(z_tilde_t, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde_t)
        degrees = jnp.arange(1, self.degree + 1, dtype=jnp.float32)     # (D,)
        powers  = jnp.power(z[:, None], degrees[None, :])                # (K, D)
        flat    = powers.T.reshape(-1)                                    # (K·D,)
        return jnp.concatenate([jnp.ones(1, dtype=jnp.float32), flat])   # (F,)

    # ── per-step evaluators ──────────────────────────────────────────────────

    def eval_p(self, z_tilde_t):
        """z_tilde_t: (K,) → A: (N, N)."""
        phi = self._poly_features(z_tilde_t)
        return jnp.einsum('f,fij->ij', phi, self.P_weights)

    def eval_q(self, z_tilde_t):
        """z_tilde_t: (K,) → q: (N,)."""
        phi = self._poly_features(z_tilde_t)
        return self.Q_weights @ phi

    # ── batched evaluators ───────────────────────────────────────────────────

    def batch_eval_p(self, z_tilde):
        """z_tilde: (T, K) → (T, N, N).  Materialises T·N² floats — see training_mode."""
        phi = jax.vmap(self._poly_features)(z_tilde)           # (T, F)
        return jnp.einsum('tf,fij->tij', phi, self.P_weights)  # (T, N, N)

    def batch_eval_q(self, z_tilde):
        """z_tilde: (T, K) → (T, N)."""
        phi = jax.vmap(self._poly_features)(z_tilde)   # (T, F)
        return phi @ self.Q_weights.T                   # (T, N)

    # ── algebraic primitives ─────────────────────────────────────────────────

    def apply(self, A, s):
        """A: (N, N), s: (N,) → (N,)."""
        return jnp.matmul(A, s)

    def combine(self, i, j):
        """Full-matrix monoid for associative_scan.

        During the parallel scan, associative_scan batches multiple pairs at once,
        so A has shape (..., N, N) and b has shape (..., N).  b_i[..., None] adds the
        column dimension needed for (..., N, N) @ (..., N, 1) → (..., N, 1).
        """
        A_i, b_i = i
        A_j, b_j = j
        A_new  = jnp.matmul(A_j, A_i)
        b_term = jnp.matmul(A_j, b_i[..., None]).squeeze(-1)
        return A_new, b_term + b_j

    def __repr__(self) -> str:
        status = f"P:{self.P_weights.shape}, Q:{self.Q_weights.shape}" if self.is_initialized() else "uninitialised"
        return (f"SparsePolyBasis(n={self._n}, n_drivers={self._n_drivers}, "
                f"degree={self.degree}, spectral_norm={self.spectral_norm}, "
                f"density_P={self.density_P}, density_Q={self.density_Q}, "
                f"mode={self.training_mode}, {status})")
