"""General B×B block-diagonal polynomial basis (n_drivers = K, N = K·B).

Generalises LRUBlockPoly to arbitrary block size B with orthogonal-matrix
initialisation instead of the fixed rotation structure:

    A_k(z_tilde[k]) = Σ_d P_weights[d, k] · z_tilde[k]^d   shape (B, B)
    q_k(z_tilde[k]) = Σ_d Q_weights[d, k] · z_tilde[k]^d   shape (B,)
"""

import jax
import jax.numpy as jnp

from .base import BaseBasis
from .q_init import q_degree_correction


@jax.tree_util.register_pytree_node_class
class BlockLinearPoly(BaseBasis):
    """
    General B×B block-diagonal polynomial basis.

    Parameters
    ----------
    n_blocks      : K — number of blocks.
    block_size    : B — size of each square block.  N = K·B.
    p_degree      : polynomial degree in P(z).
    q_degree      : polynomial degree in Q(z).
    spectral_norm : spectral norm of the degree-0 (base) matrices.
    max_input     : clip |z_tilde[k]| before polynomial evaluation.
    taylor_decay  : per-degree Q shrinkage ∈ [0, 1].
    """

    def __init__(
        self,
        n_blocks:      int        = 50,
        block_size:    int        = 4,
        p_degree:      int        = 1,
        q_degree:      int        = 1,
        spectral_norm: float      = 0.9,
        max_input:     float|None = 4.0,
        taylor_decay:  float      = 1.0,
        budget_ref:    float|None = None,
    ):
        super().__init__(p_degree, q_degree)
        self.K             = n_blocks
        self.B             = block_size
        self.spectral_norm = float(spectral_norm)
        self.max_input     = float(max_input) if max_input is not None else None
        self.taylor_decay  = float(taylor_decay)
        self.budget_ref    = float(budget_ref) if budget_ref is not None else None

    # ── dimensions ──────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return self.K * self.B

    @property
    def n_drivers(self) -> int:
        return self.K

    # ── pytree ───────────────────────────────────────────────────────────────

    def tree_flatten(self):
        return (self.P_weights, self.Q_weights), (
            self.K, self.B, self.p_degree, self.q_degree,
            self.spectral_norm, self.max_input, self.taylor_decay,
            self.budget_ref,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        K, B, p, q, sn, mi, td, br = aux
        obj = cls(K, B, p, q, sn, mi, td, br)
        obj.P_weights, obj.Q_weights = children
        return obj

    # ── factory ─────────────────────────────────────────────────────────────

    def initialize(self, key) -> "BlockLinearPoly":
        K, B = self.K, self.B
        sn   = self.spectral_norm

        key_P0, key_Pmod, key_Q = jax.random.split(key, 3)

        keys_P0 = jax.random.split(key_P0, K)

        def _rand_orth(k):
            A     = jax.random.normal(k, (B, B))
            Q, R  = jnp.linalg.qr(A)
            signs = jnp.sign(jnp.diag(R))
            return Q * signs[None, :]

        P0 = jax.vmap(_rand_orth)(keys_P0) * sn   # (K, B, B)

        gamma = (1.0 - sn ** 2) ** 0.5
        dc    = q_degree_correction(self.q_degree, self.taylor_decay)
        Q_raw = jax.random.normal(key_Q, (self.q_degree + 1, K, B)) / (B ** 0.5)
        Q     = Q_raw * dc[:, None, None] * gamma

        # Per-degree Volterra budget: ‖M_d‖ · scale_ref^d ≤ (1-sn)·0.9 / 2^(d-1)
        scale_ref = self._budget_ref()

        if self.p_degree >= 1:
            keys_mod = jax.random.split(key_Pmod, self.p_degree)
            P_mod_list = []
            for d in range(1, self.p_degree + 1):
                budget_d = (1.0 - sn) * 0.9 / (float(2 ** (d - 1)) * scale_ref ** d)
                M = jax.random.normal(keys_mod[d - 1], (K, B, B))
                norms = jax.vmap(lambda m: jnp.linalg.norm(m, ord=2))(M)  # (K,)
                M = M * (budget_d / jnp.maximum(norms, 1e-8))[:, None, None]
                P_mod_list.append(M)
            P_mod = jnp.stack(P_mod_list, axis=0)                     # (p_deg, K, B, B)
            P     = jnp.concatenate([P0[None], P_mod], axis=0)        # (p+1, K, B, B)
        else:
            P = P0[None]                                               # (1, K, B, B)

        obj = BlockLinearPoly(K, B, self.p_degree, self.q_degree,
                              self.spectral_norm, self.max_input, self.taylor_decay,
                              self.budget_ref)
        obj.P_weights, obj.Q_weights = P, Q
        return obj

    # ── per-step evaluators ──────────────────────────────────────────────────

    def eval_p(self, z_tilde_t):
        """z_tilde_t: (K,) → A: (K, B, B)."""
        z = (jnp.clip(z_tilde_t, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde_t)
        powers = jnp.arange(self.p_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[None, :], powers[:, None])              # (p+1, K)
        return jnp.einsum('dk,dkij->kij', feats, self.P_weights)

    def eval_q(self, z_tilde_t):
        """z_tilde_t: (K,) → q: (N,)."""
        z = (jnp.clip(z_tilde_t, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde_t)
        powers = jnp.arange(self.q_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[None, :], powers[:, None])              # (q+1, K)
        q      = jnp.einsum('dk,dkb->kb', feats, self.Q_weights)     # (K, B)
        return q.reshape(self.n)

    # ── batched evaluators ────────────────────────────────────────────────────

    def batch_eval_p(self, z_tilde):
        """z_tilde: (T, K) → (T, K, B, B)."""
        z = (jnp.clip(z_tilde, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde)
        powers = jnp.arange(self.p_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[:, None, :], powers[None, :, None])     # (T, p+1, K)
        return jnp.einsum('tdk,dkij->tkij', feats, self.P_weights)

    def batch_eval_q(self, z_tilde):
        """z_tilde: (T, K) → (T, N)."""
        z = (jnp.clip(z_tilde, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde)
        powers = jnp.arange(self.q_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[:, None, :], powers[None, :, None])     # (T, q+1, K)
        q      = jnp.einsum('tdk,dkb->tkb', feats, self.Q_weights)   # (T, K, B)
        return q.reshape(q.shape[0], self.n)

    # ── algebraic primitives ─────────────────────────────────────────────────

    def apply(self, A, s):
        """A: (K, B, B), s: (N,) → (N,)."""
        return jnp.matmul(A, s.reshape(self.K, self.B, 1)).reshape(self.n)

    def combine(self, i, j):
        """Block-matrix monoid for associative_scan."""
        A_i, b_i = i
        A_j, b_j = j
        A_new       = jnp.matmul(A_j, A_i)
        b_i_blocked = b_i.reshape(b_i.shape[:-1] + (self.K, self.B, 1))
        b_term      = jnp.matmul(A_j, b_i_blocked).reshape(b_j.shape)
        return A_new, b_term + b_j
