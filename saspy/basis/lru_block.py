"""
lru_block.py — LRU-inspired 2×2 rotation-block polynomial basis.

Each of the K blocks is a 2×2 real rotation block — the real representation of
a complex eigenvalue λ_k = r_k · exp(i·θ_k).  This is equivalent to DiagonalPoly
but with complex (conjugate-pair) eigenvalues, enabling oscillatory dynamics.

Projector → basis interface:
  n_drivers = K  (one scalar z_tilde[k] per block)

Recurrence for block k:
  A_k(z_tilde_t[k]) = Σ_d P_weights[d, k] · z_tilde_t[k]^d    shape (2, 2)
  q_k(z_tilde_t[k]) = Σ_d Q_weights[d, k] · z_tilde_t[k]^d    shape (2,)

  s_t[k·2 : k·2+2] = A_k · s_{t-1}[k·2 : k·2+2] + q_k

Degree-0 term (the autonomous rotation):
  P_weights[0, k] = r_k · [[cos θ_k, −sin θ_k], [sin θ_k, cos θ_k]]
  This is the constant base matrix; when z_tilde[k]=0, A_k = P_weights[0, k].

Higher-degree terms add input-modulated Volterra interactions.

N = 2·K  (total reservoir size).
"""

import jax
import jax.numpy as jnp

from .base import BaseBasis
from .q_init import q_degree_correction

_B = 2  # fixed block size for LRU


@jax.tree_util.register_pytree_node_class
class LRUBlockPoly(BaseBasis):
    """
    LRU 2×2 rotation-block polynomial basis.

    Parameters
    ----------
    n_blocks      : K — number of blocks (N = 2·K).
    p_degree      : polynomial degree of z_tilde in P(z).
    q_degree      : polynomial degree of z_tilde in Q(z).
    spectral_norm : used only for budget-allocating higher-degree P terms.
    tau_min       : shortest timescale τ; r = exp(−1/τ_min).
    tau_max       : longest  timescale τ; r = exp(−1/τ_max).
    max_input     : clips |z_tilde[k]| before polynomial evaluation.
    frac_diagonal : fraction of blocks with θ=0 (pure real eigenvalue).
    taylor_decay  : per-degree Q shrinkage ∈ [0, 1].
    """

    def __init__(
        self,
        n_blocks:      int   = 50,
        p_degree:      int   = 1,
        q_degree:      int   = 1,
        spectral_norm: float = 0.9,
        tau_min:       float = 1.0,
        tau_max:       float = 100.0,
        max_input:     float = 4.0,
        frac_diagonal: float = 0.5,
        taylor_decay:  float = 1.0,
    ):
        super().__init__(p_degree, q_degree)
        self.K             = n_blocks
        self.spectral_norm = float(spectral_norm)
        self.tau_min       = float(tau_min)
        self.tau_max       = float(tau_max)
        self.max_input     = float(max_input)
        self.frac_diagonal = float(frac_diagonal)
        self.taylor_decay  = float(taylor_decay)

    # ── dimensions ──────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return self.K * _B

    @property
    def n_drivers(self) -> int:
        return self.K

    # ── pytree ───────────────────────────────────────────────────────────────

    def tree_flatten(self):
        return (self.P_weights, self.Q_weights), (
            self.K, self.p_degree, self.q_degree,
            self.spectral_norm, self.tau_min, self.tau_max,
            self.max_input, self.frac_diagonal, self.taylor_decay,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        K, p, q, sn, tmin, tmax, mi, fd, td = aux
        obj = cls(K, p, q, sn, tmin, tmax, mi, fd, td)
        obj.P_weights, obj.Q_weights = children
        return obj

    # ── factory ─────────────────────────────────────────────────────────────

    def initialize(self, key) -> "LRUBlockPoly":
        K = self.K
        if K < 1:
            raise ValueError("n_blocks must be ≥ 1")

        key_theta, key_mod, key_q = jax.random.split(key, 3)

        # ── Step 1: degree-0 rotation blocks ─────────────────────────────
        # Log-spaced timescales → radii
        log_taus = jnp.linspace(jnp.log(self.tau_min), jnp.log(self.tau_max), K)
        taus     = jnp.exp(log_taus)               # (K,)
        r        = jnp.exp(-1.0 / taus)            # (K,) ∈ (0, 1)

        K_diag = int(round(self.frac_diagonal * K))
        K_osc  = K - K_diag
        if K_osc > 0:
            theta_osc = jax.random.uniform(key_theta, (K_osc,),
                                           minval=0.0, maxval=jnp.pi)
            theta = jnp.concatenate([jnp.zeros(K_diag), theta_osc])
        else:
            theta = jnp.zeros(K)

        def _rot(r_k, th_k):
            c = jnp.cos(th_k)
            s = jnp.sin(th_k)
            return r_k * jnp.array([[c, -s], [s, c]])

        P0 = jax.vmap(_rot)(r, theta)               # (K, 2, 2)

        # ── Step 2: Q weights with LRU gamma normalisation ───────────────
        gamma            = jnp.sqrt(1.0 - r ** 2)  # (K,)
        gamma_per_neuron = jnp.repeat(gamma, _B)    # (N,)

        dc    = q_degree_correction(self.q_degree, self.taylor_decay)  # (q+1,)
        Q_raw = jax.random.normal(key_q, (self.q_degree + 1, K, _B)) / jnp.sqrt(2.0)
        Q     = Q_raw * dc[:, None, None] * gamma[None, :, None]
        # Q shape: (q_degree+1, K, 2)

        # ── Step 3: degrees 1+ — small Volterra modulation blocks ────────
        mod_sn = (1.0 - r) * 0.9 / self.max_input  # (K,) per-block budget

        if self.p_degree >= 1:
            P_mod_raw = jax.random.normal(key_mod, (self.p_degree, K, _B, _B))

            def _scale_seq(seq, budget):
                # seq: (p_degree, 2, 2)
                norms = jax.vmap(lambda M: jnp.linalg.norm(M, ord=2))(seq)
                return seq * (budget / jnp.maximum(jnp.sum(norms), 1e-8))

            P_mod_k      = jnp.swapaxes(P_mod_raw, 0, 1)             # (K, p_deg, 2, 2)
            P_mod_scaled = jax.vmap(_scale_seq)(P_mod_k, mod_sn)      # (K, p_deg, 2, 2)
            P_mod        = jnp.swapaxes(P_mod_scaled, 0, 1)           # (p_deg, K, 2, 2)
            P            = jnp.concatenate([P0[None], P_mod], axis=0) # (p+1, K, 2, 2)
        else:
            P = P0[None]                                               # (1, K, 2, 2)

        obj = LRUBlockPoly(K, self.p_degree, self.q_degree,
                           self.spectral_norm, self.tau_min, self.tau_max,
                           self.max_input, self.frac_diagonal, self.taylor_decay)
        obj.P_weights, obj.Q_weights = P, Q
        return obj

    # ── per-step evaluators ──────────────────────────────────────────────────

    def eval_p(self, z_tilde_t):
        """z_tilde_t: (K,) → A: (K, 2, 2)."""
        z = jnp.clip(z_tilde_t, -self.max_input, self.max_input)
        powers = jnp.arange(self.p_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[None, :], powers[:, None])              # (p+1, K)
        return jnp.einsum('dk,dkij->kij', feats, self.P_weights)

    def eval_q(self, z_tilde_t):
        """z_tilde_t: (K,) → q: (N,)."""
        z = jnp.clip(z_tilde_t, -self.max_input, self.max_input)
        powers = jnp.arange(self.q_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[None, :], powers[:, None])              # (q+1, K)
        q      = jnp.einsum('dk,dkb->kb', feats, self.Q_weights)     # (K, 2)
        return q.reshape(self.n)

    # ── batched evaluators ────────────────────────────────────────────────────

    def batch_eval_p(self, z_tilde):
        """z_tilde: (T, K) → (T, K, 2, 2)."""
        z = jnp.clip(z_tilde, -self.max_input, self.max_input)
        powers = jnp.arange(self.p_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[:, None, :], powers[None, :, None])     # (T, p+1, K)
        return jnp.einsum('tdk,dkij->tkij', feats, self.P_weights)   # (T, K, 2, 2)

    def batch_eval_q(self, z_tilde):
        """z_tilde: (T, K) → (T, N)."""
        z = jnp.clip(z_tilde, -self.max_input, self.max_input)
        powers = jnp.arange(self.q_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[:, None, :], powers[None, :, None])     # (T, q+1, K)
        q      = jnp.einsum('tdk,dkb->tkb', feats, self.Q_weights)   # (T, K, 2)
        return q.reshape(q.shape[0], self.n)

    # ── algebraic primitives ─────────────────────────────────────────────────

    def apply(self, A, s):
        """A: (K, 2, 2), s: (N,) → (N,)."""
        return jnp.matmul(A, s.reshape(self.K, _B, 1)).reshape(self.n)

    def combine(self, i, j):
        """Block-matrix monoid for associative_scan."""
        A_i, b_i = i    # A: (K, 2, 2),  b: (N,)
        A_j, b_j = j
        A_new        = jnp.matmul(A_j, A_i)
        b_i_blocked  = b_i.reshape(b_i.shape[:-1] + (self.K, _B, 1))
        b_term       = jnp.matmul(A_j, b_i_blocked).reshape(b_j.shape)
        return A_new, b_term + b_j
