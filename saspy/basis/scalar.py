"""Scalar-feature structures: Diagonal (block size 1) and Block (block size B).

Both consume **one scalar driver per block** (``n_drivers = K``) and evaluate a
bounded feature per block via a :mod:`saspy.basis.feature` spec.  The transition
P and the drive Q are split into independent classes, each parametrised by its
own feature spec — so "polynomial P + trigonometric Q" (or any mix) is just a
choice of two spec objects.

  * **Diagonal**  : block size 1, scalar eigenvalue per unit, element-wise scan.
                    ``N == K``.
  * **Block**     : block size B, (B×B) matrix per block, block-matmul scan.
                    ``N == K·B``.  init_mode='rotation' (B=2, LRU complex
                    eigenvalues) or 'orthogonal' (general scaled-orthogonal base).

Contractivity is guaranteed structurally: with every feature bounded in [-1, 1],
distributing a budget of ``(1 − rho_k)·margin`` across the modulation features
keeps each block's spectral radius < 1, for *any* feature spec and *any* driver
distribution on [-1, 1].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .feature import Cheb

_EIG_CLIP = 0.9999   # hard bound on diagonal eigenvalues


# ════════════════════════════════════════════════════════════════════════════
# Diagonal  (block size 1)
# ════════════════════════════════════════════════════════════════════════════

@jax.tree_util.register_pytree_node_class
class DiagonalP:
    """Diagonal transition: A[i] = Σ_c P[c,i]·phi_c(z_tilde[i]), clipped to (−1, 1).

    n             : reservoir size N (= n_drivers).
    feature       : Cheb/Trig spec for the per-unit transition features.
    spectral_norm : base eigenvalue range — P[0,i] ∈ (−sn, sn).
    margin        : fraction of head-room (1−|P[0,i]|) given to modulation.
    """

    def __init__(self, n: int, feature=Cheb(degree=1),
                 spectral_norm: float = 0.9, margin: float = 0.95):
        self._n           = n
        self.feature      = feature
        self.spectral_norm = float(spectral_norm)
        self.margin       = float(margin)
        self.P_weights = None     # (F_s, N)
        self.frozen    = None     # feature frozen params (tuple)

    @property
    def n(self):          return self._n
    @property
    def n_drivers(self):  return self._n
    def is_initialized(self): return self.P_weights is not None

    def initialize(self, key) -> "DiagonalP":
        N, sn = self._n, self.spectral_norm
        Fs    = self.feature.n_scalar()
        k_base, k_mod, k_feat = jax.random.split(key, 3)

        lam0     = (jax.random.uniform(k_base, (N,)) * 2 - 1) * sn        # (N,)
        headroom = (1.0 - jnp.abs(lam0)) * self.margin
        rows     = [lam0]
        if Fs > 1:
            budget = headroom / (Fs - 1)                                  # per feature (N,)
            raw    = jax.random.uniform(k_mod, (Fs - 1, N), minval=-1.0, maxval=1.0)
            rows.append(raw * budget[None, :])
        P = jnp.concatenate([rows[0][None], rows[1]], 0) if Fs > 1 else lam0[None]

        obj = DiagonalP(N, self.feature, sn, self.margin)
        obj.P_weights = P
        obj.frozen    = self.feature.init_scalar(k_feat, N)
        return obj

    def eval_p(self, z_tilde_t):
        phi = self.feature.scalar_features(z_tilde_t, self.frozen)        # (N, F_s)
        a   = jnp.einsum('nc,cn->n', phi, self.P_weights)
        return jnp.clip(a, -_EIG_CLIP, _EIG_CLIP)

    def batch_eval_p(self, z_tilde):
        phi = self.feature.scalar_features(z_tilde, self.frozen)          # (T, N, F_s)
        a   = jnp.einsum('tnc,cn->tn', phi, self.P_weights)
        return jnp.clip(a, -_EIG_CLIP, _EIG_CLIP)

    def apply(self, a, s):       return a * s
    def matvec_p(self, z_tilde_t, s):  return self.apply(self.eval_p(z_tilde_t), s)
    def combine(self, i, j):
        a_i, b_i = i; a_j, b_j = j
        return a_j * a_i, a_j * b_i + b_j

    def leaky(self, P_seq, leak):
        """Diagonal of (1−leak)·I + leak·A (leak=1 → unchanged)."""
        return (1.0 - leak) + leak * P_seq

    def tree_flatten(self):
        return (self.P_weights, self.frozen), (self._n, self.feature,
                                               self.spectral_norm, self.margin)
    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.P_weights, obj.frozen = children
        return obj

    def __repr__(self):
        return f"DiagonalP(n={self._n}, feature={self.feature!r}, sn={self.spectral_norm})"


@jax.tree_util.register_pytree_node_class
class DiagonalQ:
    """Diagonal drive: q[i] = Σ_c Q[c,i]·phi_c(z_tilde[i]).

    Q is calibrated against the *nominal* spectral_norm (echo-state drive scale
    gamma = √(1−sn²)) so it never needs to see P's realised eigenvalues —
    eliminating the decoupled gamma-mismatch.
    """

    def __init__(self, n: int, feature=Cheb(degree=1), spectral_norm: float = 0.9):
        self._n           = n
        self.feature      = feature
        self.spectral_norm = float(spectral_norm)
        self.Q_weights = None      # (F_s, N)
        self.frozen    = None

    @property
    def n(self):          return self._n
    @property
    def n_drivers(self):  return self._n
    def is_initialized(self): return self.Q_weights is not None

    def initialize(self, key) -> "DiagonalQ":
        N     = self._n
        Fs    = self.feature.n_scalar()
        gamma = jnp.sqrt(1.0 - self.spectral_norm ** 2)
        k_q, k_feat = jax.random.split(key)

        Q = jax.random.normal(k_q, (Fs, N)) * (gamma / jnp.sqrt(float(Fs)))
        obj = DiagonalQ(N, self.feature, self.spectral_norm)
        obj.Q_weights = Q
        obj.frozen    = self.feature.init_scalar(k_feat, N)
        return obj

    def eval_q(self, z_tilde_t):
        phi = self.feature.scalar_features(z_tilde_t, self.frozen)        # (N, F_s)
        return jnp.einsum('nc,cn->n', phi, self.Q_weights)

    def batch_eval_q(self, z_tilde):
        phi = self.feature.scalar_features(z_tilde, self.frozen)          # (T, N, F_s)
        return jnp.einsum('tnc,cn->tn', phi, self.Q_weights)

    def tree_flatten(self):
        return (self.Q_weights, self.frozen), (self._n, self.feature, self.spectral_norm)
    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.Q_weights, obj.frozen = children
        return obj

    def __repr__(self):
        return f"DiagonalQ(n={self._n}, feature={self.feature!r}, sn={self.spectral_norm})"


# ════════════════════════════════════════════════════════════════════════════
# Block  (block size B ≥ 2)
# ════════════════════════════════════════════════════════════════════════════

def _rotation(r_k, th_k):
    c, s = jnp.cos(th_k), jnp.sin(th_k)
    return r_k * jnp.array([[c, -s], [s, c]])


@jax.tree_util.register_pytree_node_class
class BlockP:
    """Block transition: A_k = Σ_c P[c,k]·phi_c(z_tilde[k]),  A_k ∈ (B×B).

    block_size : B.  N = K·B.
    init_mode  : 'rotation' (B must be 2 — LRU complex eigenvalues, decay r_k
                 log-spaced over [tau_min, tau_max]) or 'orthogonal' (random
                 scaled-orthogonal base with spectral radius spectral_norm).
    frac_diagonal : 'rotation' only — fraction of blocks with θ=0 (real decay).
    """

    def __init__(self, n_blocks: int, block_size: int = 2, feature=Cheb(degree=1),
                 spectral_norm: float = 0.9, init_mode: str = 'rotation',
                 tau_min: float = 1.0, tau_max: float = 100.0,
                 frac_diagonal: float = 0.5, margin: float = 0.9):
        self.K            = n_blocks
        self.B            = block_size
        self.feature      = feature
        self.spectral_norm = float(spectral_norm)
        self.init_mode    = init_mode
        self.tau_min      = float(tau_min)
        self.tau_max      = float(tau_max)
        self.frac_diagonal = float(frac_diagonal)
        self.margin       = float(margin)
        self.P_weights = None      # (F_s, K, B, B)
        self.frozen    = None

    @property
    def n(self):          return self.K * self.B
    @property
    def n_drivers(self):  return self.K
    def is_initialized(self): return self.P_weights is not None

    def _base_blocks(self, key):
        """Return (P0 (K,B,B), rho (K,)) — base blocks and their spectral radii."""
        K, B = self.K, self.B
        if self.init_mode == 'rotation':
            if B != 2:
                raise ValueError("init_mode='rotation' requires block_size=2")
            r = jnp.exp(-1.0 / jnp.exp(jnp.linspace(
                jnp.log(self.tau_min), jnp.log(self.tau_max), K)))        # (K,)
            K_diag = int(round(self.frac_diagonal * K))
            theta  = jnp.concatenate([
                jnp.zeros(K_diag),
                jax.random.uniform(key, (K - K_diag,), minval=0.0, maxval=jnp.pi),
            ]) if K_diag < K else jnp.zeros(K)
            P0 = jax.vmap(_rotation)(r, theta)
            return P0, r
        elif self.init_mode == 'orthogonal':
            sn = self.spectral_norm
            def _orth(k):
                M = jax.random.normal(k, (B, B))
                Q, _ = jnp.linalg.qr(M)
                return Q * sn
            P0 = jax.vmap(_orth)(jax.random.split(key, K))
            return P0, jnp.full((K,), sn)
        raise ValueError(f"Unknown init_mode: {self.init_mode!r}")

    def initialize(self, key) -> "BlockP":
        K, B = self.K, self.B
        Fs   = self.feature.n_scalar()
        k_base, k_mod, k_feat = jax.random.split(key, 3)

        P0, rho = self._base_blocks(k_base)                              # (K,B,B), (K,)
        P_rows  = [P0]
        if Fs > 1:
            headroom = (1.0 - rho) * self.margin                         # (K,)
            budget   = headroom / (Fs - 1)                               # (K,)
            keys_mod = jax.random.split(k_mod, Fs - 1)
            for c in range(Fs - 1):
                M     = jax.random.normal(keys_mod[c], (K, B, B))
                snorm = jax.vmap(lambda m: jnp.linalg.norm(m, ord=2))(M) # (K,)
                M     = M * (budget / jnp.maximum(snorm, 1e-8))[:, None, None]
                P_rows.append(M)
            P = jnp.stack(P_rows, axis=0)                                # (F_s, K, B, B)
        else:
            P = P0[None]

        obj = BlockP(K, B, self.feature, self.spectral_norm, self.init_mode,
                     self.tau_min, self.tau_max, self.frac_diagonal, self.margin)
        obj.P_weights = P
        obj.frozen    = self.feature.init_scalar(k_feat, K)
        return obj

    def eval_p(self, z_tilde_t):
        phi = self.feature.scalar_features(z_tilde_t, self.frozen)       # (K, F_s)
        return jnp.einsum('kc,ckij->kij', phi, self.P_weights)           # (K, B, B)

    def batch_eval_p(self, z_tilde):
        phi = self.feature.scalar_features(z_tilde, self.frozen)         # (T, K, F_s)
        return jnp.einsum('tkc,ckij->tkij', phi, self.P_weights)         # (T, K, B, B)

    def apply(self, A, s):
        return jnp.matmul(A, s.reshape(self.K, self.B, 1)).reshape(self.n)

    def matvec_p(self, z_tilde_t, s):  return self.apply(self.eval_p(z_tilde_t), s)

    def combine(self, i, j):
        A_i, b_i = i; A_j, b_j = j
        A_new  = jnp.matmul(A_j, A_i)
        b_blk  = b_i.reshape(b_i.shape[:-1] + (self.K, self.B, 1))
        b_term = jnp.matmul(A_j, b_blk).reshape(b_j.shape)
        return A_new, b_term + b_j

    def leaky(self, P_seq, leak):
        """(1−leak)·I_B + leak·A per block.  P_seq: (..., K, B, B)."""
        eye = jnp.eye(self.B, dtype=P_seq.dtype)
        return leak * P_seq + (1.0 - leak) * eye

    def tree_flatten(self):
        return (self.P_weights, self.frozen), (
            self.K, self.B, self.feature, self.spectral_norm, self.init_mode,
            self.tau_min, self.tau_max, self.frac_diagonal, self.margin)
    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.P_weights, obj.frozen = children
        return obj

    def __repr__(self):
        return (f"BlockP(K={self.K}, B={self.B}, mode={self.init_mode!r}, "
                f"feature={self.feature!r}, sn={self.spectral_norm})")


@jax.tree_util.register_pytree_node_class
class BlockQ:
    """Block drive: q_k = Σ_c Q[c,k]·phi_c(z_tilde[k]),  q_k ∈ (B,).  N = K·B."""

    def __init__(self, n_blocks: int, block_size: int = 2, feature=Cheb(degree=1),
                 spectral_norm: float = 0.9):
        self.K            = n_blocks
        self.B            = block_size
        self.feature      = feature
        self.spectral_norm = float(spectral_norm)
        self.Q_weights = None       # (F_s, K, B)
        self.frozen    = None

    @property
    def n(self):          return self.K * self.B
    @property
    def n_drivers(self):  return self.K
    def is_initialized(self): return self.Q_weights is not None

    def initialize(self, key) -> "BlockQ":
        K, B  = self.K, self.B
        Fs    = self.feature.n_scalar()
        gamma = jnp.sqrt(1.0 - self.spectral_norm ** 2)
        k_q, k_feat = jax.random.split(key)

        Q = jax.random.normal(k_q, (Fs, K, B)) * (gamma / jnp.sqrt(float(Fs * B)))
        obj = BlockQ(K, B, self.feature, self.spectral_norm)
        obj.Q_weights = Q
        obj.frozen    = self.feature.init_scalar(k_feat, K)
        return obj

    def eval_q(self, z_tilde_t):
        phi = self.feature.scalar_features(z_tilde_t, self.frozen)       # (K, F_s)
        q   = jnp.einsum('kc,ckb->kb', phi, self.Q_weights)              # (K, B)
        return q.reshape(self.n)

    def batch_eval_q(self, z_tilde):
        phi = self.feature.scalar_features(z_tilde, self.frozen)         # (T, K, F_s)
        q   = jnp.einsum('tkc,ckb->tkb', phi, self.Q_weights)            # (T, K, B)
        return q.reshape(q.shape[0], self.n)

    def tree_flatten(self):
        return (self.Q_weights, self.frozen), (self.K, self.B, self.feature, self.spectral_norm)
    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.Q_weights, obj.frozen = children
        return obj

    def __repr__(self):
        return f"BlockQ(K={self.K}, B={self.B}, feature={self.feature!r})"
