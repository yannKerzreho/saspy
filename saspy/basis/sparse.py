"""Sparse full-matrix structure (n_drivers = K, N×N transitions).

Two feature regimes, dispatched on the spec type:

* **Cheb (joint monomials)** — global polynomial features mix the K drivers and
  scale a stack of N×N matrices:
      phi = [1, m_e1(z̃), …]              (F = 1 + monomial count)
      A_t = Σ_f phi_f · P_f               (N, N)
      q_t = Q_weights @ phi               (N,)

* **Trig (per-driver RFF)** — each reservoir unit gets its OWN random frequency
  vector over the K inputs, so the input enters through N independent cosines
  (rank-N), fixing the rank-deficiency of joint Trig (see note §3c-bis):
      g_h = cos(Ω_h · z̃ + φ_h) ∈ [-1,1]^N         (h = 1..H harmonics)
      A_t = M_0 + Σ_h diag(g_h) · M_h             (per-row gating)
      q_t = Σ_h W_h ⊙ cos(Ω^Q_h · z̃ + φ^Q_h)     (multivariate RFF drive)

Both regimes share the matmul monoid (combine/apply) and the sequential default
training mode (dense N×N → lax.scan, O(N²) memory).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from .feature      import Cheb, Trig
from .connectivity import log_density, connectivity_mask


# ── sparse-storage helpers (BCOO, sequential mode only) ───────────────────────
#
# In ``training_mode == 'sequential'`` the scan only ever needs the *action*
# A_t @ s, never the dense A_t.  We therefore store the P stack as a single
# BCOO (F, N, N) and compute the per-feature matvecs with one sparse contraction
# — exact, with static nnz, no fill-in (see note/sparse_bcoo.md).  This is the
# only place ``jax.experimental.sparse`` is used: we rely on ``bcoo_dot_general``
# (sparse·dense, jittable) and ``BCOO.fromdense`` only — both mature.

def _as_dense(P):
    """Densify P if it is a BCOO (sequential storage); pass through otherwise."""
    return P.todense() if isinstance(P, jsparse.BCOO) else P


def _sp_matvec(P_bcoo, s):
    """Batched sparse matvec: P (F, N, N) BCOO · s (N,) → (F, N) dense.

    out[f, i] = Σ_j P[f, i, j] · s[j].  Contracts P's last axis with s; no batch
    dims, so JAX sees a single sparse·dense contraction with a fixed nnz.
    """
    return jsparse.bcoo_dot_general(
        P_bcoo, s, dimension_numbers=(([2], [0]), ([], [])))


# ── shared helpers ────────────────────────────────────────────────────────────

def _base_matrix(key, key_mask, N, sn, density, mask_A):
    """Autonomous base matrix M_0 with spectral radius sn (diagonal kept dense)."""
    idx       = jnp.arange(N)
    mask_diag = jnp.zeros((N, N)).at[idx, idx].set(1.0)
    W = jax.random.normal(key, (N, N))
    if mask_A is not None:
        W = W * jnp.clip(mask_diag + mask_A, 0.0, 1.0)
    else:
        m = connectivity_mask(key_mask, N, N, density)
        W = W * jnp.clip(mask_diag + m, 0.0, 1.0)
    rho = jnp.max(jnp.abs(jnp.linalg.eigvals(W)))
    return W * (sn / jnp.maximum(rho, 1e-8))


def _budgeted_mods(key, key_mask, n_mod, N, density, mask_A, budget):
    """Stack of `n_mod` sparse N×N matrices, each Frobenius-scaled to `budget`."""
    keys_w = jax.random.split(key, n_mod)
    keys_m = jax.random.split(key_mask, n_mod)
    mats = []
    for f in range(n_mod):
        W = jax.random.normal(keys_w[f], (N, N))
        if mask_A is not None:
            W = W * mask_A
        else:
            W = W * connectivity_mask(keys_m[f], N, N, density)
        fro = jnp.linalg.norm(W, "fro")
        mats.append(W * (budget / jnp.maximum(fro, 1e-8)))
    return jnp.stack(mats, axis=0) if n_mod > 0 else jnp.zeros((0, N, N))


def _trig_freqs(key, feature: Trig, H, N, K):
    """Per-driver frequencies Ω (H, N, K) and phases φ (H, N) for the RFF gate."""
    k_o, k_mask, k_p = jax.random.split(key, 3)
    sigma     = feature._sigma(H)[:, None, None]                  # (H,1,1)
    omega_raw = feature._draw(k_o, (H, N, K), sigma)              # (H,N,K)
    mask      = (jax.random.uniform(k_mask, (H, N, K)) < feature.density_omega).astype(omega_raw.dtype)
    forced    = jnp.argmax(jnp.abs(omega_raw), axis=2)           # (H,N) — col to force if row empty
    fix       = jax.nn.one_hot(forced, K, dtype=omega_raw.dtype)  # (H,N,K)
    mask      = jnp.where(mask.sum(axis=2, keepdims=True) == 0, fix, mask)
    omega     = omega_raw * mask
    phase     = jax.random.uniform(k_p, (H, N), minval=0.0, maxval=2.0 * jnp.pi)
    return omega, phase


# ════════════════════════════════════════════════════════════════════════════
# SparseP
# ════════════════════════════════════════════════════════════════════════════

@jax.tree_util.register_pytree_node_class
class SparseP:
    """Sparse N×N transition basis (Cheb joint monomials | Trig per-driver RFF)."""

    def __init__(self, n: int, n_drivers: int, feature=Cheb(degree=2),
                 spectral_norm: float = 0.9, density_P: float = 0.05,
                 margin: float = 0.95, A_density: float | None = None,
                 training_mode: str = "sequential", connectivity: float | None = 1.5,
                 conn_floor: int = 6):
        self._n           = n
        self._n_drivers   = n_drivers
        self.feature      = feature
        self.spectral_norm = float(spectral_norm)
        self.density_P    = float(density_P)
        self.margin       = float(margin)
        self.A_density    = float(A_density) if A_density is not None else None
        # connectivity (ER threshold, default on): sets the *shared* recurrence
        # mask — the support of A_t = Σ_f φ_f P_f — to c·ln(N)/N (fan-in floored
        # at conn_floor). The recurrence graph lives in this shared mask, NOT in
        # the per-slice density_P: with independent per-slice masks the union over
        # F slices is near-dense, so density_P controls per-feature parameter count,
        # not the realised recurrence connectivity. See note/density_design.md.
        # Explicit A_density overrides; connectivity=None falls back to density_P.
        self.connectivity = float(connectivity) if connectivity is not None else None
        self.conn_floor   = int(conn_floor)
        self.training_mode = training_mode
        self.P_weights = None      # Cheb: (F,N,N) | Trig: (1+H,N,N)
        self.frozen    = None      # Cheb: () | Trig: (Ω (H,N,K), φ (H,N))

    @property
    def _trig(self):      return isinstance(self.feature, Trig)
    @property
    def n(self):          return self._n
    @property
    def n_drivers(self):  return self._n_drivers
    def is_initialized(self): return self.P_weights is not None

    def initialize(self, key) -> "SparseP":
        N, K, sn = self._n, self._n_drivers, self.spectral_norm
        k_base, k_basemask, k_mod, k_modmask, k_A, k_feat = jax.random.split(key, 6)
        # Shared recurrence mask (support of A_t). Precedence: explicit A_density,
        # else the ER threshold via `connectivity`, else None (per-slice density_P).
        if self.A_density is not None:
            aDens = self.A_density
        elif self.connectivity is not None:
            aDens = log_density(N, self.connectivity, self.conn_floor)
        else:
            aDens = None
        if aDens is not None:
            idx    = jnp.arange(N)
            m      = (jax.random.uniform(k_A, (N, N)) < aDens).astype(jnp.float32)
            mask_A = m.at[idx, idx].set(1.0)          # diagonal always kept
        else:
            mask_A = None
        dP = self.density_P                            # per-slice fallback (mask_A is None)
        M0 = _base_matrix(k_base, k_basemask, N, sn, dP, mask_A)

        if self._trig:
            H      = self.feature.degree
            budget = (1.0 - sn) * self.margin / max(H, 1)
            Mmod   = _budgeted_mods(k_mod, k_modmask, H, N, dP, mask_A, budget)
            P      = jnp.concatenate([M0[None], Mmod], axis=0)           # (1+H, N, N)
            frozen = _trig_freqs(k_feat, self.feature, H, N, K)
        else:
            F      = self.feature.n_joint(K)
            budget = (1.0 - sn) * self.margin / max(F - 1, 1)
            Mmod   = _budgeted_mods(k_mod, k_modmask, F - 1, N, dP, mask_A, budget)
            P      = jnp.concatenate([M0[None], Mmod], axis=0)           # (F, N, N)
            frozen = ()

        # Sequential mode never needs dense A_t — store the P stack as BCOO so
        # the (F, N, N) tensor costs ~nnz instead of F·N² (see _sp_matvec).
        if self.training_mode == 'sequential':
            P = jsparse.BCOO.fromdense(P)

        obj = SparseP(N, K, self.feature, sn, self.density_P, self.margin,
                      self.A_density, self.training_mode, self.connectivity, self.conn_floor)
        obj.P_weights, obj.frozen = P, frozen
        return obj

    # ── per-step evaluators ───────────────────────────────────────────────────
    # eval_p / batch_eval_p return the *dense* A_t (used by the parallel scan and
    # by tests).  In sequential mode P_weights is a BCOO, so they densify first;
    # the engine never calls them there — it uses the fused matvec_p below.

    def eval_p(self, z_tilde_t):
        P = _as_dense(self.P_weights)
        if self._trig:
            omega, phase = self.frozen                                   # (H,N,K), (H,N)
            g = jnp.cos(jnp.einsum('hnk,k->hn', omega, z_tilde_t) + phase)  # (H,N)
            return P[0] + jnp.einsum('hn,hnj->nj', g, P[1:])
        phi = self.feature.joint_features(z_tilde_t, self.frozen, self._n_drivers)
        return jnp.einsum('f,fij->ij', phi, P)

    def batch_eval_p(self, z_tilde):
        P = _as_dense(self.P_weights)
        if self._trig:
            omega, phase = self.frozen
            g = jnp.cos(jnp.einsum('hnk,tk->thn', omega, z_tilde) + phase[None])  # (T,H,N)
            return P[0][None] + jnp.einsum('thn,hnj->tnj', g, P[1:])
        phi = self.feature.joint_features(z_tilde, self.frozen, self._n_drivers)
        return jnp.einsum('tf,fij->tij', phi, P)

    def matvec_p(self, z_tilde_t, s):
        """A_t @ s without materialising A_t.

        Sequential mode: P_weights is a BCOO (F, N, N).  We compute the F sparse
        matvecs Ps = [P_f @ s] in one contraction, then fold in the feature
        weights — O(F·nnz) flops, no dense N×N.  Other modes fall back to the
        dense apply(eval_p(·)) path.
        """
        if self.training_mode != 'sequential':
            return self.apply(self.eval_p(z_tilde_t), s)
        Ps = _sp_matvec(self.P_weights, s)                              # (F, N)
        if self._trig:
            omega, phase = self.frozen                                   # (H,N,K), (H,N)
            g = jnp.cos(jnp.einsum('hnk,k->hn', omega, z_tilde_t) + phase)  # (H,N)
            return Ps[0] + jnp.sum(g * Ps[1:], axis=0)                  # M_0 s + Σ_h g_h⊙(M_h s)
        phi = self.feature.joint_features(z_tilde_t, self.frozen, self._n_drivers)
        return jnp.einsum('f,fn->n', phi, Ps)

    # ── standardized sequential-scan interface (batched features + lean matvec) ──
    # Same interface as LowRankP so the fast training scan is structure-agnostic.
    # Sparse is already at its kernel floor (one fused BCOO contraction), so this is
    # a code-unification path, not a speedup — scan_prep is a no-op.

    def scan_features(self, z_tilde):                   # (T,K) → (T,F) | trig (T,H,N)
        if self._trig:
            omega, phase = self.frozen                                   # (H,N,K),(H,N)
            return jnp.cos(jnp.einsum('hnk,tk->thn', omega, z_tilde) + phase[None])
        return self.feature.joint_features(z_tilde, self.frozen, self._n_drivers)

    def scan_prep(self):
        return None                                     # P_weights already stored

    def scan_matvec(self, prep, feat_t, s):             # A_t @ s, features precomputed
        Ps = _sp_matvec(self.P_weights, s)                              # (F|1+H, N)
        if self._trig:
            return Ps[0] + jnp.sum(feat_t * Ps[1:], axis=0)            # feat_t = g (H,N)
        return jnp.einsum('f,fn->n', feat_t, Ps)                       # feat_t = phi (F,)

    # ── monoid ────────────────────────────────────────────────────────────────

    def apply(self, A, s):       return jnp.matmul(A, s)
    def combine(self, i, j):
        A_i, b_i = i; A_j, b_j = j
        return jnp.matmul(A_j, A_i), jnp.matmul(A_j, b_i[..., None]).squeeze(-1) + b_j

    def leaky(self, P_seq, leak):
        """(1−leak)·I_N + leak·A on the dense A_t (parallel mode only; the
        sequential path applies the leak at the state level in SASModel.step)."""
        eye = jnp.eye(self._n, dtype=P_seq.dtype)
        return leak * P_seq + (1.0 - leak) * eye

    def tree_flatten(self):
        return (self.P_weights, self.frozen), (
            self._n, self._n_drivers, self.feature, self.spectral_norm,
            self.density_P, self.margin, self.A_density, self.training_mode,
            self.connectivity, self.conn_floor)
    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.P_weights, obj.frozen = children
        return obj

    def __repr__(self):
        feat = "trig-RFF" if self._trig else "cheb-joint"
        return (f"SparseP(n={self._n}, K={self._n_drivers}, {feat}, "
                f"feature={self.feature!r}, sn={self.spectral_norm}, mode={self.training_mode!r})")


# ════════════════════════════════════════════════════════════════════════════
# SparseQ
# ════════════════════════════════════════════════════════════════════════════

@jax.tree_util.register_pytree_node_class
class SparseQ:
    """Sparse drive basis (Cheb: Q_weights @ phi | Trig: per-unit multivariate RFF)."""

    def __init__(self, n: int, n_drivers: int, feature=Cheb(degree=2),
                 spectral_norm: float = 0.9, density_Q: float = 0.1,
                 connectivity: float | None = 1.5, conn_floor: int = 6):
        self._n           = n
        self._n_drivers   = n_drivers
        self.feature      = feature
        self.spectral_norm = float(spectral_norm)
        self.density_Q    = float(density_Q)
        # connectivity: drive-coverage density for the (N,F) row mask. Q is input
        # *coverage*, not recurrence — the ER threshold is not the native argument
        # here, but the same fan-in-floored c·ln N/N is a reasonable default and
        # keeps total nonzeros quasi-linear. None falls back to fixed density_Q.
        self.connectivity = float(connectivity) if connectivity is not None else None
        self.conn_floor   = int(conn_floor)
        self.Q_weights = None       # Cheb: (N,F) | Trig: harmonic weights (H,N)
        self.frozen    = None       # Cheb: () | Trig: (Ω (H,N,K), φ (H,N))

    @property
    def _trig(self):      return isinstance(self.feature, Trig)
    @property
    def n(self):          return self._n
    @property
    def n_drivers(self):  return self._n_drivers
    def is_initialized(self): return self.Q_weights is not None

    def initialize(self, key) -> "SparseQ":
        N, K  = self._n, self._n_drivers
        gamma = jnp.sqrt(1.0 - self.spectral_norm ** 2)
        k_q, k_mask, k_feat = jax.random.split(key, 3)

        if self._trig:
            H = self.feature.degree
            W = jax.random.normal(k_q, (H, N)) * (gamma / jnp.sqrt(float(H)))   # (H,N)
            obj = SparseQ(N, K, self.feature, self.spectral_norm, self.density_Q,
                          self.connectivity, self.conn_floor)
            obj.Q_weights = W
            obj.frozen    = _trig_freqs(k_feat, self.feature, H, N, K)
            return obj

        F  = self.feature.n_joint(K)
        dQ = (log_density(N, self.connectivity, self.conn_floor)
              if self.connectivity is not None else self.density_Q)
        col_scale = jnp.concatenate([
            jnp.ones(1), jnp.full((F - 1,), 1.0 / jnp.sqrt(max(F - 1, 1)))]).astype(jnp.float32)
        Q_raw = jax.random.normal(k_q, (N, F))
        base  = jnp.zeros((N, F)).at[jnp.arange(N), jnp.arange(N) % F].set(1.0)
        rand  = (jax.random.uniform(k_mask, (N, F)) < dQ).astype(jnp.float32)
        Q = Q_raw * jnp.clip(base + rand, 0.0, 1.0) * col_scale[None, :] * gamma

        obj = SparseQ(N, K, self.feature, self.spectral_norm, self.density_Q, self.connectivity)
        obj.Q_weights, obj.frozen = Q, ()
        return obj

    def eval_q(self, z_tilde_t):
        if self._trig:
            omega, phase = self.frozen                                   # (H,N,K), (H,N)
            g = jnp.cos(jnp.einsum('hnk,k->hn', omega, z_tilde_t) + phase)  # (H,N)
            return jnp.einsum('hn,hn->n', self.Q_weights, g)
        phi = self.feature.joint_features(z_tilde_t, self.frozen, self._n_drivers)
        return self.Q_weights @ phi

    def batch_eval_q(self, z_tilde):
        if self._trig:
            omega, phase = self.frozen
            g = jnp.cos(jnp.einsum('hnk,tk->thn', omega, z_tilde) + phase[None])  # (T,H,N)
            return jnp.einsum('hn,thn->tn', self.Q_weights, g)
        phi = self.feature.joint_features(z_tilde, self.frozen, self._n_drivers)
        return phi @ self.Q_weights.T

    def tree_flatten(self):
        return (self.Q_weights, self.frozen), (
            self._n, self._n_drivers, self.feature, self.spectral_norm,
            self.density_Q, self.connectivity, self.conn_floor)
    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.Q_weights, obj.frozen = children
        return obj

    def __repr__(self):
        feat = "trig-RFF" if self._trig else "cheb-joint"
        return (f"SparseQ(n={self._n}, K={self._n_drivers}, {feat}, feature={self.feature!r})")
