"""LowRankP / LowRankQ — CP factorisation of the transition feature-tensor.

A fourth structure family (alongside Diagonal / Block / Sparse).  It replaces
``SparseP``'s 3-tensor ``P ∈ ℝ^{F×N×N}`` (``F = 1 + C(K+D, D)`` — combinatorial
in driver count K and degree D) with a single backbone plus a rank-R, input-
modulated coupling:

        A_t = M_0  +  B · U diag(α(z̃_t)) Vᵀ ,     U, V ∈ ℝ^{N×R}

All feature dependence collapses onto a bounded R-vector ``α(z̃) ∈ [-1,1]^R``; the
only N×N-sized object is the backbone ``M_0``.  **No F anywhere** — R is a free
knob.  Drive: ``q_t = G · β(z̃)`` with ``G ∈ ℝ^{N×R}`` (Q was never the F·N²
bottleneck, but shares the same feature dictionary).

Two α/β feature modes (× the Cheb/Trig spec):

* ``alpha_mode='map'`` — α = C·φ(z̃) (Cheb) or joint cosines (Trig) over K drivers,
  R modes via a frozen mixing matrix C.  General (K, R independent).
* ``alpha_mode='driver'`` (**recommended**) — **K = R**: each mode is one univariate
  feature of its own random projection, ``α_r = T_{d_r}(z̃_r)`` (Cheb, degrees
  cycled 1..D) or ``cos(ω_r z̃_r + φ_r)`` (Trig).  The Veronese/Waring frame
  (``note/projection_vs_crossinput.md``): R projection-powers span ``Poly_{≤D}``
  once ``R ≳ C(d+D, D)``, so growing R *is* growing input coverage.  No monomial
  enumeration, no C map — drivers = modes = coupling atoms (one width).

Contractivity (distribution-free).  Orthonormal U,V (R ≤ N) ⇒ singular values of
``U diag(α) Vᵀ`` are exactly ``|α_r|``, so

        ρ(A_t) ≤ ρ(M_0) + B·‖α‖∞ ≤ sn + (1-sn)·margin < 1 ,   B = (1-sn)·margin.

The bound is the **max** of ``|α_r|`` (not the sum) — all R modes ride at full
budget, no ``1/(F-1)`` shrinkage (the key advantage over ``Σ_f φ_f P_f``).  For
R > N (overcomplete) the factors are spectral-normalised instead (``‖U‖₂ ≤ 1``),
same bound.

Cost per step (sequential): ``A_t s = M_0 s + B·U(α ⊙ (Vᵀs))`` = O(N² + N·R).
Options: ``backbone=False`` drops M_0 (memory then from ``SASModel(leak<1)``'s
(1-leak)I term — O(NR) only); ``sparse_M0=True`` stores M_0 as BCOO (O(nnz) matvec,
a memory / GPU win).  Validated config: ``alpha_mode='driver'`` + ``density_G`` on
Q + ``rank`` ~ N/8 baseline, scaled up with input coupling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from .feature      import Cheb, Trig, cheb_basis
from .connectivity import log_density, connectivity_mask


# ── shared helpers ────────────────────────────────────────────────────────────

def _as_dense(M):
    """Densify M if it is a BCOO (sequential M_0 storage); pass through otherwise."""
    return M.todense() if isinstance(M, jsparse.BCOO) else M


def _sp_matvec(M_bcoo, s):
    """Sparse matvec M (N,N) BCOO · s (N,) → (N,) dense, fixed nnz, no fill-in."""
    return jsparse.bcoo_dot_general(M_bcoo, s, dimension_numbers=(([1], [0]), ([], [])))


def _backbone(key, key_mask, N, sn, density):
    """Autonomous backbone M_0 with spectral radius sn (diagonal kept dense)."""
    idx       = jnp.arange(N)
    W         = jax.random.normal(key, (N, N))
    if density is not None:
        m = connectivity_mask(key_mask, N, N, density)
        mask = jnp.clip(m.at[idx, idx].set(1.0), 0.0, 1.0)
        W    = W * mask
    rho = jnp.max(jnp.abs(jnp.linalg.eigvals(W)))
    return W * (sn / jnp.maximum(rho, 1e-8))


def _factor(key, key_mask, N, R, density):
    """An (N, R) factor with spectral norm ≤ 1 (the contractivity hook).

      * R ≤ N and dense → orthonormal columns (QR), ‖U‖₂ = 1 exactly.
      * R > N or sparse → (masked) Gaussian ÷ its spectral norm, ‖U‖₂ ≤ 1.

    The spectral-norm route frees R from the R ≤ N cap (overcomplete dictionary).
    Keep the factors **dense** — sparsifying them zeroes columns and kills modes
    (atom = u_r v_rᵀ); put sparsity on M_0 (``connectivity``) instead.  ``density``
    is kept for ablations only; default None (dense)."""
    if density is None and R <= N:
        Q, _ = jnp.linalg.qr(jax.random.normal(key, (N, R)))
        return Q                                         # ‖Q‖₂ = 1
    W = jax.random.normal(key, (N, R))
    if density is not None:
        W = W * (jax.random.uniform(key_mask, (N, R)) < density).astype(W.dtype)
    return W / jnp.maximum(jnp.linalg.norm(W, 2), 1e-8)  # ‖W‖₂ ≤ 1


def _block_ortho(key, N, R, n_blocks):
    """**Sparse AND orthonormal** factor (N, R) via disjoint block supports.

    Partition rows (randomly permuted) and columns into ``n_blocks`` groups, then
    orthonormalise the columns *within* each block by QR.  Columns in different
    blocks have disjoint supports → orthogonal; within a block → orthonormal.  So
    UᵀU = I exactly (singular values of U diag(α) Vᵀ are exactly |α_r| — full-budget
    modes), at density ≈ 1/n_blocks.  n_blocks=1 ⇒ dense orthonormal; =R ⇒ sparsest.
    Structured (block-local) sparsity — the only sparsity compatible with exact
    orthonormality (random sparse + QR densifies).  Runs eagerly at init."""
    import numpy as _np
    n_blocks = max(1, min(int(n_blocks), R))             # ≤ R (else empty col blocks)
    perm = _np.asarray(jax.random.permutation(key, N))   # randomise row→block
    row_g = _np.array_split(perm, n_blocks)
    col_g = _np.array_split(_np.arange(R), n_blocks)
    keys  = jax.random.split(jax.random.fold_in(key, 1), n_blocks)
    U = _np.zeros((N, R), _np.float32)
    for b in range(n_blocks):
        rb, cb = row_g[b], col_g[b]
        nc = min(len(cb), len(rb))                       # need cols ≤ rows for full rank
        Q  = _np.asarray(jnp.linalg.qr(jax.random.normal(keys[b], (len(rb), nc)))[0])
        U[_np.ix_(rb, cb[:nc])] = Q
    return jnp.asarray(U)


def _alpha_freqs(key, feature: Trig, R, K):
    """Joint-RFF modes ('map'): Ω (R, K), φ (R,) for α_r = cos(Ω_r·z̃ + φ_r)."""
    k_o, k_p = jax.random.split(key)
    sigma = feature._sigma(R)[:, None]                  # (R, 1)
    omega = feature._draw(k_o, (R, K), sigma)           # (R, K)
    phase = jax.random.uniform(k_p, (R,), minval=0.0, maxval=2.0 * jnp.pi)
    return omega, phase


def _alpha_cheb_map(key, feature: Cheb, R, K):
    """Linear map C (R, F) ('map'), L1-normalised rows so α = C·φ has |α_r| ≤ 1."""
    F = feature.n_joint(K)
    C = jax.random.normal(key, (R, F))
    return C / jnp.maximum(jnp.sum(jnp.abs(C), axis=1, keepdims=True), 1e-8)


def _driver_degrees(R, D):
    """Degree assignment d_r ∈ {1..D} cycled over R modes (K=R driver mode):
    α_r = T_{d_r}(z̃_r), one univariate Cheb per random projection (Veronese)."""
    return (jnp.arange(R) % D + 1).astype(jnp.int32)            # (R,)


def _driver_freqs(key, feature: Trig, R):
    """Per-driver RFF (K=R): α_r = cos(ω_r·z̃_r + φ_r), one scalar frequency per
    projection — rank-R by construction (fixes joint-cosine rank-deficiency).
    Multiscale ω_r when ``bandwidth_min/max`` are set."""
    k_o, k_p = jax.random.split(key)
    sigma = feature._sigma(R)                                   # (R,) multiscale or const
    omega = feature._draw(k_o, (R,), sigma)                     # (R,)
    phase = jax.random.uniform(k_p, (R,), minval=0.0, maxval=2.0 * jnp.pi)
    return omega, phase


# ════════════════════════════════════════════════════════════════════════════
# LowRankP
# ════════════════════════════════════════════════════════════════════════════

@jax.tree_util.register_pytree_node_class
class LowRankP:
    """Low-rank N×N transition basis: A_t = M_0 + B·U diag(α(z̃)) Vᵀ.

    Parameters
    ----------
    n, n_drivers   : reservoir size N and projected driver count K.
    feature        : Cheb or Trig spec.
    rank           : R — modes / coupling rank / width of U,V (may exceed N).
    spectral_norm  : base radius of M_0; sets B=(1-sn)·margin and the drive γ.
    connectivity   : ER log-law density of M_0 (the recurrence graph); None →
                     ``density_P``; both None → dense M_0.
    alpha_mode     : 'map' (general) | 'driver' (K=R, recommended).
    factor_density : ablation — Bernoulli density of U,V (default dense).
    sparse_M0      : store M_0 as BCOO (sequential matvec; memory/GPU win).
    backbone       : False drops M_0 (memory from SASModel leak); matvec O(NR).
    """

    def __init__(self, n: int, n_drivers: int, feature=Cheb(degree=2), rank: int = 32,
                 spectral_norm: float = 0.9, margin: float = 0.95,
                 connectivity: float | None = 1.5, conn_floor: int = 6,
                 density_P: float | None = None, training_mode: str = "sequential",
                 factor_density: float | None = None, alpha_mode: str = "map",
                 sparse_M0: bool = False, backbone: bool = True,
                 factor_blocks: int | None = None):
        self._n           = n
        self._n_drivers   = n_drivers
        self.feature      = feature
        self.rank         = int(rank)
        self.spectral_norm = float(spectral_norm)
        self.margin       = float(margin)
        self.connectivity = float(connectivity) if connectivity is not None else None
        self.conn_floor   = int(conn_floor)
        self.density_P    = float(density_P) if density_P is not None else None
        self.training_mode = training_mode
        self.factor_density = float(factor_density) if factor_density is not None else None
        # factor_blocks: if set, U,V are SPARSE *and* orthonormal (block-disjoint
        # supports, density ≈ 1/factor_blocks) — keeps the exact spectral property
        # at sparse density. Overrides factor_density. None → dense/random per above.
        self.factor_blocks = int(factor_blocks) if factor_blocks is not None else None
        self.backbone     = bool(backbone)
        self.alpha_mode   = alpha_mode
        self.sparse_M0    = bool(sparse_M0)
        if alpha_mode == "driver" and n_drivers != rank:
            raise ValueError(f"alpha_mode='driver' requires n_drivers == rank "
                             f"(K=R); got K={n_drivers}, R={rank}")
        self.budget       = ((1.0 - self.spectral_norm) * self.margin
                              if self.backbone else self.margin)
        self.M0 = None                                  # (N, N) backbone (or BCOO, or None)
        self.U  = None                                  # (N, R)
        self.V  = None                                  # (N, R)
        self.frozen = None                              # mode-dependent α params

    @property
    def _trig(self):      return isinstance(self.feature, Trig)
    @property
    def n(self):          return self._n
    @property
    def n_drivers(self):  return self._n_drivers
    def is_initialized(self): return self.U is not None

    def _density(self, N):
        if self.density_P is not None:
            return self.density_P
        if self.connectivity is not None:
            return log_density(N, self.connectivity, self.conn_floor)
        return None                                     # dense backbone

    def initialize(self, key) -> "LowRankP":
        N, K, R = self._n, self._n_drivers, self.rank
        k_b, k_bm, k_u, k_um, k_v, k_vm, k_a = jax.random.split(key, 7)
        M0 = _backbone(k_b, k_bm, N, self.spectral_norm, self._density(N)) if self.backbone else None
        if self.factor_blocks is not None:               # sparse AND orthonormal
            U = _block_ortho(k_u, N, R, self.factor_blocks)
            V = _block_ortho(k_v, N, R, self.factor_blocks)
        else:
            U = _factor(k_u, k_um, N, R, self.factor_density)
            V = _factor(k_v, k_vm, N, R, self.factor_density)
        if self.alpha_mode == "driver":
            frozen = (_driver_freqs(k_a, self.feature, R) if self._trig
                      else (_driver_degrees(R, self.feature.degree),))
        elif self._trig:
            frozen = _alpha_freqs(k_a, self.feature, R, K)
        else:
            frozen = (_alpha_cheb_map(k_a, self.feature, R, K),)

        if self.backbone and self.training_mode == "sequential" and self.sparse_M0:
            M0 = jsparse.BCOO.fromdense(M0)

        obj = LowRankP(N, K, self.feature, R, self.spectral_norm, self.margin,
                       self.connectivity, self.conn_floor, self.density_P,
                       self.training_mode, self.factor_density, self.alpha_mode,
                       self.sparse_M0, self.backbone, self.factor_blocks)
        obj.M0, obj.U, obj.V, obj.frozen = M0, U, V, frozen
        return obj

    # ── α(z̃) ──────────────────────────────────────────────────────────────────

    def _alpha(self, z_tilde_t):                        # (K,) → (R,)
        if self.alpha_mode == "driver":                 # K=R: one feature per projection
            if self._trig:
                omega, phase = self.frozen
                return jnp.cos(omega * z_tilde_t + phase)
            (degs,) = self.frozen
            T = cheb_basis(z_tilde_t, self.feature.degree)             # (R, D+1)
            return jnp.take_along_axis(T, degs[:, None], axis=1)[:, 0]  # (R,)
        if self._trig:
            omega, phase = self.frozen
            return jnp.cos(omega @ z_tilde_t + phase)
        (C,) = self.frozen
        phi = self.feature.joint_features(z_tilde_t, (), self._n_drivers)   # (F,)
        return C @ phi

    def _alpha_batch(self, z_tilde):                    # (T,K) → (T,R)
        if self.alpha_mode == "driver":
            if self._trig:
                omega, phase = self.frozen
                return jnp.cos(z_tilde * omega[None] + phase[None])    # (T,R)
            (degs,) = self.frozen
            T = cheb_basis(z_tilde, self.feature.degree)               # (T, R, D+1)
            return jnp.take_along_axis(T, degs[None, :, None], axis=2)[..., 0]  # (T,R)
        if self._trig:
            omega, phase = self.frozen
            return jnp.cos(z_tilde @ omega.T + phase[None])
        (C,) = self.frozen
        phi = self.feature.joint_features(z_tilde, (), self._n_drivers)     # (T,F)
        return phi @ C.T

    # ── per-step evaluators ────────────────────────────────────────────────────

    def matvec_p(self, z_tilde_t, s):
        """A_t @ s = M_0 s + B·U(α ⊙ (Vᵀ s)) — never materialises the low-rank term.
        M_0 s is a sparse matvec when M_0 is BCOO; skipped when backbone=False."""
        alpha = self._alpha(z_tilde_t)                  # (R,)
        low   = self.budget * (self.U @ (alpha * (self.V.T @ s)))
        if self.M0 is None:
            return low                                  # O(NR) only — no N² term
        M0s = _sp_matvec(self.M0, s) if isinstance(self.M0, jsparse.BCOO) else self.M0 @ s
        return M0s + low

    def eval_p(self, z_tilde_t):
        alpha = self._alpha(z_tilde_t)                  # (R,)
        low   = self.budget * ((self.U * alpha[None, :]) @ self.V.T)
        return low if self.M0 is None else _as_dense(self.M0) + low

    def batch_eval_p(self, z_tilde):
        alpha = self._alpha_batch(z_tilde)              # (T,R)
        low   = self.budget * jnp.einsum('tr,nr,mr->tnm', alpha, self.U, self.V)
        return low if self.M0 is None else _as_dense(self.M0)[None] + low

    # ── standardized sequential-scan interface (batched features + lean matvec) ──
    # The fast training scan precomputes input-only features once, then scans with
    # scan_matvec.  For LowRank the win is scan_prep stacking [M_0;Vᵀ] into a single
    # row-major GEMV (one matmul + Vᵀ laid out contiguously, instead of M_0 s and a
    # cache-hostile transposed Vᵀ s every step).  See engine._fast_seq_scan.

    def scan_features(self, z_tilde):                   # (T,K) → (T,R)
        return self._alpha_batch(z_tilde)

    def scan_prep(self):
        # Stack [M_0 ; Vᵀ] → (N+R, N); only when M_0 is a dense backbone.
        if self.M0 is not None and not isinstance(self.M0, jsparse.BCOO):
            return jnp.concatenate([self.M0, self.V.T], axis=0)
        return None

    def scan_matvec(self, prep, alpha_t, s):            # A_t @ s, features precomputed
        if prep is not None:                            # stacked fast path
            st = prep @ s
            m0s, vts = st[:self._n], st[self._n:]
            return m0s + self.budget * (self.U @ (alpha_t * vts))
        low = self.budget * (self.U @ (alpha_t * (self.V.T @ s)))
        if self.M0 is None:
            return low
        M0s = _sp_matvec(self.M0, s) if isinstance(self.M0, jsparse.BCOO) else self.M0 @ s
        return M0s + low

    # ── monoid (dense matmul; used only by the parallel scan) ───────────────────

    def apply(self, A, s):       return jnp.matmul(A, s)
    def combine(self, i, j):
        A_i, b_i = i; A_j, b_j = j
        return jnp.matmul(A_j, A_i), jnp.matmul(A_j, b_i[..., None]).squeeze(-1) + b_j

    def leaky(self, P_seq, leak):
        eye = jnp.eye(self._n, dtype=P_seq.dtype)
        return leak * P_seq + (1.0 - leak) * eye

    # ── pytree ──────────────────────────────────────────────────────────────────

    def tree_flatten(self):
        return ((self.M0, self.U, self.V, self.frozen), (
            self._n, self._n_drivers, self.feature, self.rank, self.spectral_norm,
            self.margin, self.connectivity, self.conn_floor, self.density_P,
            self.training_mode, self.factor_density, self.alpha_mode, self.sparse_M0,
            self.backbone, self.factor_blocks))
    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.M0, obj.U, obj.V, obj.frozen = children
        return obj

    def __repr__(self):
        feat = "trig" if self._trig else "cheb"
        bb   = "" if self.backbone else ", no-M0"
        return (f"LowRankP(n={self._n}, K={self._n_drivers}, R={self.rank}, {feat}, "
                f"{self.alpha_mode}, sn={self.spectral_norm}{bb})")


# ════════════════════════════════════════════════════════════════════════════
# LowRankQ
# ════════════════════════════════════════════════════════════════════════════

@jax.tree_util.register_pytree_node_class
class LowRankQ:
    """Low-rank drive basis: q_t = G · β(z̃), β bounded in [-1,1] (same modes as P).

    density_G : None → dense G (entries γ/√R).  float → **sparse G** (recommended):
                Bernoulli mask (≥1/row, no dead unit) → column-L1 normalise (equal
                per-feature mass) → global rescale to the dense drive energy
                E‖q‖²≈Nγ².  G is frozen random (only the ridge readout is trained).
    """

    def __init__(self, n: int, n_drivers: int, feature=Cheb(degree=2), rank: int = 32,
                 spectral_norm: float = 0.9, alpha_mode: str = "map",
                 density_G: float | None = None, gamma: float | None = None):
        self._n           = n
        self._n_drivers   = n_drivers
        self.feature      = feature
        self.rank         = int(rank)
        self.spectral_norm = float(spectral_norm)
        self.alpha_mode   = alpha_mode
        self.density_G    = float(density_G) if density_G is not None else None
        # gamma: drive amplitude. None → echo-state calibration √(1−sn²) (stationary
        # state variance ≈1). A float overrides it (e.g. 1.0 to decouple from sn).
        self.gamma        = float(gamma) if gamma is not None else None
        if alpha_mode == "driver" and n_drivers != rank:
            raise ValueError(f"alpha_mode='driver' requires n_drivers == rank "
                             f"(K=R); got K={n_drivers}, R={rank}")
        self.G      = None                              # (N, R)
        self.frozen = None                              # mode-dependent β params

    @property
    def _trig(self):      return isinstance(self.feature, Trig)
    @property
    def n(self):          return self._n
    @property
    def n_drivers(self):  return self._n_drivers
    def is_initialized(self): return self.G is not None

    def initialize(self, key) -> "LowRankQ":
        N, K, R = self._n, self._n_drivers, self.rank
        gamma = self.gamma if self.gamma is not None else jnp.sqrt(1.0 - self.spectral_norm ** 2)
        k_g, k_a, k_m = jax.random.split(key, 3)
        G = jax.random.normal(k_g, (N, R))
        if self.density_G is not None:
            G = G * connectivity_mask(k_m, N, R, self.density_G)
            G = G / jnp.maximum(jnp.sum(jnp.abs(G), axis=0, keepdims=True), 1e-8)
            G = G * (gamma * jnp.sqrt(float(N)) / jnp.maximum(jnp.linalg.norm(G, "fro"), 1e-8))
        else:
            G = G * (gamma / jnp.sqrt(float(R)))        # dense: E‖q‖² ≈ γ²
        if self.alpha_mode == "driver":
            frozen = (_driver_freqs(k_a, self.feature, R) if self._trig
                      else (_driver_degrees(R, self.feature.degree),))
        elif self._trig:
            frozen = _alpha_freqs(k_a, self.feature, R, K)
        else:
            frozen = (_alpha_cheb_map(k_a, self.feature, R, K),)

        obj = LowRankQ(N, K, self.feature, R, self.spectral_norm, self.alpha_mode,
                       self.density_G, self.gamma)
        obj.G, obj.frozen = G, frozen
        return obj

    def _beta(self, z_tilde_t):
        if self.alpha_mode == "driver":
            if self._trig:
                omega, phase = self.frozen
                return jnp.cos(omega * z_tilde_t + phase)
            (degs,) = self.frozen
            T = cheb_basis(z_tilde_t, self.feature.degree)
            return jnp.take_along_axis(T, degs[:, None], axis=1)[:, 0]
        if self._trig:
            omega, phase = self.frozen
            return jnp.cos(omega @ z_tilde_t + phase)
        (C,) = self.frozen
        return C @ self.feature.joint_features(z_tilde_t, (), self._n_drivers)

    def _beta_batch(self, z_tilde):
        if self.alpha_mode == "driver":
            if self._trig:
                omega, phase = self.frozen
                return jnp.cos(z_tilde * omega[None] + phase[None])
            (degs,) = self.frozen
            T = cheb_basis(z_tilde, self.feature.degree)
            return jnp.take_along_axis(T, degs[None, :, None], axis=2)[..., 0]
        if self._trig:
            omega, phase = self.frozen
            return jnp.cos(z_tilde @ omega.T + phase[None])
        (C,) = self.frozen
        return self.feature.joint_features(z_tilde, (), self._n_drivers) @ C.T

    def eval_q(self, z_tilde_t):       return self.G @ self._beta(z_tilde_t)
    def batch_eval_q(self, z_tilde):   return self._beta_batch(z_tilde) @ self.G.T

    def tree_flatten(self):
        return ((self.G, self.frozen), (
            self._n, self._n_drivers, self.feature, self.rank, self.spectral_norm,
            self.alpha_mode, self.density_G, self.gamma))
    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(*aux)
        obj.G, obj.frozen = children
        return obj

    def __repr__(self):
        feat = "trig" if self._trig else "cheb"
        sg   = "" if self.density_G is None else f", sparseG({self.density_G})"
        return f"LowRankQ(n={self._n}, K={self._n_drivers}, R={self.rank}, {feat}, {self.alpha_mode}{sg})"
