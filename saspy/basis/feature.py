"""Bounded feature maps on [-1, 1] (Layer-2 building block).

Two feature *specs* — :class:`Cheb` (Chebyshev polynomial) and :class:`Trig`
(random cosine / RFF) — each usable for the transition P or the drive Q, on
either structure family:

  * **scalar** features  — one scalar driver per unit/block (Diagonal, Block).
                           ``z (..., M) -> phi (..., M, F)``
  * **joint**  features  — K drivers mixed together     (Sparse).
                           ``z (..., K) -> phi (..., F)``

The whole point of working on the compact domain [-1, 1] is that *every* feature
is bounded in [-1, 1] (``|T_d| <= 1`` for Chebyshev, ``|cos| <= 1``).  That makes
the P/Q initialisation budget uniform and **distribution-free** — no Gaussian
moments, no ``max_input**degree`` corrections, no per-degree variance matching.

Index 0 of every feature vector is the constant ``1`` (so weight row 0 is the
base eigenvalue for P / the bias for Q); the remaining entries are the bounded
modulation features.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp


# ── monomial enumeration (shared feature-count rule) ─────────────────────────

def monomial_exponents(K: int, D: int, cross_input: bool) -> np.ndarray:
    """Exponent matrix (R, K) for all monomials of total degree 1..D.

    cross_input=True  — all multivariate monomials (z_i·z_j included).
    cross_input=False — only pure-power monomials z_k^d.
    Degree-major order; each cross product appears exactly once.
    """
    rows = []
    if cross_input:
        for d in range(1, D + 1):
            for combo in itertools.combinations_with_replacement(range(K), d):
                exp = np.zeros(K, dtype=np.int32)
                for idx in combo:
                    exp[idx] += 1
                rows.append(exp)
    else:
        for d in range(1, D + 1):
            for k in range(K):
                exp = np.zeros(K, dtype=np.int32)
                exp[k] = d
                rows.append(exp)
    return np.array(rows, dtype=np.int32) if rows else np.zeros((0, K), dtype=np.int32)


# ── Chebyshev recurrence ─────────────────────────────────────────────────────

def cheb_basis(z, degree: int):
    """Chebyshev polynomials T_0..T_degree evaluated at z.

    z: (...,) → (..., degree+1).  T_0 = 1, T_1 = z, T_{k} = 2z·T_{k-1} − T_{k-2}.
    Bounded in [-1, 1] whenever |z| <= 1.
    """
    T = [jnp.ones_like(z)]
    if degree >= 1:
        T.append(z)
    for k in range(2, degree + 1):
        T.append(2.0 * z * T[-1] - T[-2])
    return jnp.stack(T, axis=-1)   # (..., degree+1)


# ════════════════════════════════════════════════════════════════════════════
# Feature specs
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Cheb:
    """Chebyshev polynomial features (bounded, deterministic — no frozen params).

    degree      : highest Chebyshev degree D.
    cross_input : joint (Sparse) only — include cross monomials z_i·z_j.
    """
    degree:      int  = 2
    cross_input: bool = True

    # ── feature counts (incl. the constant feature at index 0) ──
    def n_scalar(self) -> int:
        return self.degree + 1

    def n_joint(self, K: int) -> int:
        return 1 + len(monomial_exponents(K, self.degree, self.cross_input))

    # ── frozen parameters: none for Chebyshev ──
    def init_scalar(self, key, n_units: int):
        return ()

    def init_joint(self, key, K: int):
        return ()

    # ── scalar features: z (..., M) → (..., M, D+1) ──
    def scalar_features(self, z, frozen):
        return cheb_basis(z, self.degree)

    # ── joint features: z (..., K) → (..., F) ──
    def joint_features(self, z, frozen, K: int):
        exps  = jnp.asarray(monomial_exponents(K, self.degree, self.cross_input))  # (R, K) int
        batch = z.shape[:-1]
        Tz    = cheb_basis(z, self.degree)                      # (..., K, D+1)
        Tz_b  = Tz[..., None, :, :]                             # (..., 1, K, D+1)
        idx   = exps.reshape((1,) * len(batch) + exps.shape + (1,))  # (..1.., R, K, 1)
        # gather T_{e_{f,k}}(z_k) for each monomial f, then product over k
        gathered = jnp.take_along_axis(Tz_b, idx, axis=-1)[..., 0]   # (..., R, K)
        monos = jnp.prod(gathered, axis=-1)                     # (..., R)
        ones  = jnp.ones(batch + (1,), dtype=z.dtype)           # (..., 1)
        return jnp.concatenate([ones, monos], axis=-1)          # (..., F)


@dataclass(frozen=True)
class Trig:
    """Random cosine (RFF) features: phi = [1, cos(omega·z + phase)].

    degree        : controls feature count (same rule as Cheb); for scalar
                    features it is the number of cosine harmonics per unit.
    bandwidth     : length-scale sigma for frequency sampling.
    kernel        : 'gaussian' (N(0,1/sigma²)) or 'laplace' (Cauchy(0,1/sigma)).
    density_omega : joint only — Bernoulli density of each frequency row.
    bandwidth_min/max : if both set, per-feature bandwidth is log-spaced.
    """
    degree:        int          = 2
    bandwidth:     float        = 1.0
    kernel:        str          = "gaussian"
    density_omega: float        = 1.0
    bandwidth_min: float | None = None
    bandwidth_max: float | None = None

    def n_scalar(self) -> int:
        return self.degree + 1      # [1, cos_1, ..., cos_degree]

    def n_joint(self, K: int) -> int:
        return 1 + len(monomial_exponents(K, self.degree, True))

    # ── per-feature std of the frequency draw ──
    def _sigma(self, R: int):
        if self.bandwidth_min is not None and self.bandwidth_max is not None:
            return 1.0 / jnp.exp(jnp.linspace(
                jnp.log(self.bandwidth_min), jnp.log(self.bandwidth_max), R))
        return jnp.full((R,), 1.0 / self.bandwidth)

    def _draw(self, key, shape, sigma):
        if self.kernel == "gaussian":
            return jax.random.normal(key, shape) * sigma
        if self.kernel == "laplace":
            return jax.random.cauchy(key, shape) * sigma
        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    # ── frozen params for scalar use: (Omega (n_units, H), Phase (n_units, H)) ──
    def init_scalar(self, key, n_units: int):
        H = self.degree
        k_o, k_p = jax.random.split(key)
        sigma = self._sigma(H)[None, :]                          # (1, H)
        omega = self._draw(k_o, (n_units, H), sigma)             # (n_units, H)
        phase = jax.random.uniform(k_p, (n_units, H), minval=0.0, maxval=2.0 * jnp.pi)
        return (omega, phase)

    # ── frozen params for joint use: (Omega (R, K), Phase (R,)) ──
    def init_joint(self, key, K: int):
        R = self.n_joint(K) - 1
        k_o, k_mask, k_p = jax.random.split(key, 3)
        sigma     = self._sigma(R)[:, None]                      # (R, 1)
        omega_raw = self._draw(k_o, (R, K), sigma)               # (R, K)
        mask      = (jax.random.uniform(k_mask, (R, K)) < self.density_omega).astype(omega_raw.dtype)
        # guarantee >= 1 active frequency per row
        forced = jnp.argmax(jnp.abs(omega_raw), axis=1)
        fix    = jnp.zeros((R, K), omega_raw.dtype).at[jnp.arange(R), forced].set(1.0)
        mask   = jnp.where(mask.sum(axis=1, keepdims=True) == 0, fix, mask)
        omega  = omega_raw * mask
        phase  = jax.random.uniform(k_p, (R,), minval=0.0, maxval=2.0 * jnp.pi)
        return (omega, phase)

    # ── scalar features: z (..., M) → (..., M, H+1) ──
    def scalar_features(self, z, frozen):
        omega, phase = frozen                                   # (M, H) each
        angles = z[..., None] * omega + phase                   # (..., M, H)
        cosines = jnp.cos(angles)
        ones    = jnp.ones(z.shape + (1,), dtype=z.dtype)       # (..., M, 1)
        return jnp.concatenate([ones, cosines], axis=-1)        # (..., M, H+1)

    # ── joint features: z (..., K) → (..., F) ──
    def joint_features(self, z, frozen, K: int):
        omega, phase = frozen                                   # (R, K), (R,)
        angles  = z @ omega.T + phase                           # (..., R)
        cosines = jnp.cos(angles)
        ones    = jnp.ones(z.shape[:-1] + (1,), dtype=z.dtype)  # (..., 1)
        return jnp.concatenate([ones, cosines], axis=-1)        # (..., F)


# Type alias for annotations
Feature = (Cheb, Trig)
