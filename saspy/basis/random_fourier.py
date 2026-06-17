"""
random_fourier.py — RandomFourierBasis: RFF Q + DiagonalPoly P, one feature per unit.

Each of the N units has its own dedicated scalar driver z_tilde[k] from the projector
(n_drivers = N, same interface as DiagonalPoly) and its own frequency Omega[k].

Q side  (eval_q / batch_eval_q)
--------------------------------
    q[k] = sqrt(2) · cos(Omega[k] · z_tilde[k] + Phase[k])

With bandwidth_min / bandwidth_max, Omega[k] is drawn from a log-spaced mixture of
bandwidths:  Omega[k] ~ N(0, sigma_k²)  where sigma_k = 1 / bw_k and
bw_k = exp(linspace(log(bw_min), log(bw_max), N)).  This creates a frequency
continuum: low-bandwidth units capture high-frequency / fast features, while
high-bandwidth units capture slow / smooth features.

Without bandwidth_min / bandwidth_max, all units share the same single bandwidth.

P side  (eval_p / batch_eval_p)
--------------------------------
Exact DiagonalPoly polynomial on z_tilde:
    A_t[k] = Σ_d P_weights[d, k] · z_tilde[k]^d        clipped to (−1, 1)

Usage
-----
    basis = RandomFourierBasis(n=N, bandwidth_min=0.1, bandwidth_max=10.0, p_degree=2)
    model = SASModel(InputProjector.trivial(n_drivers=N), basis, basis)
"""

import jax
import jax.numpy as jnp

from .base import BaseBasis


@jax.tree_util.register_pytree_node_class
class RandomFourierBasis(BaseBasis):
    """
    Random Fourier Features Q + DiagonalPoly P.  n_drivers = N.

    Parameters
    ----------
    n             : reservoir size N (= n_blocks · 1 = n_drivers).
    kernel_type   : "gaussian" (RBF) or "laplace".
    bandwidth     : length-scale when bandwidth_min/max are not set.
    spectral_norm : base eigenvalue range — P[0, k] ∈ (−sn, sn).
    p_degree      : polynomial degree of z_tilde in the P transition.
    max_input     : clip |z_tilde[k]| before polynomial evaluation.  None → no clip.
    budget_ref    : scale reference for P weight budget.
                    None → falls back to max_input if set, else 1.0.
    bandwidth_min : if set together with bandwidth_max, unit k gets bandwidth
                    bw_k = exp(linspace(log(bw_min), log(bw_max), N)[k]).
    bandwidth_max : upper bound of the per-unit bandwidth range.
    """

    def __init__(
        self,
        n:             int,
        kernel_type:   str          = "gaussian",
        bandwidth:     float        = 1.0,
        spectral_norm: float        = 0.99,
        p_degree:      int          = 1,
        max_input:     float | None = 4.0,
        budget_ref:    float | None = None,
        bandwidth_min: float | None = None,
        bandwidth_max: float | None = None,
    ):
        super().__init__(p_degree=p_degree, q_degree=0)
        self._n            = n
        self.kernel_type   = kernel_type.lower()
        self.bandwidth     = float(bandwidth)
        self.spectral_norm = float(spectral_norm)
        self.max_input     = float(max_input) if max_input is not None else None
        self.budget_ref    = float(budget_ref) if budget_ref is not None else None
        self.bandwidth_min = float(bandwidth_min) if bandwidth_min is not None else None
        self.bandwidth_max = float(bandwidth_max) if bandwidth_max is not None else None

        self.Omega_weights = None   # (N,)
        self.Phase_weights = None   # (N,)
        self.P_weights     = None   # (p_degree+1, N)
        # Q_weights inherited as None from BaseBasis; unused here (RFF uses Omega/Phase)

    # ── dimensions ───────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return self._n

    @property
    def n_drivers(self) -> int:
        return self._n

    # ── factory ──────────────────────────────────────────────────────────────

    def initialize(self, key) -> "RandomFourierBasis":
        N  = self._n
        sn = self.spectral_norm

        # 2 keys for Q (freq + phase), p_degree+1 keys for P (degrees 0..D)
        keys   = jax.random.split(key, self.p_degree + 3)
        k_omega, k_phase = keys[0], keys[1]
        keys_P = keys[2:]

        # ── Q: per-unit frequencies (multi-scale if bandwidth_min/max set) ─
        use_multiscale = (self.bandwidth_min is not None
                          and self.bandwidth_max is not None)
        if use_multiscale:
            bw_k    = jnp.exp(jnp.linspace(
                jnp.log(self.bandwidth_min), jnp.log(self.bandwidth_max), N,
            ))                                      # (N,) log-spaced bandwidths
            sigma_k = 1.0 / bw_k                   # (N,) per-unit std
        else:
            sigma_k = jnp.full((N,), 1.0 / self.bandwidth)

        if self.kernel_type == "gaussian":
            omega = jax.random.normal(k_omega, (N,)) * sigma_k
        elif self.kernel_type == "laplace":
            omega = jax.random.cauchy(k_omega, (N,)) * sigma_k
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type!r}")

        phase = jax.random.uniform(k_phase, (N,), minval=0.0, maxval=2.0 * jnp.pi)

        # ── P: DiagonalPoly-style polynomial ─────────────────────────────
        p0       = (jax.random.uniform(keys_P[0], (N,)) * 2 - 1) * sn
        p_rows   = [p0]
        headroom = jnp.maximum(1.0 - jnp.abs(p0) - 0.01, 0.0)
        budget   = headroom * 0.5
        scale_ref = self._budget_ref()
        for k in range(1, self.p_degree + 1):
            scale = (budget / (2.0 ** k)) / (scale_ref ** k)
            raw   = jax.random.normal(keys_P[k], (N,))
            p_rows.append(jnp.clip(raw, -1.0, 1.0) * scale)
        P = jnp.stack(p_rows, axis=0)   # (p_degree+1, N)

        obj = RandomFourierBasis(N, self.kernel_type, self.bandwidth,
                                 sn, self.p_degree, self.max_input, self.budget_ref,
                                 self.bandwidth_min, self.bandwidth_max)
        obj.Omega_weights = omega
        obj.Phase_weights = phase
        obj.P_weights     = P
        return obj

    # ── per-step evaluators ───────────────────────────────────────────────────

    def eval_q(self, z_tilde_t):
        """z_tilde_t: (N,) → q: (N,).  Element-wise RFF."""
        angles = z_tilde_t * self.Omega_weights + self.Phase_weights
        return jnp.sqrt(2.0) * jnp.cos(angles)

    def eval_p(self, z_tilde_t):
        """z_tilde_t: (N,) → a: (N,).  DiagonalPoly polynomial per unit."""
        z = (jnp.clip(z_tilde_t, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde_t)
        powers = jnp.arange(self.p_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[None, :], powers[:, None])           # (p+1, N)
        return jnp.clip(
            jnp.einsum('dn,dn->n', feats, self.P_weights),
            -0.9999, 0.9999,
        )

    # ── batched evaluators ────────────────────────────────────────────────────

    def batch_eval_q(self, z_tilde):
        """z_tilde: (T, N) → (T, N)."""
        angles = z_tilde * self.Omega_weights[None, :] + self.Phase_weights[None, :]
        return jnp.sqrt(2.0) * jnp.cos(angles)

    def batch_eval_p(self, z_tilde):
        """z_tilde: (T, N) → (T, N)."""
        z = (jnp.clip(z_tilde, -self.max_input, self.max_input)
             if self.max_input is not None else z_tilde)
        powers = jnp.arange(self.p_degree + 1, dtype=jnp.float32)
        feats  = jnp.power(z[:, None, :], powers[None, :, None])  # (T, p+1, N)
        return jnp.clip(
            jnp.einsum('tdn,dn->tn', feats, self.P_weights),
            -0.9999, 0.9999,
        )

    # ── algebraic primitives ─────────────────────────────────────────────────

    def apply(self, A, s):
        """A: (N,) diagonal eigenvalues, s: (N,) → (N,)."""
        return A * s

    def combine(self, i, j):
        """Element-wise affine monoid for associative scan."""
        a_i, b_i = i
        a_j, b_j = j
        return a_j * a_i, a_j * b_i + b_j

    # ── pytree ────────────────────────────────────────────────────────────────

    def tree_flatten(self):
        children = (self.Omega_weights, self.Phase_weights, self.P_weights)
        aux = (
            self._n, self.kernel_type, self.bandwidth,
            self.spectral_norm, self.p_degree, self.max_input,
            self.budget_ref, self.bandwidth_min, self.bandwidth_max,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (n, kernel_type, bandwidth, spectral_norm, p_degree,
         max_input, budget_ref, bw_min, bw_max) = aux
        obj = cls(n, kernel_type, bandwidth, spectral_norm, p_degree,
                  max_input, budget_ref, bw_min, bw_max)
        obj.Omega_weights, obj.Phase_weights, obj.P_weights = children
        return obj

    def __repr__(self) -> str:
        bw_str = (f"bw=[{self.bandwidth_min},{self.bandwidth_max}]"
                  if self.bandwidth_min is not None
                  else f"bw={self.bandwidth}")
        return (f"RandomFourierBasis(n={self._n}, kernel={self.kernel_type!r}, "
                f"{bw_str}, p_degree={self.p_degree}, sn={self.spectral_norm})")
