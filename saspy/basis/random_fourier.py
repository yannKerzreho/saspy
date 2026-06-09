"""
random_fourier.py — RandomFourierBasis: kernel approximation via Bochner's Theorem.

The feature map per driver k at time t is:
    phi_q(z_tilde[k]) = sqrt(2/B) * cos(Omega[k] * z_tilde[k] + Phase[k])

The transition (eval_p) uses log-spaced fixed eigenvalues (Rho_base), NOT
cosines.  Root cause of the original underperformance:

  - Cosine-based eval_p had ~46% negative eigenvalues and mean effective
    timescale of only ~6 steps, destroying long-range memory.
  - Fixed log-spaced Rho_base gives controlled timescales from tau_min to
    tau_max (identical philosophy to LRUBlockPoly).

Multi-scale bandwidth
---------------------
Setting bandwidth_min < bandwidth_max assigns each driver k its own bandwidth
sampled log-uniformly in [bandwidth_min, bandwidth_max].  This creates a
frequency mixture: low-bandwidth drivers (large sigma) pick up high-frequency
features (useful for chaotic/nonlinear tasks), while high-bandwidth drivers
(small sigma) pick up slow, smooth features (useful for periodic/PDE tasks).
Without this option, all K drivers share the same single bandwidth.

Representational limits and the KRR connection
-----------------------------------------------
RFF standalone approximates the Echo State Kernel, NOT standard KRR on sliding
windows:

    K_echo(T,S) = Σ_{t,u} ρ^{T-t} · ρ^{S-u} · k_gauss(z_t, z_u)

Standard KRR can represent arbitrary periodic functions (learned weights can
have any sign/magnitude).  The Echo State Kernel with real positive Rho
restricts the function class to positive-weighted sums of decaying exponentials
— which CANNOT resonate with arbitrary periodic frequencies.

Empirically confirmed: changing Rho sign (positive/negative/mixed) has zero
effect on MSO-8 performance.  The limitation is architectural: real diagonal
eigenvalues can only produce DC (ρ>0) or period-2 (ρ<0) oscillations.  For
signals with arbitrary periods (MSO-8: 8 frequencies, periods 1.08–5 steps),
complex eigenvalues are required (LRUBlockPoly rotation blocks).

Rule of thumb:
  - Chaotic / bounded nonlinear (Lorenz, KS): RFF Q + diagonal P → excellent.
  - Strictly periodic (MSO-8, harmonic signals): LRU P required; Q type secondary.

When to prefer RFF Q over polynomial Q
---------------------------------------
RFF Q is worth considering when:
1. Inputs have large magnitude (bounded cosines vs unbounded polynomial Q).
2. Multivariate inputs (d>1) with nonlinear cross-channel interactions.
3. Multi-scale bandwidth is needed (set bandwidth_min / bandwidth_max).
For univariate smooth tasks, DiagonalPoly(q_degree=1) generally outperforms.
"""

import jax
import jax.numpy as jnp

from .base import BaseBasis


@jax.tree_util.register_pytree_node_class
class RandomFourierBasis(BaseBasis):
    """
    Random Fourier Features (RFF) basis emulating RBF or Laplace kernels.

    Parameters
    ----------
    n_blocks          : K — number of parallel input channels (= n_drivers).
    features_per_block: B — embedding dimension per channel; total N = K * B.
    kernel_type       : "gaussian" (RBF) or "laplace".
    bandwidth         : length-scale of the target kernel (used when
                        bandwidth_min/max are not set).
    spectral_norm     : maximum eigenvalue for eval_p.
    tau_min           : shortest timescale; eigenvalue = exp(-1/tau_min).
    tau_max           : longest  timescale; eigenvalue = exp(-1/tau_max).
    bandwidth_min     : if set together with bandwidth_max, each driver k
                        gets its own bandwidth drawn log-uniformly from
                        [bandwidth_min, bandwidth_max].  Enables multi-scale
                        frequency coverage.
    bandwidth_max     : upper bound of the per-driver bandwidth range.
    """

    def __init__(
        self,
        n_blocks:           int          = 50,
        features_per_block: int          = 4,
        kernel_type:        str          = "gaussian",
        bandwidth:          float        = 1.0,
        spectral_norm:      float        = 0.99,
        tau_min:            float        = 1.0,
        tau_max:            float        = 100.0,
        bandwidth_min:      float | None = None,
        bandwidth_max:      float | None = None,
    ):
        super().__init__(p_degree=0, q_degree=0)
        self.K             = n_blocks
        self.B             = features_per_block
        self.kernel_type   = kernel_type.lower()
        self.bandwidth     = float(bandwidth)
        self.spectral_norm = float(spectral_norm)
        self.tau_min       = float(tau_min)
        self.tau_max       = float(tau_max)
        self.bandwidth_min = float(bandwidth_min) if bandwidth_min is not None else None
        self.bandwidth_max = float(bandwidth_max) if bandwidth_max is not None else None

        self.Omega_weights = None  # (K, B) — Q frequencies
        self.Phase_weights = None  # (K, B) — Q phases
        self.Rho_base      = None  # (N,)   — P eigenvalues, log-spaced

    # ── initialization check override ────────────────────────────────────────

    def is_initialized(self) -> bool:
        return self.Rho_base is not None

    # ── dimensions ───────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return self.K * self.B

    @property
    def n_drivers(self) -> int:
        return self.K

    # ── factory ──────────────────────────────────────────────────────────────

    def initialize(self, key) -> "RandomFourierBasis":
        k1, k2 = jax.random.split(key)

        # ── Q frequencies ────────────────────────────────────────────────────
        use_multiscale = (self.bandwidth_min is not None
                          and self.bandwidth_max is not None)

        if use_multiscale:
            # Per-driver bandwidth: log-spaced from bandwidth_min to bandwidth_max.
            # Large bandwidth → small sigma → low frequencies (slow features).
            # Small bandwidth → large sigma → high frequencies (fast features).
            bw_k = jnp.exp(
                jnp.linspace(jnp.log(self.bandwidth_min),
                             jnp.log(self.bandwidth_max), self.K)
            )                                       # (K,), ascending
            sigma_k = (1.0 / bw_k)[:, None]        # (K, 1)
            raw = jax.random.normal(k1, (self.K, self.B))
            if self.kernel_type == "gaussian":
                omega = raw * sigma_k
            elif self.kernel_type == "laplace":
                # Cauchy: reuse normal pair trick (ratio of two normals)
                raw2 = jax.random.normal(k1, (self.K, self.B))
                omega = (raw / (raw2 + 1e-8)) * sigma_k
            else:
                raise ValueError(f"Unknown kernel type: {self.kernel_type!r}")
        else:
            if self.kernel_type == "gaussian":
                sigma = 1.0 / self.bandwidth
                omega = jax.random.normal(k1, (self.K, self.B)) * sigma
            elif self.kernel_type == "laplace":
                gamma = 1.0 / self.bandwidth
                omega = jax.random.cauchy(k1, (self.K, self.B)) * gamma
            else:
                raise ValueError(f"Unknown kernel type: {self.kernel_type!r}")

        # ── Q phases: uniform in [0, 2π) ─────────────────────────────────────
        b = jax.random.uniform(k2, (self.K, self.B), minval=0.0, maxval=2.0 * jnp.pi)

        # ── P eigenvalues: log-spaced timescales ─────────────────────────────
        N    = self.K * self.B
        taus = jnp.exp(
            jnp.linspace(jnp.log(self.tau_min), jnp.log(self.tau_max), N)
        )
        rho = jnp.exp(-1.0 / taus) * self.spectral_norm

        obj = RandomFourierBasis(
            self.K, self.B, self.kernel_type, self.bandwidth,
            self.spectral_norm, self.tau_min, self.tau_max,
            self.bandwidth_min, self.bandwidth_max,
        )
        obj.Omega_weights = omega
        obj.Phase_weights = b
        obj.Rho_base      = rho
        return obj

    # ── per-step evaluators ───────────────────────────────────────────────────

    def eval_q(self, z_tilde_t):
        """z_tilde_t: (K,) → q: (N,)  — random Fourier kernel feature map."""
        angles = z_tilde_t[:, None] * self.Omega_weights + self.Phase_weights
        return (jnp.sqrt(2.0 / self.B) * jnp.cos(angles)).reshape(self.n)

    def eval_p(self, z_tilde_t):
        """
        z_tilde_t: (K,) → A: (N,)  — FIXED log-spaced eigenvalues.

        Input-independent: all positive, multi-scale timescales in
        (exp(-1/tau_max)*sn, exp(-1/tau_min)*sn].
        """
        return self.Rho_base

    # ── batched evaluators ────────────────────────────────────────────────────

    def batch_eval_q(self, z_tilde):
        """z_tilde: (T, K) → Q_seq: (T, N)."""
        angles = (z_tilde[:, :, None] * self.Omega_weights[None, :, :]
                  + self.Phase_weights[None, :, :])
        return (jnp.sqrt(2.0 / self.B) * jnp.cos(angles)).reshape(z_tilde.shape[0], self.n)

    def batch_eval_p(self, z_tilde):
        """z_tilde: (T, K) → P_seq: (T, N) — constant across time."""
        T = z_tilde.shape[0]
        return jnp.broadcast_to(self.Rho_base[None, :], (T, self.n))

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
        children = (self.Omega_weights, self.Phase_weights, self.Rho_base)
        aux = (
            self.K, self.B, self.kernel_type, self.bandwidth,
            self.spectral_norm, self.tau_min, self.tau_max,
            self.bandwidth_min, self.bandwidth_max,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (K, B, kernel_type, bandwidth, spectral_norm,
         tau_min, tau_max, bandwidth_min, bandwidth_max) = aux
        obj = cls(K, B, kernel_type, bandwidth, spectral_norm,
                  tau_min, tau_max, bandwidth_min, bandwidth_max)
        obj.Omega_weights, obj.Phase_weights, obj.Rho_base = children
        return obj

    def __repr__(self) -> str:
        bw_str = (f"bw=[{self.bandwidth_min},{self.bandwidth_max}]"
                  if self.bandwidth_min is not None
                  else f"bw={self.bandwidth}")
        return (f"RandomFourierBasis(n_blocks={self.K}, features_per_block={self.B}, "
                f"kernel={self.kernel_type!r}, {bw_str}, sn={self.spectral_norm}, "
                f"tau=[{self.tau_min},{self.tau_max}])")
