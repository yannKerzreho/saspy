"""Static random input projection (Layer 1).

Maps the input sequence z of shape (T, d) to a projected sequence
z_tilde of shape (T, n_drivers) via a fixed random matrix W_in:

    z_tilde = z @ W_in        W_in: (d, n_drivers)

For d == 1, W_in = ones(1, n_drivers) so every driver sees the same
scalar input; diversity comes from the basis P/Q weights.
For d > 1, W_in is a randomly drawn, L2-column-normalised sparse matrix.
"""

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
class InputProjector:
    """
    Static random input projection.

    Parameters
    ----------
    d                  : input feature dimension (1 for univariate).
    n_drivers          : output dimension fed to the basis
                         (N for DiagonalPoly, K for block bases).
    density            : fraction of non-zero entries per column for d > 1.
                         Forced to 1.0 when d == 1.
    seed               : kept for repr / serialisation; key is passed to initialize().
    mixing_strategy    : strategy used to build the sparsity mask for d > 1.
                         "hybrid"     — cyclic base mask (no dead neurons) plus
                                        random sparse overlay (default).
                         "pure_random"— original Bernoulli mask; may produce dead
                                        columns at low density.
    normalize_columns  : if True (default), each column of W is rescaled to unit
                         L2 norm after construction.  This ensures z_tilde[i] has
                         the same variance as the input when d > 1, matching the
                         N(0,1) assumption baked into DiagonalPoly/LRU weight init.
                         Set False to preserve the raw JL scaling (entries ∝ 1/√K),
                         which preserves pairwise distances but shrinks z_tilde std
                         to ≈ √(d/K) — safe for RFF bases (cosine is scale-free)
                         but degrades polynomial bases.
    """

    def __init__(
        self,
        d:                  int,
        n_drivers:          int,
        density:            float = 1.0,
        seed:               int   = 0,
        mixing_strategy:    str   = "hybrid",
        normalize_columns:  bool  = True,
    ):
        self.d                 = d
        self.n_drivers         = n_drivers
        self.density           = float(max(0.0, min(1.0, density)))
        self.seed              = seed
        self.mixing_strategy   = mixing_strategy
        self.normalize_columns = bool(normalize_columns)
        self.W                 = None   # (d, n_drivers), set by initialize()

    # ── special constructors ─────────────────────────────────────────────────

    @classmethod
    def trivial(cls, n_drivers: int) -> "InputProjector":
        """
        W_in = ones(1, n_drivers).

        z_tilde_t[i] = z_t for every driver i — exactly the old
        "broadcast scalar" behaviour.  No random key needed.
        """
        obj   = cls(d=1, n_drivers=n_drivers, density=1.0)
        obj.W = jnp.ones((1, n_drivers), dtype=jnp.float32)
        return obj

    @classmethod
    def identity(cls, d: int) -> "InputProjector":
        """
        W_in = eye(d),  n_drivers = d.

        z_tilde_t[i] = z_t[i] — passes each input dimension through unchanged.
        Required by SparsePolyBasis so that polynomial features phi(z_tilde)
        are exactly the per-dimension powers of the raw input z.

        initialize() preserves the identity because mixing_strategy='identity'
        is detected as a no-op reinit path.
        """
        obj   = cls(d=d, n_drivers=d, density=1.0, mixing_strategy='identity')
        obj.W = jnp.eye(d, dtype=jnp.float32)
        return obj

    # ── factory ──────────────────────────────────────────────────────────────

    def initialize(self, key) -> "InputProjector":
        """Return a new InputProjector with W initialised.

        d == 1: W = ones(1, n_drivers) — all drivers broadcast the same scalar.
        d > 1:  W ~ N(0,1), masked to `density` non-zeros per column, then
                L2-normalised so each column is a unit-direction projection.
        """
        d         = self.d
        n_drivers = self.n_drivers

        # ── identity: preserve eye(d) — no random reinit ─────────────────
        if self.mixing_strategy == 'identity':
            obj   = InputProjector(d, n_drivers, 1.0, self.seed, 'identity',
                                   normalize_columns=False)
            obj.W = jnp.eye(d, dtype=jnp.float32)
            return obj

        if d == 1:
            obj   = InputProjector(d, n_drivers, self.density, self.seed, self.mixing_strategy)
            obj.W = jnp.ones((1, n_drivers), dtype=jnp.float32)
            return obj

        k_W, k_mask = jax.random.split(key)
        W_raw = jax.random.normal(k_W, (d, n_drivers), dtype=jnp.float32)

        if self.mixing_strategy == "hybrid":
            # Cyclic base mask guarantees ≥1 non-zero per column; random overlay adds mixing.
            cols          = jnp.arange(n_drivers)
            assigned_rows = cols % d
            mask_base     = jnp.zeros((d, n_drivers), dtype=bool).at[assigned_rows, cols].set(True)
            mask_rand     = jax.random.uniform(k_mask, (d, n_drivers)) < self.density
            mask          = mask_base | mask_rand
        else:
            mask = jax.random.uniform(k_mask, (d, n_drivers)) < self.density

        W = W_raw * mask

        if self.normalize_columns:
            col_norms = jnp.linalg.norm(W, axis=0, keepdims=True)  # (1, n_drivers)
            W         = W / jnp.maximum(col_norms, 1e-8)            # zero cols stay zero

        obj   = InputProjector(d, n_drivers, self.density, self.seed,
                               self.mixing_strategy, self.normalize_columns)
        obj.W = W
        return obj

    # ── forward ──────────────────────────────────────────────────────────────

    def project(self, z: jnp.ndarray) -> jnp.ndarray:
        """z: (T, d) → z_tilde: (T, n_drivers)."""
        return z @ self.W

    def project_single(self, z_t: jnp.ndarray) -> jnp.ndarray:
        """z_t: (d,) → z_tilde_t: (n_drivers,)."""
        return jnp.dot(z_t, self.W)

    # ── pytree ───────────────────────────────────────────────────────────────

    def tree_flatten(self):
        return (self.W,), (self.d, self.n_drivers, self.density, self.seed,
                           self.mixing_strategy, self.normalize_columns)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj   = cls(*aux)
        obj.W = children[0]
        return obj

    def __repr__(self) -> str:
        status = f"W:{self.W.shape}" if self.W is not None else "uninitialised"
        return (f"InputProjector(d={self.d}, n_drivers={self.n_drivers}, "
                f"density={self.density:.2f}, strategy={self.mixing_strategy}, "
                f"normalize_columns={self.normalize_columns}, {status})")
