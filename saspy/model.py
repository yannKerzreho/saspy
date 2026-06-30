"""SASModel: composable (basis_p, basis_q) JAX pytree with built-in projection.

Wires the input projection (Layer 1) to two independent Layer-2 bases:
  - basis_p: governs the transition representation A_t (and the scan monoid)
  - basis_q: governs the input-drive vector q_t

Projection is folded into the model (the old standalone InputProjector class is
gone).  Two regimes, selected by whether ``d`` is given:

  * **scalar structures (Diagonal / Block)** — pass ``d`` (input feature dim).
    A random, column-normalised W_in : (d, K) expands the d inputs to the K
    block drivers.  ``SASModel(p, q, d=3)``.

  * **sparse structure** — omit ``d``.  The joint feature map already mixes the
    K inputs, so projection is the identity (z_tilde = z) and the input
    dimension must equal ``n_drivers``.  ``SASModel(p, q)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .basis.connectivity import sparse_input_matrix


def build_input_matrix(key, d: int, n_drivers: int, density: float = 1.0,
                       normalize: bool = True, fan_in: int | None = None,
                       banded_halfwidth: int | None = None):
    """Random (d, n_drivers) projection.  Three strategies (see note/density_design.md):

      * d == 1 : unit gain (±1) — z_tilde = ±z, full [-1,1] range, no saturation.
      * fan_in / banded_halfwidth set : sparse-JL (fixed nonzeros per driver) or
        banded (local lattice, requires n_drivers == d) — `sparse_input_matrix`.
      * otherwise (hybrid, default) : cyclic base mask (≥1 per column, no dead
        driver) + Bernoulli overlay of `density`, L1-normalised columns so
        |z_tilde[k]| ≤ 1 for z ∈ [-1,1]^d.
    """
    if d == 1:
        W = jax.random.normal(key, (1, n_drivers), dtype=jnp.float32)
        return jnp.sign(W) if normalize else W
    if fan_in is not None or banded_halfwidth is not None:
        return sparse_input_matrix(key, d, n_drivers, fan_in=fan_in,
                                   normalize=normalize, banded_halfwidth=banded_halfwidth)
    k_W, k_mask = jax.random.split(key)
    W    = jax.random.normal(k_W, (d, n_drivers), dtype=jnp.float32)
    cols = jnp.arange(n_drivers)
    base = jnp.zeros((d, n_drivers), bool).at[cols % d, cols].set(True)
    rand = jax.random.uniform(k_mask, (d, n_drivers)) < density
    W    = W * (base | rand)
    if normalize:
        l1 = jnp.sum(jnp.abs(W), axis=0, keepdims=True)   # L1 column norm
        W  = W / jnp.maximum(l1, 1e-8)
    return W


@jax.tree_util.register_pytree_node_class
class SASModel:
    """Composable projection + dual-basis module.

    Parameters
    ----------
    basis_p   : transition basis — defines A_t, apply(), combine().
    basis_q   : drive basis — defines q_t.
    d         : input feature dim.  None → identity projection (sparse regime,
                input dim must equal n_drivers).  int → random W_in (d, K).
    density   : projection density for d > 1.
    normalize : L2-normalise projection columns.
    leak      : leaky-integrator rate ∈ (0, 1].  The state update becomes
                x_{t+1} = (1−leak)·x_t + leak·[P(z_t)·x_t + Q(z_t)], a first-order
                low-pass that preserves the steady-state response (same fixed
                point) and only sets the integration timescale — the SAS analogue
                of an ESN leak rate.  leak=1.0 (default) is the plain SAS update.

    Constraints
    -----------
    basis_p.n_drivers == basis_q.n_drivers ;  basis_p.n == basis_q.n
    """

    def __init__(self, basis_p, basis_q, d: int | None = None,
                 density: float = 1.0, normalize: bool = True,
                 proj_fan_in: int | None = None, proj_banded: int | None = None,
                 leak: float = 1.0, bias: bool = False):
        if basis_p.n_drivers != basis_q.n_drivers:
            raise ValueError(
                f"basis_p.n_drivers ({basis_p.n_drivers}) != "
                f"basis_q.n_drivers ({basis_q.n_drivers})")
        if basis_p.n != basis_q.n:
            raise ValueError(
                f"basis_p.n ({basis_p.n}) != basis_q.n ({basis_q.n}). "
                "Both bases must produce the same reservoir dimension N.")
        if not (0.0 < leak <= 1.0):
            raise ValueError(f"leak must be in (0, 1], got {leak}")
        self.basis_p     = basis_p
        self.basis_q     = basis_q
        self.d           = d
        self.density     = float(density)
        self.normalize   = bool(normalize)
        self.leak        = float(leak)
        # W_in strategy: None → hybrid (cyclic+overlay); fan_in → sparse-JL;
        # banded → local lattice (requires d == n_drivers).
        self.proj_fan_in = proj_fan_in
        self.proj_banded = proj_banded
        self.bias        = bool(bias)   # augment input with a constant channel [1, z]
        self.W_in        = None     # (d, K), or (d+1, K) when bias, or None (identity)

    # ── convenience ───────────────────────────────────────────────────────────

    @property
    def n(self):          return self.basis_p.n
    @property
    def n_drivers(self):  return self.basis_p.n_drivers
    @property
    def input_dim(self):  return self.n_drivers if self.d is None else self.d

    def is_initialized(self):
        return (self.basis_p.is_initialized() and self.basis_q.is_initialized()
                and (self.d is None or self.W_in is not None))

    # ── factory ───────────────────────────────────────────────────────────────

    def initialize(self, key) -> "SASModel":
        k_p, k_q, k_w = jax.random.split(key, 3)
        obj = SASModel(self.basis_p.initialize(k_p), self.basis_q.initialize(k_q),
                       self.d, self.density, self.normalize,
                       self.proj_fan_in, self.proj_banded, leak=self.leak,
                       bias=self.bias)
        if self.d is not None:
            # bias → prepend a constant channel: W_in acts on [1, z], so the bias is
            # just another input row, subject to the same hybrid mask + L1 norm.
            # Only drivers whose constant-row weight survives the mask get a bias.
            d_proj = self.d + 1 if self.bias else self.d
            obj.W_in = build_input_matrix(k_w, d_proj, self.n_drivers,
                                          self.density, self.normalize,
                                          fan_in=self.proj_fan_in,
                                          banded_halfwidth=self.proj_banded)
        return obj

    # ── projection ────────────────────────────────────────────────────────────

    def project(self, z):
        """z: (T, d) → (T, K), clipped to the bounded domain [-1, 1].

        With normalised W_in (and z already in [-1, 1]) the projection is bounded
        by construction, so this clip is a cheap safety net — it only engages on
        *out-of-domain* inputs (e.g. test values beyond the train range).
        """
        if self.W_in is None:
            z_tilde = z
        else:
            if self.bias:                                  # [1, z] — constant channel
                z = jnp.concatenate([jnp.ones((z.shape[0], 1), z.dtype), z], axis=1)
            z_tilde = z @ self.W_in
        return jnp.clip(z_tilde, -1.0, 1.0)

    def project_single(self, z_t):
        """z_t: (d,) → (K,), clipped to [-1, 1]."""
        if self.W_in is None:
            z_tilde = z_t
        else:
            if self.bias:                                  # [1, z] — constant channel
                z_t = jnp.concatenate([jnp.ones((1,), z_t.dtype), z_t])
            z_tilde = jnp.dot(z_t, self.W_in)
        return jnp.clip(z_tilde, -1.0, 1.0)

    # ── encode / step ─────────────────────────────────────────────────────────

    def encode(self, z):
        z_tilde = self.project(z)
        P_seq = self.basis_p.batch_eval_p(z_tilde)
        Q_seq = self.basis_q.batch_eval_q(z_tilde)
        if self.leak != 1.0:
            # Fold the leak into the affine map fed to the parallel scan:
            # A' = (1−leak)·I + leak·A,  q' = leak·q.
            P_seq = self.basis_p.leaky(P_seq, self.leak)
            Q_seq = self.leak * Q_seq
        return P_seq, Q_seq

    def step(self, z_t, s):
        z_tilde_t = self.project_single(z_t)
        As = self.basis_p.matvec_p(z_tilde_t, s)   # A_t @ s (sparse when sequential)
        q  = self.basis_q.eval_q(z_tilde_t)
        new = As + q
        if self.leak != 1.0:
            new = (1.0 - self.leak) * s + self.leak * new
        return new

    # ── pytree ────────────────────────────────────────────────────────────────

    def tree_flatten(self):
        return ((self.basis_p, self.basis_q, self.W_in),
                (self.d, self.density, self.normalize, self.proj_fan_in,
                 self.proj_banded, self.leak, self.bias))
    @classmethod
    def tree_unflatten(cls, aux, children):
        basis_p, basis_q, W_in = children
        obj = object.__new__(cls)
        obj.basis_p, obj.basis_q = basis_p, basis_q
        (obj.d, obj.density, obj.normalize, obj.proj_fan_in,
         obj.proj_banded, obj.leak, obj.bias) = aux
        obj.W_in = W_in
        return obj

    def __repr__(self):
        proj = "identity" if self.d is None else f"W_in(d={self.d}, density={self.density})"
        leak = "" if self.leak == 1.0 else f"\n  leak={self.leak},"
        return (f"SASModel(\n  proj={proj},{leak}\n  basis_p={self.basis_p!r},\n"
                f"  basis_q={self.basis_q!r}\n)")
