"""SASModel: composable (projector, basis_p, basis_q) JAX pytree.

Wires Layer 1 (InputProjector) to two independent Layer 2 bases:
  - basis_p: governs the transition matrix A_t
  - basis_q: governs the input-drive vector q_t
"""

import jax
import jax.numpy as jnp

from .projector  import InputProjector
from .basis.base import BaseBasis


@jax.tree_util.register_pytree_node_class
class SASModel:
    """
    Composable projector + dual-basis module.

    Parameters
    ----------
    projector : InputProjector
    basis_p   : BaseBasis — transition basis, defines A_t and the scan monoid.
    basis_q   : BaseBasis — drive basis, defines q_t.

    Constraints
    -----------
    projector.n_drivers == basis_p.n_drivers == basis_q.n_drivers
    basis_p.n == basis_q.n  (total reservoir dimension must agree)
    """

    def __init__(
        self,
        projector: InputProjector,
        basis_p:   BaseBasis,
        basis_q:   BaseBasis,
    ):
        if projector.n_drivers != basis_p.n_drivers:
            raise ValueError(
                f"projector.n_drivers ({projector.n_drivers}) != "
                f"basis_p.n_drivers ({basis_p.n_drivers})"
            )
        if basis_p.n_drivers != basis_q.n_drivers:
            raise ValueError(
                f"basis_p.n_drivers ({basis_p.n_drivers}) != "
                f"basis_q.n_drivers ({basis_q.n_drivers})"
            )
        if basis_p.n != basis_q.n:
            raise ValueError(
                f"basis_p.n ({basis_p.n}) != basis_q.n ({basis_q.n}). "
                "Both bases must produce the same total reservoir dimension N."
            )
        self.projector = projector
        self.basis_p   = basis_p
        self.basis_q   = basis_q

    # ── convenience properties ───────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Total reservoir dimension N."""
        return self.basis_p.n

    @property
    def d(self) -> int:
        """Input feature dimension."""
        return self.projector.d

    def is_initialized(self) -> bool:
        return (self.projector.W is not None
                and self.basis_p.is_initialized()
                and self.basis_q.is_initialized())

    # ── factory ──────────────────────────────────────────────────────────────

    def initialize(self, key) -> "SASModel":
        """Return a new SASModel with projector, basis_p, and basis_q initialised."""
        k1, k2, k3 = jax.random.split(key, 3)
        return SASModel(
            projector=self.projector.initialize(k1),
            basis_p=self.basis_p.initialize(k2),
            basis_q=self.basis_q.initialize(k3),
        )

    # ── encode: Layer 1 + 2 (called inside _forward) ─────────────────────────

    def encode(self, z: jnp.ndarray):
        """
        z: (T, d) → (P_seq, Q_seq)

        P_seq: (T, *A_shape)   transition representations from basis_p
        Q_seq: (T, N)          input-drive vectors from basis_q
        """
        z_tilde = self.projector.project(z)            # (T, n_drivers)
        P_seq   = self.basis_p.batch_eval_p(z_tilde)   # (T, *A_shape)
        Q_seq   = self.basis_q.batch_eval_q(z_tilde)   # (T, N)
        return P_seq, Q_seq

    # ── single step (called inside _step_once) ───────────────────────────────

    def step(self, z_t: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
        """
        z_t: (d,), s: (N,) → s_new: (N,)

        Projects z_t, evaluates A from basis_p and q from basis_q, advances state.
        """
        z_tilde_t = self.projector.project_single(z_t)  # (n_drivers,)
        A         = self.basis_p.eval_p(z_tilde_t)
        q         = self.basis_q.eval_q(z_tilde_t)
        return self.basis_p.apply(A, s) + q

    # ── pytree ───────────────────────────────────────────────────────────────

    def tree_flatten(self):
        return (self.projector, self.basis_p, self.basis_q), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        proj, basis_p, basis_q = children
        obj = object.__new__(cls)
        obj.projector = proj
        obj.basis_p   = basis_p
        obj.basis_q   = basis_q
        return obj

    def __repr__(self) -> str:
        return (f"SASModel(\n  {self.projector!r},\n"
                f"  basis_p={self.basis_p!r},\n"
                f"  basis_q={self.basis_q!r}\n)")
