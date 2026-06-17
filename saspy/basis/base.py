"""
base.py — Abstract base class for SAS polynomial bases (Layer 2).

A basis receives the *projected* input z_tilde of shape (n_drivers,) and
evaluates the recurrence polynomials:

    A_t  = eval_p(z_tilde_t)   — transition representation
    q_t  = eval_q(z_tilde_t)   — input-drive vector, shape (N,)

so that the reservoir advances as   s_t = apply(A_t, s_{t-1}) + q_t.

The two algebraic primitives — combine() and apply() — power the parallel
associative scan in engine.py.  They must be pure JAX operations so they
are differentiable and JIT-traceable.

All concrete subclasses must be registered JAX pytree nodes.
"""

import abc
import jax
import jax.numpy as jnp


class BaseBasis(abc.ABC):

    def __init__(self, p_degree: int = 1, q_degree: int = 1):
        self.p_degree  = p_degree
        self.q_degree  = q_degree
        self.P_weights = None   # set by initialize()
        self.Q_weights = None

    def is_initialized(self) -> bool:
        return self.P_weights is not None

    def _budget_ref(self) -> float:
        """Scale reference for P weight budget.
        Subclasses set self.budget_ref and self.max_input in __init__."""
        if getattr(self, 'budget_ref', None) is not None:
            return self.budget_ref
        mi = getattr(self, 'max_input', None)
        return mi if mi is not None else 1.0

    # ── dimensions ──────────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def n(self) -> int:
        """Total reservoir dimension N."""

    @property
    @abc.abstractmethod
    def n_drivers(self) -> int:
        """Size of z_tilde expected from the projector."""

    # ── factory ─────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def initialize(self, key) -> "BaseBasis":
        """Return a NEW, fully initialised instance (leaves P/Q_weights set)."""

    # ── per-step evaluators (JAX-traceable) ─────────────────────────────────

    @abc.abstractmethod
    def eval_p(self, z_tilde_t):
        """z_tilde_t: (n_drivers,) → A_rep with shape defined by the subclass."""

    @abc.abstractmethod
    def eval_q(self, z_tilde_t):
        """z_tilde_t: (n_drivers,) → q: (N,)."""

    # ── batched evaluators (full sequence) ──────────────────────────────────

    def batch_eval_p(self, z_tilde):
        """z_tilde: (T, n_drivers) → (T, *A_shape).  Override for efficiency."""
        return jax.vmap(self.eval_p)(z_tilde)

    def batch_eval_q(self, z_tilde):
        """z_tilde: (T, n_drivers) → (T, N).  Override for efficiency."""
        return jax.vmap(self.eval_q)(z_tilde)

    # ── algebraic primitives (must use only JAX ops) ────────────────────────

    @abc.abstractmethod
    def apply(self, A_rep, s):
        """Apply A_rep to state vector s → (N,)."""

    @abc.abstractmethod
    def combine(self, i, j):
        """
        Monoid for jax.lax.associative_scan.

        i = (A_i, q_i),  j = (A_j, q_j)
        Returns the composition representing "apply i then j":
            A_new = A_j ∘ A_i
            q_new = apply(A_j, q_i) + q_j
        """

    # ── pytree protocol ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def tree_flatten(self): ...

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux, children): ...
