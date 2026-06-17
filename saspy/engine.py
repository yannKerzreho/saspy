"""
engine.py — Pure algebraic scan engine (Layer 3).

This module knows nothing about input dimensionality, projection, or
polynomial structure.  It receives pre-materialised arrays:

    P_seq : (T, *A_shape)   — transition representations
    Q_seq : (T, N)          — input-drive vectors

and a basis instance (for its combine/apply operators), then runs the
two-level parallel associative scan.

Public API
----------
scan_states(P_seq, Q_seq, s0, basis, chunk_size)
    Pure function (no JIT).  Call from JIT-wrapped entry points.

_forward(model, z, s0, chunk_size)
    JIT-compiled full forward pass:
        1. project z → z_tilde   (Layer 1)
        2. evaluate P_seq, Q_seq  (Layer 2)
        3. scan → states          (Layer 3)

_step_once(model, s, z_t)
    JIT-compiled single streaming step.
"""

from __future__ import annotations

import functools
import numpy as np
import jax
import jax.numpy as jnp


# ══════════════════════════════════════════════════════════════════════════════
# Core scan — pure function, no JIT decorator
# ══════════════════════════════════════════════════════════════════════════════

def scan_states(
    P_seq,
    Q_seq,
    s0,
    basis,
    chunk_size: int,
):
    """
    Two-level parallel associative scan.

    Parameters
    ----------
    P_seq      : (T, *A_shape)   pre-evaluated transition representations
    Q_seq      : (T, N)          pre-evaluated input-drive vectors
    s0         : (N,)            initial reservoir state
    basis      : BaseBasis pytree — supplies combine() and apply()
    chunk_size : static int B    — intra-chunk parallelism granularity

    Returns
    -------
    all_states : (T, N)
    s_last     : (N,)  state at the last real timestep
    """
    T = Q_seq.shape[0]
    N = Q_seq.shape[1]
    B = chunk_size

    pad = (B - T % B) % B
    P_pad = jnp.pad(P_seq, [(0, pad)] + [(0, 0)] * (P_seq.ndim - 1))
    Q_pad = jnp.pad(Q_seq, [(0, pad), (0, 0)])

    K = P_pad.shape[0] // B                           # number of chunks

    # Reshape into (K, B, *A_shape) and (K, B, N)
    P_chunks = P_pad.reshape((K, B) + P_seq.shape[1:])
    Q_chunks = Q_pad.reshape(K, B, N)

    # ── Phase 1: intra-chunk cumulative scans (K chunks, vmapped) ────────────
    def _chunk_scan(pq):
        P_c, Q_c = pq
        return jax.lax.associative_scan(basis.combine, (P_c, Q_c))

    Acum, bcum = jax.vmap(_chunk_scan)((P_chunks, Q_chunks))
    # Acum: (K, B, *A_shape),  bcum: (K, B, N)

    # ── Phase 2: inter-chunk scan over last-step transforms ──────────────────
    A_inter, b_inter = jax.lax.associative_scan(
        basis.combine, (Acum[:, -1], bcum[:, -1])
    )
    # A_inter: (K, *A_shape),  b_inter: (K, N)

    # ── Phase 3: carries — state at the START of each chunk ──────────────────
    rest    = jax.vmap(lambda A, b: basis.apply(A, s0) + b)(
        A_inter[:-1], b_inter[:-1]
    )                                                  # (K-1, N)
    carries = jnp.concatenate([s0[None], rest], axis=0)  # (K, N)

    # ── Phase 4: resolve all states (K chunks in parallel) ───────────────────
    all_s = jax.vmap(
        lambda Ac, bc, c: jax.vmap(
            lambda A, b: basis.apply(A, c) + b
        )(Ac, bc)
    )(Acum, bcum, carries).reshape(K * B, N)           # (K*B, N)

    return all_s[:T], all_s[T - 1]


# ══════════════════════════════════════════════════════════════════════════════
# JIT-compiled entry points
# ══════════════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, static_argnames=("chunk_size",))
def _forward(model, z, s0, chunk_size: int):
    """
    Full forward pass (JIT-compiled).

    model      : SASModel pytree (projector + basis, both initialised)
    z          : (T, d) input sequence, float32
    s0         : (N,) initial state
    chunk_size : static int
    """
    P_seq, Q_seq = model.encode(z)                      # Layer 1 + 2
    return scan_states(P_seq, Q_seq, s0, model.basis_p, chunk_size)  # Layer 3


@jax.jit
def _step_once(model, s, z_t):
    """
    Single streaming step (JIT-compiled).

    model : SASModel pytree
    s     : (N,) current state
    z_t   : (d,) new input
    """
    return model.step(z_t, s)


@jax.jit
def _stream_scan(model, s0, z_seq):
    """
    Sequential streaming via lax.scan — single JIT call for T steps.

    model : SASModel pytree, s0 : (N,), z_seq : (T, d) → (all_states (T, N), s_last (N,))
    """
    def body(s, z_t):
        s_new = model.step(z_t, s)
        return s_new, s_new

    s_last, all_states = jax.lax.scan(body, s0, z_seq)
    return all_states, s_last
