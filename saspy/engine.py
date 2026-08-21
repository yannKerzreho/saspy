"""
engine.py — Pure algebraic scan engine (Layer 3).

This module knows nothing about input dimensionality, projection, or
polynomial structure.  It receives pre-materialised arrays:

    P_seq : (T, *A_shape)   — transition representations
    Q_seq : (T, N)          — input-drive vectors

and a basis instance (for its combine/apply operators), then runs the
chunked parallel scan (sequential inside a chunk, prefix scan across chunks).

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

# scan_states(inner="auto") switches to the sequential intra-chunk pass once one
# composed transition costs more than this many states.  Block banks sit at 2-4,
# dense/filled-in hulls at N.
_DENSE_CARRY_RATIO = 4

def scan_states(
    P_seq,
    Q_seq,
    s0,
    basis,
    chunk_size: int,
    inner: str = "auto",
):
    """
    Chunked parallel scan over the affine recurrence x_t = A_t x_{t-1} + q_t.

    The recurrence is affine, hence associative under
    (A,b) o (A',b') = (A A', A b' + b), so the trajectory is a prefix scan.  The
    scan composes transitions, so its carry lives in the smallest matrix family
    CLOSED under product — itself for a diagonal (O(N) numbers), but the dense
    M_N for a sparse pattern (fill-in) or for a low-rank term plus a backbone.
    That closure, not the arithmetic, is what makes the parallel path expensive:
    a scan straight over the T steps materialises one composed transition per
    step, i.e. a (T, N, N) array whenever the family densifies.

    We split into K = ceil(T/B) chunks and choose the intra-chunk pass to match:

      1. reduce each chunk to a single (A, b), vmapped over the K chunks;
      2. prefix-scan the K chunk summaries (one associative_scan, always);
      3. turn the running summaries into the state entering each chunk;
      4. resolve the states inside each chunk from that entry state, vmapped.

    Steps 1 and 4 come in two flavours, selected by ``inner``:

    ``"assoc"``
        associative_scan inside the chunk (step 1 keeps every partial
        composition, step 4 is then a broadcast ``apply``).  Depth O(log B), but
        it materialises T composed transitions.  Right when the carry is
        state-sized, since the array costs no more than the states themselves.

    ``"seq"``
        a sequential fold inside the chunk (step 1 carries only the running
        composition; step 4 replays the chunk with matvecs).  Depth O(B), but
        only K = T/B composed transitions are ever live and only O(T/B) combines
        are performed instead of O(T).  Right when the carry is a matrix.

    ``"auto"`` (default)
        ``"seq"`` when one transition is bigger than one state, ``"assoc"``
        otherwise.  On CPU this picks the faster kernel in both regimes:
        measured against the all-associative version, 1.2-2.2x faster for a
        dense (T, N, N) carry (Sparse, LowRank with backbone) and no slower for
        the diagonal, which keeps its low-depth inner scan.

    B therefore trades depth against the number of composed transitions:
    depth O(B + log(T/B)) and memory O((T/B) c), where c is the hull dimension.

    Parameters
    ----------
    P_seq      : (T, *A_shape)   pre-evaluated transition representations
    Q_seq      : (T, N)          pre-evaluated input-drive vectors
    s0         : (N,)            initial reservoir state
    basis      : BaseBasis pytree — supplies combine() and apply()
    chunk_size : static int B    — chunk length
    inner      : "auto" | "seq" | "assoc" — intra-chunk pass (see above)

    Returns
    -------
    all_states : (T, N)
    s_last     : (N,)  state at the last real timestep
    """
    T = Q_seq.shape[0]
    N = Q_seq.shape[1]
    B = min(int(chunk_size), T)

    # P_seq may be a pytree (LowRankHullP carries the pair (c, C)); one "element"
    # is the per-step slice of every leaf.
    carry_size = sum(x[0].size for x in jax.tree_util.tree_leaves(P_seq))

    if inner == "auto":
        # Is the composed transition state-sized, or a matrix?  Diagonal gives N
        # numbers and a 2x2-block bank 2N, both cheap to keep for every step;
        # a dense or filled-in hull gives N^2, i.e. N times the states.  The
        # ratio separates the two regimes by orders of magnitude, so the exact
        # threshold does not matter.
        inner = "seq" if carry_size > _DENSE_CARRY_RATIO * N else "assoc"
    if inner not in ("seq", "assoc"):
        raise ValueError(f"inner must be 'auto', 'seq' or 'assoc'; got {inner!r}")

    pad = (B - T % B) % B
    K   = (T + pad) // B                              # number of chunks

    def _chunk(x):                                    # (T, ...) -> (K, B, ...)
        x = jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
        return x.reshape((K, B) + x.shape[1:])

    P_chunks = jax.tree_util.tree_map(_chunk, P_seq)
    Q_chunks = _chunk(Q_seq)

    # ── Phase 1: reduce each chunk to one (A, b) — K chunks in parallel ───────
    # The zero padding of the final chunk zeroes that chunk's summary, which
    # phase 3 discards anyway (only summaries 0..K-2 are used).
    if inner == "assoc":
        Acum, bcum = jax.vmap(
            lambda pq: jax.lax.associative_scan(basis.combine, pq)
        )((P_chunks, Q_chunks))                        # (K, B, *A_shape), (K, B, N)
        A_sum = jax.tree_util.tree_map(lambda x: x[:, -1], Acum)
        b_sum = bcum[:, -1]
    else:
        def _fold(P_c, Q_c):
            def body(acc, pq):
                return basis.combine(acc, pq), None
            head = jax.tree_util.tree_map(lambda x: x[0], P_c)
            tail = jax.tree_util.tree_map(lambda x: x[1:], P_c)
            (A, b), _ = jax.lax.scan(body, (head, Q_c[0]), (tail, Q_c[1:]))
            return A, b
        A_sum, b_sum = jax.vmap(_fold)(P_chunks, Q_chunks)   # (K, *A_shape), (K, N)

    # ── Phase 2: prefix scan over the K chunk summaries ──────────────────────
    A_inter, b_inter = jax.lax.associative_scan(basis.combine, (A_sum, b_sum))

    # ── Phase 3: carries — state at the START of each chunk ──────────────────
    rest    = jax.vmap(lambda A, b: basis.apply(A, s0) + b)(
        jax.tree_util.tree_map(lambda x: x[:-1], A_inter), b_inter[:-1]
    )                                                  # (K-1, N)
    carries = jnp.concatenate([s0[None], rest], axis=0)  # (K, N)

    # ── Phase 4: resolve all states (K chunks in parallel) ───────────────────
    if inner == "assoc":
        all_s = jax.vmap(
            lambda Ac, bc, c: jax.vmap(
                lambda A, b: basis.apply(A, c) + b
            )(Ac, bc)
        )(Acum, bcum, carries).reshape(K * B, N)
    else:
        def _replay(P_c, Q_c, c):
            def body(s, pq):
                P_t, Q_t = pq
                s_new = basis.apply(P_t, s) + Q_t
                return s_new, s_new
            _, S = jax.lax.scan(body, c, (P_c, Q_c))
            return S
        all_s = jax.vmap(_replay)(P_chunks, Q_chunks, carries).reshape(K * B, N)

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


@jax.jit
def _fast_seq_scan(model, s0, z_seq):
    """
    Fast teacher-forced state-building scan for sequential bases (Sparse, LowRank).

    The transition features and the drive depend only on the *input*, never the
    state, so under teacher forcing they precompute batched ONCE, leaving the scan
    to do only the state recurrence via the basis' lean ``scan_matvec`` (which for
    LowRank stacks [M_0;Vᵀ] into a single row-major GEMV).  Exactly equals
    ``_stream_scan`` but with the per-step feature/projection work hoisted out.

    Requires basis_p to implement scan_features / scan_prep / scan_matvec.
    """
    z_tilde = model.project(z_seq)                          # (T, K)
    bp      = model.basis_p
    feat    = bp.scan_features(z_tilde)                     # (T, *feat)
    q_seq   = model.basis_q.batch_eval_q(z_tilde)          # (T, N)
    prep    = bp.scan_prep()                                # static per-scan (e.g. W_stack)
    leak    = model.leak

    def body(s, fq):
        feat_t, q_t = fq
        raw = bp.scan_matvec(prep, feat_t, s) + q_t
        s_new = (1.0 - leak) * s + leak * raw
        return s_new, s_new

    s_last, all_states = jax.lax.scan(body, s0, (feat, q_seq))
    return all_states, s_last
