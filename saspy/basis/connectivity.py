"""Structured connectivity / density generators for the SAS weight tensors.

Implements the density design in `note/density_design.md`:

  * `log_density`      — Erdős–Rényi connectivity-threshold density  c·ln(N)/N.
  * `connectivity_mask`— sparse Bernoulli mask with a guaranteed ≥1 per row.
  * `banded_mask`      — local (lattice) support for spatially-extended systems.
  * `sparse_input_matrix` — sparse-JL `W_in` (fan-in per driver, ≥2 for cross terms).

These are pure helpers; the bases call them so density scales with `(N, K, D, d)`
instead of being a fixed constant.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def log_density(n: int, c: float = 1.0, floor: int = 1) -> float:
    """Erdős–Rényi connectivity-threshold density: min(1, max(floor, c·ln n)/n).

    At/above c·ln(n)/n a random graph on n nodes is a.s. connected; below it
    fragments. Gives fan-in ~ c·ln(n) (grows slowly, total nonzeros ~ n·ln n).

    The `floor` is a *minimum fan-in*: the bare ER threshold is marginal at small
    n (e.g. c=1.5 puts fan-in ≈ 8 only at n≈210), and exp 04 shows the marginally
    connected regime is high-variance there. Flooring fan-in at ~6–8 sits the graph
    comfortably above threshold for small n while still growing like ln n for large
    n — empirically the constant-fan-in and log laws are indistinguishable in
    [200, 600] and both beat fixed density.
    """
    if n <= 2:
        return 1.0
    return float(min(1.0, max(float(floor), c * math.log(n)) / n))


def connectivity_mask(key, rows: int, cols: int, density: float,
                      ensure_row: bool = True) -> jnp.ndarray:
    """Bernoulli {0,1} mask of shape (rows, cols).

    With ensure_row, any all-zero row gets one random active entry (no dead unit),
    so the realised fan-in is ≥1 even at very low density.
    """
    k_m, k_f = jax.random.split(key)
    m = (jax.random.uniform(k_m, (rows, cols)) < density).astype(jnp.float32)
    if ensure_row:
        forced = jax.random.randint(k_f, (rows,), 0, cols)
        fix    = jax.nn.one_hot(forced, cols, dtype=jnp.float32)
        m = jnp.where(m.sum(axis=1, keepdims=True) == 0, fix, m)
    return m


def banded_mask(n: int, halfwidth: int, periodic: bool = True) -> jnp.ndarray:
    """Local lattice support: entry (i,j) active iff spatial distance ≤ halfwidth.

    For spatially-extended systems (KS, Lorenz-96) where nearby sites interact and
    far ones don't. `periodic` wraps the distance (ring topology).
    """
    i = jnp.arange(n)[:, None]
    j = jnp.arange(n)[None, :]
    dist = jnp.abs(i - j)
    if periodic:
        dist = jnp.minimum(dist, n - dist)
    return (dist <= halfwidth).astype(jnp.float32)


def sparse_input_matrix(key, d: int, K: int, fan_in: int | None = None,
                        normalize: bool = True, banded_halfwidth: int | None = None):
    """Random projection W_in ∈ ℝ^{d×K}, sparse (fan-in inputs per driver).

    fan_in           : nonzeros per column (driver).  Clamped to [1, d].  None →
                       dense.  Sparse-JL: O(log d) preserves geometry; use ≥2 for a
                       driver to carry a cross term z_i·z_j (fan_in=1 → no cross).
    banded_halfwidth : if set (requires K==d), each driver k mixes only inputs in a
                       window |i−k|≤w (periodic) — local structure for PDEs.
    L1-normalised columns keep z̃ = z·W_in in [-1,1] for z ∈ [-1,1]^d.
    """
    k_w, k_mask = jax.random.split(key)
    W = jax.random.normal(k_w, (d, K), dtype=jnp.float32)

    if banded_halfwidth is not None:
        if K != d:
            raise ValueError("banded W_in requires K == d (one driver per site)")
        mask = banded_mask(d, banded_halfwidth, periodic=True)          # (d,d)
    elif fan_in is not None and fan_in < d:
        f = max(min(fan_in, d), 1)
        # choose `f` distinct rows per column via top-f of random scores
        scores = jax.random.uniform(k_mask, (d, K))
        thresh = jnp.sort(scores, axis=0)[d - f][None, :]               # f-th largest per col
        mask   = (scores >= thresh).astype(jnp.float32)
    else:
        mask = jnp.ones((d, K), dtype=jnp.float32)

    W = W * mask
    if normalize:
        l1 = jnp.sum(jnp.abs(W), axis=0, keepdims=True)
        W  = W / jnp.maximum(l1, 1e-8)
    return W
