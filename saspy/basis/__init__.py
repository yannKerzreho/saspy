"""Bounded-feature bases for the SAS reservoir (Layer 2).

Two axes compose freely:

  * **feature** — :class:`Cheb` (Chebyshev polynomial) or :class:`Trig` (random
    cosine / RFF).  Both bounded in [-1, 1].
  * **structure / role** — Diagonal, Block or Sparse, split into an independent
    transition (P) and drive (Q) class.

Pick any P and any Q with any feature:

    SASModel(DiagonalP(N, feature=Cheb(1)), DiagonalQ(N, feature=Trig(2)), d=1)
    SASModel(SparseP(N, K, feature=Trig(2)), SparseQ(N, K, feature=Cheb(2)))
"""

from .feature      import Cheb, Trig, monomial_exponents, cheb_basis
from .scalar       import DiagonalP, DiagonalQ, BlockP, BlockQ
from .sparse       import SparseP, SparseQ
from .lowrank      import LowRankP, LowRankQ
from .connectivity import (
    log_density, connectivity_mask, banded_mask, sparse_input_matrix,
)

__all__ = [
    "Cheb", "Trig", "monomial_exponents", "cheb_basis",
    "DiagonalP", "DiagonalQ",
    "BlockP", "BlockQ",
    "SparseP", "SparseQ",
    "LowRankP", "LowRankQ",
    "log_density", "connectivity_mask", "banded_mask", "sparse_input_matrix",
]
