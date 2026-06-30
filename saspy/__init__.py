"""saspy — SAS (Spectral Associative Scan) reservoir forecaster."""

from .base        import BaseForecaster
from .basis       import (
    Cheb, Trig,
    DiagonalP, DiagonalQ,
    BlockP, BlockQ,
    SparseP, SparseQ,
    LowRankP, LowRankQ,
)
from .model       import SASModel, build_input_matrix
from .engine      import _forward, _step_once, _stream_scan, scan_states  # noqa: F401
from .forecaster  import SASForecaster

__version__ = "0.3.0"

__all__ = [
    "SASForecaster",
    "SASModel",
    "build_input_matrix",
    "Cheb", "Trig",
    "DiagonalP", "DiagonalQ",
    "BlockP", "BlockQ",
    "SparseP", "SparseQ",
    "LowRankP", "LowRankQ",
    "scan_states",
    "BaseForecaster",
]
