"""saspy — SAS (Spectral Associative Scan) reservoir forecaster."""

from .base        import BaseForecaster
from .projector   import InputProjector
from .basis       import BaseBasis, DiagonalPoly, LRUBlockPoly, BlockLinearPoly, RandomFourierBasis, SparsePolyBasis
from .model       import SASModel
from .engine      import _forward, _step_once, _stream_scan, scan_states
from .forecaster  import SASForecaster

__version__ = "0.2.0"

__all__ = [
    "SASForecaster",
    "SASModel",
    "InputProjector",
    "BaseBasis",
    "DiagonalPoly",
    "LRUBlockPoly",
    "BlockLinearPoly",
    "RandomFourierBasis",
    "SparsePolyBasis",
    "scan_states",
    "_forward",
    "_step_once",
    "_stream_scan",
    "BaseForecaster",
]
