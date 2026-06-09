"""saspy — SAS (Spectral Associative Scan) reservoir forecaster."""

from .base        import BaseForecaster
from .projector   import InputProjector
from .basis       import BaseBasis, DiagonalPoly, LRUBlockPoly, BlockLinearPoly, RandomFourierBasis
from .model       import SASModel
from .engine      import _forward, _step_once, _stream_scan, scan_states
from .forecaster  import SASForecaster

__version__ = "0.2.0"

__all__ = [
    # Forecaster
    "SASForecaster",
    # Model
    "SASModel",
    # Layer 1
    "InputProjector",
    # Layer 2
    "BaseBasis",
    "DiagonalPoly",
    "LRUBlockPoly",
    "BlockLinearPoly",
    "RandomFourierBasis",
    # Layer 3 (advanced use)
    "scan_states",
    "_forward",
    "_step_once",
    "_stream_scan",
    # Base
    "BaseForecaster",
]
