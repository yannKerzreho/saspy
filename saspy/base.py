"""Abstract base for streaming forecasters."""
from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """
    Streaming forecaster protocol.

    fit(history, horizons)  : calibrate on a (T,) array.
    update(x)               : ingest one new observation without refitting.
    predict(h)              : h-step-ahead point forecast in the input scale.
    """

    @abstractmethod
    def fit(self, history: np.ndarray, horizons: list[int]) -> "BaseForecaster":
        """Fit on history (T,) and prepare to forecast all requested horizons."""
        ...

    @abstractmethod
    def update(self, x: float) -> "BaseForecaster":
        """Incorporate one new observation without refitting."""
        ...

    @abstractmethod
    def predict(self, h: int) -> float:
        """Point forecast h steps ahead (in the same scale as the input)."""
        ...

    # ── min-max [-1, 1] scaling helpers (shared by all subclasses) ─────────────
    # The reservoir lives on the compact domain [-1, 1]; data is mapped there by
    # an affine min→−1 / max→+1 transform, stored as (center, half) so the same
    # (x − center)/half form drives both scalar and per-channel paths.

    @staticmethod
    def _fit_scaler(history: np.ndarray) -> tuple[float, float]:
        """Return (center, half) mapping [min, max] → [-1, 1]; half floored at 1e-8."""
        lo, hi = float(np.min(history)), float(np.max(history))
        return (lo + hi) / 2.0, max((hi - lo) / 2.0, 1e-8)

    @staticmethod
    def _scale(x, center, half):
        """Map x into [-1, 1] with pre-fitted (center, half)."""
        return (np.asarray(x) - center) / half

    @staticmethod
    def _unscale(z, center, half):
        """Inverse of _scale."""
        return np.asarray(z) * half + center

    # backward-compatible aliases (kept so external callers don't break)
    _zscore   = _scale
    _unzscore = _unscale

    def __repr__(self) -> str:
        return self.__class__.__name__
