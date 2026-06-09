"""Abstract base for all univariate forecasters."""
from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """
    Streaming-friendly univariate forecaster protocol.

    Contract
    --------
    fit(history, horizons)  : calibrate on 1-D numpy array (T,).
    update(x)               : ingest one new scalar; update state without refit.
    predict(h)              : scalar point forecast h steps ahead, in the same
                              scale as the input.

    Any preprocessing other than z-scoring (log-transform, differencing,
    residualisation, …) is the caller's responsibility — feed the already-
    transformed series to ``fit``.
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

    # ── z-score helpers (shared by all subclasses) ────────────────────────────

    @staticmethod
    def _fit_scaler(history: np.ndarray) -> tuple[float, float]:
        """Return (mean, std) from training window; std floored at 1e-8."""
        mu    = float(np.mean(history))
        sigma = float(np.std(history))
        return mu, max(sigma, 1e-8)

    @staticmethod
    def _zscore(x, mu: float, sigma: float):
        """Standardise x with pre-fitted (mu, sigma)."""
        return (np.asarray(x) - mu) / sigma

    @staticmethod
    def _unzscore(z, mu: float, sigma: float):
        """Inverse standardisation."""
        return np.asarray(z) * sigma + mu

    def __repr__(self) -> str:
        return self.__class__.__name__
