"""SASForecaster: streaming time-series forecaster.

Wraps SASModel + ridge regression into the BaseForecaster protocol.

    fit(history, horizons, context=None) → run scan, fit ridge readout(s).
    update(x)                            → single streaming step.
    predict(h)                           → h-step forecast, in the input scale.
    transform(history, context=None)     → (T, N) reservoir states.

Inputs are min-max scaled into the reservoir's compact domain [-1, 1]; other
preprocessing is the caller's responsibility.
"""

from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp

from .base       import BaseForecaster
from .model      import SASModel
from .engine     import _forward, _step_once, _stream_scan, _fast_seq_scan
from .ridge      import (
    ALPHAS          as _ALPHAS,
    ridge_cv_select as _ridge_cv,
    ridge_fit       as _ridge_fit,
)


# ── Autoregressive rollout kernel ────────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=('n_steps', 'clip'))
def _autoreg_rollout(model, s0, W1, n_steps: int, clip: bool = False):
    """Advance the reservoir n_steps autoregressively; non-destructive.

    n_steps=0 returns s0 unchanged (lax.scan with length=0 is the identity).
    With clip=True the fed-back prediction is clamped to the [-1, 1] domain
    before re-entering the reservoir — this keeps the closed loop in-domain and
    prevents the autonomous divergence that unbounded feedback can trigger.
    """
    def body(s, _):
        y_z = s @ W1                                   # () or (D,)
        if clip:
            y_z = jnp.clip(y_z, -1.0, 1.0)
        z_t = jnp.atleast_1d(y_z).astype(jnp.float32) # (1,) or (D,)
        return model.step(z_t, s), None
    s_final, _ = jax.lax.scan(body, s0, None, length=n_steps)
    return s_final


class SASForecaster(BaseForecaster):
    """
    SAS reservoir-computing forecaster.

    Two forecast modes
    ------------------
    direct  (default)
        Fit one ridge readout W[h] per requested horizon.
        predict(h) = s_last @ W[h] — a single linear readout, no reservoir advance.

    autoreg
        Fit only W[1] (one-step readout).  predict(h) advances the reservoir
        h-1 steps by feeding its own predictions back as inputs, then reads out
        once.  Non-destructive: s_last is never mutated by predict().

        Constraint: the readout and the reservoir input must live in the same
        z-score space.  This holds automatically when context is None (univariate)
        or when history and context are the same (T, D) array.

    Parameters
    ----------
    model       : SASModel(projector, basis).  Should be un-initialised
                  (fit() will call model.initialize(key)).
    washout     : steps discarded before ridge regression.
    chunk_size  : static chunk size B for the associative scan.
    n_cv_folds  : rolling-window CV folds for ridge alpha selection.
    seed        : JAX PRNG seed for model initialisation.
    alphas      : ridge penalty candidates (default: log-spaced 1e-4…1e5).
    mode        : 'direct' | 'autoreg'
    scale_input : if True, min-max scale context and history into [-1, 1] before
                  feeding the reservoir; predictions are mapped back to the
                  original scale.  Default False — the data is assumed to already
                  live in [-1, 1] (a DGP normalised to its known bounds) and the
                  reservoir sees and predicts in the raw data units.
    clip_output : if True, clamp predictions to the [-1, 1] domain (in the
                  reservoir scale).  In autoreg mode this also clamps the fed-back
                  prediction each step, keeping the closed loop in-domain and
                  guarding against autonomous divergence.  Use when the DGP is
                  known to be bounded in [-1, 1].  Default False.
    """

    def __init__(
        self,
        model:        SASModel,
        washout:      int        = 50,
        chunk_size:   int        = 64,
        n_cv_folds:   int        = 5,
        seed:         int        = 42,
        alphas:       list | None = None,
        mode:         str        = 'direct',
        scale_input:  bool       = False,
        clip_output:  bool       = False,
    ):
        if not isinstance(model, SASModel):
            raise TypeError(
                f"model must be a SASModel instance, got {type(model).__name__}."
            )
        if mode not in ('direct', 'autoreg'):
            raise ValueError(f"mode must be 'direct' or 'autoreg', got {mode!r}")
        self._model      = model
        self.washout     = washout
        self.chunk_size  = chunk_size
        self.n_cv_folds  = n_cv_folds
        self.seed        = seed
        self.alphas      = list(alphas) if alphas is not None else _ALPHAS
        self.mode        = mode
        self.scale_input = scale_input
        self.clip_output = clip_output

        self._W:            dict[int, np.ndarray] = {}
        self._s_last:       np.ndarray | None    = None
        self._states_train: np.ndarray | None    = None   # (T, N) cached after fit
        self._mu:           float | None         = None
        self._sigma:        float | None         = None
        # Multivariate context z-score params (None when d=1)
        self._ctx_mu:       np.ndarray | None    = None
        self._ctx_sigma:    np.ndarray | None    = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        history:  np.ndarray,
        horizons: list[int],
        context:  np.ndarray | None = None,
    ) -> "SASForecaster":
        """
        Fit the forecaster.

        Parameters
        ----------
        history  : (T,) univariate target series, or (T, D) for multi-output.
                   When 2-D, self._mu and self._sigma are (D,) arrays and
                   self._W[h] stores an (N, D) weight matrix per horizon.
        horizons : list of integer forecast horizons.
        context  : (T, d) optional multivariate context for the reservoir.
                   When given, the reservoir is driven by these d channels
                   instead of by history alone.  d must equal
                   projector.d as set in the SASModel.
                   When None, history is used as the sole context (d=1).
        """
        history = np.asarray(history, dtype=np.float64)
        if history.ndim not in (1, 2):
            raise ValueError(f"history must be 1-D or 2-D, got shape {history.shape}")
        T = history.shape[0]

        if history.ndim == 2:
            if self.scale_input:
                lo, hi      = history.min(axis=0), history.max(axis=0)   # (D,)
                self._mu    = (lo + hi) / 2.0                            # center
                self._sigma = np.maximum((hi - lo) / 2.0, 1e-8)         # half-range
            else:
                self._mu    = np.zeros(history.shape[1])
                self._sigma = np.ones(history.shape[1])
            Y_z = ((history - self._mu) / self._sigma).astype(np.float32)
        else:
            if self.scale_input:
                self._mu, self._sigma = self._fit_scaler(history)
            else:
                self._mu, self._sigma = 0.0, 1.0
            Y_z = self._zscore(history, self._mu, self._sigma).astype(np.float32)

        if context is not None:
            ctx = np.asarray(context, dtype=np.float64)
            if ctx.ndim == 1:
                ctx = ctx[:, None]
            if ctx.shape[0] != T:
                raise ValueError(
                    f"context.shape[0]={ctx.shape[0]} != len(history)={T}"
                )
            if self.scale_input:
                lo, hi          = ctx.min(axis=0), ctx.max(axis=0)        # (d,)
                self._ctx_mu    = (lo + hi) / 2.0                         # center
                self._ctx_sigma = np.maximum((hi - lo) / 2.0, 1e-8)       # half-range
            else:
                self._ctx_mu    = np.zeros(ctx.shape[1])
                self._ctx_sigma = np.ones(ctx.shape[1])
            ctx_z = ((ctx - self._ctx_mu) / self._ctx_sigma).astype(np.float32)
        else:
            self._ctx_mu    = None
            self._ctx_sigma = None
            ctx_z = Y_z[:, None] if Y_z.ndim == 1 else Y_z  # (T,1) or (T,D)

        key         = jax.random.PRNGKey(self.seed)
        self._model = self._model.initialize(key)

        z      = jnp.array(ctx_z)
        s0     = jnp.zeros(self._model.n, dtype=jnp.float32)
        if getattr(self._model.basis_p, 'training_mode', 'parallel') == 'sequential':
            # fast teacher-forced scan (precomputed features) when the basis supports
            # it (Sparse, LowRank); else the per-step streaming scan.
            scan = _fast_seq_scan if hasattr(self._model.basis_p, 'scan_matvec') else _stream_scan
            states, s_last = scan(self._model, s0, z)
        else:
            states, s_last = _forward(self._model, z, s0, self.chunk_size)

        states_np          = np.asarray(states, dtype=np.float32)
        self._s_last       = np.asarray(s_last, dtype=np.float32)
        self._states_train = states_np

        N  = self._model.n
        wo = self.washout
        self._W         = {}
        self.alpha_log_: dict[int, float] = {}

        fit_horizons = [1] if self.mode == 'autoreg' else horizons
        for h in fit_horizons:
            S = states_np[wo: T - h]
            Y = Y_z      [wo + h: T]
            if len(S) < 5:
                self._W[h] = np.zeros((N,) + Y_z.shape[1:], dtype=np.float32)
                continue
            alpha              = _ridge_cv(S, Y, self.n_cv_folds, self.alphas)
            self._W[h]         = _ridge_fit(S, Y, alpha).astype(np.float32)
            self.alpha_log_[h] = alpha

        return self

    def update(self, x) -> "SASForecaster":
        """
        Ingest one new observation without refitting.

        Parameters
        ----------
        x : float (univariate) or array-like of shape (d,) (multivariate).
            In multivariate mode (context was passed to fit), x must be
            the full d-dimensional context vector for this time step.
        """
        if self._mu is None:
            raise RuntimeError("SASForecaster must be fit before update().")

        if self._ctx_mu is not None:
            # Multivariate context
            x_arr = np.asarray(x, dtype=np.float64).ravel()
            x_z   = (x_arr - self._ctx_mu) / self._ctx_sigma     # (d,)
        else:
            # Univariate
            x_z = np.array([float(self._zscore(np.float64(x), self._mu, self._sigma))])

        z_t   = jnp.array(x_z.astype(np.float32))               # (d,)
        s_new = _step_once(self._model, jnp.array(self._s_last), z_t)
        self._s_last = np.asarray(s_new, dtype=np.float32)
        return self

    def predict(self, h: int):
        """
        Return the h-step forecast. Non-destructive: s_last is never modified.

        Returns a Python float for univariate (1-D history) targets, or a
        numpy array of shape (D,) for multi-output (2-D history) targets.

        direct mode  : s_last @ W[h] — single linear readout.
        autoreg mode : advance reservoir h-1 steps autoregressively via
                       lax.scan, then read out once from the resulting state.
                       Only W[1] must exist (fitted by fit()).
        """
        if self._s_last is None:
            raise RuntimeError("SASForecaster must be fit before predict().")

        if self.mode == 'direct':
            if h not in self._W:
                raise KeyError(
                    f"Horizon {h} not trained. Available: {sorted(self._W)}"
                )
            y_z = self._s_last @ self._W[h]   # scalar or (D,)

        else:  # autoreg
            if 1 not in self._W:
                raise RuntimeError(
                    "autoreg mode requires W[1]; call fit() first."
                )
            W1     = jnp.array(self._W[1], dtype=jnp.float32)   # (N,) or (N, D)
            s0     = jnp.array(self._s_last, dtype=jnp.float32)  # (N,)
            # Advance h-1 steps; lax.scan with length=0 returns s0 (h=1 case).
            s_prev = _autoreg_rollout(self._model, s0, W1, h - 1, clip=self.clip_output)
            y_z    = (np.asarray(s_prev, dtype=np.float64)
                      @ np.asarray(self._W[1], dtype=np.float64))  # scalar or (D,)

        # Optional clamp to the [-1, 1] domain (use when the DGP is known bounded).
        if self.clip_output:
            y_z = np.clip(y_z, -1.0, 1.0)
        out = self._unscale(y_z, self._mu, self._sigma)
        return out if np.ndim(out) > 0 else float(out)

    def transform(
        self,
        history: np.ndarray,
        context: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Run the reservoir on *history* (or *context* if provided) and
        return the (T, N) state matrix.

        Uses the trained z-score scalers — model must be fit first.
        The reservoir always restarts from s0 = 0.
        """
        if self._mu is None:
            raise RuntimeError("SASForecaster must be fit before transform().")

        if context is not None:
            ctx = np.asarray(context, dtype=np.float64)
            if ctx.ndim == 1:
                ctx = ctx[:, None]
            ctx_z = ((ctx - self._ctx_mu) / self._ctx_sigma).astype(np.float32)
        else:
            history = np.asarray(history, dtype=np.float64).ravel()
            h_z     = self._zscore(history, self._mu, self._sigma).astype(np.float32)
            ctx_z   = h_z[:, None]

        z  = jnp.array(ctx_z)
        s0 = jnp.zeros(self._model.n, dtype=jnp.float32)
        if getattr(self._model.basis_p, 'training_mode', 'parallel') == 'sequential':
            scan = _fast_seq_scan if hasattr(self._model.basis_p, 'scan_matvec') else _stream_scan
            states, _ = scan(self._model, s0, z)
        else:
            states, _ = _forward(self._model, z, s0, self.chunk_size)
        return np.asarray(states, dtype=np.float32)
