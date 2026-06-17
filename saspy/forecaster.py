"""
forecaster.py — SASForecaster: streaming time-series forecaster.

Wraps SASModel + ridge regression into the BaseForecaster protocol.

    fit(history, horizons, context=None)
        Initialise model, run scan, fit ridge per horizon.
        Pass `context` (T, d) to drive the reservoir with a multivariate
        signal while forecasting a univariate target (`history`).
        When context is None, history itself (d=1) drives the reservoir.

    update(x)   → single streaming step; x is scalar or (d,) vector.
    predict(h)  → s_last · W[h], un-z-scored to history scale.
    transform(history, context=None) → (T, N) raw reservoir states.

Pre-processing contract
-----------------------
The forecaster z-scores internally.  Any other transformation (log,
differencing, deseasonalisation) is the caller's responsibility.
"""

from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp

from .base       import BaseForecaster
from .model      import SASModel
from .engine     import _forward, _step_once, _stream_scan
from .ridge      import (
    ALPHAS          as _ALPHAS,
    ridge_cv_select as _ridge_cv,
    ridge_fit       as _ridge_fit,
)


# ── Autoregressive rollout kernel ────────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=('n_steps',))
def _autoreg_rollout(model, s0, W1, n_steps: int):
    """
    Advance the SAS reservoir n_steps autoregressively.

    At each step the readout W1 is applied to the current state to produce
    the predicted z-scored input for the next step.  Non-destructive: only
    the final state is returned; the caller's s_last is never mutated.

    model   : SASModel pytree (initialised)
    s0      : (N,) starting state
    W1      : (N,) or (N, D) one-step ridge readout (z-score space)
    n_steps : static int — 0 returns s0 unchanged (lax.scan identity)
    """
    def body(s, _):
        y_z = s @ W1                                   # () or (D,)
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
    scale_input : if True (default) z-score context and history before feeding
                  the reservoir; predictions are unzscored back to the original
                  scale.  Set False when the data already lives on a compact
                  attractor and no re-scaling is needed — the reservoir then
                  sees and predicts in the raw data units.
    """

    def __init__(
        self,
        model:        SASModel,
        washout:      int        = 50,
        chunk_size:   int        = 64,
        n_cv_folds:   int        = 4,
        seed:         int        = 42,
        alphas:       list | None = None,
        mode:         str        = 'direct',
        scale_input:  bool       = True,
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
        if history.ndim == 1:
            pass                         # keep as (T,)
        elif history.ndim == 2:
            pass                         # keep as (T, D)
        else:
            raise ValueError(f"history must be 1-D or 2-D, got shape {history.shape}")
        T = history.shape[0]

        # 1. scale target — scalar params for 1-D, (D,) arrays for 2-D
        if history.ndim == 2:
            if self.scale_input:
                self._mu    = history.mean(axis=0)                   # (D,)
                self._sigma = np.maximum(history.std(axis=0), 1e-8)  # (D,)
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

        # 2. prepare reservoir input z: (T, d)
        if context is not None:
            ctx = np.asarray(context, dtype=np.float64)
            if ctx.ndim == 1:
                ctx = ctx[:, None]
            if ctx.shape[0] != T:
                raise ValueError(
                    f"context.shape[0]={ctx.shape[0]} != len(history)={T}"
                )
            if self.scale_input:
                self._ctx_mu    = ctx.mean(axis=0)                        # (d,)
                self._ctx_sigma = np.maximum(ctx.std(axis=0), 1e-8)       # (d,)
            else:
                self._ctx_mu    = np.zeros(ctx.shape[1])
                self._ctx_sigma = np.ones(ctx.shape[1])
            ctx_z = ((ctx - self._ctx_mu) / self._ctx_sigma).astype(np.float32)
        else:
            self._ctx_mu    = None
            self._ctx_sigma = None
            ctx_z = Y_z[:, None] if Y_z.ndim == 1 else Y_z  # (T,1) or (T,D)

        # 3. initialise model (projector + basis)
        key         = jax.random.PRNGKey(self.seed)
        self._model = self._model.initialize(key)

        # 4. run reservoir
        z      = jnp.array(ctx_z)                              # (T, d)
        s0     = jnp.zeros(self._model.n, dtype=jnp.float32)
        if getattr(self._model.basis_p, 'training_mode', 'parallel') == 'sequential':
            states, s_last = _stream_scan(self._model, s0, z)
        else:
            states, s_last = _forward(self._model, z, s0, self.chunk_size)

        states_np         = np.asarray(states, dtype=np.float32)
        self._s_last      = np.asarray(s_last, dtype=np.float32)
        self._states_train = states_np   # cached for multi-output readout fitting

        # 5. ridge per horizon
        # autoreg mode only needs h=1; other horizons are ignored.
        N  = self._model.n
        wo = self.washout
        self._W        = {}
        self.alpha_log_: dict[int, float] = {}

        fit_horizons = [1] if self.mode == 'autoreg' else horizons
        for h in fit_horizons:
            S = states_np[wo: T - h]
            Y = Y_z      [wo + h: T]
            if len(S) < 5:
                # shape (N,) for 1-D target, (N, D) for multi-output
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
            s_prev = _autoreg_rollout(self._model, s0, W1, h - 1)
            y_z    = (np.asarray(s_prev, dtype=np.float64)
                      @ np.asarray(self._W[1], dtype=np.float64))  # scalar or (D,)

        out = self._unzscore(y_z, self._mu, self._sigma)
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
            states, _ = _stream_scan(self._model, s0, z)
        else:
            states, _ = _forward(self._model, z, s0, self.chunk_size)
        return np.asarray(states, dtype=np.float32)
