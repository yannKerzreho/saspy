"""
Modular SAS vs ESN benchmark runner.

Usage
-----
    python run_benchmark.py                     # use default config.yaml
    python run_benchmark.py --config my.yaml    # custom config
    python run_benchmark.py --dgps lorenz mso8  # subset of DGPs
    python run_benchmark.py --models esn sas_diagonal  # subset of models
    python run_benchmark.py --out results/      # output directory

Results are saved as JSON files in the output directory, one per DGP.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import yaml

# ── Insert parent dir so saspy is importable when run from benchmarks/ ───────
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import reservoirpy as rpy
import reservoirpy.datasets as rpy_datasets

import saspy
from saspy import SASForecaster, SASModel, InputProjector
from saspy.basis import DiagonalPoly, LRUBlockPoly, BlockLinearPoly, RandomFourierBasis
from saspy.engine import _stream_scan as _sas_stream_scan
from saspy.ridge import ridge_cv_select as _ridge_cv_select, ridge_fit as _ridge_fit
from esn import JaxESN

try:
    rpy.verbosity(0)
except AttributeError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# DGP loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dgp(dgp_cfg: dict, data_seed: int | None) -> np.ndarray:
    """
    Return a (T, D) array, each column normalised to [-1, 1] over the training
    window.  D = len(channels) when the config has a `channels` key, else D=1.

    Column 0 is always the forecast target channel.

    seed_strategy
    -------------
    "seed_param"  (default)
        Pass ``seed=data_seed`` to the loader (e.g. mackey_glass).

    "window"
        Generate a longer trajectory and slice at data_seed * window_stride.
        Use for deterministic loaders (lorenz, kuramoto_sivashinsky, mso8)
        whose seed kwarg is silently ignored.
    """
    loader_fn = getattr(rpy_datasets, dgp_cfg["loader"])
    params    = dict(dgp_cfg.get("params", {}))
    n_total   = dgp_cfg["n_total"]
    channels  = dgp_cfg.get("channels", None)       # list → multivariate
    channel   = dgp_cfg.get("channel",  0)          # int  → target column
    col_idx   = channels if channels is not None else [channel]
    strategy  = dgp_cfg.get("seed_strategy", "seed_param")

    def _extract(raw: np.ndarray) -> np.ndarray:
        """raw: (T, C) or (T,) → (T, D) using col_idx."""
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[:, None]
        return raw[:, col_idx]   # (T, D)

    if strategy == "window" and data_seed is not None and data_seed > 0:
        stride = int(dgp_cfg.get("window_stride", n_total // 2))
        n_gen  = n_total + data_seed * stride
        raw    = loader_fn(n_gen, **params)
        data   = _extract(raw)[data_seed * stride : data_seed * stride + n_total]
    else:
        if data_seed is not None and strategy == "seed_param":
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = loader_fn(n_total, seed=data_seed, **params)
            except TypeError:
                raw = loader_fn(n_total, **params)
        else:
            raw = loader_fn(n_total, **params)
        data = _extract(raw)[:n_total]

    # Normalise each column to [-1, 1] over the training window.
    n_train = dgp_cfg["n_train"]
    lo  = data[:n_train].min(axis=0)                         # (D,)
    hi  = data[:n_train].max(axis=0)                         # (D,)
    rng = np.where(hi != lo, hi - lo, 1.0)
    data = 2.0 * (data - lo) / rng - 1.0                    # (T, D)

    return data   # (T, D)


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

_BASIS_CLASSES = {
    "DiagonalPoly":      DiagonalPoly,
    "LRUBlockPoly":      LRUBlockPoly,
    "BlockLinearPoly":   BlockLinearPoly,
    "RandomFourierBasis": RandomFourierBasis,
}


def _n_drivers_from_cfg(basis_cfg: dict) -> int:
    """Infer n_drivers from a basis config block."""
    params = dict(basis_cfg.get("params", {}))
    btype  = basis_cfg["type"]
    if btype == "DiagonalPoly":
        return params["n"]
    # LRUBlockPoly, BlockLinearPoly, RandomFourierBasis all use n_blocks
    return params["n_blocks"]


def _make_basis(basis_cfg: dict):
    """Instantiate an un-initialised basis from a config block."""
    BasisCls     = _BASIS_CLASSES[basis_cfg["type"]]
    basis_params = dict(basis_cfg.get("params", {}))
    return BasisCls(**basis_params)


def make_sas_forecaster(
    model_cfg:  dict,
    washout:    int,
    chunk_size: int,
    seed:       int,
    d:          int = 1,
) -> SASForecaster:
    """
    Build a SASForecaster from a model config dict.

    Supports two config layouts:

    1. Single shared basis (backward-compatible):
         basis:
           type: DiagonalPoly
           params: {n: 100}

       Both basis_p and basis_q use the same class with the same params.

    2. Decoupled basis_p / basis_q:
         basis_p:
           type: LRUBlockPoly
           params: {n_blocks: 50}
         basis_q:
           type: RandomFourierBasis
           params: {n_blocks: 50, features_per_block: 2}

    Optional projector block (if absent: density=1.0, strategy=hybrid):
         projector:
           density:  0.0      # 0=cyclic channel assignment, 1=dense (default)
           strategy: hybrid   # "hybrid" | "pure_random"

    d : input dimension (number of context channels).
    """
    if "basis" in model_cfg:
        # Shared basis: both p and q use the same class/params
        basis_cfg = model_cfg["basis"]
        basis_p   = _make_basis(basis_cfg)
        basis_q   = _make_basis(basis_cfg)
        n_drivers = _n_drivers_from_cfg(basis_cfg)
    elif "basis_p" in model_cfg and "basis_q" in model_cfg:
        # Decoupled bases
        basis_p   = _make_basis(model_cfg["basis_p"])
        basis_q   = _make_basis(model_cfg["basis_q"])
        n_drivers = _n_drivers_from_cfg(model_cfg["basis_p"])
    else:
        raise ValueError(
            "model config must have either a 'basis' key (shared) "
            "or both 'basis_p' and 'basis_q' keys (decoupled)."
        )

    # Optional projector config — defaults preserve backward-compatible behaviour
    proj_cfg  = model_cfg.get("projector", {})
    density   = float(proj_cfg.get("density",  1.0))
    strategy  = str(  proj_cfg.get("strategy", "hybrid"))

    projector = InputProjector(d=d, n_drivers=n_drivers,
                               density=density, mixing_strategy=strategy)
    model     = SASModel(projector=projector, basis_p=basis_p, basis_q=basis_q)
    return SASForecaster(model=model, washout=washout, chunk_size=chunk_size, seed=seed)


def make_esn(model_cfg: dict, washout: int, seed: int):
    p = model_cfg["params"]
    res = JaxESN(int(p["units"]), lr=float(p["lr"]), sr=float(p["sr"]), seed=seed)
    return res, washout


# ─────────────────────────────────────────────────────────────────────────────
# Valid Prediction Time (VPT)
# ─────────────────────────────────────────────────────────────────────────────

def compute_vpt(nrmse_series: np.ndarray, threshold: float = 0.4) -> int:
    """
    Valid Prediction Time: index of the first step where NRMSE ≥ threshold.

    NRMSE(t) = sqrt( mean_d[(ŷ_{t,d} - y_{t,d})²] ) / σ_target
    where σ_target = sqrt( mean_d[Var_t(y_{t,d})] ).

    Returns the full test-window length when the threshold is never crossed.
    NaN / Inf values count as a threshold crossing (diverged prediction).
    """
    arr = np.asarray(nrmse_series, dtype=np.float64)
    bad = ~np.isfinite(arr) | (arr >= threshold)
    idx = np.where(bad)[0]
    return int(idx[0]) if len(idx) > 0 else int(len(arr))


def _autonomous_nrmse(preds_cl: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Per-step NRMSE for autonomous (closed-loop) predictions.

    Parameters
    ----------
    preds_cl, targets : (T, D)

    Returns
    -------
    nrmse : (T,) — normalised RMSE at each time step, using the RMS of
            per-channel variances of `targets` as the normalisation factor.
    """
    sigma = float(np.sqrt(targets.var(axis=0).mean() + 1e-12))
    return np.sqrt(((preds_cl - targets) ** 2).mean(axis=1)) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# Single evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _fit_multiout_readout(
    states_tr:  np.ndarray,
    Y_tr_all:   np.ndarray,
    washout:    int,
    horizon:    int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit a single (N, D) ridge readout for all D output channels at once.

    Y_tr_all : (T, D) — all D output channels on the NORMALISED [-1,1] scale.

    Returns
    -------
    W_mat  : (N, D)  readout weight matrix
    mu     : (D,)    per-channel training mean
    sigma  : (D,)    per-channel training std
    alpha  : float   ridge alpha selected by global MSE across all channels
    """
    T, D = Y_tr_all.shape
    S    = states_tr[washout: T - horizon].astype(np.float64)   # (T-wo-h, N)

    mu    = Y_tr_all.mean(axis=0)                               # (D,)
    sigma = np.maximum(Y_tr_all.std(axis=0), 1e-8)              # (D,)
    Y_z   = (Y_tr_all[washout + horizon:] - mu) / sigma         # (T-wo-h, D)

    # CV selects alpha using global MSE across all D channels simultaneously.
    alpha = _ridge_cv_select(S, Y_z)
    W_mat = _ridge_fit(S, Y_z, alpha).astype(np.float32)        # (N, D)
    return W_mat, mu.astype(np.float64), sigma.astype(np.float64), alpha


def eval_sas(
    forecaster:    SASForecaster,
    X_tr:          np.ndarray,
    X_te:          np.ndarray,
    horizon:       int,
    vpt_threshold: float = 0.4,
) -> dict:
    """
    Multivariate-output SAS evaluation.

    X_tr, X_te : (T, D) — all D channels, already normalised to [-1, 1].
                 When D > 1, column 0 drives the reservoir context AND is one
                 of the D output channels.  When D = 1, same as old univariate.

    One readout vector is fitted per output channel (shared ridge α).
    RMSE is averaged across all D channels.
    """
    D          = X_tr.shape[1]
    multivar   = D > 1
    context_tr = X_tr if multivar else None
    context_te = X_te if multivar else None

    # ── fit reservoir (channel 0 as nominal target for internal z-score) ──
    t0 = time.perf_counter()
    forecaster.fit(X_tr[:, 0], horizons=[horizon], context=context_tr)
    jax.block_until_ready(forecaster._s_last)
    t_fit = time.perf_counter() - t0

    # ── fit D readouts on cached training states ───────────────────────────
    W_mat, mu, sigma, _ = _fit_multiout_readout(
        forecaster._states_train, X_tr, forecaster.washout, horizon
    )

    # ── stream (accuracy): Python loop — correct sequential predictions ──────
    T_te    = len(X_te)
    s_start = forecaster._s_last.copy()      # save state before streaming
    preds_all = np.empty((T_te, D), dtype=np.float64)
    for t in range(T_te):
        preds_all[t] = forecaster._s_last.astype(np.float64) @ W_mat * sigma + mu
        forecaster.update(context_te[t] if multivar else X_te[t, 0])
    jax.block_until_ready(forecaster._s_last)

    # ── stream (timing): single _stream_scan call — apples-to-apples with ───
    # ESN's reservoir.run(X_tr[-1:]) + reservoir.run(X_te[:-1]).
    # Both are single-dispatch batched calls over T_te steps.
    if multivar:
        z_te = ((X_te - forecaster._ctx_mu) / forecaster._ctx_sigma).astype(np.float32)
    else:
        z_te = (((X_te[:, 0] - forecaster._mu) / forecaster._sigma)
                .astype(np.float32)[:, None])
    s0_jx   = jnp.array(s_start)
    z_te_jx = jnp.array(z_te)
    t0 = time.perf_counter()
    _, _s_end = _sas_stream_scan(forecaster._model, s0_jx, z_te_jx)
    jax.block_until_ready(_s_end)
    t_stream = time.perf_counter() - t0

    # ── autonomous rollout (VPT) from end-of-training state ──────────────
    # Reset to s_start (saved before teacher-forcing loop) and feed back
    # the model's own prediction at each step (no ground-truth input).
    forecaster._s_last = s_start.copy()
    preds_cl = np.empty((T_te, D), dtype=np.float64)
    for t in range(T_te):
        preds_cl[t] = forecaster._s_last.astype(np.float64) @ W_mat * sigma + mu
        y_fb = np.clip(preds_cl[t], -10.0, 10.0)   # prevent NaN explosion
        forecaster.update(y_fb if multivar else float(y_fb[0]))
    nrmse_cl = _autonomous_nrmse(preds_cl, X_te)
    vpt      = compute_vpt(nrmse_cl, vpt_threshold)

    rmse_ch = np.sqrt(np.mean((preds_all - X_te) ** 2, axis=0))   # (D,)
    return dict(
        t_fit=t_fit,
        t_stream=t_stream,
        rmse=float(rmse_ch.mean()),
        rmse_per_channel=rmse_ch.tolist(),
        mae=float(np.mean(np.abs(preds_all - X_te))),
        vpt=vpt,
    )


def eval_esn(reservoir, washout: int, X_tr: np.ndarray, X_te: np.ndarray,
             vpt_threshold: float = 0.4) -> dict:
    """
    Multivariate-output ESN evaluation (same ridge_cv readout as SAS).

    X_tr, X_te : (T, D) — all D channels, normalised to [-1, 1].

    Fit:   reservoir.run(X_tr[:-1]) → states; fit D readouts (state_t → X_tr[t+1, :]).
    Stream: teacher-forcing; state after X_te[t-1] predicts X_te[t, :].
    RMSE averaged across all D channels.
    """
    D = X_tr.shape[1]

    # ── fit ───────────────────────────────────────────────────────────────
    t0        = time.perf_counter()
    states_tr = reservoir.run(X_tr[:-1].astype(np.float32))    # (T-1, N)
    W_mat, mu, sigma, _ = _fit_multiout_readout(
        states_tr, X_tr[1:], washout, horizon=0   # target = X_tr[t+1] = X_tr[1:][t]
    )
    t_fit = time.perf_counter() - t0

    # ── stream (teacher forcing for RMSE) ─────────────────────────────────
    t0            = time.perf_counter()
    s_T           = reservoir.run(X_tr[-1:].astype(np.float32))   # (1, N)
    s_after_train = np.asarray(reservoir._state, dtype=np.float32) # save for VPT
    s_te          = reservoir.run(X_te[:-1].astype(np.float32))    # (T_te-1, N)
    states_pred   = np.vstack([s_T, s_te]).astype(np.float64)      # (T_te, N)
    preds_all     = states_pred @ W_mat * sigma + mu                # (T_te, D)
    t_stream      = time.perf_counter() - t0

    # ── autonomous rollout (VPT) from end-of-training state ───────────────
    # Restore state to right after the last training step, then feed the
    # model's own predictions back as input (no teacher forcing).
    T_te = X_te.shape[0]
    reservoir._state = jnp.array(s_after_train)
    preds_cl = np.empty((T_te, D), dtype=np.float64)
    W_mat_np = np.asarray(W_mat)   # avoid repeated JAX↔numpy conversion in loop
    for t in range(T_te):
        s = np.asarray(reservoir._state, dtype=np.float64)
        y = s @ W_mat_np * sigma + mu                              # (D,)
        preds_cl[t] = y
        reservoir.run(np.clip(y, -10.0, 10.0)[None, :].astype(np.float32))
    nrmse_cl = _autonomous_nrmse(preds_cl, X_te)
    vpt      = compute_vpt(nrmse_cl, vpt_threshold)

    rmse_ch = np.sqrt(np.mean((preds_all - X_te) ** 2, axis=0))   # (D,)
    return dict(
        t_fit=t_fit,
        t_stream=t_stream,
        rmse=float(rmse_ch.mean()),
        rmse_per_channel=rmse_ch.tolist(),
        mae=float(np.mean(np.abs(preds_all - X_te))),
        vpt=vpt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# JIT warm-up for SAS (excludes cold-compile from timing)
# ─────────────────────────────────────────────────────────────────────────────

def warmup_sas(
    model_cfg: dict,
    washout:   int,
    chunk_size: int,
    y_warmup:  np.ndarray,
    horizon:   int,
    X_warmup:  np.ndarray | None = None,
    d:         int = 1,
) -> float:
    fc = make_sas_forecaster(model_cfg, washout, chunk_size, seed=0, d=d)
    t0 = time.perf_counter()
    fc.fit(y_warmup, horizons=[horizon], context=X_warmup)
    jax.block_until_ready(fc._s_last)
    # warm up _step_once (online) and _stream_scan (batched timing)
    dummy = np.zeros(d, dtype=np.float32)
    fc.update(dummy if d > 1 else 0.0)
    jax.block_until_ready(fc._s_last)
    T_te_warmup = len(y_warmup) // 4          # short dummy stream for JIT compile
    z_warm  = jnp.zeros((T_te_warmup, d), dtype=jnp.float32)
    s_warm  = jnp.array(fc._s_last)
    _, _tmp = _sas_stream_scan(fc._model, s_warm, z_warm)
    jax.block_until_ready(_tmp)
    elapsed = time.perf_counter() - t0
    del fc
    return elapsed


def warmup_esn(model_cfg: dict, d: int, n_train: int) -> float:
    """
    Trigger JIT compilation for JaxESN's lax.scan kernel (excluded from timing).

    Pre-compiles for the three distinct sequence lengths used during evaluation:
      - n_train - 1  : training scan in eval_esn
      - 1            : single last-step carry in eval_esn
      - n_train // 4 - 1  : stream scan (= n_test - 1 for all DGPs in config)
    """
    p = model_cfg["params"]
    esn = JaxESN(int(p["units"]), lr=float(p["lr"]), sr=float(p["sr"]), seed=0)
    t0 = time.perf_counter()
    esn.run(np.zeros((n_train - 1, d), dtype=np.float32))
    esn.run(np.zeros((1, d), dtype=np.float32))
    n_stream = max(1, n_train // 4 - 1)
    esn.run(np.zeros((n_stream, d), dtype=np.float32))
    elapsed = time.perf_counter() - t0
    del esn
    return elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Per-DGP benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_dgp_benchmark(dgp_name: str, dgp_cfg: dict, models_cfg: dict,
                      active_models: list[str], benchmark_cfg: dict,
                      verbose: bool = True) -> dict:
    washout       = benchmark_cfg["washout"]
    horizon       = benchmark_cfg["horizon"]
    chunk_size    = benchmark_cfg["chunk_size"]
    vpt_threshold = float(benchmark_cfg.get("vpt_threshold", 0.4))
    n_train       = dgp_cfg["n_train"]
    data_seeds = dgp_cfg["data_seeds"]
    model_seeds = dgp_cfg["model_seeds"]

    results: dict[str, list] = {m: [] for m in active_models}
    cold_jit_times: dict[str, float] = {}

    if verbose:
        header = f"\n{'='*60}\n DGP: {dgp_cfg['label']}\n{'='*60}"
        print(header)
        col_w = 12
        cols = ["data_seed"] + [models_cfg[m]["label"][:col_w] for m in active_models]
        print("  ".join(f"{c:>{col_w}}" for c in cols))
        print("  " + "-" * (col_w * (len(cols) + 1)))

    # ── Input dimensionality from DGP config ──────────────────────────────
    d_input = len(dgp_cfg["channels"]) if "channels" in dgp_cfg else 1

    # ── Cold JIT warm-up for all JIT-compiled models (SAS and JaxESN) ────────
    for mkey in active_models:
        mtype = models_cfg[mkey]["type"]
        if mtype == "sas":
            first_seed = data_seeds[0]
            data_warm  = load_dgp(dgp_cfg, first_seed)          # (T, D)
            y_warmup   = data_warm[:n_train, 0]                 # target col
            X_warmup   = data_warm[:n_train]                    # full context
            cold_t = warmup_sas(
                models_cfg[mkey], washout, chunk_size,
                y_warmup, horizon,
                X_warmup=X_warmup if d_input > 1 else None,
                d=d_input,
            )
            cold_jit_times[mkey] = cold_t
            if verbose:
                label = models_cfg[mkey]["label"]
                print(f"  {label} cold JIT: {cold_t*1e3:.0f} ms (excluded from timing)")
        elif mtype == "esn":
            cold_t = warmup_esn(models_cfg[mkey], d_input, n_train)
            cold_jit_times[mkey] = cold_t
            if verbose:
                label = models_cfg[mkey]["label"]
                print(f"  {label} cold JIT: {cold_t*1e3:.0f} ms (excluded from timing)")

    # ── Main loop ─────────────────────────────────────────────────────────
    for d_seed in data_seeds:
        data = load_dgp(dgp_cfg, d_seed)                        # (T, D)
        X_tr = data[:n_train]                                   # (n_train, D)
        X_te = data[n_train:]                                   # (n_test,  D)

        row_rmse: dict[str, list] = {m: [] for m in active_models}

        for m_seed in model_seeds:
            for mkey in active_models:
                mcfg = models_cfg[mkey]
                if mcfg["type"] == "esn":
                    esn_model, wo = make_esn(mcfg, washout, m_seed)
                    rec = eval_esn(esn_model, wo, X_tr, X_te,
                                   vpt_threshold=vpt_threshold)
                elif mcfg["type"] == "sas":
                    fc = make_sas_forecaster(mcfg, washout, chunk_size, m_seed, d=d_input)
                    rec = eval_sas(fc, X_tr, X_te, horizon,
                                   vpt_threshold=vpt_threshold)
                else:
                    raise ValueError(f"Unknown model type: {mcfg['type']}")

                rec["data_seed"]  = d_seed
                rec["model_seed"] = m_seed
                results[mkey].append(rec)
                row_rmse[mkey].append(rec["rmse"])

        if verbose:
            row_vals = [str(d_seed)]
            for mkey in active_models:
                mu = np.mean(row_rmse[mkey])
                row_vals.append(f"{mu:.5f}")
            print("  ".join(f"{v:>{col_w}}" for v in row_vals))

    return {
        "dgp": dgp_name,
        "label": dgp_cfg["label"],
        "lyapunov_time_steps": dgp_cfg.get("lyapunov_time_steps"),
        "cold_jit_times": cold_jit_times,
        "records": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], models_cfg: dict) -> None:
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY — RMSE (mean ± std across all evaluations)")
    print("=" * 70)

    model_keys = list(models_cfg.keys())
    col_w = 20

    header = f"{'DGP':<25}" + "".join(
        f"  {models_cfg[m]['label'][:col_w]:>{col_w}}" for m in model_keys
        if m in all_results[0]["records"]
    )
    print(header)
    print("-" * len(header))

    for res in all_results:
        row = f"{res['label']:<25}"
        for mkey in model_keys:
            if mkey not in res["records"]:
                continue
            recs = res["records"][mkey]
            rmses = np.array([r["rmse"] for r in recs])
            row += f"  {rmses.mean():.5f}±{rmses.std():.5f}"
        print(row)

    # ── VPT summary ───────────────────────────────────────────────────────
    has_vpt = any(
        any("vpt" in r for r in res["records"].get(list(models_cfg.keys())[0], []))
        for res in all_results
    )
    if not has_vpt:
        return

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY — VPT in steps (mean ± std, ε=0.4, autonomous rollout)")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for res in all_results:
        row = f"{res['label']:<25}"
        lyt = res.get("lyapunov_time_steps")
        for mkey in model_keys:
            if mkey not in res["records"]:
                continue
            recs = res["records"][mkey]
            vpts = np.array([r["vpt"] for r in recs if "vpt" in r], dtype=float)
            if len(vpts) == 0:
                row += f"  {'n/a':>{col_w}}"; continue
            if lyt:
                row += f"  {vpts.mean()/lyt:.2f}±{vpts.std()/lyt:.2f} T_L"
            else:
                row += f"  {vpts.mean():.1f}±{vpts.std():.1f} steps"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SAS vs ESN benchmark")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config (default: config.yaml)")
    parser.add_argument("--dgps", nargs="*", default=None,
                        help="Subset of DGP keys to run (default: all)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model keys to run (default: all)")
    parser.add_argument("--out", default="results",
                        help="Output directory for JSON results (default: results/)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-row output")
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = pathlib.Path(__file__).parent / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    benchmark_cfg = cfg["benchmark"]
    dgps_cfg      = cfg["dgps"]
    models_cfg    = cfg["models"]

    active_dgps   = args.dgps   or list(dgps_cfg.keys())
    active_models = args.models or list(models_cfg.keys())

    out_dir = pathlib.Path(args.out)
    if not out_dir.is_absolute():
        # Resolve relative to CWD (where the user invoked the script),
        # not relative to the script file, so `--out results` always lands
        # in the current working directory regardless of how the script is called.
        out_dir = pathlib.Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"saspy {saspy.__version__}  |  JAX backend: {jax.default_backend()}")
    print(f"DGPs   : {active_dgps}")
    print(f"Models : {active_models}")

    all_results = []
    for dgp_name in active_dgps:
        if dgp_name not in dgps_cfg:
            print(f"[WARN] DGP '{dgp_name}' not found in config, skipping.")
            continue

        res = run_dgp_benchmark(
            dgp_name=dgp_name,
            dgp_cfg=dgps_cfg[dgp_name],
            models_cfg=models_cfg,
            active_models=active_models,
            benchmark_cfg=benchmark_cfg,
            verbose=not args.quiet,
        )
        all_results.append(res)

        out_path = out_dir / f"{dgp_name}.json"
        # Don't save raw predictions to keep files manageable
        slim = {
            "dgp": res["dgp"],
            "label": res["label"],
            "lyapunov_time_steps": res.get("lyapunov_time_steps"),
            "cold_jit_times": res["cold_jit_times"],
            "records": {
                mkey: [
                    {k: v for k, v in r.items() if k != "preds"}
                    for r in recs
                ]
                for mkey, recs in res["records"].items()
            },
        }
        with open(out_path, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"  → saved {out_path}")

    print_summary(all_results, models_cfg)


if __name__ == "__main__":
    main()
