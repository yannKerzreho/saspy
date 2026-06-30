"""
SAS vs ESN benchmark — prints NRMSE/VPT/SWD tables to stdout.

Usage
-----
    python main.py                       # all DGPs and models from config.yaml
    python main.py --config my.yaml      # custom config
    python main.py --dgps lorenz mso8    # subset of DGPs
    python main.py --models esn sas_lru  # subset of models
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
import warnings

import numpy as np
import yaml
from tqdm import tqdm

# saspy is editable-installed and the repo root is on sys.path, so both `saspy`
# and the `benchmark` package import without any path manipulation.
_HERE = pathlib.Path(__file__).resolve().parent

import jax
import jax.numpy as jnp
import reservoirpy as rpy

import saspy
from saspy import (
    SASForecaster, SASModel,
    DiagonalP, DiagonalQ, BlockP, BlockQ, SparseP, SparseQ,
    LowRankP, LowRankQ,
    Cheb, Trig,
)
from benchmark.esn   import JaxESN, JaxESNForecaster
from benchmark.utils import load_dgp, autonomous_nrmse, compute_vpt, sliced_wasserstein

try:
    rpy.verbosity(0)
except AttributeError:
    pass

# SWD values above this threshold indicate a diverged trajectory and are excluded
SWD_DIVERGE = 1e3

# Benchmark-wide leaky-integrator rate for every SAS model (matches the ESN's lr).
# Set from cfg['benchmark']['leak'] in main(); a per-model `leak:` key overrides it.
_BENCH_LEAK = 1.0


# ── data ──────────────────────────────────────────────────────────────────────

def _extract_window(
    data: np.ndarray,
    seed: int,
    n_train: int,
    n_test: int,
    col_idx: list[int],
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = seed * stride
    w = data[start: start + n_train + n_test][:, col_idx]
    return w[:n_train], w[n_train:]


# ── model factories ───────────────────────────────────────────────────────────

def _make_feature(fcfg: dict):
    """Build a bounded feature spec (Cheb / Trig) from a config dict."""
    kind = fcfg.get("kind", "cheb")
    if kind == "cheb":
        return Cheb(degree=int(fcfg.get("degree", 2)),
                    cross_input=bool(fcfg.get("cross_input", True)))
    if kind == "trig":
        return Trig(degree=int(fcfg.get("degree", 2)),
                    bandwidth=float(fcfg.get("bandwidth", 1.0)),
                    kernel=str(fcfg.get("kernel", "gaussian")),
                    density_omega=float(fcfg.get("density_omega", 1.0)),
                    bandwidth_min=fcfg.get("bandwidth_min"),
                    bandwidth_max=fcfg.get("bandwidth_max"))
    raise ValueError(f"Unknown feature kind: {kind!r}")


def _make_role(cfg: dict, role: str, d: int):
    """Build a P (role='p') or Q (role='q') basis for a structure config."""
    structure = cfg["structure"]
    feat      = _make_feature(cfg.get("feature", {}))
    sn        = float(cfg.get("spectral_norm", 0.9))
    if structure == "diagonal":
        cls = DiagonalP if role == "p" else DiagonalQ
        return cls(int(cfg["n"]), feature=feat, spectral_norm=sn)
    if structure == "block":
        K, B = int(cfg["n_blocks"]), int(cfg.get("block_size", 2))
        if role == "p":
            return BlockP(K, B, feature=feat, spectral_norm=sn,
                          init_mode=str(cfg.get("init_mode", "rotation")),
                          tau_min=float(cfg.get("tau_min", 1.0)),
                          tau_max=float(cfg.get("tau_max", 100.0)),
                          frac_diagonal=float(cfg.get("frac_diagonal", 0.5)))
        return BlockQ(K, B, feature=feat, spectral_norm=sn)
    if structure == "sparse":
        # n_drivers: int, or "binom" → C(d+degree, degree) (match the cross-input
        # monomial count, for a W_in linear-mix model with comparable capacity).
        Kc = cfg.get("n_drivers", d)
        if Kc == "binom":
            deg = int(cfg.get("feature", {}).get("degree", 2))
            Kc  = math.comb(d + deg, deg)
        K    = int(Kc)
        conn = cfg.get("connectivity")
        if role == "p":
            return SparseP(int(cfg["n"]), K, feature=feat, spectral_norm=sn,
                           density_P=float(cfg.get("density_P", 0.05)),
                           A_density=cfg.get("A_density"), connectivity=conn)
        return SparseQ(int(cfg["n"]), K, feature=feat, spectral_norm=sn,
                       density_Q=float(cfg.get("density_Q", 0.1)), connectivity=conn)
    if structure == "lowrank":
        # K=R driver mode: W_in projects d → R drivers, α_r = T_{d_r}(z̃_r).
        R     = int(cfg["rank"])
        amode = str(cfg.get("alpha_mode", "driver"))
        if role == "p":
            return LowRankP(int(cfg["n"]), R, feature=feat, rank=R, spectral_norm=sn,
                            alpha_mode=amode, connectivity=cfg.get("connectivity", 1.5),
                            backbone=bool(cfg.get("backbone", True)))
        return LowRankQ(int(cfg["n"]), R, feature=feat, rank=R, spectral_norm=sn,
                        alpha_mode=amode, density_G=cfg.get("density_G"))
    raise ValueError(f"Unknown structure: {structure!r}")


def make_sas(model_cfg: dict, washout: int, chunk_size: int, seed: int, d: int = 1) -> SASForecaster:
    # 'basis' = shared config for P and Q; or separate 'basis_p'/'basis_q'.
    pcfg = model_cfg.get("basis_p", model_cfg.get("basis"))
    qcfg = model_cfg.get("basis_q", model_cfg.get("basis"))
    basis_p = _make_role(pcfg, "p", d)
    basis_q = _make_role(qcfg, "q", d)

    # Leaky-integrator rate: benchmark-wide default (_BENCH_LEAK), per-model override.
    leak = float(model_cfg.get("leak", _BENCH_LEAK))

    # Sparse defaults to an identity projection (the joint cross-input features mix
    # the d inputs).  But if a `proj` block is given, build a random W_in instead —
    # this is what lets a cross_input=False sparse model mix inputs *linearly*.
    if pcfg["structure"] == "sparse" and "proj" not in model_cfg:
        model = SASModel(basis_p, basis_q, leak=leak)      # identity projection
    else:
        proj    = model_cfg.get("proj", {})
        density = float(proj.get("density", 1.0))
        bias    = bool(proj.get("bias", False))
        model   = SASModel(basis_p, basis_q, d=d, density=density, bias=bias, leak=leak)

    # data is pre-normalised to [-1, 1] by load_dgp, so no internal rescaling.
    return SASForecaster(model=model, washout=washout, chunk_size=chunk_size,
                         seed=seed, scale_input=False, clip_output=True, mode="autoreg")


def make_esn(model_cfg: dict, washout: int, seed: int) -> JaxESNForecaster:
    p      = model_cfg["params"]
    kwargs = {k: v for k, v in p.items() if k not in ("units", "lr", "sr")}
    esn    = JaxESN(int(p["units"]), lr=float(p["lr"]), sr=float(p["sr"]),
                    seed=seed, **kwargs)
    return JaxESNForecaster(esn, washout=washout, mode="autoreg")


# ── evaluation ────────────────────────────────────────────────────────────────

def eval_sas(
    fc: SASForecaster,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    vpt_threshold: float = 0.4,
) -> dict:
    # Stability of the autonomous rollout is handled by clip_output=True
    # (predictions clamped to the [-1, 1] reservoir domain each step).
    D, T_te = X_tr.shape[1], len(X_te)

    # Training time (model already JIT-warmed in run_dgp, so this is steady-state)
    t0 = time.perf_counter()
    fc.fit(X_tr, horizons=[1], context=X_tr)
    jax.block_until_ready(fc._s_last)
    t_train = time.perf_counter() - t0

    preds = np.empty((T_te, D), dtype=np.float64)
    t0 = time.perf_counter()
    for t in range(T_te):
        if not np.isfinite(fc._s_last).all():
            fc._s_last = np.zeros_like(fc._s_last)
        pred     = np.atleast_1d(fc.predict(1))
        preds[t] = pred
        fc.update(pred)
    t_infer = time.perf_counter() - t0

    nrmse = autonomous_nrmse(preds, X_te)

    # Pure state-building (scan) time — no ridge CV.  Measured last; transform()
    # restarts from s0 and does not touch the post-fit state used above.
    t0 = time.perf_counter()
    _ = fc.transform(X_tr, context=X_tr)
    t_scan = time.perf_counter() - t0

    return dict(
        vpt     = compute_vpt(nrmse, vpt_threshold),
        swd     = sliced_wasserstein(preds, X_te),
        nrmse_h = float(np.mean(nrmse[:10])),
        t_train = t_train,
        t_scan  = t_scan,
        t_infer = t_infer,
    )


def eval_esn(
    fc: JaxESNForecaster,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    vpt_threshold: float = 0.4,
) -> dict:
    D, T_te = X_tr.shape[1], len(X_te)

    # Training time (model already JIT-warmed in run_dgp, so this is steady-state)
    t0 = time.perf_counter()
    fc.fit(X_tr, horizons=[1])
    jax.block_until_ready(np.asarray(fc._esn._state))
    t_train = time.perf_counter() - t0

    preds = np.empty((T_te, D), dtype=np.float64)
    t0 = time.perf_counter()
    for t in range(T_te):
        pred    = np.atleast_1d(fc.predict(1))
        preds[t] = pred
        fc.update(pred)
    t_infer = time.perf_counter() - t0

    nrmse = autonomous_nrmse(preds, X_te)

    # Pure state-building (scan) time — no ridge CV.  Reset to s0 then run (raw,
    # no z-scoring — the data is already in [-1, 1]).
    fc._esn._state = jnp.zeros(fc._esn.units, dtype=jnp.float32)
    t0 = time.perf_counter()
    _ = fc._esn.run(np.asarray(X_tr, dtype=np.float32))
    t_scan = time.perf_counter() - t0

    return dict(
        vpt     = compute_vpt(nrmse, vpt_threshold),
        swd     = sliced_wasserstein(preds, X_te),
        nrmse_h = float(np.mean(nrmse[:10])),
        t_train = t_train,
        t_scan  = t_scan,
        t_infer = t_infer,
    )


# ── JIT warm-up ───────────────────────────────────────────────────────────────

def _warmup_sas(model_cfg: dict, washout: int, chunk_size: int, d: int, n_train: int) -> None:
    X = np.zeros((n_train, d), dtype=np.float64)
    fc = make_sas(model_cfg, washout, chunk_size, seed=0, d=d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc.fit(X, horizons=[1], context=X)
    jax.block_until_ready(fc._s_last)
    p = np.atleast_1d(fc.predict(1))
    fc.update(p)
    jax.block_until_ready(fc._s_last)


def _warmup_esn(model_cfg: dict, washout: int, d: int, n_train: int) -> None:
    X = np.zeros((n_train, d), dtype=np.float64)
    fc = make_esn(model_cfg, washout, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc.fit(X, horizons=[1])
    p = np.atleast_1d(fc.predict(1))
    fc.update(np.zeros(d))


# ── per-DGP runner ────────────────────────────────────────────────────────────

def run_dgp(
    dgp_name: str,
    dgp_cfg: dict,
    dgp_defaults: dict,
    models_cfg: dict,
    active_models: list[str],
    bench_cfg: dict,
) -> dict:
    washout       = bench_cfg["washout"]
    chunk_size    = bench_cfg["chunk_size"]
    vpt_threshold = float(bench_cfg.get("vpt_threshold", 0.4))

    n_seed = dgp_cfg.get("n_seed",  bench_cfg.get("n_seed", 10))
    n_train = dgp_cfg.get("n_train", dgp_defaults.get("n_train", 5000))
    n_test  = dgp_cfg.get("n_test",  dgp_defaults.get("n_test",  2000))
    stride  = dgp_cfg.get("stride",  dgp_defaults.get("stride",  n_test))
    n_total = n_train + n_test + (n_seed - 1) * stride

    data    = load_dgp(dgp_cfg["loader"], n_total, **(dgp_cfg.get("args") or {}))
    col_idx = dgp_cfg.get("channels", list(range(data.shape[1])))
    d       = len(col_idx)

    # JIT warm-up on a zero sequence so compilation is excluded from eval
    for mkey in active_models:
        mcfg = models_cfg[mkey]
        if mcfg["type"] == "sas":
            _warmup_sas(mcfg, washout, chunk_size, d, n_train)
        elif mcfg["type"] == "esn":
            _warmup_esn(mcfg, washout, d, n_train)

    records = {m: [] for m in active_models}
    for seed in tqdm(range(n_seed), desc=dgp_cfg["label"], leave=True):
        X_tr, X_te = _extract_window(data, seed, n_train, n_test, col_idx, stride)

        if not (np.isfinite(X_tr).all() and np.isfinite(X_te).all()):
            nan_rec = dict(vpt=0, swd=float("nan"), nrmse_h=float("nan"),
                           t_train=float("nan"), t_scan=float("nan"),
                           t_infer=float("nan"))
            for mkey in active_models:
                records[mkey].append(nan_rec.copy())
            continue

        for mkey in active_models:
            mcfg = models_cfg[mkey]
            if mcfg["type"] == "esn":
                rec = eval_esn(make_esn(mcfg, washout, seed), X_tr, X_te, vpt_threshold)
            else:
                rec = eval_sas(make_sas(mcfg, washout, chunk_size, seed, d),
                               X_tr, X_te, vpt_threshold)
            records[mkey].append(rec)

    return dict(
        label               = dgp_cfg["label"],
        lyapunov_time_steps = dgp_cfg.get("lyapunov_time_steps"),
        records             = records,
    )


# ── results table ─────────────────────────────────────────────────────────────

def print_results(all_results: list[dict], models_cfg: dict, active_models: list[str]) -> None:
    labels = [models_cfg[m]["label"] for m in active_models]

    metrics = [
        ("nrmse_h", "NRMSE h=10  (mean ± std, autonomous rollout)"),
        ("vpt",     "VPT  (mean ± std, ε=0.4)"),
        ("swd",     f"SWD  (mean ± std, diverged seeds excluded, cap={SWD_DIVERGE:.0e})"),
        ("t_train", "Training time  (s, mean ± std, scan + ridge CV)"),
        ("t_scan",  "Scan time  (s, mean ± std, build states only — no ridge)"),
        ("t_infer", "Inference time  (s, mean ± std, autonomous rollout)"),
    ]

    def _cell(res: dict, key: str, mkey: str) -> str:
        vals = np.array([r[key] for r in res["records"].get(mkey, [])], dtype=float)
        if key == "swd":
            vals = vals[np.isfinite(vals) & (vals < SWD_DIVERGE)]
        else:
            vals = vals[np.isfinite(vals)]
        lyt = res.get("lyapunov_time_steps")
        if len(vals) == 0:
            return "n/a"
        if key == "vpt" and lyt:
            return f"{vals.mean()/lyt:.1f}±{vals.std()/lyt:.1f}TL"
        return f"{vals.mean():.3f}±{vals.std():.3f}"

    # Size columns to the widest *rendered* cell (+ gap), not just the label,
    # so large values (e.g. "555.300±196.818") never spill into the next column.
    dgp_w = max(20, max(len(r["label"]) for r in all_results) + 2)
    col_w = max(len(l) for l in labels)
    for key, _ in metrics:
        for res in all_results:
            for mkey in active_models:
                col_w = max(col_w, len(_cell(res, key, mkey)))
    col_w  += 3
    total_w = dgp_w + col_w * len(active_models)

    header = f"{'DGP':<{dgp_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    for key, title in metrics:
        print(f"\n{'═' * total_w}\n  {title}\n{'═' * total_w}")
        print(header)
        print("─" * total_w)
        for res in all_results:
            line = f"{res['label']:<{dgp_w}}"
            for mkey in active_models:
                line += f"{_cell(res, key, mkey):>{col_w}}"
            print(line)
    print("═" * total_w)


# ── results plot ──────────────────────────────────────────────────────────────

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

# Paired-vs-ESN metrics.  VPT → per-seed difference (handles VPT=0/divergence);
# NRMSE/SWD → per-seed log2 ratio (strictly positive).  `better` = the side of 0
# that means the model beats ESN.
_PAIRED = [
    ("vpt",     "ΔVPT vs ESN",          "diff",  "greater"),
    ("nrmse_h", "log2(NRMSE / ESN)",    "ratio", "less"),
    ("swd",     "log2(SWD / ESN)",      "ratio", "less"),
]

# Timing metrics — all strictly positive → log2 ratio; faster (ratio<1) is better.
_TIMES = [
    ("t_train", "log2(train / ESN)",    "ratio", "less"),
    ("t_scan",  "log2(scan / ESN)",     "ratio", "less"),
    ("t_infer", "log2(infer / ESN)",    "ratio", "less"),
]


def _paired_vals(res: dict, key: str, mkey: str, base: str, mode: str) -> np.ndarray:
    """Per-seed paired metric of `mkey` vs `base` (seed-aligned records).

    Same seed ⇒ same forecasting window for both models, so this cancels the
    window-difficulty variance and isolates the model effect.
    """
    rm = res["records"].get(mkey, [])
    rb = res["records"].get(base, [])
    out = []
    for a, b in zip(rm, rb):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            continue
        va, vb = float(va), float(vb)
        if not (np.isfinite(va) and np.isfinite(vb)):
            continue
        if mode == "diff":
            out.append(va - vb)
        else:                                            # log2 ratio
            if va <= 0 or vb <= 0:
                continue
            if key == "swd" and (va >= SWD_DIVERGE or vb >= SWD_DIVERGE):
                continue
            out.append(float(np.log2(va / vb)))
    return np.array(out, dtype=float)


def _winrate(vals: np.ndarray, better: str) -> float:
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals > 0) if better == "greater" else np.mean(vals < 0))


def _paired_plot(
    all_results:   list[dict],
    models_cfg:    dict,
    active_models: list[str],
    metrics:       list,
    out_path:      str,
    title_prefix:  str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plot.")
        return

    # Baseline = the ESN model (by type), so the paired view is "SAS vs ESN".
    esn_models = [m for m in active_models if models_cfg.get(m, {}).get("type") == "esn"]
    base = esn_models[0] if esn_models else active_models[0]
    rel  = [m for m in active_models if m != base]
    if not rel:
        print("[WARN] need ≥1 non-baseline model for the paired plot — skipping.")
        return

    n_m    = len(metrics)
    n_d    = len(all_results)
    labels = [models_cfg[m]["label"] for m in rel]
    colors = _PALETTE[:len(rel)]
    pos    = np.arange(len(rel))
    rng    = np.random.default_rng(0)

    fig, axes = plt.subplots(n_m, n_d, figsize=(2.7 * n_d, 2.6 * n_m), squeeze=False)
    fig.suptitle(f"{title_prefix} — paired vs {models_cfg[base]['label']}  "
                 f"(per-seed; %% above = win-rate; dashed 0 = baseline)",
                 fontsize=9, fontweight="bold", y=1.005)

    for mi, (key, ylabel, mode, better) in enumerate(metrics):
        for di, res in enumerate(all_results):
            ax   = axes[mi, di]
            data = [_paired_vals(res, key, m, base, mode) for m in rel]

            ax.axhline(0.0, ls="--", lw=0.8, color="0.5", zorder=1)   # = baseline

            valid = [(i, d) for i, d in enumerate(data) if len(d) >= 3]
            if valid:
                vp = ax.violinplot([d for _, d in valid],
                                   positions=[pos[i] for i, _ in valid],
                                   widths=0.55, showmedians=False, showextrema=False)
                for pc, (i, _) in zip(vp["bodies"], valid):
                    pc.set_facecolor(colors[i]); pc.set_edgecolor("none"); pc.set_alpha(0.35)

            non_empty = [d if len(d) else np.array([np.nan]) for d in data]
            bp = ax.boxplot(non_empty, positions=pos, widths=0.22, patch_artist=True,
                            medianprops=dict(color="white", linewidth=1.8, zorder=5),
                            boxprops=dict(linewidth=0),
                            whiskerprops=dict(linewidth=0.9, color="#333333"),
                            capprops=dict(linewidth=0.9, color="#333333"),
                            flierprops=dict(marker=".", markersize=3, alpha=0.4,
                                            markeredgewidth=0), zorder=3)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.85); patch.set_linewidth(0)

            for i, (vals, color) in enumerate(zip(data, colors)):
                if len(vals) == 0:
                    continue
                ax.scatter(pos[i] + rng.uniform(-0.08, 0.08, len(vals)), vals,
                           s=14, color=color, alpha=0.65, zorder=6, linewidths=0)
                # win-rate above the violin
                wr = _winrate(vals, better)
                ax.text(pos[i], 0.99, f"{wr:.0%}", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=6.5, fontweight="bold",
                        color=color)

            ax.set_xticks(pos)
            ax.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=6, va="top")
            ax.tick_params(axis="x", length=0, pad=2)
            ax.tick_params(axis="y", labelsize=6.5)
            ax.spines[["right", "top"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_linewidth(0.6)
            ax.set_xlim(-0.6, len(rel) - 0.4)
            if di == 0:
                ax.set_ylabel(ylabel + f"\n(↓ better)" if better == "less"
                              else ylabel + f"\n(↑ better)", fontsize=7, labelpad=4)
            if mi == 0:
                ax.set_title(res["label"], fontsize=7.5, fontweight="bold", pad=5)

    patches = [mpatches.Patch(color=c, alpha=0.8, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=patches, loc="lower center", ncol=len(rel),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"  → plot: {out_path}")
    plt.close(fig)


def plot_results(all_results, models_cfg, active_models, out_path="benchmark.png"):
    """Forecast-quality paired plot (ΔVPT, log2 NRMSE/SWD)."""
    _paired_plot(all_results, models_cfg, active_models, _PAIRED, out_path,
                 "Forecast quality")


def plot_times(all_results, models_cfg, active_models, out_path="benchmark_times.png"):
    """Compute-time paired plot (log2 train / scan / infer)."""
    _paired_plot(all_results, models_cfg, active_models, _TIMES, out_path,
                 "Compute time")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SAS vs ESN benchmark")
    parser.add_argument("--config",   default="config.yaml",
                        help="YAML config path (default: config.yaml next to this script)")
    parser.add_argument("--dgps",    nargs="*", default=None,
                        help="DGP keys to run (default: all)")
    parser.add_argument("--models",  nargs="*", default=None,
                        help="Model keys to run (default: all)")
    parser.add_argument("--plot-out", default="benchmark.png", metavar="PATH",
                        help="Output path for the figure (default: benchmark.png)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the matplotlib figure")
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = _HERE / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bench_cfg     = cfg["benchmark"]
    dgps_cfg      = cfg["dgps"]
    models_cfg    = cfg["models"]

    # Benchmark-wide leaky rate for SAS models (default 0.25, matching the ESN's lr).
    global _BENCH_LEAK
    _BENCH_LEAK = float(bench_cfg.get("leak", 0.25))
    dgp_defaults  = {k: v for k, v in dgps_cfg.items() if not isinstance(v, dict)}
    all_dgp_keys  = [k for k, v in dgps_cfg.items() if isinstance(v, dict)]
    active_dgps   = args.dgps   or all_dgp_keys
    # Skip commented-out models (key present but body commented → YAML null).
    active_models = [m for m in (args.models or list(models_cfg.keys()))
                     if isinstance(models_cfg.get(m), dict)]

    print(f"saspy {saspy.__version__}  |  JAX: {jax.default_backend()}")
    print(f"DGPs   : {active_dgps}")
    print(f"Models : {active_models}")

    all_results = []
    for dgp_name in active_dgps:
        if dgp_name not in dgps_cfg:
            print(f"[WARN] DGP '{dgp_name}' not in config, skipping.")
            continue
        res = run_dgp(dgp_name, dgps_cfg[dgp_name], dgp_defaults,
                      models_cfg, active_models, bench_cfg)
        all_results.append(res)

    if all_results:
        print_results(all_results, models_cfg, active_models)
        if not args.no_plot:
            out = args.plot_out
            if not pathlib.Path(out).is_absolute():
                out = str(_HERE / out)
            plot_results(all_results, models_cfg, active_models, out)
            # second figure: timing (same path with a _times suffix)
            t_out = out[:-4] + "_times.png" if out.endswith(".png") else out + "_times.png"
            plot_times(all_results, models_cfg, active_models, t_out)


if __name__ == "__main__":
    main()
