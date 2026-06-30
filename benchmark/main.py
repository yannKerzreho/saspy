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
import pathlib
import sys
import warnings

import numpy as np
import yaml
from tqdm import tqdm

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # repo root → saspy importable
sys.path.insert(0, str(_HERE))          # benchmarks/ → esn.py, utils.py importable

import jax
import jax.numpy as jnp
import reservoirpy as rpy
import reservoirpy.datasets as rpy_datasets

import saspy
from saspy import SASForecaster, SASModel, InputProjector
from saspy.basis import (
    DiagonalPoly, LRUBlockPoly, BlockLinearPoly,
    RandomFourierBasis, SparsePolyBasis,
)
from esn import JaxESN, JaxESNForecaster
from utils import autonomous_nrmse, compute_vpt, sliced_wasserstein

try:
    rpy.verbosity(0)
except AttributeError:
    pass

# SWD values above this threshold indicate a diverged trajectory and are excluded
SWD_DIVERGE = 1e3

_BASIS_CLASSES = {
    "DiagonalPoly":       DiagonalPoly,
    "LRUBlockPoly":       LRUBlockPoly,
    "BlockLinearPoly":    BlockLinearPoly,
    "RandomFourierBasis": RandomFourierBasis,
    "SparsePolyBasis":    SparsePolyBasis,
}


# ── data ──────────────────────────────────────────────────────────────────────

def _load_data(loader_name: str, n_total: int, loader_kwargs: dict | None = None) -> np.ndarray:
    kw = {k: v for k, v in (loader_kwargs or {}).items() if v is not None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = getattr(rpy_datasets, loader_name)(n_total, **kw)
    if isinstance(result, tuple):
        result = result[1]
    raw = np.asarray(result, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, None]
    return raw


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

def _make_basis(cfg: dict, d: int = 1):
    params = dict(cfg.get("params", {}))
    if cfg["type"] == "SparsePolyBasis" and params.get("n_drivers") is None:
        params["n_drivers"] = d
    return _BASIS_CLASSES[cfg["type"]](**params)


def _n_drivers(cfg: dict, d: int = 1) -> int:
    p = cfg.get("params", {})
    t = cfg["type"]
    if t in ("DiagonalPoly", "RandomFourierBasis"):
        return p["n"]
    if t == "SparsePolyBasis":
        return d if p.get("n_drivers") is None else p["n_drivers"]
    return p["n_blocks"]


def make_sas(model_cfg: dict, washout: int, chunk_size: int, seed: int, d: int = 1) -> SASForecaster:
    if "basis" in model_cfg:
        basis_p = _make_basis(model_cfg["basis"], d)
        basis_q = _make_basis(model_cfg["basis"], d)
        n_dr    = _n_drivers(model_cfg["basis"], d)
    else:
        basis_p = _make_basis(model_cfg["basis_p"], d)
        basis_q = _make_basis(model_cfg["basis_q"], d)
        n_dr    = _n_drivers(model_cfg["basis_p"], d)

    pcfg  = model_cfg.get("projector", {})
    strat = str(pcfg.get("strategy", "hybrid"))
    proj  = (InputProjector.identity(d) if strat == "identity"
             else InputProjector(d=d, n_drivers=n_dr,
                                 density=float(pcfg.get("density", 1.0)),
                                 mixing_strategy=strat))

    model = SASModel(projector=proj, basis_p=basis_p, basis_q=basis_q)
    return SASForecaster(model=model, washout=washout, chunk_size=chunk_size,
                         seed=seed, scale_input=True, mode="autoreg")


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
    auto_clip_z: float | None = None,
) -> dict:
    D, T_te = X_tr.shape[1], len(X_te)
    fc.fit(X_tr, horizons=[1], context=X_tr)
    jax.block_until_ready(fc._s_last)

    mu, sigma = fc._ctx_mu, fc._ctx_sigma
    preds = np.empty((T_te, D), dtype=np.float64)

    for t in range(T_te):
        if not np.isfinite(fc._s_last).all():
            fc._s_last = np.zeros_like(fc._s_last)
        pred = np.atleast_1d(fc.predict(1))
        if auto_clip_z is not None:
            pz   = np.nan_to_num((pred - mu) / sigma, nan=0.0,
                                 posinf=auto_clip_z, neginf=-auto_clip_z)
            pred = np.clip(pz, -auto_clip_z, auto_clip_z) * sigma + mu
        preds[t] = pred
        fc.update(pred)

    nrmse = autonomous_nrmse(preds, X_te)
    return dict(
        vpt     = compute_vpt(nrmse, vpt_threshold),
        swd     = sliced_wasserstein(preds, X_te),
        nrmse_h = float(np.mean(nrmse[:10])),
    )


def eval_esn(
    fc: JaxESNForecaster,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    vpt_threshold: float = 0.4,
) -> dict:
    D, T_te = X_tr.shape[1], len(X_te)
    fc.fit(X_tr, horizons=[1])
    jax.block_until_ready(np.asarray(fc._esn._state))

    preds = np.empty((T_te, D), dtype=np.float64)
    for t in range(T_te):
        pred    = np.atleast_1d(fc.predict(1))
        preds[t] = pred
        fc.update(pred)

    nrmse = autonomous_nrmse(preds, X_te)
    return dict(
        vpt     = compute_vpt(nrmse, vpt_threshold),
        swd     = sliced_wasserstein(preds, X_te),
        nrmse_h = float(np.mean(nrmse[:10])),
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
    auto_clip_z   = bench_cfg.get("auto_clip_z")
    if auto_clip_z is not None:
        auto_clip_z = float(auto_clip_z)

    n_seed  = dgp_cfg.get("n_seed",  bench_cfg.get("n_seed", 10))
    n_train = dgp_cfg.get("n_train", dgp_defaults.get("n_train", 5000))
    n_test  = dgp_cfg.get("n_test",  dgp_defaults.get("n_test",  2000))
    stride  = dgp_cfg.get("stride",  dgp_defaults.get("stride",  n_test))
    n_total = n_train + n_test + (n_seed - 1) * stride

    data    = _load_data(dgp_cfg["loader"], n_total, dgp_cfg.get("args"))
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
            nan_rec = dict(vpt=0, swd=float("nan"), nrmse_h=float("nan"))
            for mkey in active_models:
                records[mkey].append(nan_rec.copy())
            continue

        for mkey in active_models:
            mcfg = models_cfg[mkey]
            if mcfg["type"] == "esn":
                rec = eval_esn(make_esn(mcfg, washout, seed), X_tr, X_te, vpt_threshold)
            else:
                rec = eval_sas(make_sas(mcfg, washout, chunk_size, seed, d),
                               X_tr, X_te, vpt_threshold, auto_clip_z)
            records[mkey].append(rec)

    return dict(
        label               = dgp_cfg["label"],
        lyapunov_time_steps = dgp_cfg.get("lyapunov_time_steps"),
        records             = records,
    )


# ── results table ─────────────────────────────────────────────────────────────

def print_results(all_results: list[dict], models_cfg: dict, active_models: list[str]) -> None:
    labels = [models_cfg[m]["label"] for m in active_models]
    col_w  = max(12, max(len(l) for l in labels) + 2)
    dgp_w  = max(20, max(len(r["label"]) for r in all_results) + 2)
    total_w = dgp_w + col_w * len(active_models)

    def _header() -> str:
        return f"{'DGP':<{dgp_w}}" + "".join(f"{l:>{col_w}}" for l in labels)

    def _row(res: dict, key: str) -> str:
        line = f"{res['label']:<{dgp_w}}"
        lyt  = res.get("lyapunov_time_steps")
        for mkey in active_models:
            recs = res["records"].get(mkey, [])
            vals = np.array([r[key] for r in recs], dtype=float)
            if key == "swd":
                vals = vals[np.isfinite(vals) & (vals < SWD_DIVERGE)]
            else:
                vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                cell = "n/a"
            elif key == "vpt" and lyt:
                cell = f"{vals.mean()/lyt:.1f}±{vals.std()/lyt:.1f}TL"
            else:
                cell = f"{vals.mean():.3f}±{vals.std():.3f}"
            line += f"{cell:>{col_w}}"
        return line

    metrics = [
        ("nrmse_h", "NRMSE h=10  (mean ± std, autonomous rollout)"),
        ("vpt",     "VPT  (mean ± std, ε=0.4)"),
        ("swd",     f"SWD  (mean ± std, diverged seeds excluded, cap={SWD_DIVERGE:.0e})"),
    ]

    for key, title in metrics:
        print(f"\n{'═' * total_w}\n  {title}\n{'═' * total_w}")
        print(_header())
        print("─" * total_w)
        for res in all_results:
            print(_row(res, key))
    print("═" * total_w)


# ── results plot ──────────────────────────────────────────────────────────────

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

_METRICS = [
    ("nrmse_h", "−NRMSE h=10",    True),   # negated: higher = better
    ("vpt",     "VPT",             False),
    ("swd",     "−SWD (filtered)", True),   # negated: higher = better
]


def _collect(res: dict, key: str, active_models: list[str]) -> list[np.ndarray]:
    out = []
    for mkey in active_models:
        vals = np.array([r[key] for r in res["records"].get(mkey, [])], dtype=float)
        if key == "swd":
            vals = vals[np.isfinite(vals) & (vals < SWD_DIVERGE)]
        else:
            vals = vals[np.isfinite(vals)]
        out.append(vals)
    return out


def plot_results(
    all_results:   list[dict],
    models_cfg:    dict,
    active_models: list[str],
    out_path:      str = "benchmark.png",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plot.")
        return

    n_m     = len(_METRICS)
    n_d     = len(all_results)
    labels  = [models_cfg[m]["label"] for m in active_models]
    colors  = _PALETTE[:len(active_models)]
    pos     = np.arange(len(active_models))

    fig, axes = plt.subplots(
        n_m, n_d,
        figsize=(2.6 * n_d, 2.5 * n_m),
        squeeze=False,
    )
    fig.suptitle("SAS vs ESN — autonomous forecast benchmark",
                 fontsize=9, fontweight="bold", y=1.01)

    rng = np.random.default_rng(0)

    for mi, (key, ylabel, negate) in enumerate(_METRICS):
        for di, res in enumerate(all_results):
            ax   = axes[mi, di]
            data = _collect(res, key, active_models)
            if negate:
                data = [-d for d in data]

            # ── violin ───────────────────────────────────────────────────
            valid = [(i, d) for i, d in enumerate(data) if len(d) >= 3]
            if valid:
                vp = ax.violinplot(
                    [d for _, d in valid],
                    positions=[pos[i] for i, _ in valid],
                    widths=0.55,
                    showmedians=False,
                    showextrema=False,
                )
                for pc, (i, _) in zip(vp["bodies"], valid):
                    pc.set_facecolor(colors[i])
                    pc.set_edgecolor("none")
                    pc.set_alpha(0.35)

            # ── box ───────────────────────────────────────────────────────
            non_empty = [d if len(d) else np.array([np.nan]) for d in data]
            bp = ax.boxplot(
                non_empty,
                positions=pos,
                widths=0.22,
                patch_artist=True,
                medianprops=dict(color="white", linewidth=1.8, zorder=5),
                boxprops=dict(linewidth=0),
                whiskerprops=dict(linewidth=0.9, color="#333333"),
                capprops=dict(linewidth=0.9, color="#333333"),
                flierprops=dict(marker=".", markersize=3, alpha=0.4,
                                markeredgewidth=0),
                zorder=3,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
                patch.set_linewidth(0)

            # ── jittered strip ────────────────────────────────────────────
            for i, (vals, color) in enumerate(zip(data, colors)):
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.08, 0.08, len(vals))
                ax.scatter(pos[i] + jitter, vals,
                           s=14, color=color, alpha=0.65, zorder=6,
                           linewidths=0)

            # ── axes styling ──────────────────────────────────────────────
            ax.set_xticks(pos)
            ax.set_xticklabels(
                [l.replace("-", "\n") for l in labels],
                fontsize=6, va="top",
            )
            ax.tick_params(axis="x", length=0, pad=2)
            ax.tick_params(axis="y", labelsize=6.5)
            ax.spines[["right", "top"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_linewidth(0.6)
            ax.set_xlim(-0.6, len(active_models) - 0.4)

            if di == 0:
                ax.set_ylabel(ylabel, fontsize=7.5, labelpad=4)
            if mi == 0:
                title = res["label"]
                lyt   = res.get("lyapunov_time_steps")
                if lyt and key == "vpt":
                    title += f"\n(1 TL = {lyt} steps)"
                ax.set_title(title, fontsize=7.5, fontweight="bold", pad=5)

    # ── legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=c, alpha=0.8, label=l)
        for c, l in zip(colors, labels)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=len(active_models),
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"  → plot: {out_path}")
    plt.close(fig)


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
    dgp_defaults  = {k: v for k, v in dgps_cfg.items() if not isinstance(v, dict)}
    all_dgp_keys  = [k for k, v in dgps_cfg.items() if isinstance(v, dict)]
    active_dgps   = args.dgps   or all_dgp_keys
    active_models = args.models or list(models_cfg.keys())

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


if __name__ == "__main__":
    main()
