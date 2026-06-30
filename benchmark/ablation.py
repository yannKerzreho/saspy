"""Reusable ablation harness for saspy experiments.

Builds (structure × P-feature × Q-feature) SAS forecasters, evaluates them with
the benchmark's autonomous rollout (NRMSE / VPT / SWD), prints grouped tables,
and saves raw per-seed results to JSON.  Experiments under experiments/ supply a
grid and call run_grid / print_tables / save_json.

All models run on [-1,1]-normalised data (scale_input=False, clip_output=True).
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import sys

import numpy as np

from saspy import (
    SASForecaster, SASModel,
    DiagonalP, DiagonalQ, SparseP, SparseQ, Cheb, Trig,
)
from benchmark.utils import load_dgp
from benchmark.main  import eval_sas, _extract_window


# ── DGP registry (chaotic / periodic / discrete spread) ──────────────────────

DGPS = {
    "mackey_glass": dict(loader="mackey_glass", channels=None,      lyap=None),
    "mso8":         dict(loader="mso8",         channels=None,      lyap=None),
    "lorenz":       dict(loader="lorenz",       channels=[0, 1, 2], lyap=37),
    "doublescroll": dict(loader="doublescroll", channels=[0, 1, 2], lyap=None),
    "logistic_map": dict(loader="logistic_map", channels=None,      lyap=None),
    "henon_map":    dict(loader="henon_map",    channels=[0, 1],    lyap=None),
    "rossler":      dict(loader="rossler",      channels=[0, 1, 2], lyap=None),
}

METRICS = [
    ("vpt",     "VPT  (mean ± std, higher better)"),
    ("nrmse_h", "NRMSE h=10  (mean ± std, lower better)"),
    ("swd",     "SWD  (mean ± std, lower better)"),
]


# ── labelling ────────────────────────────────────────────────────────────────

def fname(f) -> str:
    """Compact feature label, e.g. Cheb(2)→'C2', Trig(3)→'T3'."""
    return f"C{f.degree}" if isinstance(f, Cheb) else f"T{f.degree}"


def config_label(structure, fp, fq) -> str:
    return f"{structure[:4]} P:{fname(fp)} Q:{fname(fq)}"


# ── model builder ────────────────────────────────────────────────────────────

def build(structure, fp, fq, n, d, washout, chunk, seed) -> SASForecaster:
    if structure == "diagonal":
        model = SASModel(DiagonalP(n, feature=fp), DiagonalQ(n, feature=fq),
                         d=d, density=0.1)
    elif structure == "sparse":
        model = SASModel(SparseP(n, d, feature=fp), SparseQ(n, d, feature=fq))
    else:
        raise ValueError(f"Unknown structure: {structure!r}")
    return SASForecaster(model=model, washout=washout, chunk_size=chunk,
                         seed=seed, scale_input=False, clip_output=True, mode="autoreg")


# ── runner ───────────────────────────────────────────────────────────────────

def run_grid(grid, dgps, *, n, seeds, n_train, n_test, washout, chunk) -> dict:
    """grid = list of (structure, feature_P, feature_Q).  Returns
    records[dgp][config_idx] = list of per-seed metric dicts."""
    records = {dg: [[] for _ in grid] for dg in dgps}
    for dg in dgps:
        meta    = DGPS[dg]
        n_total = n_train + n_test + (seeds - 1) * n_test
        data    = load_dgp(meta["loader"], n_total, channels=meta["channels"])
        d       = data.shape[1]
        print(f"\n[{dg}] d={d}  data∈[{data.min():.2f},{data.max():.2f}]  "
              f"configs={len(grid)}  seeds={seeds}")
        for ci, (structure, fp, fq) in enumerate(grid):
            for seed in range(seeds):
                X_tr, X_te = _extract_window(data, seed, n_train, n_test,
                                             list(range(d)), n_test)
                fc  = build(structure, fp, fq, n, d, washout, chunk, seed)
                records[dg][ci].append(eval_sas(fc, X_tr, X_te, vpt_threshold=0.4))
            print(f"  · {config_label(structure, fp, fq)}  done")
    return records


# ── reporting ────────────────────────────────────────────────────────────────

def _agg(recs, key) -> tuple[float, float]:
    v = np.array([r[key] for r in recs], dtype=float)
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))


def print_tables(grid, dgps, records) -> None:
    labels = [config_label(*c) for c in grid]
    row_w  = max(len(l) for l in labels) + 2
    for key, title in METRICS:
        cells = {(ci, dg): "%.3f±%.3f" % _agg(records[dg][ci], key)
                 for ci in range(len(grid)) for dg in dgps}
        col_w  = max(max(len(c) for c in cells.values()), max(len(d) for d in dgps)) + 3
        total  = row_w + col_w * len(dgps)
        print(f"\n{'═'*total}\n  {title}\n{'═'*total}")
        print(f"{'config':<{row_w}}" + "".join(f"{d:>{col_w}}" for d in dgps))
        print("─"*total)
        for ci in range(len(grid)):
            line = f"{labels[ci]:<{row_w}}" + "".join(f"{cells[(ci, dg)]:>{col_w}}" for dg in dgps)
            print(line)
        print("═"*total)


def save_json(path, grid, dgps, records, meta) -> None:
    out = {
        "meta":    meta,
        "configs": [{"structure": s, "P": fname(p), "Q": fname(q)} for s, p, q in grid],
        "results": {
            dg: {str(ci): {k: [r[k] for r in records[dg][ci]] for k in ("vpt", "nrmse_h", "swd")}
                 for ci in range(len(grid))}
            for dg in dgps
        },
    }
    pathlib.Path(path).write_text(json.dumps(out, indent=2))
    print(f"\n  → json: {path}")


# ── tee: mirror stdout to a file ─────────────────────────────────────────────

class _Tee:
    def __init__(self, path):
        self.file   = open(path, "w")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
    def flush(self):
        self.stdout.flush()
        self.file.flush()


@contextlib.contextmanager
def tee(path):
    """Within the block, everything printed is also written to `path`."""
    t, old = _Tee(path), sys.stdout
    sys.stdout = t
    try:
        yield
    finally:
        sys.stdout = old
        t.file.close()
        print(f"  → log:  {path}")
