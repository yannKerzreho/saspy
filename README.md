# saspy

![Benchmark](benchmarks/benchmark.png)

SAS (Sate Affine Systems) reservoir computing for univariate time-series forecasting.

The reservoir state evolves as a polynomial recurrence

    s_t = P(z_t) ⊛ s_{t-1} + Q(z_t)

and is computed in `O(log T)` depth via a two-level parallel associative scan
in JAX. Forecasts are produced by per-horizon ridge regression on the state
vector.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from saspy import SASForecaster, DiagonalPoly

# Synthetic AR(1)
rng = np.random.default_rng(0)
y = np.zeros(1000)
for t in range(1, 1000):
    y[t] = 0.7 * y[t-1] + rng.normal(0, 1)

# Train on first 800 points
basis = DiagonalPoly(p_degree=1, q_degree=1)
model = SASForecaster(basis=basis, n_reservoir=100, washout=50)
model.fit(y[:800], horizons=[1, 5, 10])

# Streaming forecast over the remaining 200 points
# At each step: predict h=1 (forecast for y[t]), then ingest y[t].
preds = []
for t in range(800, 1000):
    preds.append(model.predict(1))
    model.update(y[t])

preds = np.asarray(preds)
truth = y[800:1000]
mse   = np.mean((preds - truth) ** 2)
print(f"h=1 MSE: {mse:.3f}  (mean-predictor baseline ~{np.var(truth):.3f})")
```

See `examples/etth1_example.ipynb` for a real-data walkthrough on ETTh1.

## API

* `SASForecaster(basis, n_reservoir, ...)` — fit / update / predict / transform
* `DiagonalPoly(p_degree, q_degree, ...)` — O(n) diagonal basis
* `LRUBlockPoly(p_degree, q_degree, ...)` — O(n) LRU-style rotation-block basis
* `BasePoly` — abstract base for custom bases

## Tests

```bash
pip install -e ".[dev]"
pytest
```
