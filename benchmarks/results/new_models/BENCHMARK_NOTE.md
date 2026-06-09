# Benchmark: New Architectures — Full Results (10×10 windowed)

**Date:** 2026-06-04/05  
**Protocol:** 10 data seeds (windowed) × 10 model seeds = 100 evaluations per model  

---

## Full summary table (RMSE mean ± std, 100 evaluations)

| Model | Lorenz | MSO-8 | KS |
|---|---|---|---|
| **SAS-Diagonal (p=1,q=2)** | **0.00036 ± 0.00003** | 0.00181 ± 0.00025 | 0.01449 ± 0.01837 |
| SAS-Diag+RFF (p=0,bw=5) | 0.00038 ± 0.00002 | 0.00387 ± 0.00049 | 0.00740 ± 0.00432 |
| SAS-Diag-Q1 (prior) | 0.00459 ± 0.00021 | 0.00107 ± 0.00009 | 0.00407 ± 0.01170 |
| SAS-LRU+RFF | 0.00051 ± 0.00005 | 0.00775 ± 0.00253 | 0.02308 ± 0.01036 |
| **SAS-LRU-PolyQ1** (NEW) | 0.00513 ± 0.00042 | **0.00006 ± 0.00007** | **0.00722 ± 0.01485** |
| SAS-LRU-PolyQ2 (NEW) | 0.00074 ± 0.00008 | 0.00271 ± 0.00148 | 0.02622 ± 0.02221 |

---

## Key findings

### 1. LRU-PolyQ1: new MSO-8 SOTA (0.00006), competitive on KS

MSO-8: 18× improvement over prior best (SAS-Diag-Q1: 0.00107).
KS: 0.00722 — essentially tied with SAS-Diag+RFF (0.00740) and better than SAS-Diagonal.
Lorenz: 0.00513 — poor (14× worse than SAS-Diagonal).

### 2. LRU-PolyQ2: good Lorenz, bad everything else

Lorenz: 0.00074 — 7× better than PolyQ1, only 2× worse than SAS-Diagonal.
MSO-8: 0.00271 — 45× WORSE than PolyQ1. The quadratic term z² adds always-positive
DC bias that disrupts pure amplitude tracking for periodic signals.
KS: 0.02622 — poor.

**q=1 vs q=2 trade-off for LRU:**
- q=1 is optimal for periodic tasks (MSO-8) and KS
- q=2 improves chaotic prediction (Lorenz) at the cost of periodic performance
- No q_degree value is universal for LRU

### 3. Architecture-task matrix

| Task | Best | RMSE | 2nd best | RMSE |
|---|---|---|---|---|
| Lorenz | SAS-Diagonal | 0.00036 | SAS-Diag+RFF | 0.00038 |
| MSO-8 | **LRU-PolyQ1** | **0.00006** | SAS-Diagonal | 0.00181 |
| KS | SAS-Diag-Q1* | 0.00407* | SAS-Diag+RFF | 0.00740 |

*SAS-Diag-Q1 KS result from earlier run; LRU-PolyQ1 = 0.00722 ≈ SAS-Diag+RFF.
KS variance is very high — rankings may shift with more seeds.

### 4. SAS-Diagonal remains the best single architecture

Best Lorenz (0.00036), 2nd on MSO-8 (0.00181), OK on KS.
No new architecture beats SAS-Diagonal across all three tasks simultaneously.

### 5. KS evaluation is unreliable with current 5-channel setup

Standard deviation is often 50-100% of the mean for KS (e.g., SAS-Diagonal: 0.01449 ± 0.01837).
Exp 10 showed that 10 channels with sparse W_in (density=0) dramatically improves
single-window KS RMSE (poly-q1: 0.00304 → 0.00094 with 10 channels). However,
the benchmark uses dense W_in (density=1.0) with 5 channels — this setup may
be suboptimal and artificially noisy.

---

## Q-degree interpretation for LRU

| q_degree | MSO-8 | Lorenz | Interpretation |
|---|---|---|---|
| q=0 | — | — | No Q drive; degenerate |
| **q=1** | **0.00006** | 0.00513 | Linear: pure amplitude tracking. Optimal for periodic |
| q=2 | 0.00271 | 0.00074 | Quadratic: adds nonlinear feature. Better Lorenz, worse periodic |
| (RFF) | 0.00775 | 0.00051 | Kernel: bounded cosine. Best Lorenz h=1, worst periodic |

For Q basis selection: **choose q_degree based on expected signal type, not architecture preference.**
