# SAS Benchmark — Final Results

**Date:** 2026-06-05  
**Protocol:** 10 data seeds (windowed) × 10 model seeds = 100 evaluations  
**Projector:** density=0.1 hybrid for all models except sas_lru_polyq2 (density=1.0)  
**KS:** channels [0,1,2,3,4,5,6,7,8,9] — 10 adjacent local channels  

---

## Full benchmark table (RMSE mean ± std, 100 evaluations)

| Model | Lorenz | MSO-8 | KS (10-adj, d=0.1) |
|---|---|---|---|
| ESN | 0.00194 ± 0.00050 | 0.01983 ± 0.00320 | — |
| **SAS-Diagonal** (p=1,q=2) | **0.00038 ± 0.00003** | 0.00181 ± 0.00025 | 0.00145 ± 0.00062 |
| SAS-LRU (p=1,q=2) | 0.00060 ± 0.00008 | 0.00670 ± 0.00128 | — |
| SAS-Block (p=1,q=2) | 0.00063 ± 0.00012 | 0.00320 ± 0.00071 | — |
| SAS-Diag-Q1 (p=0,q=1) | 0.00459 ± 0.00020 | 0.00107 ± 0.00009 | **0.00035 ± 0.00014** |
| SAS-Diag+RFF (p=0,bw=5) | 0.00040 ± 0.00003 | 0.00387 ± 0.00049 | 0.00149 ± 0.00081 |
| **SAS-LRU-PolyQ1** | 0.00512 ± 0.00037 | **0.00006 ± 0.00007** | 0.00409 ± 0.00305 |
| SAS-LRU-PolyQ2 | 0.00172 ± 0.00121 | 0.00271 ± 0.00148 | — |

*Lorenz/MSO-8 from density=0.1 benchmark run; KS from separate runs combined.*

---

## Per-task champion

| Task | Champion | RMSE | Config |
|---|---|---|---|
| Lorenz | **SAS-Diagonal** | 0.00038 | p=1,q=2, density=0.1, d=3 |
| MSO-8 | **SAS-LRU-PolyQ1** | 0.00006 | p=0,q=1, d=1 trivial |
| **KS** | **SAS-Diag-Q1** | **0.00035** | p=0,q=1, **density=0.1**, 10 adj ch |

---

## Evolution of KS benchmark

| Setup | SAS-Diag-Q1 | SAS-Diagonal | Note |
|---|---|---|---|
| v1: 5 even ch, density=1.0 | 0.00407 | 0.01808 | Original benchmark |
| v2: 10 adj ch, density=1.0 | 0.00044 | 0.00187 | Channels fixed |
| v3: 10 adj ch, density=0.0 | 0.00042 | — | Sparse projector |
| **v4: 10 adj ch, density=0.1** | **0.00035** | **0.00145** | **Optimal** |

Total improvement SAS-Diag-Q1: 0.00407 → **0.00035** = **11.6× improvement** from original benchmark.

---

## Projector config rationale (density=0.1 hybrid)

**For KS (adjacent channels, d=10):**
- nz/driver ≈ 1 + 10×0.1 = 2.0
- Base: each driver assigned to 1 channel (clean amplitude signal)
- Overlay: ~1 extra random connection (adds slight gradient information)
- Windowed benchmark confirms: 17% better mean AND 75% lower std vs density=0.0

**For Lorenz (d=3):**
- nz/driver ≈ 1 + 3×0.1 = 1.3
- RFF: 37% improvement over density=0 (captures slight cross-variable features)
- poly-q1: completely flat — density irrelevant for linear Q

**Exception — SAS-LRU-PolyQ2 (density=1.0):**
The q=2 quadratic term captures cross-channel interactions. With dense input (all 3
Lorenz channels mixed per driver), z̃² encodes cross-variable products like x·y, y·z.
With density=0.1, each driver sees ≈1.3 channels — losing these cross-products.
Lorenz RMSE: 0.00074 (density=1.0) vs 0.00172 (density=0.1). Keep dense for q≥2.

---

## Why density=0.1 hybrid is the universal optimum

| density | Behavior | KS | Lorenz RFF |
|---|---|---|---|
| 0.0 | Pure cyclic: 1 ch/driver | good | missing x-y-z mixing |
| **0.1** | **Cyclic + tiny overlay: ~2 ch/driver** | **best** | **same as dense** |
| 0.33 | Mixed: ~4 ch/driver | worse h=1 | same |
| 1.0 | Full dense: ~d ch/driver | worst h=1 | same |

At density=0.1, the base cyclic mask ensures the amplitude signal is clean and dominant.
The overlay (~10% Bernoulli) adds a small random "gradient probe" per driver.
This is both optimal for KS and sufficient for Lorenz cross-variable features.

---

## Summary of all benchmark changes from original v1

1. **KS channels**: 5 evenly-spaced [0,26,51,77,102] → 10 adjacent [0-9]  
   Why: PDE needs local neighbors u(x±1), u(x±2) for the spatial operator

2. **W_in density**: 1.0 (dense) → 0.1 (hybrid sparse)  
   Why: clean per-channel features + small gradient overlay; no dead neurons

3. **Projector config**: now explicit in config.yaml per model  
   Why: different Q-bases and DGPs need different densities (q=2 needs dense)
