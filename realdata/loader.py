"""
Unified dataset loader for long-term time-series forecasting benchmarks.

Supported datasets
------------------
ETTh1, ETTh2   hourly   Electricity Transformer Temperature  (7 ch)  target: OT
ETTm1, ETTm2   15-min   Electricity Transformer Temperature  (7 ch)  target: OT
ECL            hourly   Electricity Consuming Load (321 ch)          target: MT_320
WTH            hourly   Weather                   (12 ch)            target: WetBulbCelsius

Splits  (Informer / NLinear / SpaceTime convention)
------
  ETTh  : 12 / 4 / 4   months  (hourly)   →  8640  / 2880  / 2880
  ETTm  : 12 / 4 / 4   months  (15-min)   → 34560  / 11520 / 11520
  ECL   : 15 / 3 / 4   months  (hourly)   → 10957  / 2192  / 2922
  WTH   : 28 / 10 / 10 months  (hourly)   → 20454  / 7305  / 7305

Default forecast horizons
-------------------------
  ETT : 96, 192, 336, 720          (NLinear / SpaceTime)
  ECL : 24, 48, 168, 336, 720, 960 (Informer)
  WTH : 24, 48, 168, 336, 720      (Informer)

Normalisation
-------------
Per-channel z-score computed on the TRAIN split only (matching Autoformer /
PatchTST / iTransformer).  Returned array is float32 in the normalised space.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DatasetConfig:
    filename      : str
    T_train       : int
    T_val         : int
    T_test        : int
    cols          : Optional[List[str]]   # None → all numeric columns
    target        : str                   # univariate (S) target column
    pred_lens     : List[int]             # default forecast horizons


# ------------------------------------------------------------------
# Splits computed from months × average hours per month (730.5 h/mo)
#   ETTh  : 12/4/4  mo  →  8640 / 2880 / 2880
#   ETTm  : same timing, ×4 resolution
#   ECL   : 15/3/4  mo  of a 3-year (36-mo) dataset
#             15 × 730.5 = 10957 ;  3 × 730.5 = 2192 ;  4 × 730.5 = 2922
#   WTH   : 28/10/10 mo of a 4-year (48-mo) dataset
#             28 × 730.5 = 20454 ; 10 × 730.5 = 7305 ; 10 × 730.5 = 7305
# ------------------------------------------------------------------

DATASETS: dict = {
    'ETTh1': DatasetConfig(
        'ETTh1.csv', 8640, 2880, 2880,
        ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'], 'OT',
        [96, 192, 336, 720],
    ),
    'ETTh2': DatasetConfig(
        'ETTh2.csv', 8640, 2880, 2880,
        ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'], 'OT',
        [96, 192, 336, 720],
    ),
    'ETTm1': DatasetConfig(
        'ETTm1.csv', 34560, 11520, 11520,
        ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'], 'OT',
        [96, 192, 336, 720],
    ),
    'ETTm2': DatasetConfig(
        'ETTm2.csv', 34560, 11520, 11520,
        ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'], 'OT',
        [96, 192, 336, 720],
    ),
    'ECL': DatasetConfig(
        'ECL.csv', 10957, 2192, 2922,
        None, 'MT_320',
        [24, 48, 168, 336, 720, 960],
    ),
    'WTH': DatasetConfig(
        'WTH.csv', 20454, 7305, 7305,
        None, 'WetBulbCelsius',
        [24, 48, 168, 336, 720],
    ),
}


def load_dataset(
    name     : str,
    data_dir : str = '.',
    features : str = 'S',    # 'S' = univariate | 'M' = multivariate (CI)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Load and normalise a dataset.

    Returns
    -------
    data_norm : (T_total, d)  float32, globally z-scored on train split
    mu        : (d,)
    sigma     : (d,)
    T_train, T_val, T_test
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")

    cfg  = DATASETS[name]
    path = os.path.join(data_dir, cfg.filename)
    df   = pd.read_csv(path, index_col=0, parse_dates=True)

    all_cols = cfg.cols if cfg.cols is not None else list(df.columns)
    cols     = [cfg.target] if features == 'S' else all_cols

    data    = df[cols].values.astype(np.float32)
    T_total = cfg.T_train + cfg.T_val + cfg.T_test
    assert len(data) >= T_total, (
        f"{name}: need >= {T_total} rows, found {len(data)}"
    )
    data = data[:T_total]

    mu    = data[:cfg.T_train].mean(axis=0)
    sigma = data[:cfg.T_train].std(axis=0)
    sigma[sigma == 0] = 1.0

    data_norm = (data - mu) / sigma
    return data_norm, mu, sigma, cfg.T_train, cfg.T_val, cfg.T_test
