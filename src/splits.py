"""Time-ordered splits for forecasting (no shuffling)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def time_ordered_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitIndices:
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = np.arange(0, n_train)
    val = np.arange(n_train, n_train + n_val)
    test = np.arange(n_train + n_val, n)
    return SplitIndices(train=train, val=val, test=test)


def expanding_window_folds(
    train_indices: np.ndarray,
    n_folds: int = 5,
    min_train_frac: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV inside the train portion. Returns list of (train_idx, val_idx).

    Each fold: train on [0, k*step + base), validate on next step-sized block.
    """
    n = len(train_indices)
    base = int(n * min_train_frac)
    remaining = n - base
    step = remaining // n_folds
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        train_end = base + k * step
        val_end = train_end + step if k < n_folds - 1 else n
        tr = train_indices[:train_end]
        va = train_indices[train_end:val_end]
        if len(va) == 0:
            continue
        folds.append((tr, va))
    return folds
