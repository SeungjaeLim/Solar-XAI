"""MAE-first metric helpers, all zero-safe."""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps_ratio: float = 0.05) -> float:
    """MAPE with a denominator floor to avoid blow-up at near-zero power.

    `eps_ratio` * max(y_true) is used as the minimum denominator. Documented in REPORT.md.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom_floor = eps_ratio * float(np.max(np.abs(y_true)) + 1e-9)
    denom = np.maximum(np.abs(y_true), denom_floor)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
