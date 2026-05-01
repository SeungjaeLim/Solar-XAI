"""Persistence and Ridge baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def persistence_predict(
    df_with_time: pd.DataFrame,
    target_col: str = "AC_POWER",
    horizon: int = 1,
) -> np.ndarray:
    """Yesterday-same-hour persistence baseline.

    Predicts the target as the value from 24 hours ago (already shifted-by-1 in features pipeline,
    so we just look up `power_lag_24` if present, else fall back to `power_lag_1`).
    """
    if "power_lag_24" in df_with_time.columns:
        return df_with_time["power_lag_24"].to_numpy()
    return df_with_time["AC_POWER"].shift(24).fillna(0.0).to_numpy()


class RidgeBaseline:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)
