"""Free-MAE post-processing: clip and night-zero."""
from __future__ import annotations

import numpy as np
import pandas as pd


def postprocess(
    preds: np.ndarray,
    timestamps: pd.DatetimeIndex | pd.Series | None = None,
    irradiation: np.ndarray | pd.Series | None = None,
    clip_max: float | None = None,
    daylight_hours: tuple[int, int] = (5, 20),
    irrad_threshold: float = 0.01,
) -> np.ndarray:
    """Clip predictions to a non-negative range and force-zero at night.

    A prediction is night-zeroed if either:
      - timestamp is outside the daylight window, OR
      - irradiation feature is below threshold (when provided).
    """
    p = np.asarray(preds, dtype=float).copy()
    p = np.clip(p, 0.0, None)
    if clip_max is not None:
        p = np.clip(p, None, clip_max)

    night_mask = np.zeros_like(p, dtype=bool)
    if timestamps is not None:
        ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
        hours = ts.dt.hour.to_numpy()
        night_mask |= (hours < daylight_hours[0]) | (hours >= daylight_hours[1])
    if irradiation is not None:
        irr = np.asarray(irradiation, dtype=float)
        night_mask |= irr < irrad_threshold

    p[night_mask] = 0.0
    return p
