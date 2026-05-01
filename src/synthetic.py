"""Synthetic PV plant generator — fallback when Kaggle data is unavailable.

Outputs a long-format DataFrame mirroring the anikannal Plant 1 schema:
    DATE_TIME, PLANT_ID, AC_POWER, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_plant(
    start: str = "2025-05-01",
    days: int = 730,
    freq: str = "15min",
    plant_id: int = 1,
    capacity_kw: float = 1500.0,
    seed: int = 42,
    cloudy_frac: float = 0.05,
    outage_frac: float = 0.005,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=days * 24 * 4, freq=freq)
    n = len(idx)

    hour = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    dayofyear = np.asarray(idx.dayofyear, dtype=float)

    # Per-day clearness (sunny-skewed Beta)
    daily_clearness_full = rng.beta(2.0, 1.2, size=days)
    cloudy_days = rng.random(days) < cloudy_frac
    daily_clearness_full[cloudy_days] *= 0.4
    day_idx = (idx.normalize() - idx.normalize()[0]).days.to_numpy()
    daily_clearness = np.asarray(daily_clearness_full[day_idx], dtype=float)

    # Solar angle proxy: positive only between 6 and 18, peak at 12
    solar = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
    irradiation = np.asarray(solar * daily_clearness, dtype=float)  # 0..1 scale

    # Seasonal + diurnal ambient temperature
    ambient = np.asarray(
        22.0
        + 8.0 * np.sin(2 * np.pi * dayofyear / 365.0)
        + 4.0 * np.sin(2 * np.pi * (hour - 14) / 24.0)
        + rng.normal(0, 1.0, size=n),
        dtype=float,
    )
    module = np.asarray(ambient + 30.0 * irradiation + rng.normal(0, 0.5, size=n), dtype=float)

    # Power model: capacity * irradiation * temp_factor + noise
    temp_factor = 1.0 - 0.004 * (module - 25.0)
    ac_power = np.asarray(
        capacity_kw * irradiation * temp_factor + rng.normal(0, 0.02 * capacity_kw, size=n),
        dtype=float,
    )

    # Random outages
    outage_mask = rng.random(n) < outage_frac
    ac_power[outage_mask] = 0.0
    ac_power = np.clip(ac_power, 0.0, capacity_kw)

    # Force night zero
    night_mask = (hour < 5) | (hour >= 20)
    ac_power[night_mask] = 0.0
    irradiation[night_mask] = 0.0

    return pd.DataFrame(
        {
            "DATE_TIME": idx,
            "PLANT_ID": plant_id,
            "AC_POWER": ac_power,
            "AMBIENT_TEMPERATURE": ambient,
            "MODULE_TEMPERATURE": module,
            "IRRADIATION": irradiation,
        }
    )


def make_synthetic_dataset(
    out_path: str,
    days: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    p1 = make_synthetic_plant(plant_id=1, days=days, seed=seed, capacity_kw=1500.0)
    p2 = make_synthetic_plant(plant_id=2, days=days, seed=seed + 1, capacity_kw=1300.0)
    df = pd.concat([p1, p2], ignore_index=True).sort_values(["DATE_TIME", "PLANT_ID"])
    df.to_parquet(out_path, index=False)
    return df
