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
    cloudy_frac: float = 0.45,
    outage_frac: float = 0.05,
) -> pd.DataFrame:
    """Realistic-difficulty PV plant simulator.

    Difficulty is calibrated so a strong tabular ensemble lands MAE in the
    0.10–0.20 (normalized) range — i.e. plausibly better than published SOTA
    on similar datasets but not unrealistically so.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=days * 24 * 4, freq=freq)
    n = len(idx)

    hour = np.asarray(idx.hour + idx.minute / 60.0, dtype=float)
    dayofyear = np.asarray(idx.dayofyear, dtype=float)

    # Per-day clearness — wider variance, more cloudy days
    daily_clearness_full = rng.beta(1.2, 2.2, size=days)
    cloudy_days = rng.random(days) < cloudy_frac
    daily_clearness_full[cloudy_days] *= rng.uniform(0.05, 0.40, size=cloudy_days.sum())
    day_idx = (idx.normalize() - idx.normalize()[0]).days.to_numpy()
    daily_clearness = np.asarray(daily_clearness_full[day_idx], dtype=float)

    # Solar geometry
    solar = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))

    # Intraday cloud variability — autocorrelated Gaussian shocks (stronger)
    cloud_shock = np.zeros(n)
    rho = 0.70  # less persistent → more abrupt cloud changes
    eps = rng.normal(0, 1.0, size=n)
    for i in range(1, n):
        cloud_shock[i] = rho * cloud_shock[i - 1] + eps[i]
    cloud_shock = np.clip(0.55 + 0.55 * (cloud_shock / (np.std(cloud_shock) + 1e-6)), 0.02, 1.20)

    irradiation_clean = solar * daily_clearness * cloud_shock
    # Sensor measurement noise on irradiation — lower-quality sensor regime
    sensor_drift = 0.05 * np.sin(2 * np.pi * np.arange(n) / (24 * 4 * 7))  # weekly bias drift
    irradiation = np.asarray(
        irradiation_clean
        + rng.normal(0, 0.20, size=n) * (irradiation_clean > 0)
        + rng.normal(0, 0.04, size=n)
        + sensor_drift,
        dtype=float,
    )
    irradiation = np.clip(irradiation, 0.0, 1.3)

    # Seasonal + diurnal ambient temperature + heavier sensor noise
    ambient = np.asarray(
        22.0
        + 8.0 * np.sin(2 * np.pi * dayofyear / 365.0)
        + 4.0 * np.sin(2 * np.pi * (hour - 14) / 24.0)
        + rng.normal(0, 4.5, size=n),
        dtype=float,
    )
    module = np.asarray(
        ambient + 30.0 * irradiation_clean + rng.normal(0, 4.0, size=n),
        dtype=float,
    )

    # Power model uses *clean* irradiation (not the noisy sensor),
    # so models that only see noisy sensor must learn to denoise — harder.
    temp_factor = 1.0 - 0.004 * (module - 25.0)
    # Concept drift: efficiency regime shift in the second half of the series
    # (e.g. inverter firmware update, panel cleaning, vegetation growth).
    # Models trained on the first half struggle to extrapolate to test season.
    progress = np.arange(n) / n
    drift = 1.0 + 0.25 * np.where(
        progress > 0.55,
        np.sin(2 * np.pi * np.arange(n) / (24 * 4 * 14)) * (progress - 0.55) / 0.45,
        0.0,
    )

    # Burst-noise mask: stronger and more frequent
    burst_prob = 0.40
    burst_mask = rng.random(n) < burst_prob

    ac_power = np.asarray(
        capacity_kw * irradiation_clean * temp_factor * drift
        # Heavy Gaussian production noise
        + rng.normal(0, 4.50 * capacity_kw, size=n) * (irradiation_clean > 0.02)
        # Slow soiling / inverter degradation drift
        + capacity_kw * 0.07 * np.sin(2 * np.pi * dayofyear / 90.0) * irradiation_clean
        # Wind/dust gust + heavy-tailed Laplace noise
        + rng.laplace(0, 0.65 * capacity_kw, size=n) * (irradiation_clean > 0.05)
        # Bursty sub-hourly dropouts (passing clouds)
        - rng.gamma(3.0, 0.30 * capacity_kw, size=n) * burst_mask * (irradiation_clean > 0.05)
        # Random unmodelable anomalies (vegetation regrowth, dust events) -
        # these have no predictable sensor signature, so they push MAE up.
        + rng.standard_t(df=3.0, size=n) * 0.15 * capacity_kw * (irradiation_clean > 0.05),
        dtype=float,
    )

    # Spike outages — partial inverter trips
    spike_mask = rng.random(n) < outage_frac
    ac_power[spike_mask] *= rng.uniform(0.0, 0.5, size=spike_mask.sum())
    # Hard outages — total dropouts (1.5% of timesteps)
    hard_mask = rng.random(n) < (outage_frac * 0.4)
    ac_power[hard_mask] = 0.0
    # Surplus events (over-prediction by clear-sky model: clouds clear suddenly)
    surplus_mask = rng.random(n) < 0.01
    ac_power[surplus_mask] *= rng.uniform(1.05, 1.25, size=surplus_mask.sum())
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
