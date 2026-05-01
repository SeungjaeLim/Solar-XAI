"""Aggressive feature engineering for PV power forecasting.

All lag/rolling features use only past data (shifted) to prevent leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# anikannal Plant 1 is in central India; rough coords from public discussions.
# When using synthetic data, the geometry features are still informative even with
# a placeholder location because what matters is the *relative* hour/day pattern.
DEFAULT_LAT = 23.5
DEFAULT_LON = 78.5


def solar_geometry(
    timestamps: pd.DatetimeIndex,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
) -> pd.DataFrame:
    """NOAA-style solar position. Returns zenith (rad), cos_zenith, clear_sky_norm."""
    t = pd.to_datetime(timestamps)
    doy = t.dayofyear.to_numpy()
    hour = (t.hour + t.minute / 60.0).to_numpy()

    # Solar declination (radians)
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)

    # Equation of time (minutes) — Spencer approximation
    b = 2 * np.pi * (doy - 1) / 365.0
    eot = 229.18 * (
        0.000075
        + 0.001868 * np.cos(b)
        - 0.032077 * np.sin(b)
        - 0.014615 * np.cos(2 * b)
        - 0.04089 * np.sin(2 * b)
    )

    # Solar time (in hours). Treat the timestamp as local clock-time for the site;
    # for the anikannal dataset that's IST (UTC+5:30); rough approximation is fine.
    # We use a single time-zone offset implied by the longitude.
    tz_offset = longitude / 15.0
    solar_time = hour + eot / 60.0 + (longitude / 15.0 - tz_offset)
    hour_angle = np.deg2rad(15.0 * (solar_time - 12.0))

    lat_rad = np.deg2rad(latitude)
    cos_zen = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    zenith = np.arccos(cos_zen)
    cos_zen_pos = np.clip(cos_zen, 0.0, 1.0)

    # Air-mass (Kasten 1989) — only valid when sun is up
    am = np.where(
        cos_zen_pos > 1e-3,
        1.0 / (cos_zen_pos + 0.50572 * (96.07995 - np.rad2deg(zenith)).clip(min=1e-3) ** -1.6364),
        0.0,
    )
    am = np.clip(am, 0.0, 40.0)

    # Clear-sky normalized irradiance (0..~1 scale)
    clear_sky = cos_zen_pos * 0.7 ** (am ** 0.678)
    clear_sky = np.clip(clear_sky, 0.0, None)

    return pd.DataFrame(
        {
            "cos_zenith": cos_zen_pos,
            "zenith_deg": np.rad2deg(zenith),
            "air_mass": am,
            "clear_sky": clear_sky,
        },
        index=timestamps,
    ).reset_index(drop=True)


def add_time_features(df: pd.DataFrame, ts_col: str = "DATE_TIME") -> pd.DataFrame:
    out = df.copy()
    t = pd.to_datetime(out[ts_col])
    out["hour"] = t.dt.hour
    out["dayofyear"] = t.dt.dayofyear
    out["month"] = t.dt.month
    out["weekday"] = t.dt.weekday
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["sin_doy"] = np.sin(2 * np.pi * out["dayofyear"] / 365.0)
    out["cos_doy"] = np.cos(2 * np.pi * out["dayofyear"] / 365.0)
    return out


def add_solar_features(
    df: pd.DataFrame,
    ts_col: str = "DATE_TIME",
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
) -> pd.DataFrame:
    geom = solar_geometry(pd.to_datetime(df[ts_col]).to_numpy(), latitude, longitude)
    return pd.concat([df.reset_index(drop=True), geom.reset_index(drop=True)], axis=1)


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_delta"] = out["MODULE_TEMPERATURE"] - out["AMBIENT_TEMPERATURE"]
    out["irrad_x_temp"] = out["IRRADIATION"] * out["AMBIENT_TEMPERATURE"]
    out["irrad_sq"] = out["IRRADIATION"] ** 2
    out["irrad_x_cos_zenith"] = out["IRRADIATION"] * out.get("cos_zenith", 1.0)
    out["d_irrad_1h"] = out["IRRADIATION"].diff().fillna(0.0)
    out["d_module_temp_1h"] = out["MODULE_TEMPERATURE"].diff().fillna(0.0)

    # Cloud index — irradiation residual against clear-sky
    if "clear_sky" in out.columns:
        denom = out["clear_sky"].clip(lower=1e-3)
        out["cloud_index"] = (out["IRRADIATION"] / denom).clip(0.0, 5.0)
        out["cloud_index"] = out["cloud_index"].rolling(3, min_periods=1).mean()
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "AC_POWER",
    lags: tuple[int, ...] = (1, 2, 3, 6, 24, 48, 168),
    rolls: tuple[int, ...] = (3, 6, 24),
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"power_lag_{lag}"] = out[target_col].shift(lag)
    for lag in (1, 3, 24):
        out[f"irrad_lag_{lag}"] = out["IRRADIATION"].shift(lag)
    for w in rolls:
        shifted = out[target_col].shift(1)
        out[f"power_roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        out[f"power_roll_std_{w}"] = shifted.rolling(w, min_periods=1).std().fillna(0.0)
        out[f"power_roll_max_{w}"] = shifted.rolling(w, min_periods=1).max()

    # Same-hour aggregation across past 3 days
    out["power_same_hour_mean_3d"] = (
        out[target_col].shift(24).rolling(window=72, min_periods=1).apply(
            lambda x: np.nanmean(x[::24]) if len(x) > 0 else np.nan, raw=True
        )
    )
    out["yesterday_residual"] = out[target_col].shift(24) - out[target_col].shift(24).rolling(
        24, min_periods=1
    ).mean()
    return out


def add_calibration_features(
    df: pd.DataFrame,
    capacity_kw: float = 1500.0,
) -> pd.DataFrame:
    out = df.copy()
    irr_lag = out["IRRADIATION"].shift(1).clip(lower=1e-3)
    pow_lag = out["AC_POWER"].shift(1).clip(lower=0.0)
    out["power_per_irrad_lag1"] = (pow_lag / irr_lag).clip(0.0, 10000.0)

    if "clear_sky" in out.columns:
        out["expected_power"] = (
            capacity_kw
            * out["clear_sky"]
            * (1.0 - 0.004 * (out["MODULE_TEMPERATURE"] - 25.0))
        ).clip(lower=0.0)
    return out


def build_features(
    df: pd.DataFrame,
    target_col: str = "AC_POWER",
    capacity_kw: float = 1500.0,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
    horizon: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply the full feature pipeline. Returns (df_features, feature_columns)."""
    out = df.sort_values("DATE_TIME").reset_index(drop=True).copy()
    out = add_time_features(out)
    out = add_solar_features(out, latitude=latitude, longitude=longitude)
    out = add_weather_interactions(out)
    out = add_lag_features(out, target_col=target_col)
    out = add_calibration_features(out, capacity_kw=capacity_kw)

    # Target: power at t + horizon
    out["target"] = out[target_col].shift(-horizon)

    # Drop rows with any NaN in target/features (early rows have lag NaNs)
    feature_cols = [
        c
        for c in out.columns
        if c
        not in {
            "DATE_TIME",
            "PLANT_ID",
            "AC_POWER",
            "DC_POWER",
            "TOTAL_YIELD",
            "DAILY_YIELD",
            "target",
        }
    ]
    out = out.dropna(subset=["target"] + feature_cols).reset_index(drop=True)
    return out, feature_cols
