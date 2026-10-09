"""Feature engineering shared by training and live forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def solar_elevation(index: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Sun elevation in degrees at the middle of each hour starting at ``index`` (UTC)."""
    t = index + pd.Timedelta(minutes=30)
    doy = t.dayofyear.to_numpy()
    hour = (t.hour + t.minute / 60).to_numpy()
    g = 2 * np.pi * (doy - 1 + (hour - 12) / 24) / 365
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    solar_time = hour * 60 + eqtime + 4 * lon  # minutes
    hour_angle = np.radians(solar_time / 4 - 180)
    lat_r = np.radians(lat)
    sin_el = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))


def build_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Model inputs from the wide ``var__location`` weather frame.

    Uses only information available from a weather forecast, so the same function
    serves training (archived forecasts) and live prediction.
    """
    X = pd.DataFrame(index=weather.index)
    for var in config.WEATHER_VARS:
        cols = [f"{var}__{loc}" for loc in config.LOCATIONS]
        X[f"{var}_mean"] = weather[cols].mean(axis=1)
        if var in ("shortwave_radiation", "cloud_cover"):
            X[f"{var}_std"] = weather[cols].std(axis=1)
    # Per-location irradiance lets the model learn where the capacity actually sits.
    for loc in config.LOCATIONS:
        X[f"ghi_{loc}"] = weather[f"shortwave_radiation__{loc}"]
    X["snow_share"] = (weather[[f"snow_depth__{loc}" for loc in config.LOCATIONS]] > 0.02).mean(axis=1)

    elevation = solar_elevation(weather.index, config.REF_LAT, config.REF_LON)
    X["sun_elevation"] = elevation
    X["sun_up"] = np.clip(np.sin(np.radians(elevation)), 0, None)
    doy = weather.index.dayofyear.to_numpy()
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["hour_utc"] = weather.index.hour
    return X


def is_night(X: pd.DataFrame) -> np.ndarray:
    """Hours with the sun well below the horizon everywhere in Germany."""
    return (X["sun_elevation"] < -6).to_numpy() & (X["shortwave_radiation_mean"].fillna(0) <= 0).to_numpy()
