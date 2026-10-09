"""Data access: energy-charts.info (generation, capacity, prices) and Open-Meteo (weather).

Both APIs are free and need no key. Everything is returned on an hourly UTC index
where the row labelled ``h`` describes the interval ``[h, h + 1h)``.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from . import config

_session = requests.Session()
_session.headers["User-Agent"] = "solarcast-de/0.1 (portfolio project)"


def _get(url: str, params: dict, retries: int = 5) -> object:
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=120)
            if r.status_code == 429:  # rate limited: wait for the next minute window
                time.sleep(65)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"rate limit not lifted after {retries} attempts: {url}")


def _chunks(start: str, end: str, freq: str):
    """Yield (start, end) date strings covering [start, end] in ``freq`` pieces."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    edges = list(pd.date_range(start_ts, end_ts, freq=freq))
    if not edges or edges[0] != start_ts:
        edges.insert(0, start_ts)
    for i, lo in enumerate(edges):
        hi = edges[i + 1] - pd.Timedelta(days=1) if i + 1 < len(edges) else end_ts
        yield lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")


def _hourly(unix_seconds, values) -> pd.Series:
    s = pd.Series(values, index=pd.to_datetime(unix_seconds, unit="s", utc=True), dtype="float64")
    s = s[~s.index.duplicated()]
    return s.resample("1h").mean()


# --- energy-charts.info -------------------------------------------------------

def fetch_solar_quarter_hourly(start: str, end: str) -> pd.Series:
    """Actual German solar net generation in MW at the published 15-minute resolution."""
    parts = []
    for lo, hi in _chunks(start, end, "QS"):
        d = _get(f"{config.ENERGY_CHARTS_API}/public_power", {"country": "de", "start": lo, "end": hi})
        solar = next(p for p in d["production_types"] if p["name"] == "Solar")
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
        parts.append(pd.Series(solar["data"], index=idx, dtype="float64"))
    s = pd.concat(parts)
    return s[~s.index.duplicated()].sort_index().rename("solar_mw")


def fetch_solar_generation(start: str, end: str) -> pd.Series:
    """Actual German solar net generation in MW, hourly mean."""
    return fetch_solar_quarter_hourly(start, end).resample("1h").mean()


def fetch_installed_capacity() -> pd.Series:
    """Installed solar AC capacity in GW, one value per month (month-start UTC index)."""
    d = _get(
        f"{config.ENERGY_CHARTS_API}/installed_power",
        {"country": "de", "time_step": "monthly", "installation_decommission": "false"},
    )
    solar = next(p for p in d["production_types"] if p["name"] == "Solar AC")
    idx = pd.to_datetime(d["time"], format="%m.%Y", utc=True)
    return pd.Series(solar["data"], index=idx, dtype="float64").dropna().rename("capacity_gw")


def fetch_prices(start: str, end: str) -> pd.Series:
    """Day-ahead price for the DE-LU bidding zone in EUR/MWh, hourly mean."""
    parts = []
    for lo, hi in _chunks(start, end, "YS"):
        d = _get(f"{config.ENERGY_CHARTS_API}/price", {"bzn": config.BIDDING_ZONE, "start": lo, "end": hi})
        if d.get("unix_seconds"):
            parts.append(_hourly(d["unix_seconds"], d["price"]))
    if not parts:
        return pd.Series(dtype="float64", name="price_eur_mwh")
    s = pd.concat(parts)
    return s[~s.index.duplicated()].sort_index().rename("price_eur_mwh")


def capacity_for(index: pd.DatetimeIndex, capacity: pd.Series) -> pd.Series:
    """Look up the installed capacity valid at each timestamp (last known month)."""
    return capacity.reindex(index.union(capacity.index)).ffill().reindex(index)


# --- Open-Meteo ---------------------------------------------------------------

def _weather_frame(payload) -> pd.DataFrame:
    """Turn a multi-location Open-Meteo response into one wide frame ``var__location``."""
    if isinstance(payload, dict):
        payload = [payload]
    cols = {}
    for name, loc in zip(config.LOCATIONS, payload):
        hourly = loc["hourly"]
        idx = pd.to_datetime(hourly["time"], utc=True)
        for var in config.WEATHER_VARS:
            cols[f"{var}__{name}"] = pd.Series(hourly[var], index=idx, dtype="float64")
    df = pd.DataFrame(cols)
    # Open-Meteo labels radiation with the END of its averaging hour; move it to the
    # start so it lines up with the generation interval [h, h + 1h).
    rad = [c for c in df.columns if c.split("__")[0] in config.RADIATION_VARS]
    df[rad] = df[rad].shift(-1)
    return df.iloc[:-1]


def _weather_params() -> dict:
    lats, lons = zip(*config.LOCATIONS.values())
    return {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "hourly": ",".join(config.WEATHER_VARS),
        "timezone": "UTC",
    }


def fetch_weather_history(start: str, end: str) -> pd.DataFrame:
    """Archived numerical-weather-prediction output for all locations (not observations)."""
    parts = []
    for lo, hi in _chunks(start, end, "6MS"):
        params = _weather_params() | {"start_date": lo, "end_date": hi}
        parts.append(_weather_frame(_get(config.OPEN_METEO_HISTORICAL, params)))
        time.sleep(20)  # stay under the free tier's per-minute weight limit
    df = pd.concat(parts)
    return df[~df.index.duplicated()].sort_index()


def fetch_weather_forecast(past_days: int = 2, forecast_days: int = 3) -> pd.DataFrame:
    """Live forecast for all locations, including a little history for context."""
    params = _weather_params() | {"past_days": past_days, "forecast_days": forecast_days}
    return _weather_frame(_get(config.OPEN_METEO_FORECAST, params))


# --- cached training history --------------------------------------------------

def load_history(refresh: bool = False) -> pd.DataFrame:
    """Weather features, solar generation, capacity and price on one hourly index."""
    path = config.DATA_DIR / "history.csv.gz"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    config.DATA_DIR.mkdir(exist_ok=True)
    print("Downloading solar generation ...")
    solar = fetch_solar_generation(config.HISTORY_START, config.HISTORY_END)
    print("Downloading installed capacity and prices ...")
    capacity = fetch_installed_capacity()
    prices = fetch_prices(config.HISTORY_START, config.HISTORY_END)
    print("Downloading archived weather forecasts (takes a few minutes) ...")
    weather = fetch_weather_history(config.HISTORY_START, config.HISTORY_END)

    df = weather.join(solar, how="inner").join(prices, how="left")
    df["capacity_gw"] = capacity_for(df.index, capacity)
    df = df.loc[config.HISTORY_START:config.HISTORY_END]
    df.to_csv(path)
    return df
