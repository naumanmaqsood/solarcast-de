import numpy as np
import pandas as pd

from solarcast import config
from solarcast.features import build_features, is_night, solar_elevation
from solarcast.forecast import best_window, outcome


def test_solar_elevation_matches_known_values():
    # Kassel: solar noon on the June solstice is ~62 deg, at midnight the sun is below the horizon.
    lat, lon = config.LOCATIONS["kassel"]
    idx = pd.DatetimeIndex(["2025-06-21 11:00", "2025-06-21 23:00", "2025-12-21 11:00"], tz="UTC")
    noon_summer, midnight, noon_winter = solar_elevation(idx, lat, lon)
    assert abs(noon_summer - 62.1) < 1.0
    assert midnight < -10
    assert abs(noon_winter - 15.3) < 1.0


def _weather(index, ghi):
    cols = {}
    for var in config.WEATHER_VARS:
        for loc in config.LOCATIONS:
            cols[f"{var}__{loc}"] = ghi if var == "shortwave_radiation" else 0.0
    return pd.DataFrame(cols, index=index)


def test_features_are_complete_and_flag_night():
    idx = pd.date_range("2025-06-21", periods=24, freq="h", tz="UTC")
    ghi = np.where((idx.hour >= 4) & (idx.hour <= 18), 400.0, 0.0)
    X = build_features(_weather(idx, ghi))
    assert not X.isna().any().any()
    night = is_night(X)
    assert night[0] and not night[11]


def test_best_window_finds_cheapest_and_sunniest():
    idx = pd.date_range("2025-06-21", periods=24, freq="h", tz="Europe/Berlin")
    price = pd.Series(100.0, index=idx)
    price.iloc[12:15] = [10, -5, 20]
    start, mean = best_window(price, 3, lowest=True)
    assert start.hour == 12 and abs(mean - 25 / 3) < 1e-9
    start, _ = best_window(price, 2, lowest=False)
    assert price[start] == 100.0


def test_best_window_ignores_gaps():
    idx = pd.date_range("2025-06-21", periods=6, freq="h", tz="UTC")
    s = pd.Series([1.0, 1.0, np.nan, 1.0, 5.0, 5.0], index=idx)
    start, mean = best_window(s, 2, lowest=True)
    assert start == idx[0] and mean == 1.0  # the 1.0 at idx[1] and idx[3] are not adjacent
    assert best_window(s, 4) is None


def test_outcome_scores_latest_day_with_measured_daylight():
    idx = pd.date_range("2025-06-20", periods=48, freq="h", tz="Europe/Berlin")
    df = pd.DataFrame({"p10": 8.0, "p50": 10.0, "p90": 12.0, "actual": 10.0}, index=idx)
    df["day"] = (idx.hour >= 6) & (idx.hour < 20)
    df.loc[idx[24 + 6], "actual"] = 9.0     # inside the band, forecast 1 GW too high
    df.loc[idx[24 + 7], "actual"] = 13.0    # above the band, forecast 3 GW too low
    df.loc[idx[24 + 8:], "actual"] = np.nan  # not published yet
    o = outcome(df)
    assert o["date"] == "2025-06-21" and o["hours"] == 2 and o["hours_in_band"] == 1
    assert o["mae_gw"] == 2.0 and o["bias_gw"] == -1.0
    assert o["last_hour"]["start"] == "07:00" and o["last_hour"]["actual_gw"] == 13.0
    assert outcome(df.assign(actual=np.nan)) is None
