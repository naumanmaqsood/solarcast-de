"""Live forecast for the coming days, plus the decision support built on top of it."""
from __future__ import annotations

import json

import pandas as pd

from . import config, data
from .features import build_features, is_night
from .model import load_models, predict_capacity_factor

FORECAST_PATH = config.OUTPUT_DIR / "forecast.json"
LOCAL_TZ = "Europe/Berlin"


def best_window(values: pd.Series, hours: int, lowest: bool = True) -> tuple[pd.Timestamp, float] | None:
    """Start and mean of the best contiguous ``hours``-long window in an hourly series."""
    values = values.dropna()
    if len(values) < hours:
        return None
    rolling = values.rolling(hours).mean().dropna()
    # Only keep windows whose hours are truly consecutive.
    span = pd.Series(values.index, index=values.index).diff(hours - 1) if hours > 1 else None
    if span is not None:
        rolling = rolling[span.reindex(rolling.index) == pd.Timedelta(hours=hours - 1)]
    if rolling.empty:
        return None
    end = rolling.idxmin() if lowest else rolling.idxmax()
    return end - pd.Timedelta(hours=hours - 1), float(rolling[end])


def outcome(local: pd.DataFrame) -> dict | None:
    """Forecast against measurement for the latest local day that has measured daylight hours.

    ``local`` needs the columns p10, p50, p90, actual and a boolean ``day``.
    """
    g = local[local["day"] & local["actual"].notna()]
    if g.empty:
        return None
    g = g[g.index.date == g.index[-1].date()]
    err = g["p50"] - g["actual"]
    last = g.iloc[-1]
    return {
        "date": str(g.index[-1].date()),
        "hours": int(len(g)),
        "mae_gw": round(float(err.abs().mean()), 2),
        "bias_gw": round(float(err.mean()), 2),
        "hours_in_band": int(((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).sum()),
        "energy_gwh": {"actual": round(float(g["actual"].sum()), 1), "p50": round(float(g["p50"].sum()), 1)},
        "last_hour": {
            "start": g.index[-1].strftime("%H:%M"),
            "end": (g.index[-1] + pd.Timedelta(hours=1)).strftime("%H:%M"),
            "actual_gw": round(float(last["actual"]), 2),
            **{k: round(float(last[k]), 2) for k in ("p10", "p50", "p90")},
        },
    }


def make_forecast() -> dict:
    bundle = load_models()
    weather = data.fetch_weather_forecast(past_days=2, forecast_days=3)
    X = build_features(weather)[bundle["features"]]
    capacity = data.fetch_installed_capacity()
    cap = data.capacity_for(X.index, capacity)
    pred = predict_capacity_factor(bundle["models"], X).mul(cap, axis=0)  # GW

    start = (X.index[0] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (X.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    readings = data.fetch_solar_quarter_hourly(start, end).dropna() / 1000  # GW
    # Only hours with all four 15-minute readings published count as measured.
    per_hour = readings.resample("1h")
    actual = per_hour.mean()[per_hour.count() == 4].reindex(X.index)
    prices = data.fetch_prices(start, end).reindex(X.index)

    now = pd.Timestamp.now(tz="UTC").floor("h")

    frame = pred.assign(actual=actual, price=prices, ghi=X["shortwave_radiation_mean"],
                        cloud=X["cloud_cover_mean"], day=~is_night(X))
    local = frame.tz_convert(LOCAL_TZ)

    days = []
    for day, g in local.groupby(local.index.date):
        if len(g) < 20 or g.index[-1] < now:
            continue  # partial or fully past days
        entry = {
            "date": str(day),
            "energy_gwh": {k: round(float(g[k].sum()), 1) for k in ("p10", "p50", "p90")},
            "peak_gw": round(float(g["p50"].max()), 1),
            "peak_time": g["p50"].idxmax().strftime("%H:%M"),
            "mean_cloud_cover_pct": round(float(g["cloud"].mean())),
            "prices_published": bool(g["price"].notna().sum() >= 20),
        }
        sunny = best_window(g["p50"], 3, lowest=False)
        if sunny:
            entry["sunniest_3h"] = {"start": sunny[0].strftime("%H:%M"), "mean_solar_gw": round(sunny[1], 1)}
        cheap = best_window(g["price"], 3, lowest=True)
        if cheap and entry["prices_published"]:
            entry["cheapest_3h"] = {"start": cheap[0].strftime("%H:%M"), "mean_price_eur_mwh": round(cheap[1], 1)}
        days.append(entry)

    def col(name, digits):
        return [None if pd.isna(v) else round(float(v), digits) for v in frame[name]]

    result = {
        "generated_at": pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "now_utc": now.isoformat(),
        "capacity_gw": round(float(cap.iloc[-1]), 1),
        "timezone": LOCAL_TZ,
        "time_utc": [t.isoformat() for t in frame.index],
        "p10": col("p10", 2), "p50": col("p50", 2), "p90": col("p90", 2),
        "actual": col("actual", 2), "price": col("price", 1),
        "cloud": col("cloud", 0),
        "days": days,
        "latest_reading": None if readings.empty else {
            "time_utc": readings.index[-1].isoformat(), "gw": round(float(readings.iloc[-1]), 2)},
        "outcome": outcome(local),
    }
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    FORECAST_PATH.write_text(json.dumps(result, indent=1))
    return result


def load_forecast() -> dict:
    if not FORECAST_PATH.exists():
        raise FileNotFoundError("No forecast found. Run `python run.py forecast` first.")
    return json.loads(FORECAST_PATH.read_text())


def hourly_frame(forecast: dict) -> pd.DataFrame:
    """The saved forecast as a DataFrame indexed by local time."""
    idx = pd.to_datetime(forecast["time_utc"], utc=True).tz_convert(forecast["timezone"])
    cols = {k: forecast[k] for k in ("p10", "p50", "p90", "actual", "price", "cloud")}
    return pd.DataFrame(cols, index=idx).astype("float64")
