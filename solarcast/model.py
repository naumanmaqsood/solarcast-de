"""Train, evaluate and apply the capacity-factor forecasting models.

Target is the capacity factor (generation / installed capacity), which keeps the
model valid while Germany's PV fleet keeps growing. Three quantile models give a
P10 / P50 / P90 band; the P50 is the point forecast.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from . import config
from .features import build_features, is_night

MODEL_PATH = config.MODEL_DIR / "quantile_models.joblib"
METRICS_PATH = config.MODEL_DIR / "metrics.json"


def _target(history: pd.DataFrame) -> pd.Series:
    return history["solar_mw"] / (history["capacity_gw"] * 1000)


def _fit(X: pd.DataFrame, y: pd.Series) -> dict:
    # Night hours are set to zero by rule, so the models only learn from daylight.
    # (Half the raw target is exactly zero, which collapses the lower quantile.)
    day = ~is_night(X)
    X, y = X[day], y[day]
    models = {}
    for q in config.QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        )
        models[q] = m.fit(X, y)
    return models


def predict_capacity_factor(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """P10/P50/P90 capacity factor, zero at night and with non-crossing quantiles."""
    raw = np.column_stack([models[q].predict(X) for q in config.QUANTILES])
    raw = np.sort(np.clip(raw, 0, None), axis=1)
    raw[is_night(X)] = 0.0
    return pd.DataFrame(raw, index=X.index, columns=["p10", "p50", "p90"])


def _scores(actual_gw: pd.Series, pred_gw: pd.Series, capacity_gw: pd.Series) -> dict:
    err = pred_gw - actual_gw
    return {
        "mae_gw": float(err.abs().mean()),
        "rmse_gw": float(np.sqrt((err ** 2).mean())),
        "nmae_pct_of_capacity": float((err.abs() / capacity_gw).mean() * 100),
        "bias_gw": float(err.mean()),
    }


def train_and_evaluate(history: pd.DataFrame) -> dict:
    """Fit on data before TEST_START, score on the held-out period, then refit on everything."""
    history = history.dropna(subset=["solar_mw", "capacity_gw"])
    X_all, y_all = build_features(history), _target(history)
    ok = X_all.notna().all(axis=1)
    X_all, y_all, history = X_all[ok], y_all[ok], history[ok]

    test = X_all.index >= pd.Timestamp(config.TEST_START, tz="UTC")
    X_tr, y_tr, X_te = X_all[~test], y_all[~test], X_all[test]
    cap_te = history.loc[test, "capacity_gw"]
    actual = history.loc[test, "solar_mw"] / 1000

    models = _fit(X_tr, y_tr)
    pred = predict_capacity_factor(models, X_te).mul(cap_te, axis=0)

    # Baselines: yesterday's value at the same hour, and a linear fit on mean irradiance.
    persistence = (history["solar_mw"] / 1000).shift(24, freq="h").reindex(X_te.index)
    lin_cols = ["shortwave_radiation_mean"]
    linear = LinearRegression().fit(X_tr[lin_cols], y_tr)
    linear_gw = pd.Series(np.clip(linear.predict(X_te[lin_cols]), 0, None), index=X_te.index) * cap_te

    valid = persistence.notna()
    day = X_te["sun_elevation"] > 0
    model_scores = _scores(actual[valid], pred["p50"][valid], cap_te[valid])
    pers_scores = _scores(actual[valid], persistence[valid], cap_te[valid])
    monthly = (pred["p50"] - actual).abs().groupby(X_te.index.strftime("%Y-%m")).mean()

    metrics = {
        "train_period": [str(X_tr.index[0].date()), str(X_tr.index[-1].date())],
        "test_period": [str(X_te.index[0].date()), str(X_te.index[-1].date())],
        "train_hours": int(len(X_tr)),
        "test_hours": int(len(X_te)),
        "model": model_scores,
        "model_daytime_only": _scores(actual[valid & day], pred["p50"][valid & day], cap_te[valid & day]),
        "baseline_persistence": pers_scores,
        "baseline_linear_ghi": _scores(actual[valid], linear_gw[valid], cap_te[valid]),
        "skill_vs_persistence_pct": float((1 - model_scores["mae_gw"] / pers_scores["mae_gw"]) * 100),
        "p10_p90_coverage_pct_daytime": float(
            ((actual[day] >= pred["p10"][day]) & (actual[day] <= pred["p90"][day])).mean() * 100),
        "monthly_mae_gw": {k: float(v) for k, v in monthly.items()},
        "mean_test_capacity_gw": float(cap_te.mean()),
    }

    # Ship a model trained on the full history.
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"models": _fit(X_all, y_all), "features": list(X_all.columns)}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # Held-out predictions, kept for inspection.
    pred.assign(actual=actual).round(3).to_csv(config.MODEL_DIR / "test_predictions.csv.gz")
    return metrics


def load_models() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `python run.py train` first.")
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())
