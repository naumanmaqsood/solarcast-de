"""Energy assistant: Claude answers questions by calling tools over the forecast.

The model never sees raw data dumps and never does arithmetic on its own memory:
every number in an answer comes from one of the tools below, which read the saved
forecast and the model's held-out evaluation.
"""
from __future__ import annotations

import json

import anthropic
import pandas as pd
from anthropic import beta_tool

from .forecast import best_window, hourly_frame, load_forecast
from .model import load_metrics

MODEL = "claude-opus-5-5"

SYSTEM = """You are the energy assistant of SolarCast DE, a day-ahead solar power \
forecast for Germany. Operators, energy managers and curious colleagues ask you when \
solar generation will be high, when electricity is cheap, and how far the forecast can \
be trusted.

Ground every number in a tool result; if the tools do not cover something, say so \
rather than estimating. Forecast values are national totals in GW with a P10-P90 \
uncertainty band - mention the band when the decision depends on it. Times are local \
German time. Day-ahead prices for tomorrow are only published around 13:00, so they \
can be missing. Keep answers short and practical, and lead with the recommendation."""


def _day(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    day = frame[frame.index.strftime("%Y-%m-%d") == date]
    if day.empty:
        available = sorted(set(frame.index.strftime("%Y-%m-%d")))
        raise ValueError(f"No data for {date}. Available dates: {', '.join(available)}")
    return day


@beta_tool
def get_outlook() -> str:
    """Get the daily outlook for every forecast day: expected solar energy (GWh) with
    P10/P90 range, peak power and time, cloud cover, and the best 3-hour windows.
    Call this first to learn which dates are available and what "today" is.
    """
    f = load_forecast()
    return json.dumps({"generated_at": f["generated_at"], "installed_capacity_gw": f["capacity_gw"],
                       "days": f["days"]})


@beta_tool
def get_hourly_forecast(date: str) -> str:
    """Get the hour-by-hour forecast for one local date.

    Args:
        date: Local German date as YYYY-MM-DD.
    """
    day = _day(hourly_frame(load_forecast()), date)
    rows = [
        {"hour": t.strftime("%H:%M"), "solar_gw_p10": r.p10, "solar_gw_p50": r.p50, "solar_gw_p90": r.p90,
         "measured_gw": None if pd.isna(r.actual) else r.actual,
         "price_eur_mwh": None if pd.isna(r.price) else r.price, "cloud_cover_pct": r.cloud}
        for t, r in day.iterrows()
    ]
    return json.dumps(rows)


@beta_tool
def find_best_window(date: str, duration_hours: int, objective: str = "cheapest",
                     earliest_hour: int = 0, latest_hour: int = 24) -> str:
    """Find the best contiguous time window on a date for running a flexible load
    (EV charging, heat pump, battery charging, industrial batch process).

    Args:
        date: Local German date as YYYY-MM-DD.
        duration_hours: Length of the window in whole hours (1-12).
        objective: "cheapest" for the lowest average day-ahead price, or "sunniest" for the highest average solar generation.
        earliest_hour: The window may not start before this local hour (0-23).
        latest_hour: The window must end by this local hour (1-24).
    """
    if objective not in ("cheapest", "sunniest"):
        raise ValueError('objective must be "cheapest" or "sunniest"')
    if not 1 <= duration_hours <= 12:
        raise ValueError("duration_hours must be between 1 and 12")
    day = _day(hourly_frame(load_forecast()), date)
    day = day[(day.index.hour >= earliest_hour) & (day.index.hour < latest_hour)]
    series = day["price"] if objective == "cheapest" else day["p50"]
    found = best_window(series, duration_hours, lowest=objective == "cheapest")
    if found is None:
        reason = "day-ahead prices are not published yet" if objective == "cheapest" else "not enough hours"
        return json.dumps({"found": False, "reason": reason})
    start, _ = found
    window = day.loc[start:start + pd.Timedelta(hours=duration_hours - 1)]
    return json.dumps({
        "found": True, "start": start.strftime("%H:%M"),
        "end": (start + pd.Timedelta(hours=duration_hours)).strftime("%H:%M"),
        "mean_solar_gw_p50": round(float(window["p50"].mean()), 2),
        "mean_solar_gw_p10": round(float(window["p10"].mean()), 2),
        "mean_price_eur_mwh": None if window["price"].isna().any() else round(float(window["price"].mean()), 1),
        "day_mean_price_eur_mwh": None if day["price"].isna().all() else round(float(day["price"].mean()), 1),
    })


@beta_tool
def get_model_accuracy() -> str:
    """Get how accurate the forecast model was on a full held-out year: MAE/RMSE in GW,
    comparison against the persistence and linear baselines, P10-P90 coverage, and MAE
    per month. Use this when asked how reliable the forecast is.
    """
    return json.dumps(load_metrics())


TOOLS = [get_outlook, get_hourly_forecast, find_best_window, get_model_accuracy]


def ask(question: str, history: list | None = None) -> str:
    """Answer one question; ``history`` (mutated in place) carries a multi-turn chat."""
    client = anthropic.Anthropic()
    messages = history if history is not None else []
    messages.append({"role": "user", "content": question})

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        tools=TOOLS,
        messages=messages,
        output_config={"effort": "medium"},
        # If a safety classifier declines a request, let the API retry on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    final = None
    for message in runner:
        final = message
        messages.append({"role": "assistant", "content": message.content})
        tool_response = runner.generate_tool_call_response()
        if tool_response is not None:
            messages.append(tool_response)

    if final is None or final.stop_reason == "refusal":
        return "The assistant declined to answer this request."
    return "".join(block.text for block in final.content if block.type == "text")


def chat() -> None:
    """Interactive terminal chat."""
    print("SolarCast assistant - ask about the solar forecast (empty line to quit).")
    history: list = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print(f"\nAssistant: {ask(question, history)}")
