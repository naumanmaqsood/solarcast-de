# SolarCast DE

Day-ahead forecasting of Germany's solar power generation from numerical weather
forecasts, with an uncertainty band, a decision layer for flexible loads, and an
LLM assistant that answers questions by calling tools over the forecast.

Built as an independent portfolio project on renewable energy, weather-driven
forecasting, and agent-style AI applications. It uses only public data.

## What it does

1. **Data** – pulls real, public data, no API keys needed:
   - measured solar feed-in, installed PV capacity and day-ahead prices for Germany
     from [energy-charts.info](https://energy-charts.info) (Fraunhofer ISE),
   - archived and live weather forecasts for ten locations (Kassel plus the main PV
     regions) from [Open-Meteo](https://open-meteo.com).
2. **Model** – gradient-boosted quantile regression (scikit-learn) predicts the fleet
   capacity factor at P10 / P50 / P90 from irradiance, cloud cover, temperature, snow
   and sun position. Predicting the capacity factor rather than GW keeps the model
   valid while installed capacity keeps growing.
3. **Decision support** – per day: expected energy, peak, and the sunniest and
   cheapest 3-hour windows for flexible loads (EV charging, heat pumps, batteries).
4. **Dashboard** – one self-contained HTML file (`output/dashboard.html`), no server.
   With `python run.py serve` the same page runs live: the forecast and the measured
   feed-in refresh every 10 minutes and the page updates itself, showing how the
   forecast is holding up against the day's measured hours so far.
5. **AI assistant** – Claude with four tools (`get_outlook`, `get_hourly_forecast`,
   `find_best_window`, `get_model_accuracy`). Every number in an answer comes from a
   tool result, not from the model's memory.

## Results

Trained on 2023-01-01 to 2025-09-30, evaluated on a held-out year
(2025-10-01 to 2026-09-30, 8,756 hours, mean installed capacity 112 GW):

| Method                               | MAE (GW) | RMSE (GW) | Bias (GW) |
|--------------------------------------|---------:|----------:|----------:|
| Gradient boosting (this model)       |     0.76 |      1.57 |     +0.30 |
| Linear regression on mean irradiance |     1.17 |      2.25 |     +0.43 |
| Persistence (same hour yesterday)    |     1.96 |      4.18 |     -0.02 |

- MAE is 0.67 % of installed capacity (1.49 GW / 1.32 % over daylight hours only).
- 61 % lower MAE than persistence, 35 % lower than the linear baseline.
- The P10–P90 band contained 75 % of daylight hours (nominal: 80 %), so it is
  slightly too narrow.

### Known limitations

- **Positive bias.** The model over-forecasts in most held-out months (up to about
  +1.2 GW on average over daylight hours). Likely contributors: curtailment at
  negative prices, which the model does not see, and reported installed capacity
  running ahead of what is actually feeding in. Adding the day-ahead price as a
  feature is the obvious next experiment.
- **Optimistic weather inputs.** Open-Meteo's archive stitches together the first
  hours of each model run, so training and evaluation use short-lead forecasts. A
  true 24–36 h-ahead forecast will be somewhat less accurate than the table shows.
- **Live outcome is delayed and flattering.** energy-charts publishes measured feed-in
  one to two hours late, and the live comparison re-forecasts past hours from the
  newest weather run, so it is not a score of the forecast as issued the day before.
- **National scope.** The target is Germany's total feed-in, not a single plant.
  Site-level forecasting needs inverter data, which is not public.
- The assistant has not been exercised end to end in this repository's test run
  (no API key was available); its tools are tested directly.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python run.py train             # download history (cached in data/), train, evaluate
python run.py all               # live forecast + dashboard, opens in the browser
python run.py serve             # live dashboard on http://127.0.0.1:8000, Ctrl+C to stop
python run.py site              # static site in site/, as published to GitHub Pages
python -m pytest                # unit tests
```

Assistant (needs an Anthropic API key in `ANTHROPIC_API_KEY`):

```bash
python run.py ask "When is the cheapest 4-hour window to charge an EV tomorrow?"
python run.py chat
```

## Public link

`.github/workflows/pages.yml` rebuilds the forecast every 15 minutes and publishes
`site/` to GitHub Pages; the open page polls `data.json` and updates itself. To set it
up, push this repository to GitHub (public, or Pages needs a paid plan) and choose
**Settings → Pages → Source: GitHub Actions**. The published build installs the pinned
versions in `requirements-site.txt`; update the scikit-learn pin there whenever the
model is retrained with a different version.

## Layout

```
run.py                        command line entry point
solarcast/config.py           locations, date ranges, endpoints
solarcast/data.py             API clients, time alignment, cached history
solarcast/features.py         feature engineering, sun position
solarcast/model.py            training, evaluation, quantile prediction
solarcast/forecast.py         live forecast, best-window search
solarcast/dashboard.py        renders dashboard_template.html with the data
solarcast/server.py           local server that keeps the dashboard live
solarcast/agent.py            Claude tool-use assistant
tests/test_core.py            unit tests
```

## Ideas for next steps

- Day-ahead price and holiday features to capture curtailment.
- Calibrate the quantiles (conformal prediction) to hit the 80 % target.
- Train on true day-ahead forecast runs (Open-Meteo previous-runs API).
- Site-level variant using inverter data, feeding a battery dispatch optimiser.
- Add document retrieval (grid codes, tariff sheets) to the assistant.

Data licences: energy-charts.info and Open-Meteo, both CC BY 4.0.
