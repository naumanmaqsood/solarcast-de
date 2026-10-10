"""Central configuration: paths, locations, date ranges."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

# Weather sampling points. Kassel first, the rest spread over
# the regions that hold most of Germany's installed PV capacity.
LOCATIONS = {
    "kassel": (51.31, 9.48),
    "munich": (48.14, 11.58),
    "nuremberg": (49.45, 11.08),
    "stuttgart": (48.78, 9.18),
    "freiburg": (47.99, 7.85),
    "cologne": (50.94, 6.96),
    "hannover": (52.37, 9.73),
    "leipzig": (51.34, 12.37),
    "cottbus": (51.76, 14.33),
    "rostock": (54.09, 12.14),
}

# Reference point for the sun-position features (roughly the centre of Germany).
REF_LAT, REF_LON = 51.0, 10.0

WEATHER_VARS = [
    "shortwave_radiation",        # GHI, W/m2, mean of the preceding hour
    "direct_normal_irradiance",   # DNI, W/m2, mean of the preceding hour
    "diffuse_radiation",          # DHI, W/m2, mean of the preceding hour
    "cloud_cover",                # %
    "temperature_2m",             # deg C
    "wind_speed_10m",             # km/h
    "snow_depth",                 # m
]
RADIATION_VARS = WEATHER_VARS[:3]

# History used for training and evaluation (UTC dates, inclusive).
HISTORY_START = "2023-01-01"
HISTORY_END = "2026-09-30"
# Everything from this date on is held out and never seen during training.
TEST_START = "2025-10-01"

QUANTILES = (0.1, 0.5, 0.9)

ENERGY_CHARTS_API = "https://api.energy-charts.info"
OPEN_METEO_HISTORICAL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
BIDDING_ZONE = "DE-LU"
