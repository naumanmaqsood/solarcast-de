"""Render the forecast and model evaluation into one self-contained HTML file."""
from __future__ import annotations

import json
from pathlib import Path

from . import config
from .forecast import load_forecast
from .model import load_metrics

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
DASHBOARD_PATH = config.OUTPUT_DIR / "dashboard.html"
SITE_DIR = config.ROOT / "site"


def render(payload: dict) -> str:
    data = json.dumps(payload).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", data)


def live_info(url: str, poll_seconds: int, refresh_minutes: float, error: str | None = None) -> dict:
    """Tells the page where to poll for new data and when to call the data stale."""
    return {"url": url, "poll_seconds": poll_seconds, "max_age_minutes": 4 * refresh_minutes, "error": error}


def build_site(refresh_minutes: float) -> Path:
    """Static site for hosting: the page plus the ``data.json`` it polls. Rebuilt on a schedule."""
    payload = {"forecast": load_forecast(), "metrics": load_metrics(),
               "live": live_info("data.json", 120, refresh_minutes)}
    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    (SITE_DIR / "index.html").write_text(render(payload), encoding="utf-8")
    return SITE_DIR


def build_dashboard() -> Path:
    html = render({"forecast": load_forecast(), "metrics": load_metrics()})
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    return DASHBOARD_PATH
