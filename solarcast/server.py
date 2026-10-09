"""Serve the dashboard locally and keep its data fresh.

The browser cannot call energy-charts.info itself (the API does not allow other
origins), so this process re-runs the forecast on a timer and the page polls
``/api/data`` for the result.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .dashboard import live_info, render
from .forecast import make_forecast
from .model import load_metrics

POLL_SECONDS = 60  # how often the page asks this server for new data

_state: dict = {"forecast": None, "error": None, "refresh_minutes": 10}


def _payload() -> dict:
    return {"forecast": _state["forecast"], "metrics": load_metrics(),
            "live": live_info("api/data", POLL_SECONDS, _state["refresh_minutes"], _state["error"])}


def _refresh_forever(interval_s: float) -> None:
    while True:
        time.sleep(interval_s)
        try:
            _state["forecast"] = make_forecast()
            _state["error"] = None
        except Exception as e:  # keep serving the last good forecast
            _state["error"] = type(e).__name__
            print(f"Refresh failed, keeping the last forecast: {e}")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/":
            self._send(render(_payload()), "text/html; charset=utf-8")
        elif path == "/api/data":
            self._send(json.dumps(_payload()), "application/json")
        else:
            self.send_error(404)

    def _send(self, body: str, content_type: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args) -> None:  # the page polls every minute; stay quiet
        pass


def serve(port: int = 8000, refresh_minutes: float = 10, on_ready=None) -> None:
    """Run until interrupted. ``on_ready`` is called with the URL once the first forecast is in."""
    _state["forecast"] = make_forecast()
    _state["refresh_minutes"] = refresh_minutes
    threading.Thread(target=_refresh_forever, args=(refresh_minutes * 60,), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Live dashboard at {url} (data refreshes every {refresh_minutes:g} min, Ctrl+C to stop)")
    if on_ready:
        on_ready(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
