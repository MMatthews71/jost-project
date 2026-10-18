"""Thin API-Football (api-sports.io v3) client with retries, paging and rate limiting."""
from __future__ import annotations

import os
import time

import requests

BASE_URL = "https://v3.football.api-sports.io"


class ApiError(RuntimeError):
    pass


class ApiFootball:
    def __init__(self, key: str | None = None, min_interval: float | None = None, session=None):
        self.key = key or os.environ.get("API_FOOTBALL_KEY")
        if not self.key:
            raise ApiError("API_FOOTBALL_KEY is not set (env var or .env file)")
        # Free plan allows 10 req/min, Pro 300/min. Default is safe for Pro; set 6.5 for Free.
        self.min_interval = float(min_interval if min_interval is not None
                                  else os.environ.get("API_FOOTBALL_MIN_INTERVAL", 0.25))
        self.session = session or requests.Session()
        self._last = 0.0
        self.calls = 0

    def _request(self, path: str, params: dict) -> dict:
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self.session.get(f"{BASE_URL}/{path}", params=params,
                                 headers={"x-apisports-key": self.key}, timeout=30)
            self.calls += 1
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 3)
                continue
            r.raise_for_status()
            body = r.json()
            errors = body.get("errors")
            if errors:
                if isinstance(errors, dict) and "rateLimit" in errors:
                    time.sleep(2 ** attempt * 10)
                    continue
                raise ApiError(f"{path} {params}: {errors}")
            return body
        raise ApiError(f"{path} {params}: gave up after retries")

    def get(self, path: str, **params) -> list:
        """GET an endpoint, following `paging` when the endpoint is paginated."""
        body = self._request(path, params)
        out = list(body.get("response", []))
        total = (body.get("paging") or {}).get("total", 1) or 1
        for page in range(2, total + 1):
            out.extend(self._request(path, {**params, "page": page}).get("response", []))
        return out

    def status(self) -> dict:
        return self._request("status", {}).get("response", {})
