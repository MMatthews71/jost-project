"""Delivery of picks and results to the vowtips.com public API.

Everything about the wire format lives in this module: `picks_payload` and `result_payload`
build the JSON bodies, `post` sends them.

Environment:
    PICKS_API_KEY         sent as the `x-api-key` header (required for real sends)
    PICKS_ENDPOINT_URL    override for the picks URL (default: config.PICKS_URL)
    RESULTS_ENDPOINT_URL  override for the results URL (default: config.RESULTS_URL)
    PICKS_TEST_MODE       "true" -> every pick is sent with "test": true (stored, never shown)
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

from . import config

LOGO = "https://media.api-sports.io/football/{kind}/{id}.png"   # API-Football's logo CDN


class DeliveryError(RuntimeError):
    pass


def _logo(url, kind: str, api_id: int) -> str:
    return url if isinstance(url, str) and url else LOGO.format(kind=kind, id=api_id)


def _utc_z(ts: str) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def pick_view(p: dict) -> dict:
    league_id = config.LEAGUES[p["league"]].api_id
    out = {
        "match_id": p["fixture_id"],
        "league_id": league_id,
        "league": p["league_name"],
        "league_logo_url": _logo(p.get("league_logo"), "leagues", league_id),
        "kickoff_time": _utc_z(p["kickoff_utc"]),
        "home_team_id": p["home_id"],
        "home_team": p["home_name"],
        "home_logo_url": _logo(p.get("home_logo"), "teams", p["home_id"]),
        "away_team_id": p["away_id"],
        "away_team": p["away_name"],
        "away_logo_url": _logo(p.get("away_logo"), "teams", p["away_id"]),
        "market": "1X2",
        "selection": p["selection"],                 # "1" home, "X" draw, "2" away
        "odds": p["odds"],                           # median price across bookmakers
        "model_probability": round(p["p_model"], 4),
        "ev_percent": round(p["ev"] * 100, 1),
        "units": int(p["units"]) if float(p["units"]).is_integer() else p["units"],
        "category": p["tier"],                       # "vip" or "free"
        "analysis_text": None,
        "key_stats": {
            "home_elo": round(p["elo_home"]),
            "away_elo": round(p["elo_away"]),
            "home_win_probability": p["p_home"],
            "draw_probability": p["p_draw"],
            "away_win_probability": p["p_away"],
            "market_probability": round(p["p_implied"], 4),
            "bookmakers": p["n_bookmakers"],
        },
    }
    if p.get("test"):
        out["test"] = True
    return out


def picks_payload(picks: list[dict]) -> list[dict]:
    return [pick_view(p) for p in picks]


def result_payload(results: list[dict]) -> list[dict]:
    return [{
        "match_id": r["fixture_id"],
        "market": "1X2",
        "selection": r["selection"],
        "result": r["status"],              # "win" | "loss" | "void"
        "final_score": r["score"],          # 90-minute score "2-1", or null when void
        "closing_odds": r["closing_odds"],  # last pre-match median price, or null if unavailable
    } for r in results]


def _problems(body) -> list:
    """Rejections/errors reported in the site's response ({inserted, updated, drafted, rejected, errors})."""
    if not isinstance(body, dict):
        return []
    out = []
    rejected = body.get("rejected")
    if rejected:   # a count or a list
        out.append(f"rejected: {rejected}")
    if body.get("errors"):
        out.append(f"errors: {body['errors']}")
    return out


def post(kind: str, payload: list[dict], dry_run: bool = False, retries: int = 4):
    """POST a payload; returns the parsed response. Raises DeliveryError if the request fails
    or the site reports rejected items (after the caller has had the response logged)."""
    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return None
    url = config.PICKS_URL if kind == "picks" else config.RESULTS_URL
    if not url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise DeliveryError(f"refusing to POST to non-HTTPS URL: {url}")
    key = os.environ.get("PICKS_API_KEY")
    if not key:
        raise DeliveryError("PICKS_API_KEY is not set; cannot send (use --dry-run to preview)")
    headers = {"Content-Type": "application/json", "x-api-key": key}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
            if r.status_code < 300:
                try:
                    body = r.json()
                except ValueError:
                    body = r.text
                print(f"[delivery] {kind}: HTTP {r.status_code} {json.dumps(body, ensure_ascii=False)[:1000]}")
                return body
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 401:
                raise DeliveryError(f"POST {kind} unauthorised (401): check the PICKS_API_KEY secret")
            if 400 <= r.status_code < 500 and r.status_code != 429:
                break   # client error: retrying will not help
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 ** attempt * 2)
    raise DeliveryError(f"POST {kind} failed: {last}")


def check(kind: str, body) -> None:
    problems = _problems(body)
    if problems:
        raise DeliveryError(f"site reported problems with {kind}: {'; '.join(problems)}")
