"""Static configuration: leagues, paths, environment-driven settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env_file(path=ROOT / ".env") -> None:
    """Populate os.environ from a local .env (KEY=VALUE lines) without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.split("#")[0].strip().strip('"').strip("'"))


# Settings below read the environment, so a local .env must be loaded first.
load_env_file()
DATA_DIR = ROOT / "data"
FIXTURES_DIR = DATA_DIR / "fixtures"
MODEL_PATH = ROOT / "models" / "model.json"
PICKS_LOG = DATA_DIR / "picks.jsonl"
RESULTS_LOG = DATA_DIR / "results.jsonl"
ODDS_DIR = DATA_DIR / "odds"     # daily snapshot of every match's odds + model probabilities
REPORTS_DIR = ROOT / "reports"

# Timezone the "day" of picks is defined in (the site runs on Belgrade time).
TIMEZONE = os.environ.get("PICKS_TIMEZONE", "Europe/Belgrade")

# Delivery endpoints (vowtips.com public API), authenticated with the PICKS_API_KEY secret.
PICKS_URL = os.environ.get("PICKS_ENDPOINT_URL") or "https://vowtips.com/api/public/picks"
RESULTS_URL = os.environ.get("RESULTS_ENDPOINT_URL") or "https://vowtips.com/api/public/results"


def test_mode() -> bool:
    """When on, every pick is sent with "test": true: the site stores it but never shows it to members."""
    return os.environ.get("PICKS_TEST_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class League:
    key: str
    api_id: int
    name: str
    country: str
    calendar_year: bool = False  # season runs Jan-Dec (Brasileirao) instead of Aug-May


LEAGUES: dict[str, League] = {lg.key: lg for lg in [
    League("EPL", 39, "Premier League", "England"),
    League("LALIGA", 140, "La Liga", "Spain"),
    League("SERIEA", 135, "Serie A", "Italy"),
    League("BUNDESLIGA", 78, "Bundesliga", "Germany"),
    League("LIGUE1", 61, "Ligue 1", "France"),
    League("CHAMPIONSHIP", 40, "Championship", "England"),
    League("EREDIVISIE", 88, "Eredivisie", "Netherlands"),
    League("SRB", 286, "Serbian SuperLiga", "Serbia"),
    League("HNL", 210, "HNL", "Croatia"),
    League("BRA", 71, "Brasileirão", "Brazil", calendar_year=True),
]}

LEAGUE_BY_API_ID = {lg.api_id: lg for lg in LEAGUES.values()}


def current_season(league: League, today: date | None = None) -> int:
    """API-Football season label: start year for Aug-May leagues, calendar year for Brazil."""
    today = today or date.today()
    if league.calendar_year:
        return today.year
    return today.year if today.month >= 7 else today.year - 1


def default_backfill_seasons(league: League, n: int = 4, today: date | None = None) -> list[int]:
    """The n most recent *completed* seasons plus the current one."""
    cur = current_season(league, today)
    return list(range(cur - n, cur + 1))


@dataclass(frozen=True)
class PickSettings:
    vip_max: int = int(os.environ.get("VIP_MAX", 5))
    vip_min: int = int(os.environ.get("VIP_MIN", 3))
    min_ev: float = float(os.environ.get("MIN_EV", 0.0))      # expected value per unit staked
    min_odds: float = float(os.environ.get("MIN_ODDS", 1.40))
    max_odds: float = float(os.environ.get("MAX_ODDS", 3.50))
    # "probability": among value picks (EV >= min_ev), most likely winners first.
    # "ev": highest expected value first (favours long shots, where the model is least reliable).
    rank_by: str = os.environ.get("RANK_BY", "probability")
    units: float = float(os.environ.get("UNITS", 1))
    bookmaker_id: int | None = int(os.environ["ODDS_BOOKMAKER_ID"]) if os.environ.get("ODDS_BOOKMAKER_ID") else None
