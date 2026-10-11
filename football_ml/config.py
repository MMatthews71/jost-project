"""Static configuration: leagues, paths, environment-driven settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIXTURES_DIR = DATA_DIR / "fixtures"
MODEL_PATH = ROOT / "models" / "model.json"
REPORTS_DIR = ROOT / "reports"

# Timezone used to decide which calendar day a fixture belongs to (client is in Central Europe).
TIMEZONE = os.environ.get("PICKS_TIMEZONE", "Europe/Ljubljana")


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
    League("SRB", 286, "Super Liga", "Serbia"),
    League("HNL", 210, "HNL", "Croatia"),
    League("BRA", 71, "Serie A (Brasileirao)", "Brazil", calendar_year=True),
]}

LEAGUE_BY_API_ID = {lg.api_id: lg for lg in LEAGUES.values()}


def current_season(league: League, today: date | None = None) -> int:
    """API-Football season label: start year for Aug-May leagues, calendar year for Brazil."""
    today = today or date.today()
    if league.calendar_year:
        return today.year
    return today.year if today.month >= 7 else today.year - 1


def default_backfill_seasons(league: League, n: int = 3, today: date | None = None) -> list[int]:
    """The n most recent *completed* seasons plus the current one."""
    cur = current_season(league, today)
    return list(range(cur - n, cur + 1))

