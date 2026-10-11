"""Model probabilities for upcoming fixtures, using the latest Elo ratings."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .config import TIMEZONE
from .data import finished
from .elo import run_elo
from .model import OutcomeModel


def today_local() -> str:
    return datetime.now(ZoneInfo(TIMEZONE)).date().isoformat()


def fixtures_on(fixtures: pd.DataFrame, day: str) -> pd.DataFrame:
    local_day = fixtures["kickoff_utc"].dt.tz_convert(TIMEZONE).dt.date.astype(str)
    return fixtures[(local_day == day) & fixtures["status"].isin(["NS", "TBD"])]


def predict_upcoming(upcoming: pd.DataFrame, history: pd.DataFrame, model: OutcomeModel) -> pd.DataFrame:
    """Attach current Elo ratings and model probabilities to not-yet-played fixtures."""
    _, eng = run_elo(finished(history), model.elo_params)
    rows = []
    for r in upcoming.itertuples(index=False):
        eng.start_season(r.league, r.season)
        rows.append({"elo_home": eng.rating(r.league, r.home_id), "elo_away": eng.rating(r.league, r.away_id)})
    up = upcoming.reset_index(drop=True).join(pd.DataFrame(rows, columns=["elo_home", "elo_away"]))
    return model.predict_df(up) if len(up) else up


def current_ratings(fixtures: pd.DataFrame, elo_params) -> pd.DataFrame:
    """Latest Elo rating of every team in each league's most recent season, strongest first."""
    _, eng = run_elo(finished(fixtures), elo_params)
    rows = []
    for league, g in fixtures.groupby("league"):
        cur = g[g["season"] == g["season"].max()]
        eng.start_season(league, int(cur["season"].iloc[0]))
        names = dict(zip(cur.home_id, cur.home_name)) | dict(zip(cur.away_id, cur.away_name))
        played = finished(cur)
        counts = pd.concat([played.home_id, played.away_id]).value_counts()
        for team, name in names.items():
            rows.append({"league": league, "season": int(cur["season"].iloc[0]), "team": name,
                         "rating": round(eng.rating(league, team), 1), "played": int(counts.get(team, 0))})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("league")["rating"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["league", "rank"]).reset_index(drop=True)
