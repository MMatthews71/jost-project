"""Fixture download and on-disk storage (one CSV per league-season, committed to the repo)."""
from __future__ import annotations

import pandas as pd

from . import config
from .config import LEAGUES, League, current_season

FINISHED = {"FT", "AET", "PEN"}
VOID = {"PST", "CANC", "ABD", "AWD", "WO"}

COLUMNS = ["fixture_id", "league", "season", "kickoff_utc", "status", "round",
           "home_id", "home_name", "away_id", "away_name", "home_goals", "away_goals"]


def parse_fixture(item: dict, league_key: str | None = None) -> dict:
    fx, lg, teams, score = item["fixture"], item["league"], item["teams"], item["score"]
    # 1X2 settles on the 90-minute score; score.fulltime excludes extra time.
    ft = score.get("fulltime") or {}
    hg, ag = ft.get("home"), ft.get("away")
    if hg is None:
        hg, ag = item["goals"]["home"], item["goals"]["away"]
    if league_key is None:
        from .config import LEAGUE_BY_API_ID
        league_key = LEAGUE_BY_API_ID[lg["id"]].key
    return {
        "fixture_id": fx["id"],
        "league": league_key,
        "season": lg["season"],
        "kickoff_utc": fx["date"],
        "status": fx["status"]["short"],
        "round": lg.get("round"),
        "home_id": teams["home"]["id"],
        "home_name": teams["home"]["name"],
        "away_id": teams["away"]["id"],
        "away_name": teams["away"]["name"],
        "home_goals": hg,
        "away_goals": ag,
    }


def fixtures_path(league_key: str, season: int):
    return config.FIXTURES_DIR / f"{league_key}_{season}.csv"


def fetch_season(api, league: League, season: int) -> pd.DataFrame:
    items = api.get("fixtures", league=league.api_id, season=season, timezone="UTC")
    df = pd.DataFrame([parse_fixture(i, league.key) for i in items], columns=COLUMNS)
    return df.sort_values(["kickoff_utc", "fixture_id"]).reset_index(drop=True)


def save_season(df: pd.DataFrame, league_key: str, season: int) -> None:
    config.FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(fixtures_path(league_key, season), index=False)


def refresh_current(api, leagues=None) -> None:
    """Re-download the current season of each league (new results + upcoming fixtures)."""
    for key in leagues or LEAGUES:
        lg = LEAGUES[key]
        season = current_season(lg)
        df = fetch_season(api, lg, season)
        if len(df):
            save_season(df, key, season)
        print(f"[refresh] {key} {season}: {len(df)} fixtures")


def load_fixtures(leagues=None) -> pd.DataFrame:
    leagues = set(leagues or LEAGUES)
    frames = [pd.read_csv(p) for p in sorted(config.FIXTURES_DIR.glob("*.csv"))
              if p.stem.rsplit("_", 1)[0] in leagues]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, format="ISO8601")
    df = league_matches(df)
    return df.sort_values(["kickoff_utc", "fixture_id"]).reset_index(drop=True)


def league_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Drop play-offs: knockout/neutral-venue rounds and matches against teams from another
    division (e.g. Bundesliga relegation play-off). Keeps split-league phases such as Serbia's
    championship and relegation groups, where both sides are league members."""
    rnd = df["round"].fillna("").astype(str)
    regular = rnd.str.startswith("Regular Season")
    reg = df[regular]
    members = set(zip(reg.league, reg.season, reg.home_id)) | set(zip(reg.league, reg.season, reg.away_id))
    both_members = pd.Series([(lg, s, h) in members and (lg, s, a) in members
                              for lg, s, h, a in zip(df.league, df.season, df.home_id, df.away_id)],
                             index=df.index, dtype=bool)
    knockout = rnd.str.contains("Final|Semi|Play-off", case=False)
    return df[regular | (both_members & ~knockout)]


def finished(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["status"].isin(FINISHED) & df["home_goals"].notna()].copy()
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    out["result"] = "D"
    out.loc[out.home_goals > out.away_goals, "result"] = "H"
    out.loc[out.home_goals < out.away_goals, "result"] = "A"
    return out
