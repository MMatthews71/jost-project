"""Elo rating engine.

Ratings are kept per (league, team): each league is its own rating pool, so a team promoted
from the Championship does not carry a Championship-scale rating into the Premier League.
Teams new to a league start at a low percentile of that league's current ratings (promoted
sides are usually weaker), and every rating is regressed toward the league mean between seasons.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_adv: float = 60.0       # rating points added to the home side when computing expectation
    carry: float = 0.75          # fraction of (rating - league mean) kept across a season break
    new_team_pct: float = 20.0   # percentile of the league's ratings a newcomer starts at
    base: float = 1500.0

    def to_dict(self) -> dict:
        return asdict(self)


def mov_multiplier(goal_diff: int) -> float:
    """World Football Elo goal-difference multiplier."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


class EloEngine:
    def __init__(self, params: EloParams | None = None):
        self.p = params or EloParams()
        self.ratings: dict[str, dict[int, float]] = {}
        self.season: dict[str, int] = {}

    def _pool(self, league: str) -> dict[int, float]:
        return self.ratings.setdefault(league, {})

    def rating(self, league: str, team: int) -> float:
        pool = self._pool(league)
        if team not in pool:
            pool[team] = (float(np.percentile(list(pool.values()), self.p.new_team_pct))
                          if len(pool) >= 6 else self.p.base)
        return pool[team]

    def start_season(self, league: str, season: int) -> None:
        if self.season.get(league) == season:
            return
        pool = self._pool(league)
        if pool and league in self.season:
            mean = float(np.mean(list(pool.values())))
            for t in pool:
                pool[t] = mean + self.p.carry * (pool[t] - mean)
        self.season[league] = season

    def expected_home(self, league: str, home: int, away: int) -> float:
        diff = self.rating(league, home) + self.p.home_adv - self.rating(league, away)
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    def update(self, league: str, home: int, away: int, home_goals: int, away_goals: int) -> None:
        exp = self.expected_home(league, home, away)
        actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        delta = self.p.k * mov_multiplier(home_goals - away_goals) * (actual - exp)
        pool = self._pool(league)
        pool[home] += delta
        pool[away] -= delta


def run_elo(matches: pd.DataFrame, params: EloParams | None = None) -> tuple[pd.DataFrame, EloEngine]:
    """Replay finished matches chronologically.

    Returns the matches with pre-match ratings (`elo_home`, `elo_away`) attached — these use only
    information available before kickoff — plus the engine holding the latest ratings.
    """
    eng = EloEngine(params)
    pre_h = np.empty(len(matches))
    pre_a = np.empty(len(matches))
    m = matches.sort_values(["kickoff_utc", "fixture_id"])
    for i, r in enumerate(m.itertuples(index=False)):
        eng.start_season(r.league, r.season)
        pre_h[i] = eng.rating(r.league, r.home_id)
        pre_a[i] = eng.rating(r.league, r.away_id)
        eng.update(r.league, r.home_id, r.away_id, r.home_goals, r.away_goals)
    out = m.copy()
    out["elo_home"] = pre_h
    out["elo_away"] = pre_a
    return out, eng
