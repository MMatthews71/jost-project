"""Daily pick selection: model probabilities vs market odds -> top VIP picks + one free pick."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config, delivery, journal
from .config import LEAGUES, PickSettings
from .data import load_fixtures, refresh_current
from .model import OutcomeModel
from .odds import fetch_odds
from .predict import fixtures_on, predict_upcoming, today_local

SELECTION = {"H": ("1", "Home win"), "D": ("X", "Draw"), "A": ("2", "Away win")}


def _rank_key(s: PickSettings):
    return (lambda c: c["p_model"]) if s.rank_by == "probability" else (lambda c: c["ev"])


def snapshot(pred: pd.DataFrame, odds: dict[int, dict]) -> pd.DataFrame:
    """One row per fixture: model probabilities next to the market's prices. Kept daily in
    data/odds/ so pick settings can later be tuned on real odds (none exist historically)."""
    rows = []
    for r in pred.itertuples(index=False):
        o = odds.get(r.fixture_id)
        rows.append({
            "fixture_id": int(r.fixture_id), "league": r.league, "kickoff_utc": pd.Timestamp(r.kickoff_utc).isoformat(),
            "home_name": r.home_name, "away_name": r.away_name,
            "elo_home": round(float(r.elo_home), 1), "elo_away": round(float(r.elo_away), 1),
            "p_home": round(float(r.p_home), 4), "p_draw": round(float(r.p_draw), 4), "p_away": round(float(r.p_away), 4),
            **({f"odds_{k}": o["odds"][k] for k in "HDA"} | {f"best_{k}": o["best_odds"][k] for k in "HDA"}
               | {"n_bookmakers": o["n_bookmakers"]} if o else {}),
        })
    return pd.DataFrame(rows)


def candidates(pred: pd.DataFrame, odds: dict[int, dict], s: PickSettings) -> list[dict]:
    """One candidate outcome per fixture, priced within the configured odds range.

    Prefers outcomes with EV >= min_ev and, among those, the best by `rank_by` (model probability
    by default). A fixture without any value outcome contributes its best-EV outcome, which is
    only used to top up the VIP list on thin days (flagged value_bet=False)."""
    key = _rank_key(s)
    out = []
    for r in pred.itertuples(index=False):
        o = odds.get(r.fixture_id)
        if not o:
            continue
        probs = {"H": r.p_home, "D": r.p_draw, "A": r.p_away}
        options = [{"outcome": k, "odds": o["odds"][k], "p_model": float(probs[k]), "p_implied": o["implied"][k],
                    "ev": float(probs[k] * o["odds"][k] - 1)}
                   for k in "HDA" if s.min_odds <= o["odds"][k] <= s.max_odds]
        if not options:
            continue
        value = [x for x in options if x["ev"] >= s.min_ev]
        best = max(value, key=key) if value else max(options, key=lambda x: x["ev"])
        lg = LEAGUES[r.league]
        out.append({
            "fixture_id": int(r.fixture_id), "league": r.league, "league_name": lg.name, "country": lg.country,
            "season": int(r.season), "kickoff_utc": pd.Timestamp(r.kickoff_utc).isoformat(),
            "home_id": int(r.home_id), "home_name": r.home_name, "away_id": int(r.away_id), "away_name": r.away_name,
            "league_logo": getattr(r, "league_logo", None), "home_logo": getattr(r, "home_logo", None),
            "away_logo": getattr(r, "away_logo", None),
            "elo_home": round(float(r.elo_home), 1), "elo_away": round(float(r.elo_away), 1),
            "p_home": round(float(r.p_home), 4), "p_draw": round(float(r.p_draw), 4), "p_away": round(float(r.p_away), 4),
            "selection": SELECTION[best["outcome"]][0], "selection_label": SELECTION[best["outcome"]][1],
            "outcome_key": best["outcome"], "odds": best["odds"], "p_model": best["p_model"],
            "p_implied": float(best["p_implied"]), "ev": best["ev"], "value_bet": best["ev"] >= s.min_ev,
            "n_bookmakers": o["n_bookmakers"],
        })
    return out


def select(cands: list[dict], s: PickSettings) -> list[dict]:
    """VIP: up to `vip_max` value picks (EV >= min_ev) ordered by `rank_by`, topped up to `vip_min`
    with the most likely of the remaining candidates on thin days. Free: the next one in line."""
    key = _rank_key(s)
    value = sorted((c for c in cands if c["value_bet"]), key=key, reverse=True)
    rest = sorted((c for c in cands if not c["value_bet"]), key=lambda c: c["p_model"], reverse=True)
    queue = value + rest
    n_vip = max(min(len(value), s.vip_max), min(s.vip_min, len(queue)))
    vip, free = queue[:n_vip], queue[n_vip:n_vip + 1]
    return ([{**c, "tier": "vip", "rank": i, "units": s.units} for i, c in enumerate(vip, 1)]
            + [{**c, "tier": "free", "rank": 1, "units": s.units} for c in free])


def run(api, day: str | None = None, dry_run: bool = False, refresh: bool = True,
        force: bool = False, settings: PickSettings | None = None) -> list[dict]:
    s = settings or PickSettings()
    day = day or today_local()
    if not force and not dry_run and any(p["date"] == day for p in journal.picks()):
        print(f"[picks] picks for {day} already delivered; use --force to resend")
        return []
    if refresh:
        refresh_current(api)
    model = OutcomeModel.load()
    fixtures = load_fixtures()
    upcoming = fixtures_on(fixtures, day)
    # Never pick a match that has already kicked off (or is about to) by the time we send.
    upcoming = upcoming[upcoming["kickoff_utc"] > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=15)]
    print(f"[picks] {day}: {len(upcoming)} upcoming fixtures")
    if upcoming.empty:
        return []
    pred = predict_upcoming(upcoming, fixtures, model)
    odds = {}
    wanted = set(upcoming["fixture_id"])
    for (league, season), g in upcoming.groupby(["league", "season"]):
        # The odds endpoint filters by UTC date; a local day can span two of them.
        for utc_day in sorted(set(g["kickoff_utc"].dt.strftime("%Y-%m-%d"))):
            got = fetch_odds(api, utc_day, LEAGUES[league].api_id, int(season), s.bookmaker_id)
            odds.update({k: v for k, v in got.items() if k in wanted})
        print(f"[picks] odds {league}: {sum(f in odds for f in g['fixture_id'])}/{len(g)} fixtures")
    if not dry_run:
        config.ODDS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot(pred, odds).to_csv(config.ODDS_DIR / f"{day}.csv", index=False)
    picks = select(candidates(pred, odds, s), s)
    if not picks:
        print("[picks] no fixtures with odds in range; nothing to send")
        return []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    test = config.test_mode()
    for p in picks:
        p.update({"pick_id": f"{day}:{p['fixture_id']}", "date": day, "created_utc": now, "test": test})
    body = delivery.post("picks", delivery.picks_payload(picks), dry_run=dry_run)
    if not dry_run:
        journal.append(config.PICKS_LOG, picks)
        delivery.check("picks", body)
    return picks
