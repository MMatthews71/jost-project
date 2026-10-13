"""Pre-match 1X2 odds from API-Football, summarised per fixture."""
from __future__ import annotations

import statistics

MATCH_WINNER_BET_ID = 1
OUTCOME_KEYS = {"Home": "H", "Draw": "D", "Away": "A"}


def summarise(item: dict, bookmaker_id: int | None = None) -> dict | None:
    """Median price per outcome across bookmakers (or one bookmaker's price if configured),
    plus margin-free implied probabilities from those prices."""
    prices: dict[str, list[float]] = {"H": [], "D": [], "A": []}
    books = []
    for bm in item.get("bookmakers", []):
        if bookmaker_id is not None and bm["id"] != bookmaker_id:
            continue
        for bet in bm.get("bets", []):
            if bet["id"] != MATCH_WINNER_BET_ID:
                continue
            vals = {OUTCOME_KEYS.get(v["value"]): float(v["odd"]) for v in bet["values"]}
            if all(vals.get(o, 0) > 1 for o in "HDA"):
                for o in "HDA":
                    prices[o].append(vals[o])
                books.append(bm["name"])
    if not books:
        return None
    odds = {o: round(statistics.median(prices[o]), 2) for o in "HDA"}
    inv = {o: 1 / odds[o] for o in "HDA"}
    tot = sum(inv.values())
    return {
        "fixture_id": item["fixture"]["id"],
        "odds": odds,
        "best_odds": {o: max(prices[o]) for o in "HDA"},
        "implied": {o: inv[o] / tot for o in "HDA"},
        "margin": tot - 1,
        "n_bookmakers": len(books),
    }


def fetch_odds(api, day: str, league_api_id: int, season: int, bookmaker_id: int | None = None) -> dict[int, dict]:
    params = {"date": day, "league": league_api_id, "season": season, "bet": MATCH_WINNER_BET_ID}
    if bookmaker_id is not None:
        params["bookmaker"] = bookmaker_id
    out = {}
    for item in api.get("odds", **params):
        s = summarise(item, bookmaker_id)
        if s:
            out[s["fixture_id"]] = s
    return out
