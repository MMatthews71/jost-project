"""Settle delivered picks once their matches finish and report outcomes."""
from __future__ import annotations

from datetime import datetime, timezone

from . import config, delivery, journal
from .api import ApiError
from .data import FINISHED, VOID, parse_fixture
from .odds import MATCH_WINNER_BET_ID, summarise

OUTCOME_SYMBOL = {"H": "1", "D": "X", "A": "2"}
SYMBOL_OUTCOME = {v: k for k, v in OUTCOME_SYMBOL.items()}


def closing_odds(api, pick: dict) -> float | None:
    """Last pre-match median price for the picked outcome (API-Football keeps odds ~7 days)."""
    try:
        items = api.get("odds", fixture=pick["fixture_id"], bet=MATCH_WINNER_BET_ID)
    except ApiError as e:
        print(f"[results] closing odds unavailable for {pick['fixture_id']}: {e}")
        return None
    s = summarise(items[0]) if items else None
    return s["odds"][SYMBOL_OUTCOME[pick["selection"]]] if s else None


def settle(pick: dict, fx: dict) -> dict | None:
    """Return a result record, or None if the match is not decided yet."""
    status = fx["status"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = {"pick_id": pick["pick_id"], "fixture_id": pick["fixture_id"], "date": pick["date"],
            "tier": pick["tier"], "selection": pick["selection"], "odds": pick["odds"], "settled_utc": now}
    if status in VOID:
        return {**base, "status": "void", "score": None, "outcome": None, "profit_units": 0.0,
                "match_status": status, "closing_odds": None}
    if status not in FINISHED or fx["home_goals"] is None:
        return None
    hg, ag = int(fx["home_goals"]), int(fx["away_goals"])
    outcome = OUTCOME_SYMBOL["H" if hg > ag else "A" if hg < ag else "D"]
    won = outcome == pick["selection"]
    units = pick.get("units", 1)
    return {**base, "status": "win" if won else "loss", "score": f"{hg}-{ag}", "outcome": outcome,
            "profit_units": round((pick["odds"] - 1) * units, 4) if won else -float(units),
            "match_status": status, "closing_odds": None}


def run(api, dry_run: bool = False) -> list[dict]:
    pending = journal.pending_picks()
    print(f"[results] {len(pending)} pending picks")
    if not pending:
        return []
    ids = sorted({p["fixture_id"] for p in pending})
    fixtures = {}
    for i in range(0, len(ids), 20):   # API allows up to 20 ids per call
        for item in api.get("fixtures", ids="-".join(map(str, ids[i:i + 20])), timezone="UTC"):
            f = parse_fixture(item)
            fixtures[f["fixture_id"]] = f
    settled = [r for p in pending if p["fixture_id"] in fixtures
               for r in [settle(p, fixtures[p["fixture_id"]])] if r]
    print(f"[results] {len(settled)} newly settled")
    if settled:
        by_id = {p["pick_id"]: p for p in pending}
        for r in settled:
            if r["status"] != "void":
                r["closing_odds"] = closing_odds(api, by_id[r["pick_id"]])
        body = delivery.post("results", delivery.result_payload(settled), dry_run=dry_run)
        if not dry_run:
            journal.append(config.RESULTS_LOG, settled)
            delivery.check("results", body)
    return settled
