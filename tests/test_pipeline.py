"""End-to-end: train -> daily picks POST -> results POST, against a fake API and a mock vowtips endpoint."""
import pandas as pd
import pytest

from football_ml import config, delivery, journal, picks, results
from football_ml.config import LEAGUES, PickSettings
from football_ml.data import save_season
from football_ml.elo import EloParams
from football_ml.odds import summarise
from football_ml.train import train_final

from .conftest import synthetic_fixtures

DAY = "2099-09-12"  # future, so the already-kicked-off filter keeps every fixture


def api_fixture(fid, league, season, kickoff, status, home, away, hg=None, ag=None):
    lg = LEAGUES[league]
    return {"fixture": {"id": fid, "date": kickoff, "status": {"short": status}},
            "league": {"id": lg.api_id, "season": season, "round": "Regular Season - 4"},
            "teams": {"home": {"id": home, "name": f"T{home}"}, "away": {"id": away, "name": f"T{away}"}},
            "goals": {"home": hg, "away": ag},
            "score": {"fulltime": {"home": hg, "away": ag}}}


def api_odds(fid, h, d, a, books=("Bet365", "Pinnacle")):
    return {"fixture": {"id": fid}, "bookmakers": [
        {"id": i, "name": b, "bets": [{"id": 1, "name": "Match Winner", "values": [
            {"value": "Home", "odd": str(h)}, {"value": "Draw", "odd": str(d)}, {"value": "Away", "odd": str(a)}]}]}
        for i, b in enumerate(books)]}


UPCOMING = [  # (fixture id, home, away, odds H/D/A) — teams from the synthetic EPL pool
    (900001, 1000, 1001, (2.10, 3.40, 3.60)),
    (900002, 1002, 1003, (1.80, 3.60, 4.50)),
    (900003, 1004, 1005, (2.60, 3.20, 2.80)),
    (900004, 1006, 1007, (3.10, 3.30, 2.30)),
    (900005, 1008, 1009, (1.50, 4.20, 6.50)),
    (900006, 1010, 1011, (2.40, 3.10, 3.10)),
]
FINAL_SCORES = {900001: (2, 0), 900002: (1, 1), 900003: (0, 1), 900004: None, 900005: (3, 1), 900006: (0, 0)}
CLOSING = 0.95   # closing prices in the fake API = opening * 0.95


class FakeApi:
    def __init__(self):
        self.calls = []

    def get(self, path, **params):
        self.calls.append((path, params))
        if path == "odds" and "fixture" in params:          # closing odds for one settled match
            fid, _, _, o = next(u for u in UPCOMING if u[0] == params["fixture"])
            return [api_odds(fid, *(round(x * CLOSING, 2) for x in o))]
        if path == "odds":
            assert params["date"] == DAY
            return [api_odds(fid, *o) for fid, _, _, o in UPCOMING] if params["league"] == 39 else []
        if path == "fixtures" and "ids" in params:
            out = []
            for fid in map(int, params["ids"].split("-")):
                fx = next(u for u in UPCOMING if u[0] == fid)
                sc = FINAL_SCORES[fid]
                status, hg, ag = ("PST", None, None) if sc is None else ("FT", *sc)
                out.append(api_fixture(fid, "EPL", 2025, f"{DAY}T14:00:00+00:00", status, fx[1], fx[2], hg, ag))
            return out
        raise AssertionError(f"unexpected call {path} {params}")


@pytest.fixture
def trained(tmp_paths):
    hist = synthetic_fixtures(leagues=("EPL",))
    for season, g in hist.groupby("season"):
        save_season(g, "EPL", season)
    up = pd.DataFrame([{
        "fixture_id": fid, "league": "EPL", "season": 2025, "kickoff_utc": f"{DAY}T14:00:00+00:00",
        "status": "NS", "round": "Regular Season - 4", "home_id": h, "home_name": f"T{h}", "away_id": a,
        "away_name": f"T{a}", "home_goals": None, "away_goals": None,
        "league_logo": "https://media.api-sports.io/football/leagues/39.png",
        "home_logo": f"https://media.api-sports.io/football/teams/{h}.png",
        "away_logo": None} for fid, h, a, _ in UPCOMING])
    save_season(up, "EPL", 2025)
    train_final(params=EloParams())
    return tmp_paths


PICK_FIELDS = {"match_id", "league_id", "league", "league_logo_url", "kickoff_time", "home_team_id", "home_team",
               "home_logo_url", "away_team_id", "away_team", "away_logo_url", "market", "selection", "odds",
               "model_probability", "ev_percent", "units", "category", "analysis_text", "key_stats"}


def test_daily_picks_and_results_end_to_end(trained, mock_endpoint):
    api = FakeApi()
    sent = picks.run(api, day=DAY, refresh=False)

    vip = [p for p in sent if p["tier"] == "vip"]
    assert len(vip) in range(3, 6) and len([p for p in sent if p["tier"] == "free"]) == 1
    s = PickSettings()
    assert all(s.min_odds <= p["odds"] <= s.max_odds for p in sent)
    value = [p["p_model"] for p in vip if p["value_bet"]]
    assert value == sorted(value, reverse=True)          # value picks ranked by model probability

    post = mock_endpoint[0]
    assert post["path"] == "/picks" and post["api_key"] == "test-key"
    body = post["body"]
    assert isinstance(body, list) and len(body) == len(sent)
    assert [b["category"] for b in body].count("free") == 1
    first = body[0]
    assert set(first) == PICK_FIELDS                      # exactly the site's schema, no test flag
    assert first["market"] == "1X2" and first["selection"] in {"1", "X", "2"} and first["league_id"] == 39
    assert first["kickoff_time"] == f"{DAY}T14:00:00Z"
    assert first["away_logo_url"] == f"https://media.api-sports.io/football/teams/{first['away_team_id']}.png"
    assert len(journal.picks()) == len(sent)
    assert (config.ODDS_DIR / f"{DAY}.csv").exists()       # every match's odds kept for later tuning

    # Second run the same day is a no-op (no duplicate delivery).
    assert picks.run(api, day=DAY, refresh=False) == []
    assert len(mock_endpoint) == 1

    settled = results.run(api)
    assert len(settled) == len(sent)
    res = mock_endpoint[1]
    assert res["path"] == "/results" and res["api_key"] == "test-key"
    by_id = {r["match_id"]: r for r in res["body"]}
    for p in sent:
        r = by_id[p["fixture_id"]]
        assert set(r) == {"match_id", "market", "selection", "result", "final_score", "closing_odds"}
        assert r["market"] == "1X2" and r["selection"] == p["selection"]
        if FINAL_SCORES[p["fixture_id"]] is None:
            assert r["result"] == "void" and r["final_score"] is None and r["closing_odds"] is None
        else:
            hg, ag = FINAL_SCORES[p["fixture_id"]]
            outcome = "1" if hg > ag else "2" if hg < ag else "X"
            assert r["final_score"] == f"{hg}-{ag}"
            assert r["result"] == ("win" if outcome == p["selection"] else "loss")
            assert r["closing_odds"] == pytest.approx(round(p["odds"] * CLOSING, 2))
    assert journal.pending_picks() == []
    assert results.run(api) == []        # nothing left to settle


def test_test_mode_flags_every_pick(trained, mock_endpoint, monkeypatch):
    monkeypatch.setenv("PICKS_TEST_MODE", "true")
    picks.run(FakeApi(), day=DAY, refresh=False)
    assert all(b["test"] is True for b in mock_endpoint[0]["body"])


def test_site_rejections_fail_the_run_but_are_logged(trained, mock_endpoint):
    mock_endpoint.reply = {"inserted": 0, "rejected": 2, "errors": ["bad selection"]}
    with pytest.raises(delivery.DeliveryError, match="rejected"):
        picks.run(FakeApi(), day=DAY, refresh=False)
    assert journal.picks()          # sent picks are still recorded, so results get settled


def test_wrong_api_key_is_a_clear_error(trained, mock_endpoint, monkeypatch):
    monkeypatch.setenv("PICKS_API_KEY", "wrong")
    with pytest.raises(delivery.DeliveryError, match="401"):
        picks.run(FakeApi(), day=DAY, refresh=False)
    assert journal.picks() == []


def test_dry_run_does_not_post_or_log(trained, mock_endpoint, capsys):
    picks.run(FakeApi(), day=DAY, refresh=False, dry_run=True)
    assert mock_endpoint == [] and journal.picks() == []
    assert '"category": "vip"' in capsys.readouterr().out


def cand(fid, p, ev):
    return {"fixture_id": fid, "p_model": p, "ev": ev, "value_bet": ev >= 0}


def test_select_ranks_value_by_probability_then_tops_up():
    s = PickSettings(vip_max=5, vip_min=3, min_ev=0.0, rank_by="probability")
    cands = [cand(1, 0.40, 0.20), cand(2, 0.60, 0.05), cand(3, 0.70, -0.02), cand(4, 0.30, -0.10),
             cand(5, 0.55, -0.01)]
    out = picks.select(cands, s)
    assert [(p["fixture_id"], p["tier"]) for p in out] == [(2, "vip"), (1, "vip"), (3, "vip"), (5, "free")]


def test_select_caps_vip_at_max():
    s = PickSettings(vip_max=5, vip_min=3, min_ev=0.0)
    out = picks.select([cand(i, 0.5 + i / 100, 0.1) for i in range(8)], s)
    assert [p["tier"] for p in out] == ["vip"] * 5 + ["free"]


def test_candidates_respect_odds_range_and_prefer_likely_value():
    s = PickSettings(min_odds=1.40, max_odds=3.50, min_ev=0.0, rank_by="probability")
    pred = pd.DataFrame([{"fixture_id": 1, "league": "EPL", "season": 2026, "kickoff_utc": f"{DAY}T14:00:00+00:00",
                          "home_id": 1, "home_name": "A", "away_id": 2, "away_name": "B", "elo_home": 1600,
                          "elo_away": 1500, "p_home": 0.50, "p_draw": 0.30, "p_away": 0.20}])
    # Home 2.2 (EV +10%), draw 3.4 (EV +2%), away 6.0 (EV +20% but outside the odds range).
    odds = {1: summarise(api_odds(1, 2.2, 3.4, 6.0, books=("A",)))}
    [c] = picks.candidates(pred, odds, s)
    assert c["selection"] == "1" and c["value_bet"]


def test_odds_summary_removes_margin():
    s = summarise(api_odds(1, 2.0, 3.5, 4.0, books=("A",)))
    assert sum(s["implied"].values()) == pytest.approx(1)
    assert s["margin"] == pytest.approx(1 / 2 + 1 / 3.5 + 1 / 4 - 1)


def test_refuses_plain_http(monkeypatch):
    monkeypatch.setattr(config, "PICKS_URL", "http://example.com/picks")
    monkeypatch.setenv("PICKS_API_KEY", "k")
    with pytest.raises(delivery.DeliveryError, match="non-HTTPS"):
        delivery.post("picks", [{"x": 1}])


def test_missing_api_key_refuses_to_send(monkeypatch):
    monkeypatch.delenv("PICKS_API_KEY", raising=False)
    with pytest.raises(delivery.DeliveryError, match="PICKS_API_KEY"):
        delivery.post("picks", [{"x": 1}])
