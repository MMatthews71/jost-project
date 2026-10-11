"""Command line: python -m football_ml <command>."""
from __future__ import annotations

import argparse
import json
import sys

from .config import LEAGUES, default_backfill_seasons


def cmd_status(args):
    from .api import ApiFootball
    st = ApiFootball().status()
    print(f"plan={st['subscription']['plan']} requests={st['requests']}")


def cmd_backfill(args):
    from .api import ApiFootball
    from .data import fetch_season, fixtures_path, save_season
    api = ApiFootball()
    for key in args.leagues:
        lg = LEAGUES[key]
        seasons = args.seasons or default_backfill_seasons(lg, args.n)
        for season in seasons:
            if fixtures_path(key, season).exists() and not args.overwrite:
                print(f"[backfill] {key} {season}: cached")
                continue
            df = fetch_season(api, lg, season)
            if df.empty:
                print(f"[backfill] {key} {season}: no fixtures returned")
                continue
            save_season(df, key, season)
            print(f"[backfill] {key} {season}: {len(df)} fixtures ({(df.status == 'FT').sum()} finished)")
    print(f"[backfill] API calls: {api.calls}")


def cmd_refresh(args):
    from .api import ApiFootball
    from .data import refresh_current
    refresh_current(ApiFootball(), args.leagues)


def cmd_validate(args):
    from .data import load_fixtures
    from .train import validate, write_report
    rep = validate(load_fixtures(args.leagues))
    write_report(rep)
    print(json.dumps(rep["overall"], indent=2))
    print("Wrote reports/validation.md")


def cmd_train(args):
    from .data import load_fixtures
    from .train import train_final
    m = train_final(load_fixtures(args.leagues))
    print(f"Saved models/model.json (elo={m.elo_params.to_dict()}, n_train={m.meta['n_train']})")


def cmd_predict(args):
    from .data import load_fixtures
    from .model import OutcomeModel
    from .predict import fixtures_on, predict_upcoming, today_local
    fx = load_fixtures()
    day = args.date or today_local()
    pred = predict_upcoming(fixtures_on(fx, day), fx, OutcomeModel.load())
    if pred.empty:
        print(f"No upcoming fixtures on {day} in local data (run refresh first?)")
        return
    cols = ["league", "kickoff_utc", "home_name", "away_name", "p_home", "p_draw", "p_away"]
    print(pred[cols].round(3).to_string(index=False))


def cmd_ratings(args):
    from .data import load_fixtures
    from .model import OutcomeModel
    from .predict import current_ratings
    r = current_ratings(load_fixtures(args.leagues), OutcomeModel.load().elo_params)
    for league, g in r.groupby("league", sort=False):
        print(f"\n{LEAGUES[league].name} ({LEAGUES[league].country}) {g.season.iloc[0]}")
        print(g[["rank", "team", "rating", "played"]].to_string(index=False))


def main(argv=None):
    # Team names include č/ć/đ/ã; Windows consoles default to a codepage that can't print them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="football_ml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    league_arg = dict(nargs="+", default=list(LEAGUES), choices=list(LEAGUES), metavar="LEAGUE")

    p = sub.add_parser("status", help="show API plan and request quota"); p.set_defaults(fn=cmd_status)

    p = sub.add_parser("backfill", help="download historical seasons")
    p.add_argument("--leagues", **league_arg)
    p.add_argument("--seasons", nargs="+", type=int, help="explicit season labels (default: last N + current)")
    p.add_argument("-n", type=int, default=4, help="number of completed seasons (default 4: one to warm up ratings, two to train, one to validate)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_backfill)

    p = sub.add_parser("refresh", help="re-download current seasons"); p.add_argument("--leagues", **league_arg)
    p.set_defaults(fn=cmd_refresh)

    p = sub.add_parser("validate", help="tune Elo + evaluate on held-out season, write reports/")
    p.add_argument("--leagues", **league_arg); p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("train", help="fit the production model on all completed seasons")
    p.add_argument("--leagues", **league_arg); p.set_defaults(fn=cmd_train)

    p = sub.add_parser("ratings", help="show current Elo ratings table per league")
    p.add_argument("--leagues", **league_arg); p.set_defaults(fn=cmd_ratings)

    p = sub.add_parser("predict", help="show model probabilities for a day's fixtures")
    p.add_argument("--date"); p.set_defaults(fn=cmd_predict)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
