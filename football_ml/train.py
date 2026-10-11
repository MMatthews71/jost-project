"""Elo tuning, held-out-season validation and final model training."""
from __future__ import annotations

import itertools
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .config import LEAGUES, current_season
from .data import finished, load_fixtures
from .elo import EloParams, run_elo
from .model import CLASSES, OutcomeModel

GRID = {
    "k": [15.0, 20.0, 25.0, 30.0],
    "home_adv": [40.0, 60.0, 80.0],
    "carry": [0.6, 0.75, 0.9],
}


def rate(fixtures: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    """Finished matches with pre-match Elo; the first season of each league is flagged as burn-in."""
    rated, _ = run_elo(finished(fixtures), params)
    first = rated.groupby("league")["season"].transform("min")
    rated["burn_in"] = rated["season"] == first
    return rated


def holdout_seasons(fixtures: pd.DataFrame, today: date | None = None) -> dict[str, int]:
    """Per league: the most recent fully completed season available."""
    out = {}
    for key, g in fixtures.groupby("league"):
        cur = current_season(LEAGUES[key], today)
        done = sorted(s for s in g["season"].unique() if s < cur)
        if len(done) >= 2:   # need at least a burn-in season before it
            out[key] = int(done[-1])
    return out


def onehot(y) -> np.ndarray:
    y = np.asarray(y)
    return np.column_stack([(y == c).astype(float) for c in CLASSES])


def metrics(p: np.ndarray, y) -> dict:
    Y = onehot(y)
    eps = 1e-12
    cum_p, cum_y = np.cumsum(p, axis=1)[:, :2], np.cumsum(Y, axis=1)[:, :2]
    return {
        "n": int(len(Y)),
        "accuracy": float((p.argmax(1) == Y.argmax(1)).mean()),
        "log_loss": float(-np.mean(np.log(np.clip((p * Y).sum(1), eps, 1)))),
        "brier": float(np.mean(((p - Y) ** 2).sum(1))),
        "rps": float(np.mean(((cum_p - cum_y) ** 2).sum(1) / 2)),
    }


def calibration(p: np.ndarray, y, bins=10) -> list[dict]:
    Y = onehot(y)
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for ci, c in enumerate(CLASSES):
        idx = np.clip(np.digitize(p[:, ci], edges) - 1, 0, bins - 1)
        for b in range(bins):
            m = idx == b
            if m.sum():
                rows.append({"outcome": c, "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                             "predicted": float(p[m, ci].mean()), "observed": float(Y[m, ci].mean())})
    return rows


def split(rated: pd.DataFrame, holdout: dict[str, int]):
    hs = rated["league"].map(holdout)
    test = rated[rated["season"] == hs]
    train = rated[(rated["season"] < hs) & ~rated["burn_in"]]
    return train, test


def tune(fixtures: pd.DataFrame, holdout: dict[str, int] | None = None, verbose=True) -> tuple[EloParams, list]:
    """Grid-search Elo parameters by log loss on the *training* seasons only (holdout untouched)."""
    results = []
    for k, ha, carry in itertools.product(GRID["k"], GRID["home_adv"], GRID["carry"]):
        params = EloParams(k=k, home_adv=ha, carry=carry)
        rated = rate(fixtures, params)
        train = split(rated, holdout)[0] if holdout else rated[~rated["burn_in"]]
        model = OutcomeModel.fit(train, params)
        ll = metrics(model.predict_df(train)[["p_home", "p_draw", "p_away"]].values, train.result)["log_loss"]
        results.append({**params.to_dict(), "train_log_loss": ll})
        if verbose:
            print(f"  k={k:<4} home_adv={ha:<4} carry={carry:<4} log_loss={ll:.4f}")
    best = min(results, key=lambda r: r["train_log_loss"])
    return EloParams(**{k: v for k, v in best.items() if k != "train_log_loss"}), results


def validate(fixtures: pd.DataFrame, params: EloParams | None = None, today: date | None = None) -> dict:
    holdout = holdout_seasons(fixtures, today)
    if not holdout:
        raise RuntimeError("Need >= 2 completed seasons per league to validate (burn-in + holdout).")
    if params is None:
        params, _ = tune(fixtures, holdout)
    rated = rate(fixtures, params)
    train, test = split(rated, holdout)
    no_train = sorted(set(test.league) - set(train.league))
    model = OutcomeModel.fit(train, params)
    p = model.predict_df(test)[["p_home", "p_draw", "p_away"]].values

    # Baselines: per-league outcome frequencies from training data; always pick home.
    rates = train.groupby("league")["result"].value_counts(normalize=True).unstack().reindex(columns=CLASSES).fillna(0)
    overall = train["result"].value_counts(normalize=True).reindex(CLASSES).fillna(0).values
    base = np.array([rates.loc[lg].values if lg in rates.index else overall for lg in test.league])

    per_league = {}
    for lg, idx in test.groupby("league").indices.items():
        per_league[lg] = {"season": holdout[lg], "model": metrics(p[idx], test.result.values[idx]),
                          "base_rate": metrics(base[idx], test.result.values[idx])}

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elo_params": params.to_dict(),
        "holdout_seasons": holdout,
        "leagues_without_training_data": no_train,
        "n_train": int(len(train)),
        "overall": {
            "model": metrics(p, test.result),
            "base_rate": metrics(base, test.result),
            "always_home_accuracy": float((test.result == "H").mean()),
        },
        "per_league": per_league,
        "calibration": calibration(p, test.result),
        "outcome_rates_test": test.result.value_counts(normalize=True).reindex(CLASSES).fillna(0).to_dict(),
    }


def write_report(rep: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "validation.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    o = rep["overall"]
    lines = [
        "# Model validation (held-out season)",
        "",
        f"Generated {rep['generated']}. Elo params: `{rep['elo_params']}`.",
        "",
        "Each league's most recent completed season is held out. Elo ratings are updated online "
        "(only pre-kickoff information is used for every prediction); the logistic-regression layer "
        "and the Elo parameters are fitted on earlier seasons only. The first season of each league "
        "is used purely to warm up ratings.",
        "",
        f"Training matches: {rep['n_train']}. Test matches: {o['model']['n']}.",
        "",
        "## Overall",
        "",
        "| | Accuracy | Log loss | Brier | RPS |",
        "|---|---|---|---|---|",
        f"| **Model** | {o['model']['accuracy']:.3f} | {o['model']['log_loss']:.4f} | {o['model']['brier']:.4f} | {o['model']['rps']:.4f} |",
        f"| Base rates (per-league H/D/A frequencies) | {o['base_rate']['accuracy']:.3f} | {o['base_rate']['log_loss']:.4f} | {o['base_rate']['brier']:.4f} | {o['base_rate']['rps']:.4f} |",
        f"| Always home | {o['always_home_accuracy']:.3f} | | | |",
        "",
        "Accuracy is how often the model's most likely outcome happened. Around 50% is a solid result "
        "for 1X2 football predictions: about a quarter of matches are draws, and a draw is rarely the "
        "single most likely outcome, so they are hard to call. Log loss, Brier and RPS score the "
        "probabilities themselves; lower is better.",
        "",
        "## Per league",
        "",
        "| League | Holdout | n | Accuracy | Log loss | Base-rate log loss | RPS |",
        "|---|---|---|---|---|---|---|",
    ]
    for lg, r in sorted(rep["per_league"].items()):
        m, b = r["model"], r["base_rate"]
        lines.append(f"| {LEAGUES[lg].name} ({LEAGUES[lg].country}) | {r['season']} | {m['n']} | "
                     f"{m['accuracy']:.3f} | {m['log_loss']:.4f} | {b['log_loss']:.4f} | {m['rps']:.4f} |")
    lines += ["", "## Calibration", "",
              "Predicted probability vs how often the outcome actually happened, in 10% buckets "
              "(well calibrated = the two columns are close).", "",
              "| Outcome | Bucket | n | Predicted | Observed |", "|---|---|---|---|---|"]
    for c in rep["calibration"]:
        if c["n"] >= 20:
            lines.append(f"| {c['outcome']} | {c['bin']} | {c['n']} | {c['predicted']:.3f} | {c['observed']:.3f} |")
    lines += ["", "Buckets with fewer than 20 matches are omitted here (see validation.json)."]
    if rep["leagues_without_training_data"]:
        lines += ["", f"Leagues tested without any training season: {rep['leagues_without_training_data']}."]
    (config.REPORTS_DIR / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_final(fixtures: pd.DataFrame | None = None, params: EloParams | None = None) -> OutcomeModel:
    """Fit the production model on every completed non-burn-in match."""
    fixtures = load_fixtures() if fixtures is None else fixtures
    if params is None:
        params, _ = tune(fixtures, verbose=False)
    rated = rate(fixtures, params)
    train = rated[~rated["burn_in"]]
    model = OutcomeModel.fit(train, params, meta={
        "trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_match": str(rated["kickoff_utc"].max()),
    })
    model.save()
    return model
