"""Logistic-regression calibration layer: Elo difference -> P(home win, draw, away win).

Features: the rating difference, its absolute value (draws get likelier as teams get closer),
and per-league one-hot terms so each league gets its own home advantage and draw rate.
The fitted model is saved as plain JSON so it can be inspected and does not depend on the
scikit-learn version used at prediction time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config
from .config import LEAGUES
from .elo import EloParams

CLASSES = ["H", "D", "A"]


def features(elo_home, elo_away, leagues, league_list: list[str]) -> np.ndarray:
    d = (np.asarray(elo_home, float) - np.asarray(elo_away, float)) / 100.0
    onehot = np.array([[lg == L for L in league_list] for lg in leagues], dtype=float)
    return np.column_stack([d, np.abs(d), onehot])


@dataclass
class OutcomeModel:
    coef: np.ndarray            # (3, n_features), rows ordered as CLASSES
    intercept: np.ndarray       # (3,)
    leagues: list[str]
    elo_params: EloParams = field(default_factory=EloParams)
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, matches: pd.DataFrame, elo_params: EloParams, C: float = 1.0, meta=None) -> "OutcomeModel":
        leagues = sorted(LEAGUES)
        X = features(matches.elo_home, matches.elo_away, matches.league, leagues)
        lr = LogisticRegression(C=C, max_iter=2000)
        lr.fit(X, matches.result)
        order = [list(lr.classes_).index(c) for c in CLASSES]
        return cls(lr.coef_[order], lr.intercept_[order], leagues, elo_params,
                   {"n_train": int(len(matches)), "C": C, **(meta or {})})

    def predict(self, elo_home, elo_away, leagues) -> np.ndarray:
        X = features(np.atleast_1d(elo_home), np.atleast_1d(elo_away), np.atleast_1d(leagues), self.leagues)
        z = X @ self.coef.T + self.intercept
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.predict(df.elo_home.values, df.elo_away.values, df.league.values)
        return df.assign(p_home=p[:, 0], p_draw=p[:, 1], p_away=p[:, 2])

    def save(self, path=None) -> None:
        path = path or config.MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "classes": CLASSES,
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "leagues": self.leagues,
            "elo_params": self.elo_params.to_dict(),
            "meta": self.meta,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path=None) -> "OutcomeModel":
        path = path or config.MODEL_PATH
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(np.array(d["coef"]), np.array(d["intercept"]), d["leagues"],
                   EloParams(**d["elo_params"]), d.get("meta", {}))
