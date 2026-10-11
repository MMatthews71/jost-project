from datetime import date

import numpy as np
import pytest

from football_ml import config
from football_ml.data import finished
from football_ml.elo import EloEngine, EloParams, run_elo
from football_ml.model import OutcomeModel
from football_ml.train import holdout_seasons, metrics, rate, validate

from .conftest import synthetic_fixtures


def test_current_season_rules():
    epl, bra = config.LEAGUES["EPL"], config.LEAGUES["BRA"]
    assert config.current_season(epl, date(2026, 10, 6)) == 2026
    assert config.current_season(epl, date(2027, 3, 1)) == 2026
    assert config.current_season(bra, date(2027, 3, 1)) == 2027


def test_elo_update_is_zero_sum_and_directional():
    e = EloEngine(EloParams(k=20, home_adv=0))
    e.update("EPL", 1, 2, 3, 0)
    assert e.rating("EPL", 1) > 1500 > e.rating("EPL", 2)
    assert e.rating("EPL", 1) + e.rating("EPL", 2) == pytest.approx(3000)


def test_season_break_regresses_to_mean_and_newcomers_start_low():
    e = EloEngine(EloParams(carry=0.5))
    e.start_season("EPL", 2023)
    pool = e._pool("EPL")
    pool.update({i: 1400 + 20 * i for i in range(10)})
    e.start_season("EPL", 2024)
    assert pool[9] == pytest.approx(1490 + 0.5 * 90)
    assert e.rating("EPL", 99) < np.mean([pool[i] for i in range(10)])


def test_pre_match_ratings_do_not_use_the_match_itself():
    fx = finished(synthetic_fixtures(seasons=(2023,)))
    rated, _ = run_elo(fx)
    first = rated.iloc[0]
    assert first.elo_home == first.elo_away == 1500


def test_model_beats_base_rates_on_synthetic_data(tmp_paths):
    fx = synthetic_fixtures()
    rep = validate(fx, EloParams(), today=date(2025, 10, 1))
    assert rep["holdout_seasons"] == {"EPL": 2024, "SRB": 2024}
    assert rep["overall"]["model"]["log_loss"] < rep["overall"]["base_rate"]["log_loss"]


def test_model_probabilities_sum_to_one_and_roundtrip(tmp_paths):
    rated = rate(synthetic_fixtures(), EloParams())
    m = OutcomeModel.fit(rated[~rated.burn_in], EloParams())
    p = m.predict([1700, 1500], [1500, 1700], ["EPL", "EPL"])
    assert np.allclose(p.sum(1), 1)
    assert p[0, 0] > p[1, 0]           # stronger home side -> higher home-win probability
    m.save()
    m2 = OutcomeModel.load()
    assert np.allclose(m2.predict([1600], [1550], ["SRB"]), m.predict([1600], [1550], ["SRB"]))


def test_metrics_perfect_prediction():
    p = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    m = metrics(p, ["H", "D"])
    assert m["accuracy"] == 1 and m["brier"] == 0 and m["rps"] == 0
