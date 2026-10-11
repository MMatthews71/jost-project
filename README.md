# vow-model

VOW football prediction model: match-outcome probabilities (home win / draw / away win) for
10 leagues, built from Elo team ratings and a logistic-regression calibration layer.

This is **Milestone 1: Data Pipeline & Prediction Model**. Daily picks, results reporting and
the automated scheduler come in Milestone 2.

| League | Country | API-Football id |
|---|---|---|
| Premier League | England | 39 |
| Championship | England | 40 |
| La Liga | Spain | 140 |
| Serie A | Italy | 135 |
| Bundesliga | Germany | 78 |
| Ligue 1 | France | 61 |
| Eredivisie | Netherlands | 88 |
| Super Liga | Serbia | 286 |
| HNL | Croatia | 210 |
| Serie A (Brasileirão) | Brazil | 71 |

## How it works

1. **Data**: every fixture from 2022 to the current season is pulled from API-Football and
   stored in `data/fixtures/`, one CSV per league-season. European seasons are labelled by the
   year they start (2025 = 2025/26). Brasileirão seasons follow the calendar year (2026 = Jan–Dec
   2026). Promotion/relegation play-offs and finals at neutral grounds are left out of the
   ratings. Serbia's championship and relegation groups stay in, because those are real league
   matches.
2. **Elo ratings** (`football_ml/elo.py`): after every finished match the winner gains rating
   points and the loser loses the same amount. Bigger wins move the ratings more. Each league has
   its own rating pool, so a promoted team starts near the bottom of its new league instead of
   bringing over a rating from the league below. Between seasons, ratings move 10% of the way back
   toward the league average. Ratings are rebuilt from the full match history on every run, so each
   new result is included automatically once the data is refreshed.
3. **Calibration** (`football_ml/model.py`): a multinomial logistic regression turns the
   rating difference into P(home), P(draw), P(away). It learns a separate home advantage and draw
   rate for each league, and draws become more likely the closer the two teams are. The fitted
   model is saved as readable JSON in `models/model.json`.
4. **Validation** (`football_ml/train.py`): each league's most recent completed season (2025)
   is held out and predicted using only information available before each kickoff. The Elo
   settings and the regression are fitted on earlier seasons only, and 2022 is used purely to warm
   up the ratings.

## Validation results (held-out 2025 season, 3,465 matches)

| | Accuracy | Log loss | Brier | RPS |
|---|---|---|---|---|
| **Model** | **50.0%** | **1.006** | **0.601** | **0.205** |
| Baseline: each league's historical home/draw/away rates | 44.3% | 1.073 | 0.649 | 0.228 |
| Baseline: always pick the home team | 44.3% | | | |

The model's probabilities beat the baseline (lower log loss) in all 10 leagues, and its accuracy
is higher in 9 of 10. In Brazil it matches the baseline (50.0% vs 50.3%), because home teams
win half the time there. The probabilities are well calibrated: matches where the model gave an
outcome 60–70% ended that way 62–66% of the time. The full report, with
per-league results and the calibration table, is in
[`reports/validation.md`](reports/validation.md).

*Accuracy* is how often the model's most likely outcome happened. For football 1X2 around 50%
is a solid result, because draws happen about a quarter of the time but are rarely the most
likely single outcome. *Log loss*, *Brier* and *RPS* score the probabilities themselves; lower is
better.

## Running it

Requires Python 3.11+ and an API-Football key.

```bash
pip install -r requirements.txt
cp .env.example .env              # put your APIFOOTBALL_KEY in .env
python -m football_ml status      # check plan and remaining requests
python -m football_ml backfill -n 4   # 2022 to current season, all leagues (~50 requests)
python -m football_ml refresh     # re-download current seasons (new results, upcoming fixtures)
python -m football_ml validate    # tune Elo + evaluate on held-out season -> reports/
python -m football_ml train       # fit the production model -> models/model.json
python -m football_ml predict --date 2026-10-17   # probabilities for that day's fixtures
python -m pytest -q               # tests
```

The repository already includes the downloaded data, the trained model and the validation
report. `predict` works straight after cloning, and `refresh` + `train` bring it up to date.

## Project layout

```
football_ml/
  config.py     leagues, season rules, paths
  api.py        API-Football client (retries, paging, rate limiting)
  data.py       fixture download, CSV storage, play-off filtering
  elo.py        Elo rating engine
  model.py      logistic-regression calibration layer
  train.py      Elo tuning, held-out validation, final training
  predict.py    probabilities for upcoming fixtures
data/fixtures/  match data, one CSV per league-season
models/         trained model (JSON)
reports/        validation report
tests/          unit tests
```
