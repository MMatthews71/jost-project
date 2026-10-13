# vow-model

VOW football prediction model. It predicts match outcomes (home win / draw / away win) for
10 leagues, picks the day's best tips every morning (1 free + 3–5 VIP) and sends them to
vowtips.com. Once the matches finish, it reports the results back.

| League | Country | API-Football id |
|---|---|---|
| Premier League | England | 39 |
| Championship | England | 40 |
| La Liga | Spain | 140 |
| Serie A | Italy | 135 |
| Bundesliga | Germany | 78 |
| Ligue 1 | France | 61 |
| Eredivisie | Netherlands | 88 |
| Serbian SuperLiga | Serbia | 286 |
| HNL | Croatia | 210 |
| Brasileirão | Brazil | 71 |

## How it works

1. **Data**: every fixture from 2022 to the current season is pulled from API-Football and
   stored in `data/fixtures/`, one CSV per league-season. European seasons are labelled by the
   year they start (2025 = 2025/26). Brasileirão seasons follow the calendar year. Promotion and
   relegation play-offs and finals at neutral grounds are left out of the ratings. Serbia's
   championship and relegation groups stay in, because those are real league matches.
2. **Elo ratings** (`football_ml/elo.py`): after every finished match the winner gains rating
   points and the loser loses the same amount. Bigger wins move the ratings more. Each league has
   its own rating pool, so a promoted team starts near the bottom of its new league instead of
   bringing over a rating from the league below. Between seasons, ratings move 10% of the way back
   toward the league average. Ratings are rebuilt from the full match history on every run, so
   each new result is included automatically. `python -m football_ml ratings` prints the current
   tables.
3. **Calibration** (`football_ml/model.py`): a multinomial logistic regression turns the
   rating difference into P(home), P(draw), P(away). It learns a separate home advantage and draw
   rate for each league, and draws become more likely the closer the two teams are.
4. **Daily picks** (`football_ml/picks.py`): for each of today's matches the model's
   probabilities are compared with the median 1X2 price across bookmakers.
   - A selection is a **value pick** when `model probability × odds − 1 ≥ 0`, i.e. the model rates
     it more likely than the price implies (`ev_percent` in the payload).
   - Only prices between **1.40 and 3.50** are considered. Long shots are where an Elo-only model
     is least reliable, and the bookmakers' prices are more accurate than the model there.
   - Value picks are ranked by **model probability**, most likely winners first. The top 5 become
     VIP picks and the next one is the free pick. On a thin day with fewer than 3 value picks, the
     VIP list is topped up to 3 with the most likely remaining selections, so the free pick can
     occasionally have a negative `ev_percent`.
   - All of these are settings (see [Configuration](#configuration)).
5. **Results** (`football_ml/results.py`): sent picks are settled on the 90-minute score (win /
   loss, or void if a match is postponed, cancelled or abandoned). Each result is sent with the
   closing price, which is the last pre-match median price for the selection.
6. **History**: every pick and result sent is added to `data/picks.jsonl` / `data/results.jsonl`.
   These files are only ever added to, never edited. Every day's odds and model probabilities for
   *all* matches are also kept in `data/odds/`. No historical odds data exists, so once a few weeks
   have built up, this is the data for tuning the pick settings on real prices.

> No ROI guarantee: there is no historical odds data to backtest against, so the model is
> evaluated on prediction accuracy and calibration, not on proven profit.

## Validation results (held-out 2025 season, 3,465 matches)

| | Accuracy | Log loss | Brier | RPS |
|---|---|---|---|---|
| **Model** | **50.0%** | **1.006** | **0.601** | **0.205** |
| Baseline: each league's historical home/draw/away rates | 44.3% | 1.073 | 0.649 | 0.228 |
| Baseline: always pick the home team | 44.3% | | | |

The model's probabilities beat the baseline (lower log loss) in all 10 leagues, and its accuracy
is higher in 9 of 10. In Brazil it matches the baseline (50.0% vs 50.3%), because home teams
win half the time there. The probabilities are well calibrated: matches where the model gave an
outcome 60–70% ended that way 62–66% of the time. The full report is in
[`reports/validation.md`](reports/validation.md).

*Accuracy* is how often the model's most likely outcome happened. For football 1X2 around 50%
is a solid result, because draws happen about a quarter of the time but are rarely the most
likely single outcome. *Log loss*, *Brier* and *RPS* score the probabilities themselves; lower is
better.

## Automation (GitHub Actions)

The scheduler is GitHub Actions, free for this usage. Everything runs in this repository.

| Workflow | When (UTC) | What |
|---|---|---|
| `daily` | 04:30 | Settle results, refresh fixtures, send today's picks. That is 06:30 Belgrade time in summer and 05:30 in winter. |
| `daily` | 05:15 | Backup run in case GitHub starts the first one late. Does nothing if picks were already sent. |
| `daily` | 22:30, 03:30 | Results only (late matches, incl. Brazilian evening kickoffs) |
| `retrain` | Mondays 04:00 | Re-tune and refit the model, regenerate the validation report |
| `tests` | every push | Run the test suite |

After each run the bot commits the updated pick history and odds snapshots back to the repo.

### Required repository secrets

Settings → Secrets and variables → Actions → **Secrets**:

| Secret | Purpose |
|---|---|
| `API_FOOTBALL_KEY` | API-Football key (Pro plan) |
| `PICKS_API_KEY` | vowtips.com API key, sent as the `x-api-key` header |

Settings → Actions → General → **Workflow permissions** must be set to **Read and write**, so the
bot can save the pick history.

### Test mode

The site stores picks marked `"test": true` but never shows them to members. There are two ways
to turn test mode on:
- **For one run**: Actions → `daily` → Run workflow, then tick `test_mode`.
- **For every run**: Settings → Secrets and variables → Actions → **Variables**, then add
  `PICKS_TEST_MODE` = `true`. Delete the variable to go live.

### Running a job by hand

Actions → `daily` → **Run workflow** has these options:
- `job`: `picks`, `results` or `both`.
- `dry_run`: print what would be sent, without sending anything.
- `test_mode`: send picks marked as test.
- `force`: resend today's picks. The site upserts by match, market and selection, so resending is
  safe.

### When something goes wrong

A failed run shows a red ✗ in the Actions tab, and GitHub emails the repo owner. Open the run to
see the log. Common causes:

| Log message | Fix |
|---|---|
| `POST picks unauthorised (401)` | The `PICKS_API_KEY` secret is wrong or missing |
| `site reported problems with picks: rejected …` | The site refused some picks; the log shows its errors. The picks are still recorded. |
| `API_FOOTBALL_KEY is not set` / API errors | Check the `API_FOOTBALL_KEY` secret and the API-Football subscription |
| `no fixtures with odds in range` | Not an error: no matches that day |

A failed results run is retried automatically by the next run. Picks that weren't sent can be
sent again with **Run workflow** (job `picks`).

## Configuration

These are optional environment variables. To change them for scheduled runs, add each one as a
repository **variable** and add it to the `env:` block in `.github/workflows/daily.yml`. For
local runs, put them in `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `VIP_MIN` / `VIP_MAX` | 3 / 5 | Number of VIP picks per day |
| `MIN_EV` | 0.0 | Minimum expected value for a value pick (0.05 = +5%) |
| `MIN_ODDS` / `MAX_ODDS` | 1.40 / 3.50 | Price range picks are chosen from |
| `RANK_BY` | `probability` | `probability` (most likely first) or `ev` (highest expected value first) |
| `UNITS` | 1 | Stake sent with each pick |
| `ODDS_BOOKMAKER_ID` | (median of all) | Price picks at one bookmaker instead, e.g. `8` = Bet365 |
| `PICKS_TIMEZONE` | Europe/Belgrade | Which calendar day counts as "today" |

The JSON sent to the site is built in one place,
[`football_ml/delivery.py`](football_ml/delivery.py). That's where to look if the site's API
changes. `analysis_text` is sent as `null`. `key_stats` contains the two teams' Elo ratings, the
model's three probabilities, the market's probability for the pick and the number of bookmakers.

## Running locally

Requires Python 3.11+.

```bash
pip install -r requirements.txt
cp .env.example .env              # put API_FOOTBALL_KEY (and PICKS_API_KEY to send) in .env
python -m football_ml status      # check API plan and remaining requests
python -m football_ml backfill    # 2022 to current season, all leagues (~50 requests)
python -m football_ml refresh     # re-download current seasons (new results, upcoming fixtures)
python -m football_ml validate    # tune Elo + evaluate on held-out season -> reports/
python -m football_ml train       # fit the production model -> models/model.json
python -m football_ml ratings     # current Elo ratings table for every league
python -m football_ml predict --date 2026-10-17   # probabilities for that day's fixtures
python -m football_ml picks --dry-run             # today's picks, printed instead of sent
python -m football_ml results --dry-run           # settle sent picks, printed instead of sent
python -m pytest -q               # tests, incl. end-to-end against a mock vowtips endpoint
```

The repository already includes the data, the trained model and the validation report, so
`ratings`, `predict` and `picks --dry-run` work straight after cloning.

## API usage

A daily run uses about 10 API-Football requests to refresh fixtures, 10–20 for odds and a few for
results and closing odds. That's well within the Pro plan's 7,500 a day.

## Project layout

```
football_ml/
  config.py     leagues, season rules, endpoints, pick settings
  api.py        API-Football client (retries, paging, rate limiting)
  data.py       fixture download, CSV storage, play-off filtering
  elo.py        Elo rating engine
  model.py      logistic-regression calibration layer
  train.py      Elo tuning, held-out validation, final training
  predict.py    current ratings tables, probabilities for upcoming fixtures
  odds.py       1X2 odds aggregation across bookmakers
  picks.py      daily pick selection + odds snapshots
  results.py    settlement + closing odds
  delivery.py   vowtips.com payloads + HTTPS POST
  journal.py    append-only pick/result history
data/
  fixtures/     match data, one CSV per league-season
  odds/         daily odds + model probabilities for every match
  picks.jsonl   every pick sent
  results.jsonl every result sent
models/         trained model (JSON)
reports/        validation report
tests/          unit + end-to-end tests
.github/workflows/  daily, retrain, tests
```
