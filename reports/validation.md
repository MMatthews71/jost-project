# Model validation (held-out season)

Generated 2026-10-13T03:58:25+00:00. Elo params: `{'k': 20.0, 'home_adv': 40.0, 'carry': 0.9, 'new_team_pct': 20.0, 'base': 1500.0}`.

Each league's most recent completed season is held out. Elo ratings are updated online (only pre-kickoff information is used for every prediction); the logistic-regression layer and the Elo parameters are fitted on earlier seasons only. The first season of each league is used purely to warm up ratings. The production model (`models/model.json`) is re-tuned and refitted on all completed matches, including the holdout season, so its Elo settings can differ slightly from the ones above.

Training matches: 6932. Test matches: 3465.

## Overall

| | Accuracy | Log loss | Brier | RPS |
|---|---|---|---|---|
| **Model** | 0.500 | 1.0062 | 0.6011 | 0.2050 |
| Base rates (per-league H/D/A frequencies) | 0.443 | 1.0728 | 0.6485 | 0.2276 |
| Always home | 0.443 | | | |

Accuracy is how often the model's most likely outcome happened. Around 50% is a solid result for 1X2 football predictions: about a quarter of matches are draws, and a draw is rarely the single most likely outcome, so they are hard to call. Log loss, Brier and RPS score the probabilities themselves; lower is better.

## Per league

| League | Holdout | n | Accuracy | Log loss | Base-rate log loss | RPS |
|---|---|---|---|---|---|---|
| Brasileirão (Brazil) | 2025 | 380 | 0.500 | 0.9985 | 1.0401 | 0.2029 |
| Bundesliga (Germany) | 2025 | 306 | 0.542 | 0.9684 | 1.0719 | 0.1951 |
| Championship (England) | 2025 | 552 | 0.451 | 1.0615 | 1.0839 | 0.2232 |
| Premier League (England) | 2025 | 380 | 0.476 | 1.0335 | 1.0852 | 0.2110 |
| Eredivisie (Netherlands) | 2025 | 306 | 0.497 | 0.9809 | 1.0716 | 0.1964 |
| HNL (Croatia) | 2025 | 180 | 0.511 | 0.9866 | 1.0667 | 0.1986 |
| La Liga (Spain) | 2025 | 380 | 0.518 | 0.9812 | 1.0509 | 0.2006 |
| Ligue 1 (France) | 2025 | 305 | 0.505 | 1.0003 | 1.0645 | 0.2082 |
| Serie A (Italy) | 2025 | 380 | 0.511 | 0.9989 | 1.0906 | 0.2023 |
| Serbian SuperLiga (Serbia) | 2025 | 296 | 0.527 | 1.0027 | 1.0976 | 0.1945 |

## Calibration

Predicted probability vs how often the outcome actually happened, in 10% buckets (well calibrated = the two columns are close).

| Outcome | Bucket | n | Predicted | Observed |
|---|---|---|---|---|
| H | 0.0-0.1 | 51 | 0.078 | 0.078 |
| H | 0.1-0.2 | 250 | 0.157 | 0.200 |
| H | 0.2-0.3 | 469 | 0.254 | 0.247 |
| H | 0.3-0.4 | 736 | 0.353 | 0.385 |
| H | 0.4-0.5 | 737 | 0.447 | 0.442 |
| H | 0.5-0.6 | 567 | 0.549 | 0.517 |
| H | 0.6-0.7 | 382 | 0.648 | 0.657 |
| H | 0.7-0.8 | 197 | 0.743 | 0.751 |
| H | 0.8-0.9 | 71 | 0.839 | 0.859 |
| D | 0.0-0.1 | 20 | 0.087 | 0.250 |
| D | 0.1-0.2 | 375 | 0.168 | 0.152 |
| D | 0.2-0.3 | 2709 | 0.259 | 0.271 |
| D | 0.3-0.4 | 361 | 0.312 | 0.296 |
| A | 0.0-0.1 | 262 | 0.066 | 0.080 |
| A | 0.1-0.2 | 709 | 0.153 | 0.171 |
| A | 0.2-0.3 | 879 | 0.249 | 0.235 |
| A | 0.3-0.4 | 724 | 0.347 | 0.345 |
| A | 0.4-0.5 | 416 | 0.445 | 0.382 |
| A | 0.5-0.6 | 268 | 0.545 | 0.507 |
| A | 0.6-0.7 | 149 | 0.646 | 0.617 |
| A | 0.7-0.8 | 44 | 0.741 | 0.705 |

Buckets with fewer than 20 matches are omitted here (see validation.json).
