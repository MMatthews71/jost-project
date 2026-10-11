import numpy as np
import pandas as pd
import pytest

from football_ml import config


def synthetic_fixtures(leagues=("EPL", "SRB"), seasons=(2022, 2023, 2024), n_teams=12, seed=0) -> pd.DataFrame:
    """Double round-robin seasons where outcomes depend on hidden team strength."""
    rng = np.random.default_rng(seed)
    rows, fid = [], 1
    for li, lg in enumerate(leagues):
        strength = rng.normal(0, 0.5, n_teams)
        for season in seasons:
            strength = 0.8 * strength + rng.normal(0, 0.2, n_teams)
            day = pd.Timestamp(f"{season}-08-10", tz="UTC")
            for rnd in range(2):
                for h in range(n_teams):
                    for a in range(n_teams):
                        if h == a or (h + a + rnd) % 2:
                            continue
                        lam_h = np.exp(0.25 + strength[h] - strength[a] * 0.8)
                        lam_a = np.exp(0.05 + strength[a] - strength[h] * 0.8)
                        rows.append({
                            "fixture_id": fid, "league": lg, "season": season,
                            "kickoff_utc": day + pd.Timedelta(hours=fid % 2000), "status": "FT", "round": "Regular Season - 1",
                            "home_id": 1000 * (li + 1) + h, "home_name": f"{lg}-T{h}",
                            "away_id": 1000 * (li + 1) + a, "away_name": f"{lg}-T{a}",
                            "home_goals": rng.poisson(lam_h), "away_goals": rng.poisson(lam_a)})
                        fid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    """Point every storage path at a temp dir."""
    for attr, val in [("FIXTURES_DIR", tmp_path / "data" / "fixtures"),
                      ("MODEL_PATH", tmp_path / "models" / "model.json"),
                      ("REPORTS_DIR", tmp_path / "reports")]:
        monkeypatch.setattr(config, attr, val)
    return tmp_path

