"""
Generates a synthetic dataset with the SAME schema as real downloaded
Retrosheet game logs, purely so the feature-engineering and modeling
code can be exercised end-to-end in an environment with no internet
access. This is NOT real data and produces NO meaningful accuracy
numbers -- it exists only to catch bugs (shape mismatches, leakage,
crashes) before the user runs the real pipeline on their own machine.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from download_data import GAMELOG_COLUMNS  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

TEAMS = ["NYA", "BOS", "TOR", "TBA", "BAL", "CLE", "CHA", "DET", "KCA",
         "MIN", "HOU", "LAA", "OAK", "SEA", "TEX", "ATL", "MIA", "NYN",
         "PHI", "WAS", "CHN", "CIN", "MIL", "PIT", "SLN", "ARI", "COL",
         "LAN", "SDN", "SFN"]

rng = np.random.default_rng(42)

# Give each team a fixed "true strength" so there's real signal for the
# model to find -- otherwise this synthetic test would just confirm the
# code runs without confirming the model can learn anything at all.
team_strength = {t: rng.normal(0, 0.3) for t in TEAMS}
team_pitching = {t: rng.normal(0, 0.3) for t in TEAMS}


def simulate_season(year: int, n_games: int = 800) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp(f"{year}-04-01")
    pitchers = {t: [f"{t.lower()}_sp{i}" for i in range(1, 6)] for t in TEAMS}

    for i in range(n_games):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        date = start + pd.Timedelta(days=int(i * 0.9))

        home_strength = team_strength[home] - team_pitching[away] + 0.15  # home edge
        away_strength = team_strength[away] - team_pitching[home]

        home_score = max(0, int(rng.poisson(4.3 + home_strength * 3)))
        away_score = max(0, int(rng.poisson(4.0 + away_strength * 3)))

        row = {c: "" for c in GAMELOG_COLUMNS}
        row.update({
            "date": date.strftime("%Y%m%d"),
            "game_num": "0",
            "day_of_week": date.day_name()[:3],
            "visitor_team": away, "visitor_league": "AL",
            "visitor_game_num": str(i),
            "home_team": home, "home_league": "AL",
            "home_game_num": str(i),
            "visitor_score": str(away_score),
            "home_score": str(home_score),
            "game_length_outs": "54",
            "day_night": rng.choice(["D", "N"]),
            "park_id": f"{home}01",
            "attendance": str(rng.integers(10000, 40000)),
            "game_duration_minutes": str(rng.integers(150, 220)),
            "visitor_ab": "33", "visitor_h": str(max(0, away_score + rng.integers(-2, 3))),
            "visitor_b2": "2", "visitor_b3": "0", "visitor_hr": str(rng.integers(0, 3)),
            "visitor_rbi": str(away_score), "visitor_bb": str(rng.integers(1, 5)),
            "visitor_k": str(rng.integers(4, 12)), "visitor_e": str(rng.integers(0, 2)),
            "home_ab": "34", "home_h": str(max(0, home_score + rng.integers(-2, 3))),
            "home_b2": "2", "home_b3": "0", "home_hr": str(rng.integers(0, 3)),
            "home_rbi": str(home_score), "home_bb": str(rng.integers(1, 5)),
            "home_k": str(rng.integers(4, 12)), "home_e": str(rng.integers(0, 2)),
            "visitor_sp_id": rng.choice(pitchers[away]),
            "home_sp_id": rng.choice(pitchers[home]),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for year in [2022, 2023, 2024]:
        df = simulate_season(year)
        out_path = os.path.join(OUT_DIR, f"GL{year}.csv")
        df.to_csv(out_path, index=False)
        print(f"wrote synthetic {out_path} ({len(df)} games)")


if __name__ == "__main__":
    main()
