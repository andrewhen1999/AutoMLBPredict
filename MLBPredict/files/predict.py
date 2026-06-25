"""
Trains final models on ALL available historical data, then generates
predictions for upcoming games.

LIMITATION: this pipeline has no live data feed. "Upcoming game"
predictions require you to supply the matchup (teams + probable
starters) yourself in data/upcoming_games.csv, since there's no
connected schedule/odds API here. The model will compute rolling
features for each team as of the most recent data it has, which is
only as current as your last download_data.py run.

TOTALS: the totals model predicts an expected total-runs NUMBER, not
over/under a fixed line. To get an OVER/UNDER call, put the real
sportsbook line for each game in the `total_line` column of your
upcoming-games CSV (e.g. 8.5). If you leave it blank, the script will
still show the model's expected total, but won't make an over/under
call since there's nothing real to compare it to.

Usage:
    python predict.py --upcoming-csv data/upcoming_games.csv
"""
import argparse
import glob
import os
import sys

import joblib
import pandas as pd

import subprocess

sys.path.insert(0, os.path.dirname(__file__))
from features import load_raw_gamelogs, build_game_features  # noqa: E402
from models import (  # noqa: E402
    make_winner_model, make_totals_regressor, prepare_xy,
    WINNER_FEATURE_COLS, TOTALS_FEATURE_COLS,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
UPCOMING_TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "data", "upcoming_games.csv")


def train_final_models(game_df: pd.DataFrame, model_kind: str = "xgb"):
    X_w, y_w, _ = prepare_xy(game_df, WINNER_FEATURE_COLS, "home_win")
    winner_model = make_winner_model(model_kind)
    winner_model.fit(X_w, y_w)

    # Totals: regress directly on total_runs (a number), not a fixed
    # over/under label, so any real line can be applied at predict time.
    X_t, y_t, _ = prepare_xy(game_df, TOTALS_FEATURE_COLS, "total_runs")
    totals_model = make_totals_regressor(model_kind)
    totals_model.fit(X_t, y_t)

    return winner_model, totals_model


def get_latest_team_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """For each team, grab its most recent rolling-feature snapshot
    (as of its last played game) to use as the basis for predicting
    its next game. This is necessarily a few days stale relative to
    "today" unless you've just re-downloaded data through yesterday.
    """
    # Only the engineered rolling/derived features -- NOT every raw
    # home_*/away_* column (which would include raw box-score fields
    # like home_hr that collide in name with engineered home_hr_roll30
    # once prefixes are stripped, and aren't usable as "current state"
    # anyway since they describe a single game, not a trailing trend).
    engineered_suffixes = (
        "_roll10", "_roll30", "_roll81", "_roll5", "_roll15",
        "_rest_days", "_win_streak",
    )
    rows = []
    for col_prefix in ["home", "away"]:
        cols = [
            c for c in game_df.columns
            if c.startswith(f"{col_prefix}_") and c.endswith(engineered_suffixes)
        ]
        team_col = "home_team" if col_prefix == "home" else "visitor_team"
        sub = game_df[["date", team_col] + cols].copy()
        sub = sub.rename(columns={team_col: "team"})
        sub.columns = ["date", "team"] + [c[len(col_prefix) + 1:] for c in cols]
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)
    latest = combined.sort_values("date").groupby("team").tail(1).set_index("team")
    return latest


def predict_upcoming(upcoming_df: pd.DataFrame, latest_features: pd.DataFrame,
                      winner_model, totals_model) -> pd.DataFrame:
    results = []
    for _, game in upcoming_df.iterrows():
        home, away = game["home_team"], game["visitor_team"]
        if home not in latest_features.index or away not in latest_features.index:
            print(f"WARNING: no recent data for {home} or {away}, skipping")
            continue

        home_f = latest_features.loc[home]
        away_f = latest_features.loc[away]

        feat_row = {}
        for w in [10, 30, 81]:
            feat_row[f"rs_diff_roll{w}"] = home_f[f"rs_roll{w}"] - away_f[f"rs_roll{w}"]
            feat_row[f"ra_diff_roll{w}"] = home_f[f"ra_roll{w}"] - away_f[f"ra_roll{w}"]
            feat_row[f"winpct_diff_roll{w}"] = home_f[f"winpct_roll{w}"] - away_f[f"winpct_roll{w}"]
        feat_row["rest_diff"] = 0  # unknown without schedule data; neutral default
        feat_row["home_win_streak"] = home_f["win_streak"]
        feat_row["away_win_streak"] = away_f["win_streak"]
        for w in [5, 15]:
            feat_row[f"home_sp_runs_allowed_roll{w}"] = home_f.get(f"sp_runs_allowed_roll{w}", float("nan"))
            feat_row[f"away_sp_runs_allowed_roll{w}"] = away_f.get(f"sp_runs_allowed_roll{w}", float("nan"))

        winner_X = pd.DataFrame([feat_row])[WINNER_FEATURE_COLS]
        home_win_proba = winner_model.predict_proba(winner_X)[0, 1]

        totals_row = {}
        for w in [10, 30, 81]:
            totals_row[f"home_rs_roll{w}"] = home_f[f"rs_roll{w}"]
            totals_row[f"away_rs_roll{w}"] = away_f[f"rs_roll{w}"]
            totals_row[f"home_ra_roll{w}"] = home_f[f"ra_roll{w}"]
            totals_row[f"away_ra_roll{w}"] = away_f[f"ra_roll{w}"]
        totals_row["home_hr_roll30"] = home_f["hr_roll30"]
        totals_row["away_hr_roll30"] = away_f["hr_roll30"]
        for w in [5, 15]:
            totals_row[f"home_sp_runs_allowed_roll{w}"] = home_f.get(f"sp_runs_allowed_roll{w}", float("nan"))
            totals_row[f"away_sp_runs_allowed_roll{w}"] = away_f.get(f"sp_runs_allowed_roll{w}", float("nan"))

        totals_X = pd.DataFrame([totals_row])[TOTALS_FEATURE_COLS]
        predicted_total = totals_model.predict(totals_X)[0]

        # Real sportsbook line, if the user supplied one for this game.
        real_line = game.get("total_line", None)
        has_real_line = pd.notna(real_line) and str(real_line).strip() != ""

        row = {
            "date": game.get("date", ""),
            "home_team": home, "away_team": away,
            "home_win_prob": round(home_win_proba, 3),
            "predicted_winner": home if home_win_proba >= 0.5 else away,
            "model_expected_total": round(float(predicted_total), 2),
        }

        if has_real_line:
            real_line = float(real_line)
            edge = predicted_total - real_line
            row["sportsbook_total_line"] = real_line
            row["predicted_side"] = "OVER" if edge > 0 else "UNDER"
            row["edge_vs_line"] = round(edge, 2)
        else:
            row["sportsbook_total_line"] = None
            row["predicted_side"] = "N/A (no line given)"
            row["edge_vs_line"] = None

        results.append(row)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument("--model", default="xgb", choices=["xgb", "logistic"])
    parser.add_argument(
        "--upcoming-csv", default=UPCOMING_TEMPLATE,
        help=(
            "CSV with columns: date,home_team,visitor_team,total_line "
            "(3-letter Retrosheet team codes; total_line is the real "
            "sportsbook total for that game, e.g. 8.5 -- leave blank if "
            "you don't have one, the model will still show its own "
            "expected total but won't call OVER/UNDER without a real line)"
        ),
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "GL*.csv")))
    if not paths:
        print(f"No GL*.csv files found in {args.raw_dir}. Run download_data.py first.")
        sys.exit(1)

    print(f"Loading {len(paths)} year(s) of game logs...")
    raw = load_raw_gamelogs(paths)
    game_df = build_game_features(raw)

    print(f"Training final models on {len(game_df)} historical games...")
    winner_model, totals_model = train_final_models(game_df, model_kind=args.model)

    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(winner_model, os.path.join(args.models_dir, "winner_model.pkl"))
    joblib.dump(totals_model, os.path.join(args.models_dir, "totals_model.pkl"))
    print(f"Saved trained models to {args.models_dir}/")

    if not os.path.exists(args.upcoming_csv):
        print(
            f"\nNo upcoming-games file found at {args.upcoming_csv}. "
            f"Create one with columns: date,home_team,visitor_team "
            f"(3-letter Retrosheet team codes, e.g. NYA, BOS, LAN) and "
            f"re-run with --upcoming-csv pointing to it."
        )
        sys.exit(0)

    upcoming_df = pd.read_csv(args.upcoming_csv)
    latest_features = get_latest_team_features(game_df)
    predictions = predict_upcoming(upcoming_df, latest_features, winner_model, totals_model)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "predictions.csv")
    predictions.to_csv(out_path, index=False)
    print(f"\nPredictions:\n{predictions.to_string(index=False)}")
    print(f"\nSaved to {out_path}")
    print(
        "\nNote: 'model_expected_total' is this model's own prediction of "
        "total runs. 'predicted_side' is only meaningful for games where "
        "you supplied a real sportsbook line in the total_line column -- "
        "without a real line there's nothing legitimate to call OVER or "
        "UNDER against. Also remember: this model has not been validated "
        "against real game outcomes in this environment (no internet "
        "access here) -- treat predictions as provisional until you've "
        "run backtest.py on real historical data and confirmed the "
        "accuracy numbers it reports."
    )


if __name__ == "__main__":
    main()
