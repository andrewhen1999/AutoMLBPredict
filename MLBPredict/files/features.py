"""
Feature engineering for MLB game prediction.

Core principle: every feature for a game must be computable using only
information available BEFORE that game starts (no leakage). Rolling
stats are computed as of the day before the game, using a trailing
window, and the very first games of a season/window naturally get
fewer observations (handled via min_periods, not by padding with
future data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Rolling windows for team-form features (in games)
ROLLING_WINDOWS = [10, 30, 81]  # ~last 10, last month-ish, half season

# Starting pitcher rolling window (in starts)
PITCHER_ROLLING_STARTS = [5, 15]


def load_raw_gamelogs(paths: list[str]) -> pd.DataFrame:
    """Load and concatenate one or more raw GLyyyy.csv files."""
    frames = [pd.read_csv(p, dtype=str) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    numeric_cols = [
        "visitor_score", "home_score", "attendance",
        "game_duration_minutes",
        "visitor_ab", "visitor_h", "visitor_b2", "visitor_b3",
        "visitor_hr", "visitor_rbi", "visitor_bb", "visitor_k",
        "visitor_e",
        "home_ab", "home_h", "home_b2", "home_b3", "home_hr",
        "home_rbi", "home_bb", "home_k", "home_e",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df["game_id"] = (
        df["date"].dt.strftime("%Y%m%d")
        + "_" + df["home_team"] + "_" + df["visitor_team"]
        + "_" + df["game_num"].astype(str)
    )
    return df


def _team_game_view(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape game-level (one row per game) into team-level (two rows
    per game: one for home, one for visitor), which makes rolling
    per-team stats straightforward to compute with groupby + shift.
    """
    home = pd.DataFrame({
        "game_id": df["game_id"],
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["visitor_team"],
        "is_home": 1,
        "runs_scored": df["home_score"],
        "runs_allowed": df["visitor_score"],
        "hits": df["home_h"],
        "hr": df["home_hr"],
        "bb": df["home_bb"],
        "k": df["home_k"],
        "errors": df["home_e"],
        "sp_id": df["home_sp_id"],
        "park_id": df["park_id"],
    })
    visitor = pd.DataFrame({
        "game_id": df["game_id"],
        "date": df["date"],
        "team": df["visitor_team"],
        "opponent": df["home_team"],
        "is_home": 0,
        "runs_scored": df["visitor_score"],
        "runs_allowed": df["home_score"],
        "hits": df["visitor_h"],
        "hr": df["visitor_hr"],
        "bb": df["visitor_bb"],
        "k": df["visitor_k"],
        "errors": df["visitor_e"],
        "sp_id": df["visitor_sp_id"],
        "park_id": df["park_id"],
    })
    team_games = pd.concat([home, visitor], ignore_index=True)
    team_games["won"] = (team_games["runs_scored"] > team_games["runs_allowed"]).astype(int)
    team_games = team_games.sort_values(["team", "date"]).reset_index(drop=True)
    return team_games


def _add_rolling_team_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add trailing rolling-average features per team, shifted by one
    game so the features for game N only use games strictly before N
    (no leakage of the current game's own result into its features).
    """
    g = team_games.groupby("team", group_keys=False)

    for window in ROLLING_WINDOWS:
        for col, alias in [
            ("runs_scored", "rs"), ("runs_allowed", "ra"),
            ("won", "winpct"), ("hr", "hr"), ("bb", "bb"), ("k", "k"),
            ("errors", "err"),
        ]:
            shifted = g[col].apply(lambda s: s.shift(1))
            team_games[f"{alias}_roll{window}"] = (
                shifted.groupby(team_games["team"])
                .rolling(window, min_periods=3)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # Rest days: days since this team's previous game
    team_games["prev_date"] = g["date"].shift(1)
    team_games["rest_days"] = (
        team_games["date"] - team_games["prev_date"]
    ).dt.days
    team_games["rest_days"] = team_games["rest_days"].clip(upper=10)

    # Current win streak / loss streak going into this game (sign = direction)
    # NOTE: this uses a Python-level loop per team via groupby().apply(),
    # which is correct but not vectorized. Fine for single-digit years of
    # data; if you run this across 50+ years of full Retrosheet history
    # (~100k+ team-games) and it's slow, that's this function -- the fix
    # would be a vectorized run-length encoding, not a logic change.
    def _streak(s: pd.Series) -> pd.Series:
        # +n means n consecutive wins coming in, -n means n consecutive losses
        prev = s.shift(1)
        streak = np.zeros(len(s))
        cur = 0
        for i, val in enumerate(prev):
            if pd.isna(val):
                cur = 0
            elif val == 1:
                cur = cur + 1 if cur >= 0 else 1
            else:
                cur = cur - 1 if cur <= 0 else -1
            streak[i] = cur
        return pd.Series(streak, index=s.index)

    team_games["win_streak"] = g["won"].apply(_streak)

    return team_games


def _add_pitcher_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Starting pitcher rolling run-support-allowed proxy. Without
    play-by-play pitching lines (ER, IP per start) reliably isolated in
    the game-log format, the best signal available at this granularity
    is: when this pitcher's team has started him before, how many runs
    did the opponent score in those games? This is a team-level proxy
    for pitcher quality, not a true per-pitcher ERA, and should be
    treated as a weaker feature than the team-form rolling stats.
    """
    g = team_games.groupby("sp_id", group_keys=False)
    for window in PITCHER_ROLLING_STARTS:
        shifted = g["runs_allowed"].apply(lambda s: s.shift(1))
        team_games[f"sp_runs_allowed_roll{window}"] = (
            shifted.groupby(team_games["sp_id"])
            .rolling(window, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return team_games


def build_team_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    team_games = _team_game_view(raw_df)
    team_games = _add_rolling_team_features(team_games)
    team_games = _add_pitcher_features(team_games)
    return team_games


def build_game_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Merge home/visitor team-level rolling features back onto the
    original one-row-per-game table, producing the final model-ready
    feature table with a `home_win` label and a `total_runs` label.
    """
    team_feats = build_team_feature_table(raw_df)

    feature_cols = [c for c in team_feats.columns if (
        c.endswith(tuple(f"roll{w}" for w in ROLLING_WINDOWS))
        or c.endswith(tuple(f"roll{w}" for w in PITCHER_ROLLING_STARTS))
        or c in ("rest_days", "win_streak")
    )]

    home_feats = team_feats[team_feats["is_home"] == 1][
        ["game_id"] + feature_cols
    ].add_prefix("home_")
    home_feats = home_feats.rename(columns={"home_game_id": "game_id"})

    away_feats = team_feats[team_feats["is_home"] == 0][
        ["game_id"] + feature_cols
    ].add_prefix("away_")
    away_feats = away_feats.rename(columns={"away_game_id": "game_id"})

    game_df = raw_df.merge(home_feats, on="game_id", how="left")
    game_df = game_df.merge(away_feats, on="game_id", how="left")

    # Labels
    game_df["home_win"] = (game_df["home_score"] > game_df["visitor_score"]).astype(int)
    game_df["total_runs"] = game_df["home_score"] + game_df["visitor_score"]

    # Differential features (often more predictive than raw values)
    for window in ROLLING_WINDOWS:
        game_df[f"rs_diff_roll{window}"] = (
            game_df[f"home_rs_roll{window}"] - game_df[f"away_rs_roll{window}"]
        )
        game_df[f"ra_diff_roll{window}"] = (
            game_df[f"home_ra_roll{window}"] - game_df[f"away_ra_roll{window}"]
        )
        game_df[f"winpct_diff_roll{window}"] = (
            game_df[f"home_winpct_roll{window}"] - game_df[f"away_winpct_roll{window}"]
        )

    game_df["rest_diff"] = game_df["home_rest_days"] - game_df["away_rest_days"]

    return game_df


def make_synthetic_total_line(game_df: pd.DataFrame, window: int = 81) -> pd.Series:
    """Build a synthetic O/U line as the trailing-average combined
    expected total for this matchup context, since no real historical
    closing-line data is available in this pipeline.

    THIS IS A PROXY, NOT A REAL SPORTSBOOK LINE. Accuracy against this
    proxy does NOT mean the model would beat actual Vegas totals --
    real lines incorporate sharper, more current information (weather,
    bullpen fatigue, lineup news) than season-trailing averages can
    capture. Treat this number as a way to validate that the model
    finds real signal in run-scoring environment, not as a betting
    backtest.
    """
    expected_home_rs = game_df[f"home_rs_roll{window}"]
    expected_away_rs = game_df[f"away_rs_roll{window}"]
    expected_home_ra = game_df[f"home_ra_roll{window}"]
    expected_away_ra = game_df[f"away_ra_roll{window}"]
    # Average of each team's scoring rate and their opponent's allowed
    # rate, blended, as a simple expected-total estimate.
    line = (
        (expected_home_rs + expected_away_ra) / 2
        + (expected_away_rs + expected_home_ra) / 2
    ) / 2 * 2  # average expected runs for each side, summed
    return line.round(1)
