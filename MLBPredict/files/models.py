"""
Model definitions for the two prediction tasks:
  1. Winner classifier: predict home_win (binary)
  2. Totals: two options --
       (a) classifier against a synthetic line (used in backtest.py to
           sanity-check the pipeline finds real scoring signal)
       (b) regressor predicting expected total runs directly, which is
           the one actually useful for predict.py, since it lets you
           compare against ANY real sportsbook line at prediction time
           instead of being locked to whatever line the model was
           trained against.

Both use gradient-boosted trees, which handle the kind of tabular,
moderately-sized, mixed-scale features here well without much tuning.
Logistic regression / linear regression baselines are included because
for a near-coin-flip problem like this, a simpler model is sometimes
just as good and far less prone to overfitting noise into "signal."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


WINNER_FEATURE_COLS = [
    "rs_diff_roll10", "rs_diff_roll30", "rs_diff_roll81",
    "ra_diff_roll10", "ra_diff_roll30", "ra_diff_roll81",
    "winpct_diff_roll10", "winpct_diff_roll30", "winpct_diff_roll81",
    "rest_diff",
    "home_win_streak", "away_win_streak",
    "home_sp_runs_allowed_roll5", "away_sp_runs_allowed_roll5",
    "home_sp_runs_allowed_roll15", "away_sp_runs_allowed_roll15",
]

TOTALS_FEATURE_COLS = [
    "home_rs_roll10", "home_rs_roll30", "home_rs_roll81",
    "away_rs_roll10", "away_rs_roll30", "away_rs_roll81",
    "home_ra_roll10", "home_ra_roll30", "home_ra_roll81",
    "away_ra_roll10", "away_ra_roll30", "away_ra_roll81",
    "home_hr_roll30", "away_hr_roll30",
    "home_sp_runs_allowed_roll5", "away_sp_runs_allowed_roll5",
    "home_sp_runs_allowed_roll15", "away_sp_runs_allowed_roll15",
]


def make_winner_model(kind: str = "xgb"):
    if kind == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.5)),
        ])
    if not HAS_XGB:
        raise ImportError(
            "xgboost not installed. Run: pip install xgboost --break-system-packages "
            "or use kind='logistic' instead."
        )
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=42,
    )


def make_totals_model(kind: str = "xgb"):
    # Classifier against a synthetic line -- used only in backtest.py
    # to validate the pipeline detects real scoring-environment signal.
    # NOT used for predict.py anymore; see make_totals_regressor.
    if kind == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.5)),
        ])
    if not HAS_XGB:
        raise ImportError(
            "xgboost not installed. Run: pip install xgboost --break-system-packages "
            "or use kind='logistic' instead."
        )
    return XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=42,
    )


def make_totals_regressor(kind: str = "xgb"):
    """Predicts expected total_runs directly (a number), not over/under
    a fixed line. This is what predict.py uses, because it lets you
    compare the prediction against ANY real sportsbook total you supply
    at prediction time, rather than being locked to a line baked in at
    training time.
    """
    if kind == "logistic":  # kept as a name for CLI consistency with --model
        return Pipeline([
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=5.0)),
        ])
    if not HAS_XGB:
        raise ImportError(
            "xgboost not installed. Run: pip install xgboost --break-system-packages "
            "or use kind='logistic' instead."
        )
    return XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        random_state=42,
    )


def prepare_xy(game_df: pd.DataFrame, feature_cols: list[str], label_col: str):
    """Drop rows with missing features/label (early-season warmup rows
    where rolling windows haven't filled yet) and return X, y.
    """
    cols_present = [c for c in feature_cols if c in game_df.columns]
    missing = set(feature_cols) - set(cols_present)
    if missing:
        print(f"WARNING: feature columns not found, skipping: {missing}")

    subset = game_df.dropna(subset=cols_present + [label_col]).copy()
    X = subset[cols_present]
    y = subset[label_col]
    return X, y, subset
