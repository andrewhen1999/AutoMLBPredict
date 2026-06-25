"""
Time-ordered backtest for both the winner model and the totals model.

CRITICAL DESIGN PROPERTY: this backtest NEVER trains on a game and
evaluates on a game that happened before it. We use an expanding-window
walk-forward split: train on all games up to date T, evaluate on the
next chunk of games after T, then advance T and repeat. This is the
only way to get an honest read on what "accuracy" would have meant if
you'd actually been using this model in real time, rather than letting
the model implicitly learn from the future (e.g. via global rolling
averages computed over the whole dataset, which the feature pipeline
in features.py deliberately avoids by shifting before rolling).

If you modify this file, preserve the property that for every
evaluated game, the model was fit ONLY on games with an earlier date.
Breaking this is the single most common way these projects produce
fake, inflated accuracy numbers.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from features import (  # noqa: E402
    load_raw_gamelogs, build_game_features, make_synthetic_total_line,
)
from models import (  # noqa: E402
    make_winner_model, make_totals_model, prepare_xy,
    WINNER_FEATURE_COLS, TOTALS_FEATURE_COLS,
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def walk_forward_splits(dates: pd.Series, n_folds: int = 6, min_train_frac: float = 0.4):
    """Yield (train_mask, test_mask) pairs using expanding windows over
    time. The first min_train_frac of the date range is always reserved
    purely as training data so the first fold isn't trained on almost
    nothing.
    """
    unique_dates = pd.Series(dates.unique()).sort_values().to_numpy()
    unique_dates = pd.to_datetime(unique_dates)
    n = len(unique_dates)
    start_idx = int(n * min_train_frac)
    fold_edges = np.linspace(start_idx, n - 1, n_folds + 1).astype(int)

    for i in range(n_folds):
        train_cutoff_date = unique_dates[fold_edges[i]]
        test_end_date = unique_dates[min(fold_edges[i + 1], n - 1)]
        train_mask = dates <= train_cutoff_date
        test_mask = (dates > train_cutoff_date) & (dates <= test_end_date)
        if test_mask.sum() == 0:
            continue
        yield train_mask, test_mask, train_cutoff_date, test_end_date


def run_backtest(game_df: pd.DataFrame, model_kind: str = "xgb", n_folds: int = 6):
    results = {"winner": [], "totals": []}

    # ---- Winner model ----
    X_all, y_all, subset_all = prepare_xy(game_df, WINNER_FEATURE_COLS, "home_win")
    dates = subset_all["date"]

    for train_mask, test_mask, cutoff, test_end in walk_forward_splits(dates, n_folds):
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        if len(X_test) < 20 or y_train.nunique() < 2:
            continue

        model = make_winner_model(model_kind)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan")
        ll = log_loss(y_test, proba, labels=[0, 1])

        # Baseline: always predict the home team wins (the naive
        # "home field advantage" heuristic) -- the model should beat this.
        baseline_acc = accuracy_score(y_test, np.ones(len(y_test)))

        results["winner"].append({
            "fold_end": str(test_end.date()), "n_test": len(X_test),
            "accuracy": acc, "auc": auc, "log_loss": ll,
            "home_win_baseline_acc": baseline_acc,
        })

    # ---- Totals model ----
    game_df = game_df.copy()
    game_df["synthetic_line"] = make_synthetic_total_line(game_df)
    game_df["over"] = (game_df["total_runs"] > game_df["synthetic_line"]).astype(int)

    X_all_t, y_all_t, subset_t = prepare_xy(game_df, TOTALS_FEATURE_COLS, "over")
    dates_t = subset_t["date"]

    for train_mask, test_mask, cutoff, test_end in walk_forward_splits(dates_t, n_folds):
        X_train, y_train = X_all_t[train_mask], y_all_t[train_mask]
        X_test, y_test = X_all_t[test_mask], y_all_t[test_mask]
        if len(X_test) < 20 or y_train.nunique() < 2:
            continue

        model = make_totals_model(model_kind)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan")
        ll = log_loss(y_test, proba, labels=[0, 1])
        baseline_acc = max(y_test.mean(), 1 - y_test.mean())  # always-majority baseline

        results["totals"].append({
            "fold_end": str(test_end.date()), "n_test": len(X_test),
            "accuracy": acc, "auc": auc, "log_loss": ll,
            "majority_baseline_acc": baseline_acc,
        })

    return results


def summarize(results: dict) -> pd.DataFrame:
    rows = []
    for task, folds in results.items():
        if not folds:
            continue
        df = pd.DataFrame(folds)
        baseline_col = "home_win_baseline_acc" if task == "winner" else "majority_baseline_acc"
        rows.append({
            "task": task,
            "n_folds": len(df),
            "total_test_games": df["n_test"].sum(),
            "mean_accuracy": df["accuracy"].mean(),
            "weighted_accuracy": (df["accuracy"] * df["n_test"]).sum() / df["n_test"].sum(),
            "mean_auc": df["auc"].mean(),
            "mean_log_loss": df["log_loss"].mean(),
            "mean_baseline_accuracy": df[baseline_col].mean(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--model", default="xgb", choices=["xgb", "logistic"])
    parser.add_argument("--folds", type=int, default=6)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "GL*.csv")))
    if not paths:
        print(f"No GL*.csv files found in {args.raw_dir}. Run download_data.py first.")
        sys.exit(1)

    print(f"Loading {len(paths)} year(s) of game logs...")
    raw = load_raw_gamelogs(paths)
    print(f"Loaded {len(raw)} games. Building features...")
    game_df = build_game_features(raw)

    print(f"Running {args.folds}-fold walk-forward backtest with model='{args.model}'...")
    results = run_backtest(game_df, model_kind=args.model, n_folds=args.folds)

    summary = summarize(results)
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (honest, time-ordered, out-of-sample)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()
    print(
        "Context: random guessing = 50%. Always-predict-home (winner) or "
        "always-predict-majority-class (totals) are naive baselines shown "
        "above -- the model should beat them by a meaningful margin, not "
        "just match them. If mean_accuracy is barely above the baseline, "
        "the model isn't adding much over the simplest possible heuristic."
    )

    for task, folds in results.items():
        if folds:
            pd.DataFrame(folds).to_csv(
                os.path.join(args.out_dir, f"backtest_{task}_folds.csv"), index=False
            )
    summary.to_csv(os.path.join(args.out_dir, "backtest_summary.csv"), index=False)
    print(f"\nDetailed fold-by-fold results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
