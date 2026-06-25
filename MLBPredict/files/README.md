# MLB Game Predictor (Winner + Totals)

Predicts (1) which team wins and (2) whether the final combined score goes
OVER or UNDER a line, using only historical box-score-level data (no
Statcast, no betting market data, no live odds).

## Inputting real totals lines for upcoming games

`predict.py` predicts an expected total-runs **number** for each
upcoming game (a regression, not a fixed over/under call). To get an
actual OVER/UNDER prediction, add the real sportsbook total to the
`total_line` column in `data/upcoming_games.csv`:

```csv
date,home_team,visitor_team,total_line
2026-06-24,NYA,BOS,8.5
2026-06-24,LAN,SFN,7.5
```

The script compares its own `model_expected_total` against your
`total_line` and reports OVER if the model's number is higher, UNDER if
lower, plus the size of that gap (`edge_vs_line`). If you leave
`total_line` blank for a game, you'll still see the model's expected
total, but `predicted_side` will show "N/A" — there's no legitimate
OVER/UNDER call without a real number to compare against.

Note this whole project has no live odds feed, so you'll need to type
in the current line yourself from wherever you're getting it (a
sportsbook, an odds aggregator site, etc.) each time you run this.

## Important: realistic accuracy expectations

**This will not reliably hit 60% on either target, and you should be
suspicious of any model that claims to.** Here's why, stated plainly so
you can sanity-check the backtest output yourself instead of taking it on
faith:

- MLB moneylines are heavily efficient markets. Public, reproducible
  models built on box-score-level features (the kind of data available
  here) typically land **54-57%** on winner prediction. Models claiming
  60%+ sustained over a full season are almost always either (a) using
  data not available before game time (leakage), (b) backtesting on a
  small/cherry-picked sample, or (c) including market odds as a feature
  (which works, but isn't a model "predicting" anything — it's reading
  the market).
- For totals, there is no public, free source of historical *closing*
  totals lines in this pipeline. The code generates a **synthetic line**
  (rolling average total for that matchup context) to grade against.
  This is explicitly a proxy, not a real sportsbook line — expect
  **52-55%** against it, and expect that number to NOT transfer directly
  to "beating the actual Vegas total," because the real line is set by
  sharper inputs (current bullpen usage, weather, lineup news) that
  aren't in this dataset.
- The backtest in `src/backtest.py` is **time-ordered** (train on past,
  predict on future) specifically so it can't leak future information
  into past predictions. If you modify it, preserve that property or the
  numbers become meaningless.

If you want to push toward genuinely higher accuracy, the highest-leverage
additions, in order, are: (1) real betting market odds as a feature for
the totals model, (2) Statcast-level pitcher/batter quality metrics
(xERA, barrel%, etc.) via `pybaseball`, (3) daily lineup and bullpen
fatigue data. The code is structured so these can be added as new columns
in the feature builder without restructuring anything.

## Verifying the pipeline runs (optional, no real data needed)

`src/_dev_make_synthetic_data.py` generates fake game logs with the
correct schema and injected (fake) team-strength signal, purely so you
can confirm the code runs end-to-end before pointing it at real data:

```bash
python src/_dev_make_synthetic_data.py   # writes data/raw/GL2022-2024.csv (FAKE data)
python src/backtest.py --model logistic   # confirms pipeline runs, numbers are meaningless
rm data/raw/GL202*.csv                    # remove fake data before using real data
```

**Any accuracy number from this synthetic data is meaningless for real
prediction** — it only proves the code doesn't crash and can detect
signal when signal is artificially guaranteed to exist. Delete the fake
CSVs before running the real pipeline, or `download_data.py`'s
already-downloaded check will skip re-fetching real data for those years.

## Setup

```bash
pip install -r requirements.txt
```

Requires internet access (this sandbox does not have it — run this on
your own machine).

## Usage

```bash
# 1. Download historical game logs (from Retrosheet directly, with a
#    GitHub-mirror fallback if Retrosheet's site is unreachable)
python src/download_data.py --start-year 2015 --end-year 2024

# 2. Run the time-ordered backtest -- this builds features internally
#    and reports honest, out-of-sample accuracy for both models
python src/backtest.py --model xgb --folds 6

# 3. Train final models on all available data and predict upcoming
#    games. Edit data/upcoming_games.csv first with the real matchups
#    you want predictions for (date,home_team,visitor_team columns,
#    3-letter Retrosheet team codes e.g. NYA, BOS, LAN).
python src/predict.py --model xgb --upcoming-csv data/upcoming_games.csv
```

If `xgboost` isn't installed or fails on your machine, pass
`--model logistic` to either command to use a logistic-regression
fallback that needs no extra native dependencies.

## Project structure

```
data/raw/                       # downloaded Retrosheet game logs (.csv per year)
data/upcoming_games.csv         # edit this with real matchups before running predict.py
src/download_data.py            # downloads historical game logs (Retrosheet + GitHub mirror)
src/features.py                 # feature engineering (rolling team/pitcher stats, no leakage)
src/models.py                   # model definitions for winner + totals classifiers
src/backtest.py                 # time-ordered walk-forward backtest, reports honest accuracy
src/predict.py                  # trains final models, predicts upcoming games
src/_dev_make_synthetic_data.py # dev-only: fake data to sanity-check the pipeline runs
models/                          # saved trained models (.pkl), created by predict.py
output/                          # backtest reports, prediction CSVs
```
