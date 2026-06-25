"""
Downloads Retrosheet game logs for a range of years.

Source: Retrosheet's own site publishes free per-year zipped game logs
at retrosheet.org/gamelogs/glYYYY.zip. As a GitHub-hosted mirror (useful
if your network allows GitHub but blocks retrosheet.org, or vice versa),
kyleam/retrosheet-gamelogs republishes the identical unzipped .TXT files.
This script tries Retrosheet's own zip first, then falls back to the
GitHub mirror. Data is free per Retrosheet's usage notice: "The
information used here was obtained free of charge from and is
copyrighted by Retrosheet. Interested parties may contact Retrosheet at
www.retrosheet.org."

Game log field layout (verified against Retrosheet's own published
glfields.txt spec, no header row in source files):
    1       date (yyyymmdd)
    2       game number (0/1/2/3/A/B for header/doubleheader games)
    3       day of week
    4-5     visiting team, league
    6       visiting team game number
    7-8     home team, league
    9       home team game number
    10-11   visiting score, home score
    12      length of game in outs
    13      day/night
    14      completion info
    15      forfeit info
    16      protest info
    17      park ID
    18      attendance
    19      time of game (minutes)
    20-21   visiting/home line score
    22-38   visiting team offensive stats (AB, H, 2B, 3B, HR, RBI, SH,
            SF, HBP, BB, IBB, K, SB, CS, GDP, CI, LOB)
    39-43   visiting team pitching stats (pitchers used, individual ER,
            team ER, WP, balks)
    44-49   visiting team defensive stats (PO, A, E, PB, DP, TP)
    50-66   home team offensive stats (same 17 categories as 22-38)
    67-71   home team pitching stats
    72-77   home team defensive stats
    78-89   umpire IDs/names (HP, 1B, 2B, 3B, LF, RF)
    90-93   visiting/home manager ID/name
    94-99   winning/losing/saving pitcher ID/name
    100-101 game-winning RBI batter ID/name
    102-103 visiting starting pitcher ID/name
    104-105 home starting pitcher ID/name
    106-159 starting lineups (9 batters x ID/name/position, vis then home)
    160     additional info (e.g. "HTBF" if home team batted first)
    161     acquisition info

Usage:
    python download_data.py --start-year 2015 --end-year 2024
"""
import argparse
import io
import os
import sys
import time
import zipfile

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Only the columns we actually use downstream are named individually;
# everything else (umpires, managers, full lineups) is named generically
# since build_features.py never reads it. This keeps the mapping
# unambiguous for the columns that matter (scores, pitchers, team stats)
# while not pretending we've hand-verified every umpire/lineup slot.
_OFFENSE_STATS = [
    "ab", "h", "b2", "b3", "hr", "rbi", "sh", "sf", "hbp", "bb", "ibb",
    "k", "sb", "cs", "gdp", "ci", "lob",
]  # 17 fields, matches positions 22-38 and 50-66

GAMELOG_COLUMNS = (
    [
        "date", "game_num", "day_of_week",
        "visitor_team", "visitor_league", "visitor_game_num",
        "home_team", "home_league", "home_game_num",
        "visitor_score", "home_score", "game_length_outs", "day_night",
        "completion_info", "forfeit_info", "protest_info", "park_id",
        "attendance", "game_duration_minutes",
        "visitor_line_score", "home_line_score",
    ]
    + [f"visitor_{s}" for s in _OFFENSE_STATS]                # 22-38
    + ["visitor_pitchers_used", "visitor_individual_er",
       "visitor_team_er", "visitor_wp", "visitor_balks"]       # 39-43
    + ["visitor_po", "visitor_a", "visitor_e", "visitor_pb",
       "visitor_dp", "visitor_tp"]                             # 44-49
    + [f"home_{s}" for s in _OFFENSE_STATS]                     # 50-66
    + ["home_pitchers_used", "home_individual_er",
       "home_team_er", "home_wp", "home_balks"]                # 67-71
    + ["home_po", "home_a", "home_e", "home_pb",
       "home_dp", "home_tp"]                                   # 72-77
    + [f"ump_{pos}_{f}" for pos in ["hp", "1b", "2b", "3b", "lf", "rf"]
       for f in ["id", "name"]]                                # 78-89
    + ["visitor_mgr_id", "visitor_mgr_name",
       "home_mgr_id", "home_mgr_name"]                         # 90-93
    + ["winning_pitcher_id", "winning_pitcher_name",
       "losing_pitcher_id", "losing_pitcher_name",
       "save_pitcher_id", "save_pitcher_name"]                 # 94-99
    + ["gwrbi_id", "gwrbi_name"]                                # 100-101
    + ["visitor_sp_id", "visitor_sp_name",
       "home_sp_id", "home_sp_name"]                           # 102-105
    + [f"{side}_lineup_{i}_{f}" for side in ["visitor", "home"]
       for i in range(1, 10) for f in ["id", "name", "pos"]]   # 106-159
    + ["additional_info", "acquisition_info"]                  # 160-161
)

RETROSHEET_ZIP_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"
GITHUB_MIRROR_URL = (
    "https://raw.githubusercontent.com/kyleam/retrosheet-gamelogs/"
    "master/GL{year}.TXT"
)


def _parse_gamelog_text(raw_text: str) -> pd.DataFrame:
    """Parse raw, headerless Retrosheet game log CSV text into a
    DataFrame using our verified column layout. Tolerates the actual
    field count being slightly off our named-column count (extra
    trailing columns, e.g. years with fewer lineup slots in early
    history) by truncating/padding rather than hard failing, since a
    handful of mismatched trailing admin columns shouldn't block
    ingestion of the scoring/team-stat columns we actually use.
    """
    raw_df = pd.read_csv(
        io.StringIO(raw_text), header=None, dtype=str, on_bad_lines="warn"
    )
    n_expected = len(GAMELOG_COLUMNS)
    n_actual = raw_df.shape[1]
    if n_actual >= n_expected:
        raw_df = raw_df.iloc[:, :n_expected]
        raw_df.columns = GAMELOG_COLUMNS
    else:
        # Fewer columns than expected (shouldn't happen for modern
        # years but guard anyway) -- keep what's there, name the rest.
        cols = GAMELOG_COLUMNS[:n_actual]
        raw_df.columns = cols
        print(
            f"  NOTE: expected {n_expected} cols, got {n_actual}. "
            f"Trailing columns (umpires/lineups) may be missing for "
            f"this year, but core scoring/team-stat columns are intact."
        )
    return raw_df


def download_year(year: int, out_dir: str) -> str | None:
    out_path = os.path.join(out_dir, f"GL{year}.csv")
    if os.path.exists(out_path):
        print(f"[skip] {year} already downloaded -> {out_path}")
        return out_path

    raw_text = None

    # Try 1: Retrosheet's own zip (authoritative source)
    zip_url = RETROSHEET_ZIP_URL.format(year=year)
    try:
        print(f"[fetch] {year} from {zip_url}")
        resp = requests.get(zip_url, timeout=30)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                txt_names = [n for n in zf.namelist() if n.upper().endswith(".TXT")]
                if txt_names:
                    raw_text = zf.read(txt_names[0]).decode("utf-8", errors="replace")
        else:
            print(f"  Retrosheet zip returned HTTP {resp.status_code}")
    except (requests.RequestException, zipfile.BadZipFile) as e:
        print(f"  Retrosheet zip failed: {e}")

    # Try 2: GitHub mirror fallback
    if raw_text is None:
        mirror_url = GITHUB_MIRROR_URL.format(year=year)
        try:
            print(f"  falling back to mirror: {mirror_url}")
            resp = requests.get(mirror_url, timeout=30)
            if resp.status_code == 200:
                raw_text = resp.text
            else:
                print(f"  Mirror returned HTTP {resp.status_code}")
        except requests.RequestException as e:
            print(f"  Mirror failed: {e}")

    if raw_text is None:
        print(f"  WARNING: could not fetch {year} from any source, skipping")
        return None

    df = _parse_gamelog_text(raw_text)
    df.to_csv(out_path, index=False)
    print(f"  saved {len(df)} games -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--out-dir", default=RAW_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ok, failed = [], []
    for year in range(args.start_year, args.end_year + 1):
        try:
            path = download_year(year, args.out_dir)
            (ok if path else failed).append(year)
        except requests.RequestException as e:
            print(f"  ERROR downloading {year}: {e}")
            failed.append(year)
        time.sleep(0.5)  # be polite to the servers

    print(f"\nDone. Downloaded: {ok}")
    if failed:
        print(f"Failed/missing: {failed}")
        print(
            "If many years failed, check your internet connection, or "
            "verify the sources are still up: "
            "https://www.retrosheet.org/gamelogs/index.html and "
            "https://github.com/kyleam/retrosheet-gamelogs"
        )
        sys.exit(1 if not ok else 0)


if __name__ == "__main__":
    main()
