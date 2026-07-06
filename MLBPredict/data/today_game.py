import requests
import csv
from datetime import date

TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHN",
    "Chicago White Sox": "CHA",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCA",
    "Los Angeles Angels": "ANA",
    "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYN",
    "New York Yankees": "NYA",
    "Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDN",
    "San Francisco Giants": "SFN",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "SLN",
    "Tampa Bay Rays": "TBA",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WAS"
}

today = date.today().strftime("%Y-%m-%d")

url = "https://statsapi.mlb.com/api/v1/schedule"
params = {
    "sportId": 1,
    "date": today
}

response = requests.get(url, params=params)
data = response.json()

rows = []

for game_date in data["dates"]:
    game_date_str = game_date["date"]

    for game in game_date["games"]:
        home = TEAM_MAP[game["teams"]["home"]["team"]["name"]]
        away = TEAM_MAP[game["teams"]["away"]["team"]["name"]]

        rows.append([game_date_str, home, away])

with open("upcoming_games.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "home_team", "visitor_team"])
    writer.writerows(rows)

print("CSV created successfully!")