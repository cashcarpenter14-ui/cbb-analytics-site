import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"

GAMES_PATH = RAW_DATA_DIR / "full_season_games.csv"
BOXSCORES_PATH = RAW_DATA_DIR / "team_boxscores.csv"
ELO_PATH = RAW_DATA_DIR / "elo_ratings_d1.csv"
OUTPUT_PATH = RAW_DATA_DIR / "team_boxscores_d1.csv"

TEAM_NAME_MAP = {
    "Queens Royals": "Queens University Royals",
}

games = pd.read_csv(GAMES_PATH)
boxscores = pd.read_csv(BOXSCORES_PATH)
elo = pd.read_csv(ELO_PATH)

games.columns = [str(c).strip() for c in games.columns]
boxscores.columns = [str(c).strip() for c in boxscores.columns]
elo.columns = [str(c).strip() for c in elo.columns]

elo_team_col = "team" if "team" in elo.columns else "Team"

games["game_id"] = games["game_id"].astype(str).str.strip()
games["home_team"] = games["home_team"].astype(str).str.strip().replace(TEAM_NAME_MAP)
games["away_team"] = games["away_team"].astype(str).str.strip().replace(TEAM_NAME_MAP)

boxscores["game_id"] = boxscores["game_id"].astype(str).str.strip()
boxscores["team"] = boxscores["team"].astype(str).str.strip().replace(TEAM_NAME_MAP)

elo[elo_team_col] = elo[elo_team_col].astype(str).str.strip().replace(TEAM_NAME_MAP)

d1_teams = set(elo[elo_team_col].dropna())

games_d1 = games[
    games["home_team"].isin(d1_teams) &
    games["away_team"].isin(d1_teams)
].copy()

valid_game_ids = set(games_d1["game_id"])

filtered = boxscores[boxscores["game_id"].isin(valid_game_ids)].copy()

print("Original boxscore rows:", len(boxscores))
print("Filtered D1 boxscore rows:", len(filtered))
print("Filtered D1 games:", filtered["game_id"].nunique())
print("Filtered teams:", filtered["team"].nunique())

filtered.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)