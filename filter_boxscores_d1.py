import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"

CLEAN_GAMES_PATH = RAW_DATA_DIR / "full_season_games.csv"
SOURCE_BOXSCORES_PATH = RAW_DATA_DIR / "team_boxscores.csv"
OUTPUT_PATH = RAW_DATA_DIR / "team_boxscores_d1.csv"

games = pd.read_csv(CLEAN_GAMES_PATH)

# skip bad first line: "team_boxscores"
boxscores = pd.read_csv(SOURCE_BOXSCORES_PATH, skiprows=1)

games.columns = [str(c).strip() for c in games.columns]
boxscores.columns = [str(c).strip() for c in boxscores.columns]

games["game_id"] = games["game_id"].astype(str).str.strip()
boxscores["game_id"] = boxscores["game_id"].astype(str).str.strip()

valid_game_ids = set(games["game_id"].dropna())

filtered = boxscores[boxscores["game_id"].isin(valid_game_ids)].copy()

print("Original boxscore rows:", len(boxscores))
print("Filtered D1 boxscore rows:", len(filtered))
print("Filtered D1 games:", filtered["game_id"].nunique())
print("Filtered teams:", filtered["team"].nunique())

filtered.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)