import pandas as pd
import numpy as np
import os
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"

os.makedirs(DATA_DIR, exist_ok=True)

# LOAD FILES
boxscores = pd.read_csv(RAW_DATA_DIR / "team_boxscores_d1.csv")
games = pd.read_csv(RAW_DATA_DIR / "full_season_games.csv")
elo = pd.read_csv(RAW_DATA_DIR / "elo_ratings_d1.csv")

# CLEAN COLUMN NAMES
boxscores.columns = [str(c).strip() for c in boxscores.columns]
games.columns = [str(c).strip() for c in games.columns]
elo.columns = [str(c).strip() for c in elo.columns]

# CLEAN VALUES
boxscores["team"] = boxscores["team"].astype(str).str.strip()
boxscores["game_id"] = boxscores["game_id"].astype(str).str.strip()
boxscores["points"] = pd.to_numeric(boxscores["points"], errors="coerce")
boxscores["possessions"] = pd.to_numeric(boxscores["possessions"], errors="coerce")

# BUILD OPPONENT TABLE
opp = boxscores[["game_id", "team", "points", "possessions"]].copy()
opp = opp.rename(columns={
    "team": "opponent",
    "points": "opp_points",
    "possessions": "opp_possessions"
})

# BUILD MATCHUPS
game_matchups = boxscores.merge(opp, on="game_id", how="inner")
game_matchups = game_matchups[game_matchups["team"] != game_matchups["opponent"]].copy()

# EFFICIENCIES
game_matchups["off_eff"] = np.where(
    game_matchups["possessions"] > 0,
    game_matchups["points"] / game_matchups["possessions"] * 100,
    np.nan
)

game_matchups["def_eff"] = np.where(
    game_matchups["opp_possessions"] > 0,
    game_matchups["opp_points"] / game_matchups["opp_possessions"] * 100,
    np.nan
)

# TEAM STATS
team_stats = game_matchups.groupby("team", as_index=False).agg({
    "off_eff": "mean",
    "def_eff": "mean",
    "possessions": "mean"
})

team_stats = team_stats.rename(columns={"team": "Team"})

# CLEAN ELO FILE
elo_team_col = None
elo_rating_col = None

for c in elo.columns:
    cl = c.lower()
    if cl in ["team", "school"]:
        elo_team_col = c
    if cl in ["rating", "elo"]:
        elo_rating_col = c

if elo_team_col is None or elo_rating_col is None:
    raise ValueError("elo_ratings_d1.csv must contain a team column and a rating/elo column")

elo = elo[[elo_team_col, elo_rating_col]].copy()
elo.columns = ["Team", "Elo"]
elo["Team"] = elo["Team"].astype(str).str.strip()
elo["Elo"] = pd.to_numeric(elo["Elo"], errors="coerce")

# INCLUDE ALL TEAMS FROM ELO
all_teams = pd.DataFrame({"Team": elo["Team"].dropna().unique()})
team_stats = all_teams.merge(team_stats, on="Team", how="left")

team_stats["off_eff"] = team_stats["off_eff"].fillna(100.0)
team_stats["def_eff"] = team_stats["def_eff"].fillna(100.0)
team_stats["possessions"] = team_stats["possessions"].fillna(67.0)

# MERGE ELO
team_stats = team_stats.merge(elo, on="Team", how="left")
team_stats["Elo"] = team_stats["Elo"].fillna(1500)

# SAVE
team_stats.to_csv(DATA_DIR / "team_stats_current.csv", index=False)

team_rankings = team_stats.sort_values("Elo", ascending=False).reset_index(drop=True)
team_rankings.insert(0, "Rank", range(1, len(team_rankings) + 1))
team_rankings.to_csv(DATA_DIR / "team_rankings.csv", index=False)

metadata = {
    "teams": int(len(team_stats))
}

with open(DATA_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Pipeline complete")
print("Saved:")
print(DATA_DIR / "team_stats_current.csv")
print(DATA_DIR / "team_rankings.csv")
print(DATA_DIR / "model_metadata.json")
