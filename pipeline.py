import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import requests
from datetime import datetime, timedelta

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


def fetch_espn_scoreboard_for_date(date_str):
    url = f"{ESPN_SCOREBOARD_URL}?dates={date_str}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_completed_games_from_scoreboard(scoreboard_json):
    rows = []

    events = scoreboard_json.get("events", [])
    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        team_a = competitors[0]
        team_b = competitors[1]

        home = None
        away = None
        for tm in competitors:
            if tm.get("homeAway") == "home":
                home = tm
            elif tm.get("homeAway") == "away":
                away = tm

        if home is None or away is None:
            continue

        try:
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
        except Exception:
            continue

        game_id = str(event.get("id"))
        game_date = event.get("date", "")[:10]

        home_team = home.get("team", {}).get("displayName", "").strip()
        away_team = away.get("team", {}).get("displayName", "").strip()

        neutral_site = bool(comp.get("neutralSite", False))

        winner = home_team if home_score > away_score else away_team
        loser = away_team if home_score > away_score else home_team

        rows.append({
            "game_id": game_id,
            "game_date": game_date,
            "date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "winner": winner,
            "loser": loser,
            "neutral_site": neutral_site,
            "source": "ESPN"
        })

    return pd.DataFrame(rows)


def fetch_recent_completed_games(days_back=3):
    all_rows = []
    today = datetime.utcnow().date()

    for i in range(days_back + 1):
        d = today - timedelta(days=i)
        date_str = d.strftime("%Y%m%d")
        try:
            scoreboard = fetch_espn_scoreboard_for_date(date_str)
            df = extract_completed_games_from_scoreboard(scoreboard)
            if not df.empty:
                all_rows.append(df)
        except Exception as e:
            print(f"Warning: failed to fetch ESPN data for {date_str}: {e}")

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["game_id"])
    return out

BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"

os.makedirs(DATA_DIR, exist_ok=True)

# LOAD FILES
boxscores = pd.read_csv(RAW_DATA_DIR / "team_boxscores_d1.csv")
games = pd.read_csv(RAW_DATA_DIR / "full_season_games.csv")
games = pd.read_csv(RAW_DATA_DIR / "full_season_games.csv")
# UPDATE FULL SEASON GAMES WITH RECENT ESPN RESULTS
recent_games = fetch_recent_completed_games(days_back=3)

if not recent_games.empty:
    games.columns = [str(c).strip() for c in games.columns]
    recent_games.columns = [str(c).strip() for c in recent_games.columns]

    if "game_id" in games.columns:
        games["game_id"] = games["game_id"].astype(str).str.strip()
    if "game_id" in recent_games.columns:
        recent_games["game_id"] = recent_games["game_id"].astype(str).str.strip()

    # align missing columns
    for col in games.columns:
        if col not in recent_games.columns:
            recent_games[col] = np.nan
    for col in recent_games.columns:
        if col not in games.columns:
            games[col] = np.nan

    recent_games = recent_games[games.columns]
    games = pd.concat([games, recent_games], ignore_index=True)
    games = games.drop_duplicates(subset=["game_id"], keep="last")

    games.to_csv(RAW_DATA_DIR / "full_season_games.csv", index=False)
    print(f"Added/updated {len(recent_games)} recent ESPN games")
else:
    print("No new completed ESPN games found")
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