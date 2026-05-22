import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import requests
from datetime import datetime, timedelta

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"

SEASON_WEIGHT = 0.60
LAST5_WEIGHT = 0.25
SITE_WEIGHT = 0.15

TEAM_NAME_MAP = {
    "Queens Royals": "Queens University Royals",
}


def fetch_espn_scoreboard_for_date(date_str):
    url = f"{ESPN_SCOREBOARD_URL}?dates={date_str}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_completed_games_from_scoreboard(scoreboard_json):
    rows = []

    for event in scoreboard_json.get("events", []):
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

        home, away = None, None
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

    return pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["game_id"])


def weighted_metric(season, last5, site):
    return (
        SEASON_WEIGHT * season +
        LAST5_WEIGHT * last5 +
        SITE_WEIGHT * site
    )


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

# UPDATE FULL SEASON GAMES WITH RECENT ESPN RESULTS
recent_games = fetch_recent_completed_games(days_back=3)

if not recent_games.empty:
    recent_games.columns = [str(c).strip() for c in recent_games.columns]

    games["game_id"] = games["game_id"].astype(str).str.strip()
    recent_games["game_id"] = recent_games["game_id"].astype(str).str.strip()

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

# CLEAN BOXSCORE VALUES
boxscores["team"] = boxscores["team"].astype(str).str.strip()
boxscores["team"] = boxscores["team"].replace(TEAM_NAME_MAP)
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

# BUILD TEAM-GAME MATCHUPS
game_matchups = boxscores.merge(opp, on="game_id", how="inner")
game_matchups = game_matchups[game_matchups["team"] != game_matchups["opponent"]].copy()

# CALCULATE GAME-LEVEL EFFICIENCIES
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

# ADD GAME DATE / SITE INFO
date_col = "date" if "date" in games.columns else "game_date"

games_small = games[["game_id", date_col, "home_team", "away_team", "neutral_site"]].copy()
games_small = games_small.rename(columns={date_col: "date"})

games_small["game_id"] = games_small["game_id"].astype(str).str.strip()
games_small["date"] = pd.to_datetime(games_small["date"], errors="coerce")
games_small["home_team"] = games_small["home_team"].astype(str).str.strip()
games_small["away_team"] = games_small["away_team"].astype(str).str.strip()
games_small["neutral_site"] = games_small["neutral_site"].fillna(False).astype(bool)

game_matchups = game_matchups.merge(games_small, on="game_id", how="left")


def get_site(row):
    if bool(row.get("neutral_site", False)):
        return "neutral"
    if row["team"] == row["home_team"]:
        return "home"
    if row["team"] == row["away_team"]:
        return "away"
    return "unknown"


game_matchups["site"] = game_matchups.apply(get_site, axis=1)

# SEASON STATS
season_stats = game_matchups.groupby("team", as_index=False).agg({
    "off_eff": "mean",
    "def_eff": "mean",
    "possessions": "mean"
})

season_stats = season_stats.rename(columns={
    "off_eff": "season_off_eff",
    "def_eff": "season_def_eff",
    "possessions": "season_possessions"
})

# LAST 5 STATS
game_matchups = game_matchups.sort_values(["team", "date"])

last5 = (
    game_matchups
    .dropna(subset=["date"])
    .groupby("team", group_keys=False)
    .tail(5)
)

last5_stats = last5.groupby("team", as_index=False).agg({
    "off_eff": "mean",
    "def_eff": "mean",
    "possessions": "mean"
})

last5_stats = last5_stats.rename(columns={
    "off_eff": "last5_off_eff",
    "def_eff": "last5_def_eff",
    "possessions": "last5_possessions"
})

# SITE STATS
site_stats = game_matchups.groupby(["team", "site"], as_index=False).agg({
    "off_eff": "mean",
    "def_eff": "mean",
    "possessions": "mean"
})

site_pivot = site_stats.pivot(index="team", columns="site")
site_pivot.columns = [f"{site}_{stat}" for stat, site in site_pivot.columns]
site_pivot = site_pivot.reset_index()

# COMBINE STATS
team_stats = season_stats.merge(last5_stats, on="team", how="left")
team_stats = team_stats.merge(site_pivot, on="team", how="left")

# FILL LAST 5 WITH SEASON IF MISSING
for stat in ["off_eff", "def_eff", "possessions"]:
    team_stats[f"last5_{stat}"] = team_stats[f"last5_{stat}"].fillna(
        team_stats[f"season_{stat}"]
    )

# FILL SITE SPLITS WITH SEASON IF MISSING
for site in ["home", "away", "neutral"]:
    for stat in ["off_eff", "def_eff", "possessions"]:
        col = f"{site}_{stat}"

        if col not in team_stats.columns:
            team_stats[col] = np.nan

        team_stats[col] = team_stats[col].fillna(team_stats[f"season_{stat}"])

# DEFAULT WEIGHTED STATS FOR RANKINGS
# Neutral is used as the default site for team rankings.
team_stats["off_eff"] = weighted_metric(
    team_stats["season_off_eff"],
    team_stats["last5_off_eff"],
    team_stats["neutral_off_eff"]
)

team_stats["def_eff"] = weighted_metric(
    team_stats["season_def_eff"],
    team_stats["last5_def_eff"],
    team_stats["neutral_def_eff"]
)

team_stats["possessions"] = weighted_metric(
    team_stats["season_possessions"],
    team_stats["last5_possessions"],
    team_stats["neutral_possessions"]
)

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
elo["Team"] = elo["Team"].replace(TEAM_NAME_MAP)
elo["Elo"] = pd.to_numeric(elo["Elo"], errors="coerce")

# INCLUDE ALL TEAMS FROM BOTH ELO AND BOXSCORES
elo_teams = pd.Series(elo["Team"].dropna().unique())
stat_teams = pd.Series(team_stats["Team"].dropna().unique())

all_teams = pd.DataFrame({
    "Team": pd.concat([elo_teams, stat_teams], ignore_index=True).drop_duplicates()
})

team_stats = all_teams.merge(team_stats, on="Team", how="left")

# FILL MISSING CORE VALUES
team_stats["off_eff"] = team_stats["off_eff"].fillna(100.0)
team_stats["def_eff"] = team_stats["def_eff"].fillna(100.0)
team_stats["possessions"] = team_stats["possessions"].fillna(67.0)

# FILL MISSING DETAIL COLUMNS
for col in team_stats.columns:
    if col.endswith("_off_eff"):
        team_stats[col] = team_stats[col].fillna(team_stats["off_eff"])
    elif col.endswith("_def_eff"):
        team_stats[col] = team_stats[col].fillna(team_stats["def_eff"])
    elif col.endswith("_possessions"):
        team_stats[col] = team_stats[col].fillna(team_stats["possessions"])

# MERGE ELO
team_stats = team_stats.merge(elo, on="Team", how="left")
team_stats["Elo"] = team_stats["Elo"].fillna(1500)

# MODEL POWER RATING
# Higher offense is good. Lower defense is good.
team_stats["Power_Rating"] = team_stats["off_eff"] - team_stats["def_eff"]

# Optional blended rating using both efficiency and Elo
team_stats["Elo_Normalized"] = (team_stats["Elo"] - 1500) / 25
team_stats["Overall_Rating"] = (
    0.70 * team_stats["Power_Rating"] +
    0.30 * team_stats["Elo_Normalized"]
)
# --- STRENGTH OF SCHEDULE + RESUME SCORE ---

team_rating_lookup = team_stats.set_index("Team")["Overall_Rating"].to_dict()

game_matchups["team_rating"] = game_matchups["team"].map(team_rating_lookup)
game_matchups["opponent_rating"] = game_matchups["opponent"].map(team_rating_lookup)

# Strength of schedule
sos = game_matchups.groupby("team", as_index=False).agg({
    "opponent_rating": "mean"
})

sos = sos.rename(columns={
    "team": "Team",
    "opponent_rating": "SOS_Rating"
})

team_stats = team_stats.merge(sos, on="Team", how="left")
team_stats["SOS_Rating"] = team_stats["SOS_Rating"].fillna(team_stats["SOS_Rating"].mean())

# Quality wins / bad losses
game_matchups["team_win"] = game_matchups["points"] > game_matchups["opp_points"]

quality_cutoff = game_matchups["opponent_rating"].quantile(0.80)
bad_loss_cutoff = game_matchups["opponent_rating"].quantile(0.20)

game_matchups["quality_win"] = (
    game_matchups["team_win"] &
    (game_matchups["opponent_rating"] >= quality_cutoff)
)

game_matchups["bad_loss"] = (
    (~game_matchups["team_win"]) &
    (game_matchups["opponent_rating"] <= bad_loss_cutoff)
)

resume = game_matchups.groupby("team", as_index=False).agg({
    "quality_win": "sum",
    "bad_loss": "sum"
})

resume = resume.rename(columns={"team": "Team"})

team_stats = team_stats.merge(resume, on="Team", how="left")
team_stats["quality_win"] = team_stats["quality_win"].fillna(0)
team_stats["bad_loss"] = team_stats["bad_loss"].fillna(0)

team_stats["Resume_Score"] = (
    1.8 * team_stats["quality_win"] -
    2.2 * team_stats["bad_loss"]
)

# --- TOURNAMENT PROJECTIONS ---

def calculate_recent_form(row):
    recent_off = row.get("last5_off_eff", row.get("off_eff", 100))
    season_off = row.get("season_off_eff", row.get("off_eff", 100))

    recent_def = row.get("last5_def_eff", row.get("def_eff", 100))
    season_def = row.get("season_def_eff", row.get("def_eff", 100))

    return (recent_off - season_off) - (recent_def - season_def)

# --- DIFFICULTY-ADJUSTED EFFECTIVENESS ---

team_stats["Difficulty_Adjusted_Effectiveness"] = (
    0.40 * team_stats["Power_Rating"] +
    0.35 * team_stats["SOS_Rating"] +
    0.25 * team_stats["Resume_Score"]
)

def calculate_tournament_score(row):
    overall = row.get("Overall_Rating", 0)
    power = row.get("Power_Rating", 0)
    elo = row.get("Elo", 1500)
    sos = row.get("SOS_Rating", 0)
    resume_score = row.get("Resume_Score", 0)
    recent_form = row.get("Recent_Form", 0)

    elo_component = (elo - 1500) / 25

    score = (
        0.23 * overall +
        0.10 * power +
        0.05 * elo_component +
        0.33 * sos +
        0.26 * resume_score +
        0.03 * recent_form
    )

    return score


team_stats["Recent_Form"] = team_stats.apply(calculate_recent_form, axis=1)
team_stats["Tournament_Score"] = team_stats.apply(calculate_tournament_score, axis=1)

team_stats["Tournament_Rank"] = (
    team_stats["Tournament_Score"]
    .rank(ascending=False, method="first")
    .astype(int)
)


def get_projected_seed_from_rank(rank):
    if rank <= 4:
        return "1 Seed"
    elif rank <= 8:
        return "2 Seed"
    elif rank <= 12:
        return "3 Seed"
    elif rank <= 16:
        return "4 Seed"
    elif rank <= 24:
        return "5–6 Seed"
    elif rank <= 28:
        return "7 Seed"
    elif rank <= 36:
        return "8–9 Seed"
    elif rank <= 40:
        return "10 Seed"
    elif rank <= 48:
        return "11–12 Seed"
    elif rank <= 68:
        return "Bubble / First Four"
    else:
        return "Outside Field"


def get_tournament_status_from_rank(rank):
    if rank <= 35:
        return "Lock"
    elif rank <= 52:
        return "Likely In"
    elif rank <= 69:
        return "Bubble"
    elif rank <= 78:
        return "Work To Do"
    elif rank <= 80:
        return "Likely Out"
    else:
        return "Locked Out"


def get_bubble_status_from_rank(rank):
    if rank <= 45:
        return "Safe"
    elif rank <= 68:
        return "Bubble"
    elif rank <= 85:
        return "Needs Work"
    else:
        return "Out"


team_stats["projected_seed"] = team_stats["Tournament_Rank"].apply(get_projected_seed_from_rank)
team_stats["tournament_status"] = team_stats["Tournament_Rank"].apply(get_tournament_status_from_rank)
team_stats["bubble_status"] = team_stats["Tournament_Rank"].apply(get_bubble_status_from_rank)

# SAVE TEAM STATS
team_stats.to_csv(DATA_DIR / "team_stats_current.csv", index=False)

# SAVE RANKINGS
team_rankings = team_stats.sort_values("Overall_Rating", ascending=False).reset_index(drop=True)
team_rankings.insert(0, "Rank", range(1, len(team_rankings) + 1))
team_rankings.to_csv(DATA_DIR / "team_rankings.csv", index=False)

metadata = {
    "teams": int(len(team_stats)),
    "weights": {
        "season": SEASON_WEIGHT,
        "last5": LAST5_WEIGHT,
        "site": SITE_WEIGHT
    },
    "ranking_formula": "Overall_Rating = 0.70 * Power_Rating + 0.30 * Elo_Normalized",
    "power_rating_formula": "Power_Rating = off_eff - def_eff",
    "default_site_for_rankings": "neutral"
}

with open(DATA_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Pipeline complete")
print("Saved:")
print(DATA_DIR / "team_stats_current.csv")
print(DATA_DIR / "team_rankings.csv")
print(DATA_DIR / "model_metadata.json")