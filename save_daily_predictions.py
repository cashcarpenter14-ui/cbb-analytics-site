from pathlib import Path
from datetime import datetime
import json
import requests
import pandas as pd

from model import simulate_matchup

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "raw_data"

TEAM_STATS_PATH = DATA_DIR / "team_stats_current.csv"
PREDICTIONS_PATH = DATA_DIR / "model_predictions.csv"
LOCAL_SCHEDULE_PATH = RAW_DATA_DIR / "full_season_games.csv"
ADJUSTMENTS_PATH = DATA_DIR / "model_adjustments.json"

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


def get_target_date():
    return datetime.now().strftime("%Y%m%d")


def load_model_adjustments():
    if not ADJUSTMENTS_PATH.exists():
        return {"margin_bias": 0, "total_bias": 0}

    with open(ADJUSTMENTS_PATH, "r") as f:
        return json.load(f)


def get_espn_games(target_date):
    print(f"Trying ESPN schedule for: {target_date}")

    params = {
        "dates": target_date,
        "groups": 50,
        "seasontype": 2,
        "limit": 1000,
    }

    response = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()
    games = []

    for event in data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors", [])

        if len(competitors) != 2:
            continue

        team_a = competitors[0]["team"]["displayName"]
        team_b = competitors[1]["team"]["displayName"]

        neutral_site = competition.get("neutralSite", False)

        home_team = None
        for comp in competitors:
            if comp.get("homeAway") == "home":
                home_team = comp["team"]["displayName"]

        if neutral_site:
            site = "neutral"
        elif home_team == team_a:
            site = "team1_home"
        elif home_team == team_b:
            site = "team2_home"
        else:
            site = "neutral"

        games.append({
            "game_date": f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}",
            "team1": team_a,
            "team2": team_b,
            "site": site,
        })

    return pd.DataFrame(games)


def get_local_schedule_games(target_date):
    print("ESPN returned 0 games. Trying local schedule fallback...")

    if not LOCAL_SCHEDULE_PATH.exists():
        print(f"No local schedule found at {LOCAL_SCHEDULE_PATH}")
        return pd.DataFrame()

    schedule_df = pd.read_csv(LOCAL_SCHEDULE_PATH)

    possible_date_cols = ["game_date", "date", "Date"]
    possible_team1_cols = ["team1", "Team1", "home_team", "Home", "team"]
    possible_team2_cols = ["team2", "Team2", "away_team", "Away", "opponent"]

    date_col = next((c for c in possible_date_cols if c in schedule_df.columns), None)
    team1_col = next((c for c in possible_team1_cols if c in schedule_df.columns), None)
    team2_col = next((c for c in possible_team2_cols if c in schedule_df.columns and c != team1_col), None)

    if date_col is None or team1_col is None or team2_col is None:
        print("Local schedule columns not recognized.")
        print("Columns found:", schedule_df.columns.tolist())
        return pd.DataFrame()

    target_date_dash = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"

    schedule_df[date_col] = pd.to_datetime(schedule_df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    games_df = schedule_df[schedule_df[date_col] == target_date_dash].copy()

    if games_df.empty:
        print(f"No local schedule games found for {target_date_dash}")
        return pd.DataFrame()

    output = pd.DataFrame({
        "game_date": games_df[date_col],
        "team1": games_df[team1_col],
        "team2": games_df[team2_col],
        "site": "neutral",
    })

    if "site" in games_df.columns:
        output["site"] = games_df["site"]
    elif "neutral_site" in games_df.columns:
        output["site"] = games_df["neutral_site"].apply(lambda x: "neutral" if bool(x) else "team1_home")

    return output


def get_games_for_date(target_date):
    espn_games = get_espn_games(target_date)

    if not espn_games.empty:
        print(f"Using ESPN schedule. Games found: {len(espn_games)}")
        return espn_games

    local_games = get_local_schedule_games(target_date)
    print(f"Using local fallback. Games found: {len(local_games)}")
    return local_games


def load_existing_predictions():
    if PREDICTIONS_PATH.exists():
        try:
            return pd.read_csv(PREDICTIONS_PATH)
        except pd.errors.EmptyDataError:
            pass

    return pd.DataFrame(columns=[
        "game_date",
        "team1",
        "team2",
        "site",
        "model_score1",
        "model_score2",
        "model_margin_team1",
        "model_total",
        "model_win_prob1",
        "model_win_prob2",
        "actual_score1",
        "actual_score2",
        "vegas_spread_team1",
        "vegas_total",
    ])


def main():
    print("Running prediction script...")

    target_date = get_target_date()

    if not TEAM_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing {TEAM_STATS_PATH}")

    team_stats_df = pd.read_csv(TEAM_STATS_PATH)
    existing_df = load_existing_predictions()
    adjustments = load_model_adjustments()

    margin_bias = adjustments.get("margin_bias", 0)
    total_bias = adjustments.get("total_bias", 0)

    print(f"Using model adjustments: margin_bias={margin_bias}, total_bias={total_bias}")

    games_df = get_games_for_date(target_date)

    print(f"Games pulled: {len(games_df)}")

    if games_df.empty:
        print("No games found.")
        return

    new_rows = []

    for _, game in games_df.iterrows():
        team1 = str(game["team1"]).strip()
        team2 = str(game["team2"]).strip()
        site = str(game.get("site", "neutral")).strip()
        game_date = str(game["game_date"]).strip()

        already_saved = (
            (existing_df["game_date"].astype(str) == game_date)
            & (existing_df["team1"].astype(str) == team1)
            & (existing_df["team2"].astype(str) == team2)
        ).any()

        if already_saved:
            print(f"Skipping already saved: {team1} vs {team2}")
            continue

        if team1 not in team_stats_df["Team"].values or team2 not in team_stats_df["Team"].values:
            print(f"Skipping unmatched: {team1} vs {team2}")
            continue

        try:
            result = simulate_matchup(team_stats_df, team1, team2, site)

            raw_score1 = result["proj_score1"]
            raw_score2 = result["proj_score2"]

            raw_margin = raw_score1 - raw_score2
            raw_total = raw_score1 + raw_score2

            adjusted_margin = raw_margin - margin_bias
            adjusted_total = raw_total - total_bias

            adjusted_score1 = round((adjusted_total + adjusted_margin) / 2, 1)
            adjusted_score2 = round((adjusted_total - adjusted_margin) / 2, 1)

            new_rows.append({
                "game_date": game_date,
                "team1": team1,
                "team2": team2,
                "site": site,
                "model_score1": adjusted_score1,
                "model_score2": adjusted_score2,
                "model_margin_team1": adjusted_margin,
                "model_total": adjusted_total,
                "model_win_prob1": result.get("win_prob1"),
                "model_win_prob2": result.get("win_prob2"),
                "actual_score1": None,
                "actual_score2": None,
                "vegas_spread_team1": None,
                "vegas_total": None,
            })

            print(f"Saved: {team1} vs {team2}")

        except Exception as e:
            print(f"Error: {team1} vs {team2} → {e}")

    if not new_rows:
        print("No new predictions added.")
        return

    new_df = pd.DataFrame(new_rows)

    if existing_df.empty:
        final_df = new_df
    else:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)

    final_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Added {len(new_rows)} games.")
    print("Saved to model_predictions.csv")


if __name__ == "__main__":
    main()