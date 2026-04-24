from pathlib import Path
from datetime import datetime
import requests
import pandas as pd

from model import simulate_matchup

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

TEAM_STATS_PATH = DATA_DIR / "team_stats_current.csv"
PREDICTIONS_PATH = DATA_DIR / "model_predictions.csv"

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


def get_today_espn_games():
    today = datetime.now().strftime("%Y%m%d")

    params = {
        "dates": today,
        "groups": 50,
        "limit": 500,
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
            "game_date": datetime.now().strftime("%Y-%m-%d"),
            "team1": team_a,
            "team2": team_b,
            "site": site,
        })

    return pd.DataFrame(games)


def load_existing_predictions():
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)

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

    if not TEAM_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing {TEAM_STATS_PATH}")

    team_stats_df = pd.read_csv(TEAM_STATS_PATH)
    existing_df = load_existing_predictions()
    today_games = get_today_espn_games()

    print(f"Games pulled: {len(today_games)}")

    if today_games.empty:
        print("No games found for today.")
        return

    new_rows = []

    for _, game in today_games.iterrows():
        team1 = game["team1"]
        team2 = game["team2"]
        site = game["site"]
        game_date = game["game_date"]

        already_saved = (
            (existing_df["game_date"].astype(str) == game_date)
            & (existing_df["team1"].astype(str) == team1)
            & (existing_df["team2"].astype(str) == team2)
        ).any()

        if already_saved:
            continue

        if team1 not in team_stats_df["Team"].values or team2 not in team_stats_df["Team"].values:
            print(f"Skipping unmatched: {team1} vs {team2}")
            continue

        try:
            result = simulate_matchup(team_stats_df, team1, team2, site)

            new_rows.append({
                "game_date": game_date,
                "team1": team1,
                "team2": team2,
                "site": site,
                "model_score1": result["proj_score1"],
                "model_score2": result["proj_score2"],
                "model_margin_team1": result["proj_score1"] - result["proj_score2"],
                "model_total": result["proj_score1"] + result["proj_score2"],
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
    final_df = pd.concat([existing_df, new_df], ignore_index=True)

    DATA_DIR.mkdir(exist_ok=True)
    final_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Added {len(new_rows)} games.")
    print("Saved to model_predictions.csv")


if __name__ == "__main__":
    main()
