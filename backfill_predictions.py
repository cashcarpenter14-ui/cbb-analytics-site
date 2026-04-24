print("BACKFILL SCRIPT STARTED")
from pathlib import Path
import pandas as pd

from model import simulate_matchup

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "raw_data"

TEAM_STATS_PATH = DATA_DIR / "team_stats_current.csv"
PREDICTIONS_PATH = DATA_DIR / "model_predictions.csv"
SCHEDULE_PATH = RAW_DATA_DIR / "full_season_games.csv"


def load_predictions():
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
        "data_type",
    ])


def main():
    print("Running historical backfill...")

    if not TEAM_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing {TEAM_STATS_PATH}")

    if not SCHEDULE_PATH.exists():
        raise FileNotFoundError(f"Missing {SCHEDULE_PATH}")

    team_stats_df = pd.read_csv(TEAM_STATS_PATH)
    games_df = pd.read_csv(SCHEDULE_PATH)
    predictions_df = load_predictions()

    games_df.columns = [str(c).strip() for c in games_df.columns]

    required_cols = ["date", "home_team", "away_team", "home_score", "away_score"]

    missing = [col for col in required_cols if col not in games_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in full_season_games.csv: {missing}")

    games_df["date"] = pd.to_datetime(games_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    games_df["home_score"] = pd.to_numeric(games_df["home_score"], errors="coerce")
    games_df["away_score"] = pd.to_numeric(games_df["away_score"], errors="coerce")

    completed_games = games_df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()

    print(f"Completed games found: {len(completed_games)}")

    new_rows = []

    for _, game in completed_games.iterrows():
        game_date = str(game["date"]).strip()
        team1 = str(game["home_team"]).strip()
        team2 = str(game["away_team"]).strip()

        neutral_site = bool(game.get("neutral_site", False))
        site = "neutral" if neutral_site else "team1_home"

        already_saved = (
            (predictions_df["game_date"].astype(str) == game_date)
            & (predictions_df["team1"].astype(str) == team1)
            & (predictions_df["team2"].astype(str) == team2)
        ).any()

        if already_saved:
            continue

        if team1 not in team_stats_df["Team"].values or team2 not in team_stats_df["Team"].values:
            print(f"Skipping unmatched: {team1} vs {team2}")
            continue

        try:
            result = simulate_matchup(team_stats_df, team1, team2, site)

            model_score1 = result["proj_score1"]
            model_score2 = result["proj_score2"]

            new_rows.append({
                "game_date": game_date,
                "team1": team1,
                "team2": team2,
                "site": site,
                "model_score1": model_score1,
                "model_score2": model_score2,
                "model_margin_team1": model_score1 - model_score2,
                "model_total": model_score1 + model_score2,
                "model_win_prob1": result.get("win_prob1"),
                "model_win_prob2": result.get("win_prob2"),
                "actual_score1": game["home_score"],
                "actual_score2": game["away_score"],
                "vegas_spread_team1": None,
                "vegas_total": None,
                "data_type": "backtest",
            })

            print(f"Backfilled: {team1} vs {team2}")

        except Exception as e:
            print(f"Error: {team1} vs {team2} → {e}")

    if not new_rows:
        print("No new backfilled predictions added.")
        return

    new_df = pd.DataFrame(new_rows)

    if predictions_df.empty:
        final_df = new_df
    else:
        final_df = pd.concat([predictions_df, new_df], ignore_index=True)

    final_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Added {len(new_rows)} backfilled games.")
    print("Saved to model_predictions.csv")


if __name__ == "__main__":
    main()