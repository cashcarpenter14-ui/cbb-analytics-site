from pathlib import Path
import requests
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

PREDICTIONS_PATH = DATA_DIR / "model_predictions.csv"

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"


def get_espn_results_for_date(game_date):
    espn_date = str(game_date).replace("-", "")

    params = {
        "dates": espn_date,
        "groups": 50,
        "seasontype": 2,
        "limit": 1000,
    }

    response = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()
    results = {}

    for event in data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competition = competitions[0]
        status = competition.get("status", {}).get("type", {})
        is_final = status.get("completed", False)

        if not is_final:
            continue

        competitors = competition.get("competitors", [])
        if len(competitors) != 2:
            continue

        team_a = competitors[0]["team"]["displayName"]
        team_b = competitors[1]["team"]["displayName"]

        score_a = competitors[0].get("score")
        score_b = competitors[1].get("score")

        if score_a is None or score_b is None:
            continue

        score_a = int(score_a)
        score_b = int(score_b)

        results[(team_a, team_b)] = (score_a, score_b)
        results[(team_b, team_a)] = (score_b, score_a)

    return results


def main():
    print("Running results updater...")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {PREDICTIONS_PATH}")

    df = pd.read_csv(PREDICTIONS_PATH)

    if df.empty:
        print("No predictions found.")
        return

    required_cols = ["game_date", "team1", "team2", "actual_score1", "actual_score2"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    missing_results = df[
        df["actual_score1"].isna() | df["actual_score2"].isna()
    ].copy()

    if missing_results.empty:
        print("No missing results to update.")
        return

    updated_count = 0

    for game_date in sorted(missing_results["game_date"].dropna().unique()):
        print(f"Checking results for {game_date}...")

        try:
            results = get_espn_results_for_date(game_date)
        except Exception as e:
            print(f"Could not pull results for {game_date}: {e}")
            continue

        if not results:
            print(f"No final ESPN results found for {game_date}.")
            continue

        date_mask = df["game_date"].astype(str) == str(game_date)

        for idx, row in df[date_mask].iterrows():
            if pd.notna(row.get("actual_score1")) and pd.notna(row.get("actual_score2")):
                continue

            team1 = str(row["team1"]).strip()
            team2 = str(row["team2"]).strip()

            key = (team1, team2)

            if key not in results:
                print(f"No match found: {team1} vs {team2}")
                continue

            actual_score1, actual_score2 = results[key]

            df.at[idx, "actual_score1"] = actual_score1
            df.at[idx, "actual_score2"] = actual_score2

            updated_count += 1
            print(f"Updated: {team1} {actual_score1} - {team2} {actual_score2}")

    df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Updated {updated_count} games.")
    print("Saved updated model_predictions.csv")


if __name__ == "__main__":
    main()