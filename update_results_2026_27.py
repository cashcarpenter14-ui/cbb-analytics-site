
from pathlib import Path
import requests
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "2026_27_candidate"
PREDICTIONS_PATH = DATA_DIR / "model_predictions.csv"

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/mens-college-basketball/scoreboard"
)


def get_espn_results_for_date(game_date):
    espn_date = str(game_date).replace("-", "")

    response = requests.get(
        ESPN_SCOREBOARD_URL,
        params={
            "dates": espn_date,
            "groups": 50,
            "seasontype": 2,
            "limit": 1000,
        },
        timeout=20,
    )
    response.raise_for_status()

    results = {}

    for event in response.json().get("events", []):
        game_id = str(event.get("id", "")).strip()
        competitions = event.get("competitions", [])

        if not game_id or not competitions:
            continue

        status = event.get("status", {}).get("type", {})
        if not status.get("completed", False):
            continue

        competitors = competitions[0].get("competitors", [])

        if len(competitors) != 2:
            continue

        scores = {}

        for competitor in competitors:
            team_name = competitor.get("team", {}).get("displayName")
            score = competitor.get("score")

            if not team_name or score is None:
                continue

            try:
                scores[team_name.strip()] = int(score)
            except (TypeError, ValueError):
                continue

        if len(scores) == 2:
            results[game_id] = scores

    return results


def main():
    print("Running 2026-27 results updater...")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_PATH)

    df = pd.read_csv(PREDICTIONS_PATH, dtype={"game_id": str})

    if df.empty:
        print("No predictions to update.")
        return

    required = [
        "game_id",
        "game_date",
        "team1",
        "team2",
        "actual_score1",
        "actual_score2",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    pending = df[
        df["actual_score1"].isna()
        | df["actual_score2"].isna()
    ]

    if pending.empty:
        print("All games already have results.")
        return

    updated = 0

    for game_date in sorted(pending["game_date"].dropna().unique()):
        try:
            results = get_espn_results_for_date(game_date)
        except Exception as exc:
            print(f"ESPN error for {game_date}: {exc}")
            continue

        for idx, row in pending[
            pending["game_date"] == game_date
        ].iterrows():

            game_id = str(row["game_id"]).strip()

            if game_id not in results:
                continue

            scores = results[game_id]
            team1 = str(row["team1"]).strip()
            team2 = str(row["team2"]).strip()

            if team1 not in scores or team2 not in scores:
                print(
                    f"Team-name mismatch for game {game_id}: "
                    f"{team1} vs {team2}"
                )
                continue

            df.at[idx, "actual_score1"] = scores[team1]
            df.at[idx, "actual_score2"] = scores[team2]
            updated += 1

    if updated:
        df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Updated completed games: {updated}")
    print(f"Prediction file: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()