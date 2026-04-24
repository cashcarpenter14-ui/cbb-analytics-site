from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

PREDICTIONS_PATH = DATA_DIR / "model_predictions.csv"
ADJUSTMENTS_PATH = DATA_DIR / "model_adjustments.json"


def main():
    print("Running model adjustment updater...")

    if not PREDICTIONS_PATH.exists():
        print("No predictions file found.")
        return

    df = pd.read_csv(PREDICTIONS_PATH)

    required = ["model_score1", "model_score2", "actual_score1", "actual_score2"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return

    # Force numeric
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ✅ THIS IS THE FIX
    completed = df[
        df["model_score1"].notna()
        & df["model_score2"].notna()
        & df["actual_score1"].notna()
        & df["actual_score2"].notna()
    ].copy()

    print(f"Rows in file: {len(df)}")
    print(f"Completed rows found: {len(completed)}")

    if completed.empty:
        adjustments = {
            "games_used": 0,
            "margin_bias": 0,
            "total_bias": 0,
        }

        with open(ADJUSTMENTS_PATH, "w") as f:
            json.dump(adjustments, f, indent=2)

        print("No completed games yet. Saved default adjustments.")
        return

    completed["model_margin"] = completed["model_score1"] - completed["model_score2"]
    completed["actual_margin"] = completed["actual_score1"] - completed["actual_score2"]

    completed["model_total"] = completed["model_score1"] + completed["model_score2"]
    completed["actual_total"] = completed["actual_score1"] + completed["actual_score2"]

    margin_bias = (completed["model_margin"] - completed["actual_margin"]).mean()
    total_bias = (completed["model_total"] - completed["actual_total"]).mean()

    adjustments = {
        "games_used": int(len(completed)),
        "margin_bias": round(float(margin_bias), 3),
        "total_bias": round(float(total_bias), 3),
    }

    with open(ADJUSTMENTS_PATH, "w") as f:
        json.dump(adjustments, f, indent=2)

    print("Saved model adjustments:")
    print(adjustments)


if __name__ == "__main__":
    main()