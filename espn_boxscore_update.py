import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"

GAMES_FILE = RAW_DATA_DIR / "full_season_games.csv"
BOXSCORES_FILE = RAW_DATA_DIR / "team_boxscores_d1.csv"

ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"


def clean_name(name):
    if pd.isna(name):
        return ""
    return " ".join(str(name).replace("\xa0", " ").strip().split())


def fetch_summary(game_id):
    url = f"{ESPN_SUMMARY_URL}?event={game_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_val(val):
    if val is None:
        return np.nan
    val = str(val).strip()
    if "-" in val:
        try:
            return float(val.split("-")[-1])
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan


def get_stat(stats, names):
    names = [n.lower() for n in names]
    for s in stats:
        name = str(s.get("name", "")).lower()
        disp = str(s.get("displayName", "")).lower()
        short = str(s.get("shortDisplayName", "")).lower()

        if name in names or disp in names or short in names:
            return parse_val(s.get("displayValue", s.get("value")))
    return np.nan


def estimate_possessions(fga, oreb, tov, fta):
    if any(pd.isna(x) for x in [fga, oreb, tov, fta]):
        return np.nan
    return fga - oreb + tov + 0.475 * fta


def extract_game(summary, game_id):
    rows = []
    teams = summary.get("boxscore", {}).get("teams", [])

    for t in teams:
        team = clean_name(t.get("team", {}).get("displayName"))
        stats = t.get("statistics", [])

        pts = parse_val(t.get("score"))

        fga = get_stat(stats, ["field goals", "fga", "fg"])
        fta = get_stat(stats, ["free throws", "fta", "ft"])
        oreb = get_stat(stats, ["offensive rebounds", "oreb"])
        tov = get_stat(stats, ["turnovers", "tov"])

        poss = estimate_possessions(fga, oreb, tov, fta)

        if pd.isna(pts) or pd.isna(poss):
            continue

        rows.append({
            "game_id": str(game_id),
            "team": team,
            "points": pts,
            "possessions": poss,
            "source": "ESPN_UPDATE"
        })

    return pd.DataFrame(rows)


def main():
    print("ESPN BOXSCORE UPDATE")
    print("--------------------")

    games = pd.read_csv(GAMES_FILE)
    box = pd.read_csv(BOXSCORES_FILE)

    games.columns = [c.strip() for c in games.columns]
    box.columns = [c.strip() for c in box.columns]

    games["game_id"] = games["game_id"].astype(str)
    box["game_id"] = box["game_id"].astype(str)

    games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
    games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")

    completed = games.dropna(subset=["home_score", "away_score"])

    existing = set(box["game_id"])
    missing = completed[~completed["game_id"].isin(existing)]

    print("Completed games:", len(completed))
    print("Already in boxscores:", len(existing))
    print("Missing games:", len(missing))

    new_rows = []

    for i, (_, g) in enumerate(missing.iterrows(), 1):
        gid = g["game_id"]

        try:
            summary = fetch_summary(gid)
            df = extract_game(summary, gid)

            if len(df) == 2:
                new_rows.append(df)
                print(f"[{i}] added {gid}")
            else:
                print(f"[{i}] skipped {gid}")

        except Exception as e:
            print(f"[{i}] ERROR {gid}: {e}")

    if not new_rows:
        print("No new data added.")
        return

    new_df = pd.concat(new_rows, ignore_index=True)

    for col in box.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    for col in new_df.columns:
        if col not in box.columns:
            box[col] = np.nan

    new_df = new_df[box.columns]

    backup = RAW_DATA_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    box.to_csv(backup, index=False)

    updated = pd.concat([box, new_df], ignore_index=True)
    updated = updated.drop_duplicates(subset=["game_id", "team"])

    updated.to_csv(BOXSCORES_FILE, index=False)

    print("\nDONE")
    print("Added rows:", len(new_df))
    print("Backup:", backup)


if __name__ == "__main__":
    main()
