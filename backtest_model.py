import pandas as pd
from pathlib import Path

from model import simulate_matchup


def clean_name(name):
    if pd.isna(name):
        return ""
    return " ".join(str(name).replace("\xa0", " ").strip().split())


BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"

team_stats = pd.read_csv(DATA_DIR / "team_stats_current.csv")
games = pd.read_csv(RAW_DATA_DIR / "full_season_games.csv")

games.columns = [str(c).strip() for c in games.columns]
team_stats.columns = [str(c).strip() for c in team_stats.columns]

required_cols = [
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "neutral_site"
]

missing = [c for c in required_cols if c not in games.columns]
if missing:
    raise ValueError(f"Missing required columns in full_season_games.csv: {missing}")

print("TOTAL ROWS:", len(games))

# Clean names BEFORE filtering
games["home_team"] = games["home_team"].map(clean_name)
games["away_team"] = games["away_team"].map(clean_name)
team_stats["Team"] = team_stats["Team"].map(clean_name)

games["home_score"] = pd.to_numeric(games["home_score"], errors="coerce")
games["away_score"] = pd.to_numeric(games["away_score"], errors="coerce")

print("NON-NULL HOME SCORES:", games["home_score"].notna().sum())
print("NON-NULL AWAY SCORES:", games["away_score"].notna().sum())

# Only test games where BOTH teams exist in your model
valid_teams = set(team_stats["Team"].dropna())

completed = games[
    games["home_team"].isin(valid_teams) &
    games["away_team"].isin(valid_teams) &
    games["home_score"].notna() &
    games["away_score"].notna()
].copy()

print("COMPLETED MODEL-VALID GAMES USED:", len(completed))

results = []

for _, game in completed.iterrows():
    home = clean_name(game["home_team"])
    away = clean_name(game["away_team"])

    neutral = bool(game.get("neutral_site", False))
    site_value = "neutral" if neutral else "team1_home"

    try:
        pred = simulate_matchup(
            team_stats_df=team_stats,
            team1_name=home,
            team2_name=away,
            site_value=site_value,
            n_sims=300
        )

        actual_margin = game["home_score"] - game["away_score"]
        pred_margin = pred["proj_score1"] - pred["proj_score2"]

        actual_total = game["home_score"] + game["away_score"]
        pred_total = pred["total"]

        actual_winner = home if game["home_score"] > game["away_score"] else away
        pred_winner = home if pred["proj_score1"] > pred["proj_score2"] else away

        results.append({
            "home_team": home,
            "away_team": away,
            "neutral_site": neutral,
            "actual_home_score": game["home_score"],
            "actual_away_score": game["away_score"],
            "pred_home_score": pred["proj_score1"],
            "pred_away_score": pred["proj_score2"],
            "actual_margin": actual_margin,
            "pred_margin": pred_margin,
            "margin_error": abs(pred_margin - actual_margin),
            "actual_total": actual_total,
            "pred_total": pred_total,
            "total_error": abs(pred_total - actual_total),
            "actual_winner": actual_winner,
            "pred_winner": pred_winner,
            "winner_correct": actual_winner == pred_winner,
            "home_win_prob": pred["win_prob1"],
        })

    except Exception as e:
        print(f"ERROR: {home} vs {away} -> {e}")
        results.append({
            "home_team": home,
            "away_team": away,
            "error": str(e)
        })

backtest = pd.DataFrame(results)

if "error" in backtest.columns:
    successful = backtest[backtest["error"].isna()].copy()
    failed = backtest[backtest["error"].notna()].copy()
else:
    successful = backtest.copy()
    failed = pd.DataFrame()

print("\nBACKTEST RESULTS")
print("----------------")
print(f"Games tested: {len(successful)}")
print(f"Games skipped/errors: {len(failed)}")

if len(successful) > 0:
    print(f"Winner accuracy: {successful['winner_correct'].mean() * 100:.1f}%")
    print(f"Avg margin error: {successful['margin_error'].mean():.2f}")
    print(f"Median margin error: {successful['margin_error'].median():.2f}")
    print(f"Avg total error: {successful['total_error'].mean():.2f}")
    print(f"Median total error: {successful['total_error'].median():.2f}")

backtest.to_csv(DATA_DIR / "backtest_results.csv", index=False)

if len(failed) > 0:
    failed.to_csv(DATA_DIR / "backtest_errors.csv", index=False)
    print("\nSome games failed. See data/backtest_errors.csv")

print("\nSaved:")
print(DATA_DIR / "backtest_results.csv")