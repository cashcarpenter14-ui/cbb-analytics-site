import math
import numpy as np
import pandas as pd

TEAM_NAME_MAP = {
    "St Mary's Gaels": "Saint Mary's Gaels",
    "St. Mary's Gaels": "Saint Mary's Gaels",
    "Saint Marys Gaels": "Saint Mary's Gaels",
    "Saint Josephs Hawks": "Saint Joseph's Hawks",
    "St Johns Red Storm": "St. John's Red Storm",
    "St. Johns Red Storm": "St. John's Red Storm",
}

LEAGUE_DEF_EFF = 102.5
HOME_COURT_POINTS = 3.0


def clean_team_name(name):
    if pd.isna(name):
        return np.nan
    name = str(name).replace("\xa0", " ").strip()
    name = " ".join(name.split())
    return TEAM_NAME_MAP.get(name, name)


def clamp(x, low, high):
    if pd.isna(x):
        return np.nan
    return max(low, min(high, float(x)))


def round_half(x, default=0.0):
    if pd.isna(x) or np.isinf(x):
        return default
    return round(float(x) * 2) / 2


def simulate_matchup(team_stats_df, team1_name, team2_name, site_value="neutral", n_sims=10000):
    def get_site_weighted_value(row, stat, site):
        site = str(site).strip().lower()
        candidates = [
            f"{site}_{stat}",
            f"weighted_{stat}",
            f"season_{stat}",
            stat,
        ]
        for col in candidates:
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        return 0.0

    def safe_stat(row, candidates, default=0.0):
        for col in candidates:
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        return float(default)

    def project_team_box(team_row, projected_points):
        fgm = max(12, round(projected_points / 2.15))
        fga = max(fgm + 8, round(fgm / 0.45))
        fg_pct = round(100 * fgm / fga, 1) if fga > 0 else 0.0

        three_rate = safe_stat(
            team_row,
            ["season_three_rate", "home_three_rate", "away_three_rate", "neutral_three_rate"],
            0.38
        )
        if three_rate > 1:
            three_rate = three_rate / 100

        three_att = max(8, round(fga * three_rate))
        three_made = min(three_att, round(three_att * 0.34))

        ftm = max(6, round(projected_points * 0.16))
        fta = max(ftm + 1, round(ftm / 0.74))

        oreb = round(safe_stat(
            team_row,
            ["season_offensiveRebounds", "home_offensiveRebounds", "away_offensiveRebounds"],
            9
        ))
        dreb = round(safe_stat(
            team_row,
            ["season_defensiveRebounds", "home_defensiveRebounds", "away_defensiveRebounds"],
            24
        ))
        reb = oreb + dreb

        ast = round(safe_stat(
            team_row,
            ["season_assists", "home_assists", "away_assists"],
            max(8, fgm * 0.55)
        ))
        tov = round(safe_stat(
            team_row,
            ["season_turnovers", "home_turnovers", "away_turnovers"],
            12
        ))
        stl = round(safe_stat(
            team_row,
            ["season_steals", "home_steals", "away_steals"],
            6
        ))
        blk = round(safe_stat(
            team_row,
            ["season_blocks", "home_blocks", "away_blocks"],
            4
        ))

        return {
            "PTS": int(projected_points),
            "FGM": int(fgm),
            "FGA": int(fga),
            "FG%": fg_pct,
            "3PM": int(three_made),
            "3PA": int(three_att),
            "FTM": int(ftm),
            "FTA": int(fta),
            "REB": int(reb),
            "AST": int(ast),
            "TOV": int(tov),
            "STL": int(stl),
            "BLK": int(blk),
        }

    team1_name = clean_team_name(team1_name)
    team2_name = clean_team_name(team2_name)
    site_value = str(site_value).strip().lower()

    if site_value not in ["neutral", "team1_home", "team2_home", "home", "away"]:
        site_value = "neutral"

    row1_df = team_stats_df[team_stats_df["Team"] == team1_name]
    row2_df = team_stats_df[team_stats_df["Team"] == team2_name]

    if row1_df.empty:
        raise ValueError(f"Team not found in team_stats_df: {team1_name}")
    if row2_df.empty:
        raise ValueError(f"Team not found in team_stats_df: {team2_name}")

    row1 = row1_df.iloc[0]
    row2 = row2_df.iloc[0]

    if site_value in ["team1_home", "home"]:
        site1, site2 = "home", "away"
    elif site_value == "team2_home":
        site1, site2 = "away", "home"
    else:
        site1 = site2 = "neutral"

    off1 = get_site_weighted_value(row1, "off_eff", site1)
    def1 = get_site_weighted_value(row1, "def_eff", site1)
    tempo1 = get_site_weighted_value(row1, "possessions", site1)

    off2 = get_site_weighted_value(row2, "off_eff", site2)
    def2 = get_site_weighted_value(row2, "def_eff", site2)
    tempo2 = get_site_weighted_value(row2, "possessions", site2)

    if off1 == 0:
        off1 = safe_stat(row1, ["weighted_off_eff", "season_off_eff"], 102)
    if def1 == 0:
        def1 = safe_stat(row1, ["weighted_def_eff", "season_def_eff"], 102)
    if tempo1 == 0:
        tempo1 = safe_stat(row1, ["weighted_possessions", "season_possessions"], 67)

    if off2 == 0:
        off2 = safe_stat(row2, ["weighted_off_eff", "season_off_eff"], 102)
    if def2 == 0:
        def2 = safe_stat(row2, ["weighted_def_eff", "season_def_eff"], 102)
    if tempo2 == 0:
        tempo2 = safe_stat(row2, ["weighted_possessions", "season_possessions"], 67)

    possessions = math.sqrt(max(tempo1, 1) * max(tempo2, 1))
    possessions = clamp(possessions, 58, 78)

    exp_eff1 = clamp(off1 * (def2 / LEAGUE_DEF_EFF), 80, 130)
    exp_eff2 = clamp(off2 * (def1 / LEAGUE_DEF_EFF), 80, 130)

    score1 = possessions * exp_eff1 / 100
    score2 = possessions * exp_eff2 / 100

    if site_value in ["team1_home", "home"]:
        score1 += HOME_COURT_POINTS / 2
        score2 -= HOME_COURT_POINTS / 2
    elif site_value == "team2_home":
        score1 -= HOME_COURT_POINTS / 2
        score2 += HOME_COURT_POINTS / 2

    score1 = clamp(score1, 45, 110)
    score2 = clamp(score2, 45, 110)

    sim_scores1 = np.random.normal(score1, 10, int(n_sims))
    sim_scores2 = np.random.normal(score2, 10, int(n_sims))

    sim_scores1 = np.clip(sim_scores1, 35, 130)
    sim_scores2 = np.clip(sim_scores2, 35, 130)

    win_prob1 = float(np.mean(sim_scores1 > sim_scores2) + 0.5 * np.mean(sim_scores1 == sim_scores2))
    win_prob2 = 1 - win_prob1

    spread = round_half(float(np.mean(sim_scores1 - sim_scores2)))
    total_line = round_half(float(np.mean(sim_scores1 + sim_scores2)))

    proj1 = int(round(np.mean(sim_scores1)))
    proj2 = int(round(np.mean(sim_scores2)))

    box1 = project_team_box(row1, proj1)
    box2 = project_team_box(row2, proj2)

    return {
        "team1": team1_name,
        "team2": team2_name,
        "site": site_value,
        "proj_score1": proj1,
        "proj_score2": proj2,
        "spread_team1": spread,
        "total": total_line,
        "win_prob1": round(win_prob1, 4),
        "win_prob2": round(win_prob2, 4),
        "box_score_team1": box1,
        "box_score_team2": box2,
    }