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
    "Queens Royals": "Queens University Royals",
    "Lindenwood Lions": "Lindenwood Lions",
    "Southern Indiana Screaming Eagles": "Southern Indiana Screaming Eagles",
}

LEAGUE_DEF_EFF = 102.5
HOME_COURT_POINTS = 3.0

SEASON_WEIGHT = 0.60
LAST5_WEIGHT = 0.25
SITE_WEIGHT = 0.15


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


def safe_stat(row, candidates, default=0.0):
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return float(default)


def get_weighted_stat(row, stat, site):
    site = str(site).strip().lower()
    default = 67.0 if stat == "possessions" else 100.0

    season_val = safe_stat(row, [f"season_{stat}", stat], default)
    last5_val = safe_stat(row, [f"last5_{stat}", f"season_{stat}", stat], season_val)
    site_val = safe_stat(row, [f"{site}_{stat}", f"season_{stat}", stat], season_val)

    return (
        SEASON_WEIGHT * season_val +
        LAST5_WEIGHT * last5_val +
        SITE_WEIGHT * site_val
    )


def simulate_matchup(team_stats_df, team1_name, team2_name, site_value="neutral", n_sims=10000):

    def project_team_box(team_row, projected_points, projected_possessions):
        # Possessions-based shooting volume
        shot_rate = safe_stat(
            team_row,
            ["season_shot_rate", "home_shot_rate", "away_shot_rate", "neutral_shot_rate"],
            0.92
        )
        shot_rate = clamp(shot_rate, 0.82, 1.05)

        fga = round(projected_possessions * shot_rate)

        # FG%
        fg_pct = safe_stat(
            team_row,
            ["season_fg_pct", "home_fg_pct", "away_fg_pct", "neutral_fg_pct"],
            0.45
        )
        if fg_pct > 1:
            fg_pct = fg_pct / 100
        fg_pct = clamp(fg_pct, 0.35, 0.58)

        fgm = round(fga * fg_pct)

        three_rate = safe_stat(
            team_row,
            ["season_three_rate", "home_three_rate", "away_three_rate", "neutral_three_rate"],
            0.38
        )
        if three_rate > 1:
            three_rate = three_rate / 100
        three_rate = clamp(three_rate, 0.20, 0.55)

        three_att = max(8, round(fga * three_rate))
        three_att = min(three_att, fga)

        three_pct = safe_stat(
            team_row,
            ["season_three_pct", "home_three_pct", "away_three_pct", "neutral_three_pct"],
            0.34
        )
        if three_pct > 1:
            three_pct = three_pct / 100
        three_pct = clamp(three_pct, 0.25, 0.45)

        three_made = min(three_att, round(three_att * three_pct))

        # Free throw rate: FTA / FGA
        ft_rate = safe_stat(
            team_row,
            ["season_ft_rate", "home_ft_rate", "away_ft_rate", "neutral_ft_rate"],
            0.28
        )
        if ft_rate > 2:
            ft_rate = ft_rate / 100
        ft_rate = clamp(ft_rate, 0.12, 0.55)

        fta = round(fga * ft_rate)

        ft_pct = safe_stat(
            team_row,
            ["season_ft_pct", "home_ft_pct", "away_ft_pct", "neutral_ft_pct"],
            0.74
        )
        if ft_pct > 1:
            ft_pct = ft_pct / 100
        ft_pct = clamp(ft_pct, 0.55, 0.88)

        ftm = round(fta * ft_pct)

        # Reconcile box-score shooting stats to projected points
        estimated_points = (2 * fgm) + three_made + ftm
        if estimated_points > 0:
            scale = projected_points / estimated_points
            fgm = max(12, round(fgm * scale))
            fga = max(fgm + 8, round(fga * scale))
            three_att = max(8, round(three_att * scale))
            three_att = min(three_att, fga)
            three_made = min(three_att, round(three_made * scale))
            fta = max(ftm + 1, round(fta * scale))
            ftm = min(fta, round(ftm * scale))

        fg_pct_display = round(100 * fgm / fga, 1) if fga > 0 else 0.0

        oreb = round(safe_stat(
            team_row,
            ["season_offensiveRebounds", "home_offensiveRebounds", "away_offensiveRebounds", "neutral_offensiveRebounds"],
            9
        ))

        dreb = round(safe_stat(
            team_row,
            ["season_defensiveRebounds", "home_defensiveRebounds", "away_defensiveRebounds", "neutral_defensiveRebounds"],
            24
        ))

        ast = round(safe_stat(
            team_row,
            ["season_assists", "home_assists", "away_assists", "neutral_assists"],
            max(8, fgm * 0.55)
        ))

        tov = round(safe_stat(
            team_row,
            ["season_turnovers", "home_turnovers", "away_turnovers", "neutral_turnovers"],
            12
        ))

        stl = round(safe_stat(
            team_row,
            ["season_steals", "home_steals", "away_steals", "neutral_steals"],
            6
        ))

        blk = round(safe_stat(
            team_row,
            ["season_blocks", "home_blocks", "away_blocks", "neutral_blocks"],
            4
        ))

        return {
            "PTS": int(projected_points),
            "FGM": int(fgm),
            "FGA": int(fga),
            "FG%": fg_pct_display,
            "3PM": int(three_made),
            "3PA": int(three_att),
            "FTM": int(ftm),
            "FTA": int(fta),
            "REB": int(oreb + dreb),
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
    elif site_value in ["team2_home", "away"]:
        site1, site2 = "away", "home"
    else:
        site1, site2 = "neutral", "neutral"

    off1 = get_weighted_stat(row1, "off_eff", site1)
    def1 = get_weighted_stat(row1, "def_eff", site1)
    tempo1 = get_weighted_stat(row1, "possessions", site1)

    off2 = get_weighted_stat(row2, "off_eff", site2)
    def2 = get_weighted_stat(row2, "def_eff", site2)
    tempo2 = get_weighted_stat(row2, "possessions", site2)

    geom_tempo = math.sqrt(max(tempo1, 1) * max(tempo2, 1))
    avg_tempo = (tempo1 + tempo2) / 2
    possessions = (0.60 * geom_tempo) + (0.40 * avg_tempo)
    possessions = clamp(possessions, 62, 73)

    adj1 = def2 / LEAGUE_DEF_EFF
    adj2 = def1 / LEAGUE_DEF_EFF

    adj1 = 0.7 * adj1 + 0.3 * 1.0
    adj2 = 0.7 * adj2 + 0.3 * 1.0

    exp_eff1 = clamp(off1 * adj1, 85, 125)
    exp_eff2 = clamp(off2 * adj2, 85, 125)

    score1 = possessions * exp_eff1 / 100
    score2 = possessions * exp_eff2 / 100

    if site_value in ["team1_home", "home"]:
        score1 += HOME_COURT_POINTS / 2
        score2 -= HOME_COURT_POINTS / 2
    elif site_value in ["team2_home", "away"]:
        score1 -= HOME_COURT_POINTS / 2
        score2 += HOME_COURT_POINTS / 2

    raw_total = score1 + score2
    target_total = 149
    stabilized_total = (0.40 * raw_total) + (0.60 * target_total)

    if raw_total > 0:
        scale = stabilized_total / raw_total
        score1 *= scale
        score2 *= scale

    score1 = clamp(score1, 50, 105)
    score2 = clamp(score2, 50, 105)

    avg_off = (off1 + off2) / 2
    score_gap = abs(score1 - score2)

    sim_std = 8.0
    sim_std += (possessions - 67) * 0.08
    sim_std += (avg_off - 102) * 0.03
    sim_std -= min(score_gap, 20) * 0.05
    sim_std = clamp(sim_std, 6.5, 10.5)

    sim_scores1 = np.random.normal(score1, sim_std, int(n_sims))
    sim_scores2 = np.random.normal(score2, sim_std, int(n_sims))

    sim_scores1 = np.clip(sim_scores1, 35, 130)
    sim_scores2 = np.clip(sim_scores2, 35, 130)

    win_prob1 = float(
        np.mean(sim_scores1 > sim_scores2) +
        0.5 * np.mean(sim_scores1 == sim_scores2)
    )
    win_prob2 = 1 - win_prob1

    proj1 = int(round(np.mean(sim_scores1)))
    proj2 = int(round(np.mean(sim_scores2)))

    spread = round_half(min(proj1, proj2) - max(proj1, proj2))
    total_line = round_half(proj1 + proj2)

    if proj1 > proj2:
        favorite = team1_name
    elif proj2 > proj1:
        favorite = team2_name
    else:
        favorite = "Even"

    box1 = project_team_box(row1, proj1, possessions)
    box2 = project_team_box(row2, proj2, possessions)

    return {
        "team1": team1_name,
        "team2": team2_name,
        "site": site_value,
        "team1_site_used": site1,
        "team2_site_used": site2,
        "proj_score1": proj1,
        "proj_score2": proj2,
        "favorite": favorite,
        "spread": spread,
        "total": total_line,
        "win_prob1": round(win_prob1, 4),
        "win_prob2": round(win_prob2, 4),
        "box_score_team1": box1,
        "box_score_team2": box2,
    }