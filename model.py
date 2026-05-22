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

# --- GLOBAL MODEL SETTINGS ---
LEAGUE_DEF_EFF = 102.5
HOME_COURT_POINTS = 3.0

# Default weighting profile
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
    def project_team_box(team_row, projected_points):
        # Estimated shooting profile
        fgm = max(12, round(projected_points / 2.15))
        fga = max(fgm + 8, round(fgm / 0.45))
        fg_pct = round(100 * fgm / fga, 1) if fga > 0 else 0.0

        three_rate = safe_stat(
            team_row,
            ["season_three_rate", "home_three_rate", "away_three_rate", "neutral_three_rate"],
            0.38,
        )
        if three_rate > 1:
            three_rate = three_rate / 100
        three_rate = clamp(three_rate, 0.20, 0.55)

        three_att = max(8, round(fga * three_rate))
        three_att = min(three_att, fga)
        three_made = min(three_att, round(three_att * 0.34))

        ftm = max(6, round(projected_points * 0.16))
        fta = max(ftm + 1, round(ftm / 0.74))

        oreb = round(safe_stat(
            team_row,
            [
                "season_offensiveRebounds",
                "home_offensiveRebounds",
                "away_offensiveRebounds",
                "neutral_offensiveRebounds",
                "OREB",
                "offensiveRebounds",
            ],
            9,
        ))

        dreb = round(safe_stat(
            team_row,
            [
                "season_defensiveRebounds",
                "home_defensiveRebounds",
                "away_defensiveRebounds",
                "neutral_defensiveRebounds",
                "DREB",
                "defensiveRebounds",
            ],
            24,
        ))

        ast = round(safe_stat(
            team_row,
            ["season_assists", "home_assists", "away_assists", "neutral_assists", "AST", "assists"],
            max(8, fgm * 0.55),
        ))

        tov = round(safe_stat(
            team_row,
            ["season_turnovers", "home_turnovers", "away_turnovers", "neutral_turnovers", "TO", "TOV", "turnovers"],
            12,
        ))

        stl = round(safe_stat(
            team_row,
            ["season_steals", "home_steals", "away_steals", "neutral_steals", "STL", "steals"],
            6,
        ))

        blk = round(safe_stat(
            team_row,
            ["season_blocks", "home_blocks", "away_blocks", "neutral_blocks", "BLK", "blocks"],
            4,
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
            "OREB": int(oreb),
            "DREB": int(dreb),
            "REB": int(oreb + dreb),
            "AST": int(ast),
            "TO": int(tov),
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

    # --- DYNAMIC TEMPO INTERACTION ---
    # When styles differ, the geometric mean keeps totals from getting inflated.
    geom_tempo = math.sqrt(max(tempo1, 1) * max(tempo2, 1))
    avg_tempo = (tempo1 + tempo2) / 2
    tempo_gap = abs(tempo1 - tempo2)

    tempo_weight_geom = 0.70 + min(tempo_gap / 25, 0.15)
    tempo_weight_avg = 1 - tempo_weight_geom

    possessions = (
        tempo_weight_geom * geom_tempo +
        tempo_weight_avg * avg_tempo
    )

    # --- OFFENSIVE REBOUNDING EXTRA-POSSESSION EFFECT ---
    oreb1 = safe_stat(
        row1,
        ["season_offensiveRebounds", f"{site1}_offensiveRebounds", "offensiveRebounds", "OREB"],
        9,
    )
    oreb2 = safe_stat(
        row2,
        ["season_offensiveRebounds", f"{site2}_offensiveRebounds", "offensiveRebounds", "OREB"],
        9,
    )

    oreb_bonus = ((oreb1 + oreb2) - 18) * 0.18
    possessions += oreb_bonus
    possessions = clamp(possessions, 62, 73)

    # Defense adjustment: lower defensive efficiency is better.
    adj1 = def2 / LEAGUE_DEF_EFF
    adj2 = def1 / LEAGUE_DEF_EFF

    adj1 = 0.7 * adj1 + 0.3 * 1.0
    adj2 = 0.7 * adj2 + 0.3 * 1.0

    exp_eff1 = clamp(off1 * adj1, 85, 125)
    exp_eff2 = clamp(off2 * adj2, 85, 125)

    # --- TURNOVER PRESSURE MISMATCH ---
    # Steals from one team plus turnovers from the opponent can swing margin/efficiency.
    tov1 = safe_stat(
        row1,
        ["season_turnovers", f"{site1}_turnovers", "turnovers", "TO", "TOV"],
        12,
    )
    tov2 = safe_stat(
        row2,
        ["season_turnovers", f"{site2}_turnovers", "turnovers", "TO", "TOV"],
        12,
    )
    stl1 = safe_stat(
        row1,
        ["season_steals", f"{site1}_steals", "steals", "STL"],
        6,
    )
    stl2 = safe_stat(
        row2,
        ["season_steals", f"{site2}_steals", "steals", "STL"],
        6,
    )

    turnover_edge1 = (stl1 - tov2) * 0.45
    turnover_edge2 = (stl2 - tov1) * 0.45

    exp_eff1 = clamp(exp_eff1 + turnover_edge1, 82, 128)
    exp_eff2 = clamp(exp_eff2 + turnover_edge2, 82, 128)

    score1 = possessions * exp_eff1 / 100
    score2 = possessions * exp_eff2 / 100

    if site_value in ["team1_home", "home"]:
        score1 += HOME_COURT_POINTS / 2
        score2 -= HOME_COURT_POINTS / 2
    elif site_value in ["team2_home", "away"]:
        score1 -= HOME_COURT_POINTS / 2
        score2 += HOME_COURT_POINTS / 2

    raw_total = score1 + score2

    # --- DYNAMIC TOTAL CALIBRATION ---
    # Lets high-efficiency matchups breathe while still stabilizing noisy totals.
    avg_off = (off1 + off2) / 2
    target_total = 149 + ((avg_off - 102) * 0.9)

    stabilized_total = (
        0.48 * raw_total +
        0.52 * target_total
    )

    if raw_total > 0:
        scale = stabilized_total / raw_total
        score1 *= scale
        score2 *= scale

    # --- POSTSEASON / NEUTRAL-SITE COMPRESSION ---
    # Tournament-style games tend to have tighter margins and more neutral-site variance.
    if site_value == "neutral":
        avg_score = (score1 + score2) / 2
        score1 = avg_score + ((score1 - avg_score) * 0.92)
        score2 = avg_score + ((score2 - avg_score) * 0.92)

        # Slightly suppress neutral-site scoring environment
        score1 *= 0.985
        score2 *= 0.985

    score1 = clamp(score1, 50, 105)
    score2 = clamp(score2, 50, 105)

    score_gap = abs(score1 - score2)

    sim_std = 8.0
    sim_std += (possessions - 67) * 0.08
    sim_std += (avg_off - 102) * 0.03
    sim_std -= min(score_gap, 20) * 0.04

    # Neutral-site / postseason-style games are more volatile.
    if site_value == "neutral":
        sim_std += 0.75

    # Extra volatility for three-point-heavy style.
    three_rate1 = safe_stat(
        row1,
        ["season_three_rate", f"{site1}_three_rate", "three_rate"],
        0.38,
    )
    three_rate2 = safe_stat(
        row2,
        ["season_three_rate", f"{site2}_three_rate", "three_rate"],
        0.38,
    )
    if three_rate1 > 1:
        three_rate1 /= 100
    if three_rate2 > 1:
        three_rate2 /= 100

    avg_three_rate = (three_rate1 + three_rate2) / 2
    if avg_three_rate > 0.42:
        sim_std += min((avg_three_rate - 0.42) * 8, 0.60)

    sim_std = clamp(sim_std, 7.0, 11.25)

    seed_text = f"{team1_name}-{team2_name}-{site_value}"
    seed = abs(hash(seed_text)) % (2**32)
    rng = np.random.default_rng(seed)

    sim_scores1 = rng.normal(score1, sim_std, int(n_sims))
    sim_scores2 = rng.normal(score2, sim_std, int(n_sims))

    sim_scores1 = np.clip(sim_scores1, 35, 130)
    sim_scores2 = np.clip(sim_scores2, 35, 130)

    win_prob1 = float(
        np.mean(sim_scores1 > sim_scores2) +
        0.5 * np.mean(sim_scores1 == sim_scores2)
    )
    win_prob2 = 1 - win_prob1

    proj1 = int(round(np.mean(sim_scores1)))
    proj2 = int(round(np.mean(sim_scores2)))

    total_line = round_half(proj1 + proj2)
    margin = abs(proj1 - proj2)

    if proj1 > proj2:
        favorite = team1_name
        underdog = team2_name
        spread = -round_half(margin)
        favorite_win_prob = win_prob1
    elif proj2 > proj1:
        favorite = team2_name
        underdog = team1_name
        spread = -round_half(margin)
        favorite_win_prob = win_prob2
    else:
        favorite = "Even"
        underdog = "Even"
        spread = 0.0
        favorite_win_prob = 0.5

    if favorite == "Even":
        line_display = "Even"
    else:
        line_display = f"{favorite} {spread}"

    if favorite_win_prob >= 0.70:
        confidence = "High"
    elif favorite_win_prob >= 0.58:
        confidence = "Medium"
    else:
        confidence = "Low"

    if margin <= 3 and favorite_win_prob < 0.60:
        upset_alert = "High upset potential"
    elif margin <= 6:
        upset_alert = "Moderate upset potential"
    else:
        upset_alert = "Low upset potential"

    volatility = round_half(sim_std)

    # --- GAME ARCHETYPE TAGS ---
    if possessions > 69 and avg_off > 110:
        game_style = "Shootout"

    elif possessions < 66 and avg_def < 98:
        game_style = "Grindfest"

    elif margin >= 18:
        game_style = "Mismatch"
  
    elif margin <= 5:
        game_style = "Toss-Up"

    else:
        game_style = "Balanced"

    # --- VOLATILITY TAG ---
    if sim_std >= 10:
        volatility_tag = "High Variance"

    elif sim_std >= 8.5:
        volatility_tag = "Moderate Variance"

    else:
        volatility_tag = "Low Variance"

    box1 = project_team_box(row1, proj1)
    box2 = project_team_box(row2, proj2)

    return {
        "team1": team1_name,
        "team2": team2_name,
        "site": site_value,
        "team1_site_used": site1,
        "team2_site_used": site2,
        "proj_score1": proj1,
        "proj_score2": proj2,
        "favorite": favorite,
        "underdog": underdog,
        "spread": spread,
        "line_display": line_display,
        "favorite_win_prob": round(favorite_win_prob, 4),
        "confidence": confidence,
        "upset_alert": upset_alert,
        "volatility": volatility,
        "volatility_tag": volatility_tag,
        "game_style": game_style,
        "total": total_line,
        "win_prob1": round(win_prob1, 4),
        "win_prob2": round(win_prob2, 4),
        "box_score_team1": box1,
        "box_score_team2": box2,
    }
