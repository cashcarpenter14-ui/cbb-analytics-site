import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model import simulate_matchup

# --- PATHS ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Frontera Metrics",
    layout="wide"
)

# --- LOADERS ---
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# --- DATA ---
team_stats_path = DATA_DIR / "team_stats_current.csv"
team_rankings_path = DATA_DIR / "team_rankings.csv"
metadata_path = DATA_DIR / "model_metadata.json"
branding_path = DATA_DIR / "team_branding.csv"
predictions_path = DATA_DIR / "model_predictions.csv"

team_stats_df = load_csv(team_stats_path) if team_stats_path.exists() else pd.DataFrame()
team_rankings_df = load_csv(team_rankings_path) if team_rankings_path.exists() else pd.DataFrame()
metadata = load_json(metadata_path) if metadata_path.exists() else {}
branding_df = load_csv(branding_path) if branding_path.exists() else pd.DataFrame()
predictions_df = load_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()

def get_team_logo(team_name):
    if branding_df.empty or "team" not in branding_df.columns or "logo_file" not in branding_df.columns:
        return None

    match = branding_df[
        branding_df["team"].astype(str).str.strip().str.lower()
        == str(team_name).strip().lower()
    ]

    if match.empty:
        return None

    file_name = str(match.iloc[0]["logo_file"]).strip()

    if file_name == "" or file_name.lower() in ["nan", "none"]:
        return None

    logo_path = BASE_DIR / "assets" / "team_logos" / file_name
    return logo_path if logo_path.exists() else None

# --- HEADER ---
logo_col, title_col = st.columns([1, 4])

with logo_col:
    if Path("FMLogo.svg").exists():
        st.image("FMLogo.svg", width=140)

with title_col:
    st.markdown("## Frontera Metrics")
    st.markdown("COLLEGE BASKETBALL ANALYTICS")

# --- NAV ---
page = st.sidebar.radio(
    "Go to",
    ["Home", "Ratings & Rankings", "Matchup Predictor", "Team Comparison, Model Accuracy"]
)

# --- PAGES ---
if page == "Home":
    st.subheader("Welcome to Frontera Metrics")
    st.write(
        "Frontera Metrics is a college basketball analytics platform built to provide "
        "team ratings, matchup projections, and data-driven insights through a live, "
        "continuously improving model."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Ratings & Rankings")
        st.write("View team-level rankings built from efficiency, possessions, and rating data.")

    with c2:
        st.markdown("### Matchup Predictor")
        st.write("Compare any two teams and generate projected scores, win probabilities, and expected box scores.")

    with c3:
        st.markdown("### Team Comparison")
        st.write("See side-by-side team data to evaluate strengths, weaknesses, and overall profile.")

    st.markdown("---")

    if metadata:
        st.markdown("### Current model snapshot")
        st.json(metadata)

elif page == "Ratings & Rankings":
    st.subheader("Ratings & Rankings")

    if team_rankings_df.empty:
        st.warning("No team rankings found. Run pipeline.py first.")
    else:
        metric_view = st.radio(
            "View",
            ["Overall", "Offense", "Defense", "Tempo"],
            horizontal=True
        )

        rankings_df = team_rankings_df.copy()

        if metric_view == "Overall":
            rankings_df = rankings_df.sort_values("Elo", ascending=False).reset_index(drop=True)
            show_cols = [col for col in ["Team", "Elo", "off_eff", "def_eff", "possessions"] if col in rankings_df.columns]

        elif metric_view == "Offense":
            rankings_df = rankings_df.sort_values("off_eff", ascending=False).reset_index(drop=True)
            show_cols = [col for col in ["Team", "off_eff", "def_eff", "Elo", "possessions"] if col in rankings_df.columns]

        elif metric_view == "Defense":
            rankings_df = rankings_df.sort_values("def_eff", ascending=True).reset_index(drop=True)
            show_cols = [col for col in ["Team", "def_eff", "off_eff", "Elo", "possessions"] if col in rankings_df.columns]

        else:
            rankings_df = rankings_df.sort_values("possessions", ascending=False).reset_index(drop=True)
            show_cols = [col for col in ["Team", "possessions", "off_eff", "def_eff", "Elo"] if col in rankings_df.columns]

        rankings_df["Rank"] = range(1, len(rankings_df) + 1)
        rankings_df = rankings_df[["Rank"] + show_cols]

        st.dataframe(rankings_df, use_container_width=True)

elif page == "Matchup Predictor":
    st.subheader("Matchup Predictor")

    if team_stats_df.empty:
        st.warning("No team stats found. Run pipeline.py first.")
    else:
        teams = sorted(team_stats_df["Team"].dropna().unique().tolist())

        col1, col2, col3 = st.columns(3)
        with col1:
            team1 = st.selectbox("Team 1", teams, key="matchup_team1")
        with col2:
            team2 = st.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0, key="matchup_team2")
        with col3:
            site = st.selectbox("Site", ["neutral", "team1_home", "team2_home"], key="matchup_site")

        if team1 == team2:
            st.warning("Please choose two different teams.")
        else:
            logo1 = get_team_logo(team1)
            logo2 = get_team_logo(team2)

            st.markdown("---")

            left, middle, right = st.columns([3, 1, 3])

            with left:
                lsp1, lmain1, lsp2 = st.columns([1, 2, 1])
                with lmain1:
                    if logo1 is not None:
                        st.image(logo1, width=110)
                    st.markdown(
                        f"<div style='text-align:center; font-weight:600; font-size:20px; margin-top:8px;'>{team1}</div>",
                        unsafe_allow_html=True
                    )

            with middle:
                site_text = (
                    "Neutral site"
                    if site == "neutral"
                    else f"{team1} home"
                    if site == "team1_home"
                    else f"{team2} home"
                )

                st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='text-align:center; font-size:32px; font-weight:700;'>VS</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='text-align:center; font-size:14px; color:#666; margin-top:8px;'>{site_text}</div>",
                    unsafe_allow_html=True
                )

            with right:
                rsp1, rmain1, rsp2 = st.columns([1, 2, 1])
                with rmain1:
                    if logo2 is not None:
                        st.image(logo2, width=110)
                    st.markdown(
                        f"<div style='text-align:center; font-weight:600; font-size:20px; margin-top:8px;'>{team2}</div>",
                        unsafe_allow_html=True
                    )

            st.markdown("")

            if st.button("Run Prediction", use_container_width=True):
                try:
                    result = simulate_matchup(team_stats_df, team1, team2, site)

                    st.markdown("### Projection")

                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric(f"{result['team1']} Score", result["proj_score1"])
                    p2.metric(f"{result['team2']} Score", result["proj_score2"])

                    if "spread" in result:
                        p3.metric("Spread", result["spread"])
                    elif "spread_team1" in result:
                        p3.metric("Spread (Team 1)", result["spread_team1"])
                    else:
                        p3.metric("Spread", "N/A")

                    p4.metric("Total", result["total"])

                    w1, w2 = st.columns(2)
                    w1.metric(f"{result['team1']} Win %", f"{result['win_prob1']:.1%}")
                    w2.metric(f"{result['team2']} Win %", f"{result['win_prob2']:.1%}")

                    if "box_score_team1" in result and "box_score_team2" in result:
                        st.markdown("### Predicted Box Score")
                        box_df = pd.DataFrame([
                            {"Team": result["team1"], **result["box_score_team1"]},
                            {"Team": result["team2"], **result["box_score_team2"]},
                        ])
                        st.dataframe(box_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
elif page == "Team Comparison":
    st.subheader("Team Comparison")

    if team_stats_df.empty:
        st.warning("No team stats found. Run pipeline.py first.")
    else:
        teams = sorted(team_stats_df["Team"].dropna().unique().tolist())

        c1, c2 = st.columns(2)
        with c1:
            team1 = st.selectbox("Compare Team 1", teams, key="compare_team1")
        with c2:
            team2 = st.selectbox("Compare Team 2", teams, index=1 if len(teams) > 1 else 0, key="compare_team2")

        stat_options = {
            "Overall Rating": ("Elo", True),
            "Offensive Efficiency": ("off_eff", True),
            "Defensive Efficiency": ("def_eff", False),
            "Tempo": ("possessions", True),
        }

        selected_stats = st.multiselect(
            "Choose stats to compare",
            options=list(stat_options.keys()),
            default=["Overall Rating", "Offensive Efficiency", "Defensive Efficiency", "Tempo"]
        )

        row1 = team_stats_df[team_stats_df["Team"] == team1].reset_index(drop=True)
        row2 = team_stats_df[team_stats_df["Team"] == team2].reset_index(drop=True)

        if row1.empty or row2.empty:
            st.error("One or both teams could not be found.")
        elif not selected_stats:
            st.warning("Select at least one stat to compare.")
        else:
            row1 = row1.iloc[0]
            row2 = row2.iloc[0]

            st.markdown("### Head-to-Head Comparison")

            comparison_rows = []

            def winner_label(val1, val2, higher_is_better=True):
                if pd.isna(val1) or pd.isna(val2):
                    return "—"
                if abs(float(val1) - float(val2)) < 1e-9:
                    return "Even"
                if higher_is_better:
                    return team1 if val1 > val2 else team2
                return team1 if val1 < val2 else team2

            for stat_label in selected_stats:
                col_name, higher_is_better = stat_options[stat_label]

                if col_name in row1.index and col_name in row2.index:
                    comparison_rows.append({
                        "Stat": stat_label,
                        team1: round(float(row1[col_name]), 2),
                        team2: round(float(row2[col_name]), 2),
                        "Edge": winner_label(row1[col_name], row2[col_name], higher_is_better)
                    })

            compare_df = pd.DataFrame(comparison_rows)
            st.dataframe(compare_df, use_container_width=True)

            st.markdown("---")
            st.markdown("### Advantage Summary")

            advantage_rows = []

            for stat_label in selected_stats:
                col_name, higher_is_better = stat_options[stat_label]

                if col_name in row1.index and col_name in row2.index:
                    val1 = row1[col_name]
                    val2 = row2[col_name]

                    if pd.isna(val1) or pd.isna(val2):
                        edge = "—"
                    elif abs(float(val1) - float(val2)) < 1e-9:
                        edge = "Even"
                    else:
                        if higher_is_better:
                            edge = team1 if val1 > val2 else team2
                        else:
                            edge = team1 if val1 < val2 else team2

                    advantage_rows.append({
                        "Stat": stat_label,
                        "Advantage": edge
                    })

            advantage_df = pd.DataFrame(advantage_rows)

            def highlight_advantage(row):
                if row["Advantage"] == team1:
                    return ["", "background-color: rgba(0, 200, 0, 0.25)"]
                elif row["Advantage"] == team2:
                    return ["", "background-color: rgba(0, 120, 255, 0.25)"]
                elif row["Advantage"] == "Even":
                    return ["", "background-color: rgba(200, 200, 200, 0.2)"]
                return ["", ""]

            st.dataframe(
                advantage_df.style.apply(highlight_advantage, axis=1),
                use_container_width=True
            )

elif page == "Model Accuracy":
    st.subheader("Model Accuracy Dashboard")

    if predictions_df.empty:
        st.warning("No predictions found yet.")
    else:
        df = predictions_df.copy()

        # Force numeric conversion
        for col in ["model_score1", "model_score2", "actual_score1", "actual_score2"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Only require ACTUAL scores
        completed = df[
            df["actual_score1"].notna() &
            df["actual_score2"].notna()
        ].copy()

        if completed.empty:
            st.warning("Predictions exist, but no completed game results yet.")
            st.dataframe(df, use_container_width=True)

        else:
            # Core calculations
            completed["model_margin"] = completed["model_score1"] - completed["model_score2"]
            completed["actual_margin"] = completed["actual_score1"] - completed["actual_score2"]

            completed["model_total"] = completed["model_score1"] + completed["model_score2"]
            completed["actual_total"] = completed["actual_score1"] + completed["actual_score2"]

            completed["winner_correct"] = (
                (completed["model_margin"] > 0) ==
                (completed["actual_margin"] > 0)
            )

            completed["spread_error"] = (
                completed["model_margin"] - completed["actual_margin"]
            ).abs()

            completed["total_error"] = (
                completed["model_total"] - completed["actual_total"]
            ).abs()

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games Graded", len(completed))
            m2.metric("Winner Accuracy", f"{completed['winner_correct'].mean():.1%}")
            m3.metric("Avg Spread Error", f"{completed['spread_error'].mean():.2f}")
            m4.metric("Avg Total Error", f"{completed['total_error'].mean():.2f}")

            st.markdown("---")

            # Recent performance
            st.markdown("### Recent Performance")
            recent = completed.tail(10)

            r1, r2, r3 = st.columns(3)
            r1.metric("Last 10 Accuracy", f"{recent['winner_correct'].mean():.1%}")
            r2.metric("Last 10 Spread Error", f"{recent['spread_error'].mean():.2f}")
            r3.metric("Last 10 Total Error", f"{recent['total_error'].mean():.2f}")

            st.markdown("---")

            # Game log
            st.markdown("### Game Log")

            st.dataframe(
                completed.sort_values("game_date", ascending=False),
                use_container_width=True
            )