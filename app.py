import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model import simulate_matchup

st.set_page_config(page_title="College Basketball Analytics Beta", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


team_stats_path = DATA_DIR / "team_stats_current.csv"
team_rankings_path = DATA_DIR / "team_rankings.csv"
metadata_path = DATA_DIR / "model_metadata.json"

team_stats_df = load_csv(team_stats_path) if team_stats_path.exists() else pd.DataFrame()
team_rankings_df = load_csv(team_rankings_path) if team_rankings_path.exists() else pd.DataFrame()
metadata = load_json(metadata_path) if metadata_path.exists() else {}

st.title("🏀 College Basketball Analytics Beta")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Ratings & Rankings", "Matchup Predictor", "Team Comparison"]
)

if page == "Home":
    st.subheader("Home")
    st.write("Beta college basketball analytics site built from your notebook.")
    if metadata:
        st.write("### Current model metadata")
        st.json(metadata)

elif page == "Ratings & Rankings":
    st.subheader("Ratings & Rankings")
    if team_rankings_df.empty:
        st.warning("No team rankings found. Run pipeline.py first.")
    else:
        st.dataframe(team_rankings_df, use_container_width=True)

elif page == "Matchup Predictor":
    st.subheader("Matchup Predictor")

    if team_stats_df.empty:
        st.warning("No team stats found. Run pipeline.py first.")
    else:
        teams = sorted(team_stats_df["Team"].dropna().unique().tolist())

        col1, col2, col3 = st.columns(3)
        with col1:
            team1 = st.selectbox("Team 1", teams)
        with col2:
            team2 = st.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0)
        with col3:
            site = st.selectbox("Site", ["neutral", "team1_home", "team2_home"])

        if st.button("Run Prediction"):
            try:
                result = simulate_matchup(team_stats_df, team1, team2, site)

                st.write("### Projection")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{result['team1']} Score", result["proj_score1"])
                m2.metric(f"{result['team2']} Score", result["proj_score2"])
                m3.metric("Spread (Team 1)", result["spread_team1"])
                m4.metric("Total", result["total"])

                w1, w2 = st.columns(2)
                w1.metric(f"{result['team1']} Win %", f"{result['win_prob1']:.1%}")
                w2.metric(f"{result['team2']} Win %", f"{result['win_prob2']:.1%}")

                box_df = pd.DataFrame([
                    {"Team": result["team1"], **result["box_score_team1"]},
                    {"Team": result["team2"], **result["box_score_team2"]},
                ])
                st.write("### Predicted Box Score")
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

        row1 = team_stats_df[team_stats_df["Team"] == team1].reset_index(drop=True)
        row2 = team_stats_df[team_stats_df["Team"] == team2].reset_index(drop=True)

        compare_df = pd.concat([row1, row2], ignore_index=True)
        st.dataframe(compare_df, use_container_width=True)