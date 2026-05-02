import json
from pathlib import Path
import matplotlib.pyplot as plt
import requests
from datetime import datetime

import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
from model import simulate_matchup

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def add_logo_to_chart(ax, x, y, logo_path, zoom=0.05):
    try:
        img = plt.imread(str(logo_path))
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
    except Exception:
        ax.scatter(x, y, s=25)

# --- PATHS ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

ASSETS_DIR = BASE_DIR / "assets"
WEBSITE_LOGO_PATH = ASSETS_DIR / "fronedge_logos" / "FronEdgeScriptball.svg"  # adjust name if needed

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FronEdge Metrics",
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

# ✅ ADD THIS (displays logo — does not remove anything)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if WEBSITE_LOGO_PATH.exists():
        st.image(str(WEBSITE_LOGO_PATH), width=300)

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
    if Path("FronEdgeScriptball.svg").exists():
        st.image("FronEdgeScriptball.svg", width=140)

with title_col:
    st.markdown("## FronEdge Metrics")
    st.markdown("COLLEGE BASKETBALL ANALYTICS")

# --- NAV ---
page = st.sidebar.radio(
    "Go to",
    ["Home", "Ratings & Rankings", "Matchup Predictor", "Team Comparison", "Teams", "Daily Slate", "Model Accuracy"]
)
def fetch_today_slate():
    today = datetime.now().strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={today}"

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    games = []

    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])

        if len(competitors) != 2:
            continue

        home = next((t for t in competitors if t.get("homeAway") == "home"), None)
        away = next((t for t in competitors if t.get("homeAway") == "away"), None)

        if home is None or away is None:
            continue

        games.append({
            "game_id": event.get("id"),
            "date": event.get("date"),
            "home_team": home.get("team", {}).get("displayName"),
            "away_team": away.get("team", {}).get("displayName"),
            "neutral_site": bool(comp.get("neutralSite", False)),
        })

    return pd.DataFrame(games)

# --- PAGES ---
if page == "Home":
    st.subheader("Welcome to FronEdge Metrics")
    st.write(
        "FronEdge Metrics is a college basketball analytics platform built to provide "
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

        st.markdown("### Efficiency Map")

        chart_df = team_rankings_df.copy()

        fig, ax = plt.subplots(figsize=(12, 8))

        for _, team_row in chart_df.iterrows():
            team_name = team_row["Team"]
            x = team_row.get("off_eff", None)
            y = team_row.get("def_eff", None)

            if pd.notna(x) and pd.notna(y):
                logo_path = get_team_logo(team_name)

                if logo_path:
                    add_logo_to_chart(ax, x, y, logo_path)
                else:
                    ax.scatter(x, y, s=25)

        ax.axvline(102.5, linestyle="--", linewidth=1)
        ax.axhline(102.5, linestyle="--", linewidth=1)

        ax.set_xlabel("Offensive Efficiency")
        ax.set_ylabel("Defensive Efficiency")
        ax.set_title("Team Efficiency Map")

        ax.invert_yaxis()

        st.pyplot(fig)

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
        def add_logo_to_chart(ax, x, y, logo_path, zoom=0.045):
            try:
                img = plt.imread(str(logo_path))
                imagebox = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                ax.add_artist(ab)
            except Exception:
                ax.scatter(x, y, s=25)

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

elif page == "Teams":
    st.title("Teams")

    team_stats_df["Team"] = (
        team_stats_df["Team"]
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )

    ALL_TEAMS = sorted(team_stats_df["Team"].dropna().unique())

    search = st.text_input("Search Team")

    if search:
        teams = [t for t in ALL_TEAMS if search.lower() in t.lower()]
    else:
        teams = ALL_TEAMS

    if not teams:
        st.warning("No teams found.")
    else:
        team = st.selectbox("Select a Team", teams)
        row = team_stats_df[team_stats_df["Team"] == team].iloc[0]

        st.header(team)

        rank_match = team_rankings_df[team_rankings_df["Team"] == team]

        if not rank_match.empty and "Rank" in rank_match.columns:
            national_rank = int(rank_match["Rank"].iloc[0])
            percentile = round((1 - (national_rank / len(team_rankings_df))) * 100, 1)
        else:
            national_rank = "N/A"
            percentile = "N/A"

        st.markdown("### Team Profile")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Power Rating", round(row.get("Power_Rating", 0), 2))
        c2.metric("National Rank", national_rank)
        c3.metric("Percentile", f"{percentile}%" if percentile != "N/A" else "N/A")
        c4.metric("Pace", round(row.get("possessions", 0), 1))

        st.markdown("---")

        st.markdown("### Core Efficiency")

        e1, e2, e3 = st.columns(3)
        e1.metric("Overall Rating", round(row.get("Overall_Rating", 0), 2))
        e2.metric("Offensive Efficiency", round(row.get("off_eff", 0), 2))
        e3.metric("Defensive Efficiency", round(row.get("def_eff", 0), 2))

        st.markdown("---")

        st.markdown("### Tournament Projection")

        proj1, proj2, proj3, proj4 = st.columns(4)
        proj1.metric("Tournament Status", row.get("tournament_status", "Not Calculated"))
        proj2.metric("Projected Seed", row.get("projected_seed", "N/A"))
        proj3.metric("Bubble Status", row.get("bubble_status", "N/A"))
        proj4.metric("Projected Record", row.get("projected_record", "N/A"))

        st.markdown("---")

        st.markdown("### Recent Form")

        trend = row.get("last5_off_eff", 0) - row.get("season_off_eff", 0)

        if trend > 0:
            st.success(f"Trending Up (+{round(trend, 2)} Off Eff last 5)")
        elif trend < 0:
            st.error(f"Trending Down ({round(trend, 2)} Off Eff last 5)")
        else:
            st.info("Recent form is flat compared to season average.")

        st.markdown("---")

        st.markdown("### Efficiency Profile")

        st.markdown("### Efficiency Profile")

        fig, ax = plt.subplots(figsize=(6, 4))

        labels = ["Off Eff", "Def Eff", "D1 Avg"]
        values = [
            row.get("off_eff", 0),
            row.get("def_eff", 0),
            102.5
]

        ax.bar(labels, values)
        ax.set_ylabel("Efficiency")
        ax.set_title(f"{team} Efficiency Profile")

        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Resume")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric("Tournament Rank", int(row.get("Tournament_Rank", 0)))
        r2.metric("SOS Rating", round(row.get("SOS_Rating", 0), 2))
        r3.metric("Quality Wins", int(row.get("quality_win", 0)))
        r4.metric("Bad Losses", int(row.get("bad_loss", 0)))

        r5, r6, r7, r8 = st.columns(4)

        r5.metric("Resume Score", round(row.get("Resume_Score", 0), 2))
        r6.metric("Tournament Status", row.get("tournament_status", "N/A"))
        r7.metric("Projected Seed", row.get("projected_seed", "N/A"))
        r8.metric("Bubble Status", row.get("bubble_status", "N/A"))

        st.markdown("---")

        st.subheader("Performance Splits")

        split_data = {
            "Split": ["Season", "Last 5", "Home", "Away", "Neutral"],
            "Off Eff": [
                row.get("season_off_eff", None),
                row.get("last5_off_eff", None),
                row.get("home_off_eff", None),
                row.get("away_off_eff", None),
                row.get("neutral_off_eff", None),
            ],
            "Def Eff": [
                row.get("season_def_eff", None),
                row.get("last5_def_eff", None),
                row.get("home_def_eff", None),
                row.get("away_def_eff", None),
                row.get("neutral_def_eff", None),
            ],
            "Pace": [
                row.get("season_possessions", None),
                row.get("last5_possessions", None),
                row.get("home_possessions", None),
                row.get("away_possessions", None),
                row.get("neutral_possessions", None),
            ],
        }

        split_df = pd.DataFrame(split_data)

        for col in ["Off Eff", "Def Eff", "Pace"]:
            split_df[col] = pd.to_numeric(split_df[col], errors="coerce").round(2)

        st.dataframe(
            split_df,
            use_container_width=True,
            hide_index=True
        )
        
elif page == "Daily Slate":
    st.title("Daily Slate Predictor")

    slate = fetch_today_slate()

    if slate.empty:
        st.warning("No games found for today.")
    else:
        rows = []

        for _, game in slate.iterrows():
            home = game["home_team"]
            away = game["away_team"]

            site_value = "neutral" if game["neutral_site"] else "team1_home"

            try:
                result = simulate_matchup(
                    team_stats_df=team_stats_df,
                    team1_name=home,
                    team2_name=away,
                    site_value=site_value,
                    n_sims=1000
                )

                rows.append({
                    "Game": f"{away} at {home}" if not game["neutral_site"] else f"{away} vs {home}",
                    "Projected Score": f"{result['team1']} {result['proj_score1']} - {result['team2']} {result['proj_score2']}",
                    "Favorite": result.get("favorite", "N/A"),
                    "Spread": result.get("spread", "N/A"),
                    "Total": result.get("total", "N/A"),
                    "Home Win %": f"{result['win_prob1'] * 100:.1f}%",
                })

            except Exception as e:
                rows.append({
                    "Game": f"{away} at {home}",
                    "Projected Score": "Unavailable",
                    "Favorite": "N/A",
                    "Spread": "N/A",
                    "Total": "N/A",
                    "Home Win %": "N/A",
                    "Error": str(e)
                })

        slate_df = pd.DataFrame(rows)

        st.dataframe(
            slate_df,
            use_container_width=True,
            hide_index=True
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

            
            st.markdown("---")

            st.title("Model Accuracy Dashboard")
            st.caption("Performance based on saved model predictions with completed results.")

            st.markdown("### Overall Performance")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games Graded", f"{len(completed):,}")
            m2.metric("Winner Accuracy", f"{completed['winner_correct'].mean():.1%}")
            m3.metric("Avg Spread Error", f"{completed['spread_error'].mean():.2f}")
            m4.metric("Avg Total Error", f"{completed['total_error'].mean():.2f}")

            st.markdown("---")

            left, right = st.columns(2)

            with left:
                st.markdown("### Spread Error Distribution")

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(completed["spread_error"].dropna(), bins=25)
                ax.set_xlabel("Spread Error")
                ax.set_ylabel("Games")
                ax.set_title("Spread Error")
                st.pyplot(fig)

            with right:
                st.markdown("### Total Error Distribution")

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.hist(completed["total_error"].dropna(), bins=25)
                ax2.set_xlabel("Total Error")
                ax2.set_ylabel("Games")
                ax2.set_title("Total Error")
                st.pyplot(fig2)

            st.markdown("---")

            st.markdown("### Recent Performance")

            recent = completed.tail(10)

            r1, r2, r3 = st.columns(3)
            r1.metric("Last 10 Accuracy", f"{recent['winner_correct'].mean():.1%}")
            r2.metric("Last 10 Spread Error", f"{recent['spread_error'].mean():.2f}")
            r3.metric("Last 10 Total Error", f"{recent['total_error'].mean():.2f}")

            st.markdown("---")

            st.markdown("### Game Log")
            st.dataframe(
                completed.sort_values("game_date", ascending=False),
                use_container_width=True,
                hide_index=True
)
            st.markdown("---")
            st.markdown("### Error Distribution")

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.hist(completed["spread_error"].dropna(), bins=25)
            ax.set_xlabel("Spread Error")
            ax.set_ylabel("Games")
            ax.set_title("Distribution of Spread Prediction Error")

            st.pyplot(fig)

            # Game log
            st.markdown("### Game Log")

            st.dataframe(
                completed.sort_values("game_date", ascending=False),
                use_container_width=True
            )