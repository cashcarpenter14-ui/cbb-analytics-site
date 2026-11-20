import re
import base64
import json
from io import BytesIO
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from PIL import Image, ImageChops

from model import simulate_matchup


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
LOGO_DIR = ASSETS_DIR / "fronedge_logos"
TEAM_LOGO_DIR = ASSETS_DIR / "team_logos"

MAIN_LOGO_PATH = LOGO_DIR / "TheJersey1.svg"
SIDEBAR_LOGO_PATH = LOGO_DIR / "TheJersey1.svg"
ALT_LOGO_1 = LOGO_DIR / "FronEdgeScript.svg"
ALT_LOGO_2 = LOGO_DIR / "FronEdgeMetricsLogo.svg"
ALT_LOGO_3 = LOGO_DIR / "TheSanAntone.svg"

# -----------------------------
# Page config + CSS
# -----------------------------
st.set_page_config(
    page_title="FronEdge Metrics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    :root {
        --fe-navy: #071b33;
        --fe-blue: #0b5ed7;
        --fe-gold: #f2b632;
        --fe-bg: #f6f8fb;
        --fe-card: #ffffff;
        --fe-border: #e3e9f1;
        --fe-muted: #637083;
    }

    .stApp {
        background: var(--fe-bg);
        color: var(--fe-navy);
    }

    .block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem;
    max-width: 1280px;
    }


    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061a31 0%, #020b16 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #f7fbff !important;
    }

    h1 {
        font-size: 2.15rem !important;
        font-weight: 850 !important;
        letter-spacing: -0.04em;
        color: var(--fe-navy);
        margin-bottom: 0.15rem !important;
    }

    h2, h3 {
        color: var(--fe-navy);
        letter-spacing: -0.025em;
    }

    .fe-hero-title {
        font-size: 2.25rem;
        font-weight: 900;
        letter-spacing: -0.045em;
        color: var(--fe-navy);
        margin-bottom: 0.25rem;
    }

    .fe-hero-subtitle {
        font-size: 1.02rem;
        color: #31445c;
        margin-bottom: 0.9rem;
    }

    .fe-section-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--fe-muted);
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .fe-card {
        background: var(--fe-card);
        border: 1px solid var(--fe-border);
        border-radius: 16px;
        padding: 14px 15px;
        box-shadow: 0 8px 20px rgba(7, 27, 51, 0.05);
        min-height: 76px;
        margin-bottom: 0.75rem;
    }

    .fe-card-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--fe-muted);
        font-weight: 800;
        margin-bottom: 6px;
    }

    .fe-card-value {
        font-size: 1.22rem;
        color: var(--fe-navy);
        font-weight: 850;
        line-height: 1.12;
    }

    .fe-card-sub {
        font-size: 0.76rem;
        color: var(--fe-blue);
        margin-top: 5px;
        font-weight: 650;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--fe-border);
        padding: 12px 14px;
        border-radius: 14px;
        box-shadow: 0 6px 16px rgba(7, 27, 51, 0.045);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 13px;
        border: 1px solid #e5ebf2;
        overflow: hidden;
        background: #ffffff;
    }

    .fe-insight {
        background: #ffffff;
        border: 1px solid var(--fe-border);
        border-left: 4px solid var(--fe-blue);
        border-radius: 14px;
        padding: 12px 14px;
        font-size: 0.92rem;
        color: var(--fe-navy);
        min-height: 74px;
        box-shadow: 0 6px 16px rgba(7, 27, 51, 0.04);
    }

    .fe-footer-caption {
        text-align: center;
        color: #5f6b7a;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }


    .fe-rank-card {
        background: #ffffff;
        border: 1px solid var(--fe-border);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 8px 20px rgba(7, 27, 51, 0.05);
        margin-bottom: 0.75rem;
    }

    .fe-rank-header {
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--fe-muted);
        font-weight: 850;
        border-bottom: 1px solid #edf1f6;
        padding-bottom: 0.4rem;
        margin-bottom: 0.25rem;
    }

    .fe-rank-team {
        font-weight: 750;
        color: var(--fe-navy);
        line-height: 1.1;
    }

    .fe-rank-stat-blue {
        color: var(--fe-blue);
        font-weight: 800;
    }

    .fe-rank-stat-green {
        color: #059669;
        font-weight: 800;
    }

    .fe-logo-table {
        background: #ffffff;
        border: 1px solid var(--fe-border);
        border-radius: 16px;
        padding: 12px 14px;
        box-shadow: 0 8px 20px rgba(7, 27, 51, 0.05);
        margin-bottom: 0.75rem;
        overflow: hidden;
        max-width: 100%;
    }

    .fe-logo-table {
        font-size: 0.88rem;
    }

    .fe-logo-table-team {
        font-size: 0.88rem;
    }

    .fe-logo-table-blue,
    .fe-logo-table-green {
        font-size: 0.88rem;
    }   

    .fe-logo-row {
        display: grid;
        grid-template-columns: 30px 34px minmax(135px, 1fr) 58px 58px 58px;
        align-items: center;
        column-gap: 6px;
        padding: 6px 2px;
        border-bottom: 1px solid #edf1f6;
    }

    .fe-logo-row:last-child {
        border-bottom: none;
    }

    .fe-logo-header {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--fe-muted);
        font-weight: 850;
        padding-bottom: 8px;
    }

    .fe-rank-num {
        font-weight: 850;
        color: var(--fe-navy);
        white-space: nowrap;
        text-align: left;
        min-width: 32px;
    }

    .fe-team-logo-wrap {
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .fe-team-logo-wrap img {
        max-width: 27px;
        max-height: 27px;
        object-fit: contain;
        display: block;
    }

    .fe-team-logo-placeholder {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: #eef3fb;
        border: 1px solid #d7e0ee;
    }

    .fe-logo-table-team {
        font-weight: 850;
        color: var(--fe-navy);
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .fe-logo-table-blue {
        color: var(--fe-blue);
        font-weight: 850;
        text-align: right;
    }

    .fe-logo-table-green {
        color: #059669;
        font-weight: 850;
        text-align: right;
    }



    .fe-mini-card {
        background: #ffffff;
        border: 1px solid var(--fe-border);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 8px 20px rgba(7, 27, 51, 0.045);
        min-height: 214px;
        overflow: hidden;
    }

    .fe-insight-list {
        border-left: 4px solid var(--fe-blue);
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 13px;
        font-size: 0.92rem;
        line-height: 1.35;
        color: var(--fe-navy);
    }

    .fe-insight-list b {
        font-weight: 850;
    }

    .fe-top-performer-row {
        display: grid;
        grid-template-columns: 28px minmax(0, 1fr) 58px 72px;
        align-items: center;
        gap: 8px;
        padding: 8px 0;
        border-bottom: 1px solid #edf1f6;
        font-size: 0.84rem;
        width: 100%;
        box-sizing: border-box;
    }

    .fe-top-performer-row:last-child {
        border-bottom: none;
    }

    .fe-top-performer-row img {
        max-width: 24px;
        max-height: 24px;
        object-fit: contain;
        display: block;
    }

    .fe-top-performer-team {
        font-weight: 800;
        color: var(--fe-navy);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 0;
    }

    .fe-top-performer-value {
        font-weight: 850;
        color: #059669;
        text-align: right;
        white-space: nowrap;
    }

    .fe-top-performer-cat {
        color: #31445c;
        font-size: 0.76rem;
        text-align: right;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .fe-view-link {
        margin-top: -4px;
        margin-bottom: 4px;
        color: var(--fe-blue);
        font-weight: 750;
        font-size: 0.86rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Data
# -----------------------------
team_stats_path = DATA_DIR / "team_stats_current.csv"
team_rankings_path = DATA_DIR / "team_rankings.csv"
metadata_path = DATA_DIR / "model_metadata.json"
branding_path = DATA_DIR / "team_branding.csv"
predictions_path = DATA_DIR / "model_predictions.csv"
backtest_results_path = DATA_DIR / "backtest_results_current.csv"
backtest_summary_path = DATA_DIR / "backtest_summary_current.csv"
postseason_results_path = DATA_DIR / "backtest_results_postseason.csv"
postseason_summary_path = DATA_DIR / "backtest_summary_postseason.csv"

team_stats_df = load_csv(team_stats_path) if team_stats_path.exists() else pd.DataFrame()
team_rankings_df = load_csv(team_rankings_path) if team_rankings_path.exists() else pd.DataFrame()
metadata = load_json(metadata_path) if metadata_path.exists() else {}
branding_df = load_csv(branding_path) if branding_path.exists() else pd.DataFrame()
predictions_df = load_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
backtest_results_df = load_csv(backtest_results_path) if backtest_results_path.exists() else pd.DataFrame()
backtest_summary_df = load_csv(backtest_summary_path) if backtest_summary_path.exists() else pd.DataFrame()
postseason_results_df = load_csv(postseason_results_path) if postseason_results_path.exists() else pd.DataFrame()
postseason_summary_df = load_csv(postseason_summary_path) if postseason_summary_path.exists() else pd.DataFrame()


# -----------------------------
# Helpers
# -----------------------------
def _trim_transparent_image(img):
    """Trim transparent/white padding so team logos render at a consistent visual size."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def _normalized_logo_array(logo_path, canvas_size=220, padding=18):
    """Return a square RGBA numpy-friendly image with the logo centered."""
    img = Image.open(str(logo_path)).convert("RGBA")
    img = _trim_transparent_image(img)

    max_side = canvas_size - (padding * 2)
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
    x = (canvas_size - img.width) // 2
    y = (canvas_size - img.height) // 2
    canvas.paste(img, (x, y), img)
    return canvas


def add_logo_to_chart(ax, x, y, logo_path, zoom=0.08):
    try:
        img = _normalized_logo_array(logo_path)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(
            imagebox,
            (x, y),
            frameon=False,
            zorder=6,
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(ab)
    except Exception:
        ax.scatter(x, y, s=35, zorder=6)


def clean_numeric(df):
    df = df.copy()

    for col in df.columns:

        # Skip obvious text/object columns
        if df[col].dtype == "object":

            # Only convert if MOST values look numeric
            converted = pd.to_numeric(df[col], errors="coerce")

            valid_ratio = converted.notna().mean()

            if valid_ratio > 0.8:
                df[col] = converted

        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def safe_metric_value(row, col, default=0, decimals=2):
    value = row.get(col, default)
    try:
        if pd.isna(value):
            return default
        if isinstance(value, (int, float)):
            return round(value, decimals)
        return value
    except Exception:
        return default


def render_metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="fe-card">
            <div class="fe-card-title">{title}</div>
            <div class="fe-card-value">{value}</div>
            <div class="fe-card-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(text):
    st.markdown(f"<div class='fe-insight'>{text}</div>", unsafe_allow_html=True)


def section_label(text):
    st.markdown(f"<div class='fe-section-label'>{text}</div>", unsafe_allow_html=True)


def logo_data_uri(logo_path):
    """Embed local logos as data URIs so Streamlit Cloud renders them clearly in HTML tables."""
    try:
        path = Path(logo_path)
        suffix = path.suffix.lower()
        mime = "image/png"
        if suffix == ".svg":
            mime = "image/svg+xml"
        elif suffix in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"

        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def show_top_table(title, df, sort_col, cols, ascending=False, n=10):
    section_label(title)

    if df.empty or sort_col not in df.columns:
        st.info("Data not available yet.")
        return

    display = df.sort_values(sort_col, ascending=ascending).head(n).reset_index(drop=True)
    show_cols = [c for c in cols if c in display.columns]
    display = clean_numeric(display[show_cols])
    st.dataframe(display, use_container_width=True, hide_index=True)



def render_logo_rankings_table(
    title,
    df,
    sort_col,
    ascending=False,
    n=10,
    net_col="Overall_Rating",
    net_label="Net",
):
    section_label(title)

    if df.empty or sort_col not in df.columns:
        st.info("Data not available yet.")
        return

    needed = ["Team", "off_eff", "def_eff", net_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.info("Ranking data not available yet.")
        return

    display = df.sort_values(sort_col, ascending=ascending).head(n).reset_index(drop=True)

    rows_html = [
        "<div class='fe-logo-row fe-logo-header'>"
        "<div>#</div>"
        "<div></div>"
        "<div>Team</div>"
        "<div style='text-align:right;'>Off</div>"
        "<div style='text-align:right;'>Def</div>"
        f"<div style='text-align:right;'>{net_label}</div>"
        "</div>"
    ]

    for idx, row in display.iterrows():
        team = str(row.get("Team", "N/A"))
        logo_path = get_team_logo(team)
        logo_uri = logo_data_uri(logo_path) if logo_path else None

        if logo_uri:
            logo_html = f"<img src='{logo_uri}' alt='{team} logo' />"
        else:
            logo_html = "<div class='fe-team-logo-placeholder'></div>"

        rows_html.append(
            "<div class='fe-logo-row'>"
            f"<div class='fe-rank-num'>{idx + 1}</div>"
            f"<div class='fe-team-logo-wrap'>{logo_html}</div>"
            f"<div class='fe-logo-table-team'>{team}</div>"
            f"<div class='fe-logo-table-blue'>{safe_metric_value(row, 'off_eff')}</div>"
            f"<div class='fe-logo-table-green'>{safe_metric_value(row, 'def_eff')}</div>"
            f"<div class='fe-logo-table-blue'>{safe_metric_value(row, net_col)}</div>"
            "</div>"
        )

    table_html = "<div class='fe-logo-table'>" + "".join(rows_html) + "</div>"
    st.markdown(table_html, unsafe_allow_html=True)



def render_top_performers_card(rows):
    html_rows = []
    for row in rows:
        team = str(row.get("Team", "N/A"))
        logo_path = get_team_logo(team)
        logo_uri = logo_data_uri(logo_path) if logo_path else None
        logo_html = f"<img src='{logo_uri}' alt='{team} logo' />" if logo_uri else "<div class='fe-team-logo-placeholder'></div>"
        html_rows.append(
            "<div class='fe-top-performer-row'>"
            f"<div>{logo_html}</div>"
            f"<div class='fe-top-performer-team'>{team}</div>"
            f"<div class='fe-top-performer-value'>{row.get('Value', 'N/A')}</div>"
            f"<div class='fe-top-performer-cat'>{row.get('Category', '')}</div>"
            "</div>"
        )
    st.markdown("<div class='fe-mini-card'>" + "".join(html_rows) + "</div>", unsafe_allow_html=True)


def render_insight_list_card(lines):
    if not lines:
        lines = ["Insight data unavailable."]
    items = "".join(f"<div>{line}</div>" for line in lines)
    st.markdown(f"<div class='fe-mini-card fe-insight-list'>{items}</div>", unsafe_allow_html=True)

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

    logo_path = TEAM_LOGO_DIR / file_name
    return logo_path if logo_path.exists() else None


def render_footer():
    st.markdown("---")
    f1, f2, f3 = st.columns([1.2, 1, 1])
    with f1:
        if ALT_LOGO_1.exists():
            st.image(str(ALT_LOGO_1), width=230)
    with f2:
        if ALT_LOGO_2.exists():
            st.image(str(ALT_LOGO_2), width=150)
    with f3:
        if ALT_LOGO_3.exists():
            st.image(str(ALT_LOGO_3), width=140)


def render_small_efficiency_map(df):
    if df.empty or not {"off_eff", "def_eff"}.issubset(df.columns):
        st.info("Efficiency map unavailable.")
        return

    chart_df = df.copy()
    chart_df["off_eff"] = pd.to_numeric(chart_df["off_eff"], errors="coerce")
    chart_df["def_eff"] = pd.to_numeric(chart_df["def_eff"], errors="coerce")
    chart_df = chart_df.dropna(subset=["off_eff", "def_eff"])

    if chart_df.empty:
        st.info("Efficiency map unavailable.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.8))

    x_min, x_max = chart_df["off_eff"].min(), chart_df["off_eff"].max()
    y_min, y_max = chart_df["def_eff"].min(), chart_df["def_eff"].max()
    x_pad = max((x_max - x_min) * 0.07, 1)
    y_pad = max((y_max - y_min) * 0.07, 1)

    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    off_mid = chart_df["off_eff"].median()
    def_mid = chart_df["def_eff"].median()

    # Colored performance quadrants. Lower defensive efficiency is better.
    quadrants = [
        (x_min, y_min, off_mid - x_min, def_mid - y_min, "#f9f9ea"),      # weak offense, strong defense
        (off_mid, y_min, x_max - off_mid, def_mid - y_min, "#e9f7ef"),    # strong offense, strong defense
        (x_min, def_mid, off_mid - x_min, y_max - def_mid, "#fdeeee"),    # weak offense, weak defense
        (off_mid, def_mid, x_max - off_mid, y_max - def_mid, "#d9e0ed"),  # strong offense, weak defense
    ]

    for x, y, w, h, color in quadrants:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=0))

    ax.scatter(chart_df["off_eff"], chart_df["def_eff"], s=14, alpha=0.60, zorder=2)
    ax.axvline(off_mid, linestyle="--", linewidth=1, zorder=3)
    ax.axhline(def_mid, linestyle="--", linewidth=1, zorder=3)

    top_logo_df = (
        chart_df.sort_values("Overall_Rating", ascending=False).head(10)
        if "Overall_Rating" in chart_df.columns
        else chart_df.head(0)
    )
    for _, row in top_logo_df.iterrows():
        logo_path = get_team_logo(row.get("Team", ""))
        if logo_path:
            add_logo_to_chart(ax, row["off_eff"], row["def_eff"], logo_path, zoom=0.065)

    ax.text(x_min + x_pad * 0.4, y_min + y_pad * 0.7, "Strong Defense\nWeak Offense", fontsize=8, color="#d7c60b", va="top")
    ax.text(x_max - x_pad * 0.4, y_min + y_pad * 0.7, "Strong Offense\nStrong Defense", fontsize=8, color="#047857", ha="right", va="top")
    ax.text(x_min + x_pad * 0.4, y_max - y_pad * 0.7, "Weak Offense\nWeak Defense", fontsize=8, color="#b91c1c", va="bottom")
    ax.text(x_max - x_pad * 0.4, y_max - y_pad * 0.7, "Strong Offense\nWeak Defense", fontsize=8, color="#1b0e92", ha="right", va="bottom")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_xlabel("Offensive Efficiency")
    ax.set_ylabel("Defensive Efficiency")
    ax.set_title("Efficiency Landscape")
    st.pyplot(fig, use_container_width=True)


def render_clean_efficiency_map(df):
    """Clean, product-style efficiency map used on dashboard/rankings pages.

    This version is intentionally simple:
    - quiet dots for the full D1 field
    - same-size logos for top teams only
    - no outline rings, no legend, no strength-of-schedule bubbles
    - larger plot area so it fills the card cleanly
    """
    if df.empty or not {"off_eff", "def_eff"}.issubset(df.columns):
        st.info("Efficiency map unavailable.")
        return

    chart_df = df.copy()
    chart_df["off_eff"] = pd.to_numeric(chart_df["off_eff"], errors="coerce")
    chart_df["def_eff"] = pd.to_numeric(chart_df["def_eff"], errors="coerce")
    chart_df = chart_df.dropna(subset=["off_eff", "def_eff"])

    if chart_df.empty:
        st.info("Efficiency map unavailable.")
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.65), dpi=125)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_min, x_max = chart_df["off_eff"].min(), chart_df["off_eff"].max()
    y_min, y_max = chart_df["def_eff"].min(), chart_df["def_eff"].max()

    x_pad = max((x_max - x_min) * 0.09, 1.5)
    y_pad = max((y_max - y_min) * 0.09, 1.5)

    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    off_mid = chart_df["off_eff"].median()
    def_mid = chart_df["def_eff"].median()

    # Lower defensive efficiency is better, so the y-axis is inverted below.
    quadrant_specs = [
        (x_min, y_min, off_mid - x_min, def_mid - y_min, "#f8f9e8"),
        (off_mid, y_min, x_max - off_mid, def_mid - y_min, "#eaf7f0"),
        (x_min, def_mid, off_mid - x_min, y_max - def_mid, "#fdeeee"),
        (off_mid, def_mid, x_max - off_mid, y_max - def_mid, "#fff5df"),
    ]

    for x, y, w, h, color in quadrant_specs:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=color,
                edgecolor="none",
                alpha=0.58,
                zorder=0,
            )
        )

    ax.scatter(
        chart_df["off_eff"],
        chart_df["def_eff"],
        s=8,
        alpha=0.20,
        color="#1f8ae0",
        linewidths=0,
        zorder=2,
    )

    if "Overall_Rating" in chart_df.columns:
        top_logo_df = chart_df.sort_values("Overall_Rating", ascending=False).head(10)
    else:
        top_logo_df = chart_df.head(0)

    for _, row in top_logo_df.iterrows():
        logo_path = get_team_logo(row.get("Team", ""))
        if logo_path:
            add_logo_to_chart(ax, row["off_eff"], row["def_eff"], logo_path, zoom=0.046)

    ax.axvline(off_mid, linestyle="--", linewidth=0.9, color="#3b82f6", alpha=0.65, zorder=3)
    ax.axhline(def_mid, linestyle="--", linewidth=0.9, color="#3b82f6", alpha=0.65, zorder=3)

    label_kwargs = {"fontsize": 7.0, "fontweight": "600"}
    ax.text(
        x_min + x_pad * 0.55,
        y_min + y_pad * 0.72,
        "Strong Defense\nWeak Offense",
        color="#4d7c0f",
        va="top",
        **label_kwargs,
    )
    ax.text(
        x_max - x_pad * 0.55,
        y_min + y_pad * 0.72,
        "Strong Defense\nStrong Offense",
        color="#047857",
        ha="right",
        va="top",
        **label_kwargs,
    )
    ax.text(
        x_min + x_pad * 0.55,
        y_max - y_pad * 0.72,
        "Weak Defense\nWeak Offense",
        color="#b91c1c",
        va="bottom",
        **label_kwargs,
    )
    ax.text(
        x_max - x_pad * 0.55,
        y_max - y_pad * 0.72,
        "Weak Defense\nStrong Offense",
        color="#92400e",
        ha="right",
        va="bottom",
        **label_kwargs,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_xlabel("Offensive Efficiency", fontsize=8)
    ax.set_ylabel("Defensive Efficiency", fontsize=8)
    ax.set_title("Efficiency Landscape", fontsize=10, pad=7)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.30)
    ax.spines["bottom"].set_alpha(0.30)
    ax.tick_params(axis="both", labelsize=7)

    # Tight margins keep the plot from looking tiny inside its card.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.125)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_full_efficiency_map(df):
    """Use the same clean visual language for the full map so it never falls back to the old cluttered version."""
    render_clean_efficiency_map(df)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    if SIDEBAR_LOGO_PATH.exists():
        st.image(str(SIDEBAR_LOGO_PATH), width=155)
    
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Ratings & Rankings",
            "Matchup Predictor",
            "Daily Slate",
            "Teams",
            "Bubble Board",
            "Team Comparison",
            "Model Accuracy",
        ],
        key="main_navigation",
    )

    
navigation_pages = [
    "Home",
    "Ratings & Rankings",
    "Matchup Predictor",
    "Daily Slate",
    "Teams",
    "Bubble Board",
    "Team Comparison",
    "Model Accuracy",
]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"
@st.cache_data
def fetch_today_slate(test_mode=False):
    if test_mode:
        df = pd.read_csv("raw_data/full_season_games.csv")
        test_date = "2025-11-03"
        return df[df["date"].astype(str).str.contains(test_date)].copy()

    today = datetime.now().strftime("%Y%m%d")
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"mens-college-basketball/scoreboard?dates={today}"
    )

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


# -----------------------------
# Pages
# -----------------------------
if page == "Home":
    st.markdown(
        """
        <div class="fe-hero-title">FronEdge Metrics-College Basketball Analytics</div>
        <div class="fe-hero-subtitle">Data-driven rankings, projections, and tournament insights built around predictive strength and schedule-adjusted performance.</div>
        """,
        unsafe_allow_html=True,
    )

    if not team_stats_df.empty:
        teams_count = len(team_stats_df)
        top_team = team_stats_df.sort_values("Overall_Rating", ascending=False).iloc[0]["Team"] if "Overall_Rating" in team_stats_df.columns else "N/A"
        top_dae = team_stats_df.sort_values("Difficulty_Adjusted_Effectiveness", ascending=False).iloc[0]["Team"] if "Difficulty_Adjusted_Effectiveness" in team_stats_df.columns else "N/A"
        bubble_count = team_stats_df["bubble_status"].astype(str).str.contains("Bubble", case=False, na=False).sum() if "bubble_status" in team_stats_df.columns else 0
        projected_field = int((team_stats_df["Tournament_Rank"] <= 76).sum()) if "Tournament_Rank" in team_stats_df.columns else 0
    else:
        teams_count, top_team, top_dae, bubble_count, projected_field = 0, "N/A", "N/A", 0, 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Teams Tracked", f"{teams_count:,}", "D1 teams")
    with c2:
        render_metric_card("Top Overall", top_team, "Predictive leader")
    with c3:
        render_metric_card("Top Schedule-Tested", top_dae, "Difficulty-adjusted")
    with c4:
        render_metric_card("Projected Field", projected_field, f"{bubble_count} bubble teams")

    st.markdown("")

    if team_rankings_df.empty:
        st.warning("No rankings found. Run pipeline.py first.")
    else:
        top_left, top_right = st.columns([1, 1.05])
        with top_left:
            render_logo_rankings_table(
                "Top 10 Overall",
                team_rankings_df,
                "Overall_Rating",
                ascending=False,
                n=10,
            )
        with top_right:
            section_label("Efficiency Map")
            render_small_efficiency_map(team_rankings_df)

        lower_left, lower_right = st.columns([1, 1])

        
        with lower_left:
            section_label("Today's Slate")

            slate = fetch_today_slate()

            if slate.empty:
                st.info("No games scheduled today.")

            else:
                valid_teams = (
                    set(team_stats_df["Team"].dropna())
                    if not team_stats_df.empty
                    else set()
                )

                games_shown = 0

                for game_idx, (_, game) in enumerate(
                    slate.iterrows()
                ):
                    home = game.get("home_team")
                    away = game.get("away_team")

                    if (
                        home not in valid_teams
                        or away not in valid_teams
                    ):
                        continue

                    neutral = game.get(
                        "neutral_site", False
                    )

                    if pd.isna(neutral):
                        neutral = False
                    elif isinstance(neutral, str):
                        neutral = (
                            neutral.strip().lower()
                            in ("true", "1", "yes")
                        )
                    else:
                        neutral = bool(neutral)

                    if neutral:
                        matchup = f"{away} vs. {home}"
                    else:
                        matchup = f"{away} @ {home}"

                    # Preserve the original tipoff time
                    game_date = game.get("date")

                    if pd.notna(game_date):
                        tipoff = pd.Timestamp(
                            game_date
                        ).tz_convert(
                            "America/Chicago"
                        ).strftime("%I:%M %p CT")
                    else:
                        tipoff = "TBD"

                    button_label = (
                        f"{tipoff}  |  {matchup}"
                    )

                    if st.button(
                        button_label,
                        key=f"home_slate_{game_idx}",
                        use_container_width=True,
                    ):
                        st.session_state[
                            "daily_slate_selected_game"
                        ] = {
                            "home_team": home,
                            "away_team": away,
                        }

                        st.session_state[
                            "selected_page"
                        ] = "Daily Slate"

                        st.rerun()

                    games_shown += 1

                    if games_shown >= 6:
                        break

                if games_shown == 0:
                    st.info(
                        "No D1 matchups available today."
                    )

        with lower_right:
            show_top_table(
                "Top 10 Difficulty-Adjusted",
                team_rankings_df,
                "Difficulty_Adjusted_Effectiveness",
                ["Team", "Difficulty_Adjusted_Effectiveness", "SOS_Rating", "Resume_Score"],
                ascending=False,
                n=10,
            )

        section_label("Quick Insights")
        i1, i2, i3, i4 = st.columns(4)
        if not team_stats_df.empty:
            best_off = team_stats_df.sort_values("off_eff", ascending=False).iloc[0]
            best_def = team_stats_df.sort_values("def_eff", ascending=True).iloc[0]
            best_sos = team_stats_df.sort_values("SOS_Rating", ascending=False).iloc[0] if "SOS_Rating" in team_stats_df.columns else None
            with i1:
                render_insight(f"<b>{top_team}</b><br>leads overall rating.")
            with i2:
                render_insight(f"<b>{best_off['Team']}</b><br>owns the top offense.")
            with i3:
                render_insight(f"<b>{best_def['Team']}</b><br>owns the top defense.")
            with i4:
                render_insight(f"<b>{best_sos['Team']}</b><br>is most schedule-tested." if best_sos is not None else "SOS data unavailable.")

elif page == "Ratings & Rankings":
    st.title("Ratings & Rankings")
    st.caption("Explore team ratings, efficiency metrics, and rankings that power the predictive model.")

    if team_rankings_df.empty:
        st.warning("No team rankings found. Run pipeline.py first.")
    else:
        rankings_source = team_rankings_df.copy()

        if "Difficulty_Adjusted_Effectiveness" not in rankings_source.columns:
            if "Overall_Rating" in rankings_source.columns and "SOS_Rating" in rankings_source.columns:
                rankings_source["Difficulty_Adjusted_Effectiveness"] = (
                    rankings_source["Overall_Rating"] + rankings_source["SOS_Rating"]
                )
            else:
                rankings_source["Difficulty_Adjusted_Effectiveness"] = rankings_source.get("Overall_Rating", 0)

        # --- Summary cards ---
        teams_count = len(rankings_source)
        top_overall_row = rankings_source.sort_values("Overall_Rating", ascending=False).iloc[0] if "Overall_Rating" in rankings_source.columns else None
        top_off_row = rankings_source.sort_values("off_eff", ascending=False).iloc[0] if "off_eff" in rankings_source.columns else None
        top_def_row = rankings_source.sort_values("def_eff", ascending=True).iloc[0] if "def_eff" in rankings_source.columns else None
        top_dae_row = rankings_source.sort_values("Difficulty_Adjusted_Effectiveness", ascending=False).iloc[0]

        card1, card2, card3, card4 = st.columns(4)
        with card1:
            render_metric_card("Teams Tracked", f"{teams_count:,}", "D1 teams")
        with card2:
            render_metric_card(
                "Top Overall Rating",
                top_overall_row["Team"] if top_overall_row is not None else "N/A",
                f"{safe_metric_value(top_overall_row, 'Overall_Rating') if top_overall_row is not None else 'N/A'} rating",
            )
        with card3:
            render_metric_card(
                "Top Offensive Rating",
                top_off_row["Team"] if top_off_row is not None else "N/A",
                f"{safe_metric_value(top_off_row, 'off_eff') if top_off_row is not None else 'N/A'} off eff",
            )
        with card4:
            render_metric_card(
                "Top Defensive Rating",
                top_def_row["Team"] if top_def_row is not None else "N/A",
                f"{safe_metric_value(top_def_row, 'def_eff') if top_def_row is not None else 'N/A'} def eff",
            )

        st.markdown("")

        # --- Main ratings module: one rankings card + one clean map card ---
        left, right = st.columns([1.03, 1.07])

        with left:
            tab_overall, tab_offense, tab_defense, tab_dae = st.tabs([
                "Top 10 Overall",
                "Top Offensive",
                "Top Defensive",
                "Difficulty-Adjusted",
            ])

            with tab_overall:
                render_logo_rankings_table("Top 10 Overall", rankings_source, "Overall_Rating", False, 10, "Overall_Rating", "Net")
                st.markdown("<div class='fe-view-link'>View full rankings →</div>", unsafe_allow_html=True)

            with tab_offense:
                render_logo_rankings_table("Top Offensive", rankings_source, "off_eff", False, 10, "Overall_Rating", "Net")
                st.markdown("<div class='fe-view-link'>View full offense rankings →</div>", unsafe_allow_html=True)

            with tab_defense:
                render_logo_rankings_table("Top Defensive", rankings_source, "def_eff", True, 10, "Overall_Rating", "Net")
                st.markdown("<div class='fe-view-link'>View full defense rankings →</div>", unsafe_allow_html=True)

            with tab_dae:
                render_logo_rankings_table("Difficulty-Adjusted", rankings_source, "Difficulty_Adjusted_Effectiveness", False, 10, "Difficulty_Adjusted_Effectiveness", "DAE")
                st.markdown("<div class='fe-view-link'>View difficulty-adjusted rankings →</div>", unsafe_allow_html=True)

        with right:
            section_label("Efficiency Landscape")
            with st.container(border=True):
                render_clean_efficiency_map(rankings_source)

        st.markdown("---")

        # --- Supporting analytics ---
        support_left, support_mid, support_right = st.columns([1, 1.15, 1])

        with support_left:
            section_label("Rating Distribution")
            if "Overall_Rating" in rankings_source.columns:
                with st.container(border=True):
                    fig, ax = plt.subplots(figsize=(4.8, 2.55), dpi=140)
                    values = pd.to_numeric(rankings_source["Overall_Rating"], errors="coerce").dropna()
                    ax.hist(values, bins=20)
                    ax.set_xlabel("Overall Rating", fontsize=8)
                    ax.set_ylabel("Teams", fontsize=8)
                    ax.set_title("Overall Rating Distribution", fontsize=9)
                    ax.tick_params(axis="both", labelsize=7)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    fig.tight_layout(pad=0.7)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            else:
                st.info("Rating distribution unavailable.")

        with support_mid:
            section_label("Key Insights")
            insight_lines = []
            if top_overall_row is not None:
                insight_lines.append(f"<b>{top_overall_row['Team']}</b> leads the overall model rating.")
            if top_off_row is not None:
                insight_lines.append(f"<b>{top_off_row['Team']}</b> owns the strongest offensive profile.")
            if top_def_row is not None:
                insight_lines.append(f"<b>{top_def_row['Team']}</b> owns the strongest defensive profile.")
            if top_dae_row is not None:
                insight_lines.append(f"<b>{top_dae_row['Team']}</b> leads difficulty-adjusted effectiveness.")
            render_insight_list_card(insight_lines)

        with support_right:
            section_label("Top Performers")
            performer_rows = []
            if top_overall_row is not None:
                performer_rows.append({"Team": top_overall_row["Team"], "Value": safe_metric_value(top_overall_row, "Overall_Rating"), "Category": "Overall"})
            if top_off_row is not None:
                performer_rows.append({"Team": top_off_row["Team"], "Value": safe_metric_value(top_off_row, "off_eff"), "Category": "Offense"})
            if top_def_row is not None:
                performer_rows.append({"Team": top_def_row["Team"], "Value": safe_metric_value(top_def_row, "def_eff"), "Category": "Defense"})
            if top_dae_row is not None:
                performer_rows.append({"Team": top_dae_row["Team"], "Value": safe_metric_value(top_dae_row, "Difficulty_Adjusted_Effectiveness"), "Category": "Difficulty Adj."})
            render_top_performers_card(performer_rows)

        st.markdown("---")
        st.markdown("### Full Rankings Table")

        metric_view = st.radio("View", ["Overall", "Difficulty-Adjusted", "Offense", "Defense", "Tempo"], horizontal=True)
        rankings_df = rankings_source.copy()

        if metric_view == "Overall":
            sort_col = "Overall_Rating" if "Overall_Rating" in rankings_df.columns else "Elo"
            show_cols = ["Team", "Overall_Rating", "Power_Rating", "Elo", "off_eff", "def_eff", "possessions"]
            ascending = False
        elif metric_view == "Difficulty-Adjusted":
            sort_col = "Difficulty_Adjusted_Effectiveness"
            show_cols = ["Team", "Difficulty_Adjusted_Effectiveness", "Overall_Rating", "Power_Rating", "SOS_Rating", "Resume_Score", "Tournament_Rank"]
            ascending = False
        elif metric_view == "Offense":
            sort_col = "off_eff"
            show_cols = ["Team", "off_eff", "def_eff", "Overall_Rating", "Elo", "possessions"]
            ascending = False
        elif metric_view == "Defense":
            sort_col = "def_eff"
            show_cols = ["Team", "def_eff", "off_eff", "Overall_Rating", "Elo", "possessions"]
            ascending = True
        else:
            sort_col = "possessions"
            show_cols = ["Team", "possessions", "off_eff", "def_eff", "Overall_Rating", "Elo"]
            ascending = False

        if sort_col in rankings_df.columns:
            rankings_df = rankings_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

        rankings_df["Rank"] = range(1, len(rankings_df) + 1)
        show_cols = [c for c in ["Rank"] + show_cols if c in rankings_df.columns]
        st.dataframe(clean_numeric(rankings_df[show_cols]), use_container_width=True, hide_index=True)

elif page == "Matchup Predictor":
    st.title("Matchup Predictor")
    st.caption("Clean matchup projection with score, line, total, confidence, style, and detailed tabs.")

    if team_stats_df.empty:
        st.warning("No team stats found. Run pipeline.py first.")
    else:
        teams = sorted(team_stats_df["Team"].dropna().unique().tolist())

        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 2, 1.3])
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

            with st.container(border=True):
                left, middle, right = st.columns([2.5, 1.2, 2.5])

                with left:
                    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
                    if logo1 is not None:
                        st.image(str(logo1), width=105)
                    st.markdown(f"<h3 style='text-align:center; margin-bottom:0;'>{team1}</h3>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with middle:
                    site_text = "Neutral Site" if site == "neutral" else f"{team1} Home" if site == "team1_home" else f"{team2} Home"
                    st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)
                    st.markdown("<h1 style='text-align:center;'>VS</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; color:#637083;'>{site_text}</p>", unsafe_allow_html=True)

                with right:
                    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
                    if logo2 is not None:
                        st.image(str(logo2), width=105)
                    st.markdown(f"<h3 style='text-align:center; margin-bottom:0;'>{team2}</h3>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Run Prediction", use_container_width=True):
                try:
                    result = simulate_matchup(
                        team_stats_df=team_stats_df,
                        team1_name=team1,
                        team2_name=team2,
                        site_value=site,
                        n_sims=1000,
                    )

                    st.markdown("### Projection Summary")

                    score_text = f"{result['team1']} {result['proj_score1']} - {result['team2']} {result['proj_score2']}"
                    favorite = result.get("favorite", "Even")
                    spread = result.get("spread", 0)
                    total = result.get("total", "N/A")
                    confidence = result.get("confidence", "N/A")
                    game_style = result.get("game_style", "Balanced")
                    volatility_tag = result.get("volatility_tag", "N/A")
                    upset_alert = result.get("upset_alert", "N/A")

                    s1, s2, s3, s4 = st.columns(4)
                    with s1:
                        render_metric_card("Projected Score", score_text, "Simulation average")
                    with s2:
                        line_text = "Even" if favorite == "Even" else f"{favorite} {spread}"
                        render_metric_card("Projected Line", line_text, "Spread projection")
                    with s3:
                        render_metric_card("Projected Total", total, "Combined points")
                    with s4:
                        render_metric_card("Confidence", confidence, "Model confidence")

                    b1, b2, b3 = st.columns(3)
                    with b1:
                        render_metric_card("Game Style", game_style, "Matchup archetype")
                    with b2:
                        render_metric_card("Volatility", volatility_tag, "Scoring variance")
                    with b3:
                        render_metric_card("Upset Alert", upset_alert, "Risk indicator")

                    w1, w2 = st.columns(2)
                    with w1:
                        st.metric(f"{result['team1']} Win Probability", f"{result['win_prob1']:.1%}")
                    with w2:
                        st.metric(f"{result['team2']} Win Probability", f"{result['win_prob2']:.1%}")

                    # Team-colored win probability bar
                    import html

                    def get_probability_color(team_name, fallback):
                        if branding_df.empty:
                            return fallback

                        matches = branding_df[
                            branding_df["team"].astype(str).str.strip().str.casefold()
                            == str(team_name).strip().casefold()
                        ]

                        if matches.empty:
                            return fallback

                        value = matches.iloc[0].get("primary_color")
                        if pd.isna(value):
                            return fallback

                        value = str(value).strip().lstrip("#")
                        if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
                            return fallback

                        return "#" + value.upper()

                    color1 = get_probability_color(result["team1"], "#2563EB")
                    color2 = get_probability_color(result["team2"], "#DC2626")

                    prob1 = max(0.0, min(1.0, float(result["win_prob1"])))
                    prob2 = max(0.0, min(1.0, float(result["win_prob2"])))
                    total_prob = prob1 + prob2

                    if total_prob > 0:
                        prob1 /= total_prob
                        prob2 /= total_prob

                    team1_label = html.escape(str(result["team1"]))
                    team2_label = html.escape(str(result["team2"]))

                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;
                                    font-size:13px;font-weight:600;margin:12px 0 6px;">
                            <span>{team1_label}</span>
                            <span>{team2_label}</span>
                        </div>
                        <div style="display:flex;width:100%;height:22px;
                                    overflow:hidden;border-radius:7px;
                                    background:#e5e7eb;">
                            <div style="width:{prob1 * 100:.2f}%;
                                        background:{color1};"></div>
                            <div style="width:{prob2 * 100:.2f}%;
                                        background:{color2};"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    tabs = st.tabs(["Overview", "Matchup Factors", "Projected Box Score", "Team Profiles"])

                    with tabs[0]:
                        st.markdown("### Model Read")
                        if favorite != "Even":
                            render_insight(
                                f"<b>{favorite}</b> is projected as the favorite. "
                                f"This matchup is classified as <b>{game_style}</b> with <b>{volatility_tag}</b>."
                            )
                        else:
                            render_insight(
                                f"This projects as an even matchup. "
                                f"The model classifies it as <b>{game_style}</b> with <b>{volatility_tag}</b>."
                            )

                    with tabs[1]:
                        st.markdown("### Key Swing Factors")
                        box1 = result.get("box_score_team1", {})
                        box2 = result.get("box_score_team2", {})
                        factor_rows = []

                        def get_box_value(box, possible_keys):
                            for key in possible_keys:
                                if key in box:
                                    return box.get(key)
                            return None

                        def add_factor(label, keys, lower_is_better=False):
                            v1 = get_box_value(box1, keys)
                            v2 = get_box_value(box2, keys)
                            if v1 is None or v2 is None:
                                return

                            if lower_is_better:
                                edge = result["team1"] if v1 < v2 else result["team2"] if v2 < v1 else "Even"
                            else:
                                edge = result["team1"] if v1 > v2 else result["team2"] if v2 > v1 else "Even"

                            factor_rows.append({
                                "Factor": label,
                                result["team1"]: round(v1, 1),
                                result["team2"]: round(v2, 1),
                                "Edge": edge,
                                "Difference": round(abs(v1 - v2), 1),
                            })

                        add_factor("Assists", ["AST", "assists", "Assists"])
                        add_factor("Turnovers", ["TO", "TOV", "turnovers", "Turnovers"], lower_is_better=True)
                        add_factor("Offensive Rebounds", ["OREB", "offensive_rebounds", "Offensive Rebounds", "off_reb"])
                        add_factor("Defensive Rebounds", ["DREB", "defensive_rebounds", "Defensive Rebounds", "def_reb"])
                        add_factor("Total Rebounds", ["REB", "totalRebounds", "Total Rebounds"])
                        add_factor("Free Throw Attempts", ["FTA", "free_throw_attempts", "Free Throw Attempts"])

                        if factor_rows:
                            st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
                        else:
                            st.info("Key factor data is not available for this matchup yet.")

                    with tabs[2]:
                        st.markdown("### Projected Box Score")
                        box_df = pd.DataFrame([
                            {"Team": result["team1"], **result["box_score_team1"]},
                            {"Team": result["team2"], **result["box_score_team2"]},
                        ])
                        st.dataframe(box_df, use_container_width=True, hide_index=True)

                    with tabs[3]:
                        st.markdown("### Team Profile Snapshot")

                        row1 = team_stats_df[team_stats_df["Team"] == team1].iloc[0]
                        row2 = team_stats_df[team_stats_df["Team"] == team2].iloc[0]

                        profile_rows = []
                        for label, col, lower_better in [
                            ("Overall Rating", "Overall_Rating", False),
                            ("Offensive Efficiency", "off_eff", False),
                            ("Defensive Efficiency", "def_eff", True),
                            ("Tempo", "possessions", False),
                            ("SOS Rating", "SOS_Rating", False),
                            ("Difficulty Adj.", "Difficulty_Adjusted_Effectiveness", False),
                        ]:
                            if col in row1.index and col in row2.index:
                                v1 = row1[col]
                                v2 = row2[col]
                                if pd.notna(v1) and pd.notna(v2):
                                    edge = team1 if (v1 < v2 if lower_better else v1 > v2) else team2 if (v2 < v1 if lower_better else v2 > v1) else "Even"
                                    profile_rows.append({
                                        "Metric": label,
                                        team1: round(float(v1), 2),
                                        team2: round(float(v2), 2),
                                        "Edge": edge,
                                    })

                        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")


elif page == "Daily Slate":
    from daily_schedule import get_daily_schedule
    from datetime import date, timedelta
    from zoneinfo import ZoneInfo

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def slate_format_number(value, decimals=1):
        try:
            if value is None or pd.isna(value):
                return "N/A"
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)

    def slate_tipoff(value):
        try:
            timestamp = pd.to_datetime(value, utc=True)
            return timestamp.tz_convert(
                "America/Chicago"
            ).strftime("%I:%M %p CT")
        except (ValueError, TypeError):
            return "TBD"

    def slate_matchup(game):
        separator = "vs" if game["neutral"] else "@"
        return f"{game['away']} {separator} {game['home']}"

    def slate_site(game):
        return (
            "Neutral"
            if bool(game.get("neutral_site", False))
            else "Home"
        )

    def slate_team_profile(team_name):
        matches = team_stats_df[
            team_stats_df["Team"] == team_name
        ]
        if matches.empty:
            return None
        return matches.iloc[0]

    def slate_profile_comparison(home, away):
        home_row = slate_team_profile(home)
        away_row = slate_team_profile(away)

        if home_row is None or away_row is None:
            st.info("Team profile data is unavailable.")
            return

        metrics = [
            ("Overall Rating", "Overall_Rating", False),
            ("Offensive Efficiency", "off_eff", False),
            ("Defensive Efficiency", "def_eff", True),
            ("Tempo", "possessions", None),
            ("Strength of Schedule", "SOS_Rating", False),
            (
                "Difficulty-Adjusted",
                "Difficulty_Adjusted_Effectiveness",
                False,
            ),
        ]

        comparison = []

        for label, column, lower_better in metrics:
            if (
                column not in home_row.index
                or column not in away_row.index
            ):
                continue

            try:
                home_value = float(home_row[column])
                away_value = float(away_row[column])
            except (ValueError, TypeError):
                continue

            if pd.isna(home_value) or pd.isna(away_value):
                continue

            if lower_better is None:
                edge = "Style metric"
            elif home_value == away_value:
                edge = "Even"
            elif lower_better:
                edge = (
                    home if home_value < away_value
                    else away
                )
            else:
                edge = (
                    home if home_value > away_value
                    else away
                )

            comparison.append({
                "Metric": label,
                away: round(away_value, 2),
                home: round(home_value, 2),
                "Advantage": edge,
            })

        if comparison:
            st.dataframe(
                pd.DataFrame(comparison),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No comparable team metrics available.")

    # --------------------------------------------------
    # DATE NAVIGATION
    # --------------------------------------------------
    today_ct = datetime.now(
        ZoneInfo("America/Chicago")
    ).date()

    if "daily_slate_selected_date" not in st.session_state:
        st.session_state["daily_slate_selected_date"] = today_ct

    st.markdown("### Game Schedule")

    date_left, date_middle, date_right = st.columns(
        [1, 2, 1]
    )

    with date_left:
        if st.button(
            "Previous Day",
            use_container_width=True,
            key="slate_previous",
        ):
            st.session_state["daily_slate_selected_date"] -= (
                timedelta(days=1)
            )

    with date_right:
        if st.button(
            "Next Day",
            use_container_width=True,
            key="slate_next",
        ):
            st.session_state["daily_slate_selected_date"] += (
                timedelta(days=1)
            )

    with date_middle:
        st.date_input(
            "Select Game Date",
            key="daily_slate_selected_date",
        )

    selected_date = st.session_state[
        "daily_slate_selected_date"
    ]

    # --------------------------------------------------
    # FETCH SCHEDULE
    # --------------------------------------------------
    @st.cache_data(ttl=300, show_spinner=False)
    def load_daily_slate(game_date):
        return get_daily_schedule(game_date)

    try:
        with st.spinner("Loading daily games..."):
            slate = load_daily_slate(selected_date)
        schedule_error = None
    except Exception as e:
        slate = pd.DataFrame()
        schedule_error = str(e)

    if schedule_error:
        st.error(
            f"Unable to retrieve the schedule: {schedule_error}"
        )

    elif slate.empty:
        st.info("No games scheduled for this date.")

    else:
        # --------------------------------------------------
        # TEAM ELIGIBILITY
        # --------------------------------------------------
        valid_teams = (
            set(team_stats_df["Team"].dropna())
            if not team_stats_df.empty
            else set()
        )

        slate = slate.copy()

        slate["model_eligible"] = (
            slate["home_team"].isin(valid_teams)
            & slate["away_team"].isin(valid_teams)
        )

        # --------------------------------------------------
        # SUMMARY METRICS
        # --------------------------------------------------
        total_games = len(slate)
        d1_games = int(slate["model_eligible"].sum())
        other_games = total_games - d1_games

        neutral_games = int(
            slate["neutral_site"].fillna(False).sum()
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Total Games", total_games)

        with m2:
            st.metric("Model-Eligible Matchups", d1_games)

        with m3:
            st.metric("Without Model Data", other_games)

        with m4:
            st.metric("Neutral-Site Games", neutral_games)

        # --------------------------------------------------
        # FILTERS
        # --------------------------------------------------
        st.markdown("### Browse Matchups")

        filter_left, filter_middle, filter_right = (
            st.columns([2, 1.2, 1.2])
        )

        all_teams = sorted(
            set(slate["home_team"].dropna())
            | set(slate["away_team"].dropna())
        )

        with filter_left:
            selected_team = st.selectbox(
                "Filter by Team",
                ["All Teams"] + all_teams,
                key="slate_team_filter",
            )

        with filter_middle:
            status_filter = st.selectbox(
                "Game Status",
                ["All Games", "Scheduled", "Live", "Final"],
                key="slate_status_filter",
            )

        with filter_right:
            site_filter = st.selectbox(
                "Game Site",
                ["All Sites", "Neutral", "Home"],
                key="slate_site_filter",
            )

        filtered = slate.copy()

        if selected_team != "All Teams":
            filtered = filtered[
                (filtered["home_team"] == selected_team)
                | (filtered["away_team"] == selected_team)
            ]

        if site_filter == "Neutral":
            filtered = filtered[
                filtered["neutral_site"] == True
            ]
        elif site_filter == "Home":
            filtered = filtered[
                filtered["neutral_site"] == False
            ]

        if status_filter != "All Games":
            status_names = filtered["status"].fillna(
                ""
            ).astype(str).str.upper()

            if status_filter == "Scheduled":
                filtered = filtered[
                    status_names.str.contains(
                        "SCHEDULED|PRE",
                        regex=True,
                    )
                ]
            elif status_filter == "Live":
                filtered = filtered[
                    status_names.str.contains(
                        "IN_PROGRESS|INPROGRESS|HALFTIME",
                        regex=True,
                    )
                ]
            elif status_filter == "Final":
                filtered = filtered[
                    status_names.str.contains("FINAL")
                ]

        # --------------------------------------------------
        # GENERATE PROJECTIONS
        # --------------------------------------------------
        projected_games = []
        other_matchups = []

        for _, game in filtered.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            neutral = bool(game["neutral_site"])

            game_record = {
                "game_id": str(game["game_id"]),
                "home": home,
                "away": away,
                "tipoff": slate_tipoff(game["date"]),
                "site": slate_site(game),
                "status": game.get(
                    "status_detail", "Scheduled"
                ),
                "neutral": neutral,
                "away_score": game.get("away_score"),
                "home_score": game.get("home_score"),
                "result": None,
                "error": None,
            }

            if not game["model_eligible"]:
                other_matchups.append(game_record)
                continue

            try:
                result = simulate_matchup(
                    team_stats_df=team_stats_df,
                    team1_name=home,
                    team2_name=away,
                    site_value=(
                        "neutral"
                        if neutral
                        else "team1_home"
                    ),
                    n_sims=1000,
                )

                game_record["result"] = result

            except Exception as e:
                game_record["error"] = str(e)

            projected_games.append(game_record)

        # --------------------------------------------------
        # D1 MATCHUP OVERVIEW
        # --------------------------------------------------
        st.markdown("### Game Projections")

        st.caption(
            "Select a matchup below for expanded "
            "FronEdge analysis."
        )

        if not projected_games:
            st.info(
                "No model-eligible matchups match these filters."
            )

        else:
            overview_rows = []

            for game in projected_games:
                result = game["result"]

                if result is not None:
                    projected_score = (
                        f"{slate_format_number(result.get('proj_score2'))}"
                        " - "
                        f"{slate_format_number(result.get('proj_score1'))}"
                    )

                    favorite = result.get(
                        "favorite", "N/A"
                    )
                    spread = result.get("spread", "N/A")
                    total = result.get("total", "N/A")

                else:
                    projected_score = "N/A"
                    favorite = "N/A"
                    spread = "N/A"
                    total = "N/A"

                overview_rows.append({
                    "Tipoff": game["tipoff"],
                    "Matchup": slate_matchup(game),
                    "Site": game["site"],
                    "Status": game["status"],
                    "Projected Score (Away-Home)": (
                        projected_score
                    ),
                    "Favorite": favorite,
                })

            st.dataframe(
                pd.DataFrame(overview_rows),
                hide_index=True,
                use_container_width=True,
            )

            # --------------------------------------------------
            # EXPANDABLE MATCHUP ANALYSIS
            # --------------------------------------------------
            st.markdown("### Matchup Analysis")

            for game in projected_games:
                home = game["home"]
                away = game["away"]
                result = game["result"]

                heading = (
                    f"{game['tipoff']} | "
                    f"{slate_matchup(game)}"
                )

                with st.expander(heading):
                    st.caption(
                        f"{game['site']} Site | "
                        f"{game['status']}"
                    )

                    if result is None:
                        st.warning(
                            "A projection could not be generated "
                            "for this matchup."
                        )
                        if game["error"]:
                            st.caption(
                                f"Model error: {game['error']}"
                            )
                        continue

                    # --------------------------------------
                    # TEAM HEADER
                    # --------------------------------------
                    left, center, right = st.columns(
                        [2, 1, 2]
                    )

                    with left:
                        away_logo = get_team_logo(away)
                        if away_logo:
                            st.image(
                                str(away_logo),
                                width=75,
                            )
                        st.markdown(f"**{away}**")
                        st.metric(
                            "Projected Points",
                            slate_format_number(
                                result.get("proj_score2")
                            ),
                        )

                    with center:
                        st.markdown("### VS")
                        st.caption(game["site"])

                    with right:
                        home_logo = get_team_logo(home)
                        if home_logo:
                            st.image(
                                str(home_logo),
                                width=75,
                            )
                        st.markdown(f"**{home}**")
                        st.metric(
                            "Projected Points",
                            slate_format_number(
                                result.get("proj_score1")
                            ),
                        )

                    # --------------------------------------
                    # PROJECTION METRICS
                    # --------------------------------------
                    st.markdown("#### Projection Summary")

                    p1, p2, p3 = st.columns(3)

                    with p1:
                        st.metric(
                            "Favorite",
                            str(
                                result.get(
                                    "favorite", "N/A"
                                )
                            ),
                        )

                    with p2:
                        st.metric(
                            "Projected Spread",
                            slate_format_number(
                                result.get("spread")
                            ),
                        )

                    with p3:
                        st.metric(
                            "Projected Total",
                            slate_format_number(
                                result.get("total")
                            ),
                        )

                    # --------------------------------------
                    # WIN PROBABILITIES
                    # --------------------------------------
                    st.markdown("#### Win Probability")

                    away_probability = result.get(
                        "win_prob2"
                    )
                    home_probability = result.get(
                        "win_prob1"
                    )

                    if (
                        away_probability is not None
                        and home_probability is not None
                    ):
                        w1, w2 = st.columns(2)

                        with w1:
                            st.metric(
                                f"{away} Win Probability",
                                f"{float(away_probability):.1%}",
                            )

                        with w2:
                            st.metric(
                                f"{home} Win Probability",
                                f"{float(home_probability):.1%}",
                            )

                    # --------------------------------------
                    # ANALYSIS TABS
                    # --------------------------------------
                    tab1, tab2, tab3 = st.tabs(
                        [
                            "Matchup Factors",
                            "Projected Box Score",
                            "Model Insights",
                        ]
                    )

                    with tab1:
                        st.markdown(
                            "#### Team Statistical Comparison"
                        )
                        slate_profile_comparison(
                            home, away
                        )

                    with tab2:
                        st.markdown(
                            "#### Projected Box Score"
                        )

                        home_box = result.get(
                            "box_score_team1", {}
                        )
                        away_box = result.get(
                            "box_score_team2", {}
                        )

                        box_rows = [
                            {"Team": away, **away_box},
                            {"Team": home, **home_box},
                        ]

                        st.dataframe(
                            pd.DataFrame(box_rows),
                            hide_index=True,
                            use_container_width=True,
                        )

                    with tab3:
                        st.markdown(
                            "#### FronEdge Model Read"
                        )

                        insight_rows = [
                            (
                                "Confidence",
                                result.get(
                                    "confidence", "N/A"
                                ),
                            ),
                            (
                                "Game Style",
                                result.get(
                                    "game_style", "N/A"
                                ),
                            ),
                            (
                                "Volatility",
                                result.get(
                                    "volatility_tag", "N/A"
                                ),
                            ),
                            (
                                "Upset Alert",
                                result.get(
                                    "upset_alert", "N/A"
                                ),
                            ),
                        ]

                        for label, value in insight_rows:
                            st.markdown(
                                f"**{label}:** {value}"
                            )

        # --------------------------------------------------
        # GAMES WITHOUT MODEL PROJECTIONS
        # --------------------------------------------------
        if other_matchups:
            st.markdown("---")

            with st.expander(
                f"Other Scheduled Matchups "
                f"({len(other_matchups)})"
            ):
                st.caption(
                    "At least one team is not in the "
                    "current FronEdge model dataset. "
                    "These games are shown without projections."
                )

                other_rows = []

                for game in other_matchups:
                    other_rows.append({
                        "Tipoff": game["tipoff"],
                        "Away": game["away"],
                        "Home": game["home"],
                        "Site": game["site"],
                        "Status": game["status"],
                    })

                st.dataframe(
                    pd.DataFrame(other_rows),
                    hide_index=True,
                    use_container_width=True,
                )

        st.markdown("---")
        st.caption(
            "FronEdge Metrics projections are model "
            "estimates, not guaranteed outcomes. "
            "Game schedule and status data provided by ESPN."
        )

elif page == "Teams":
    st.title("Teams")
    st.caption("Cleaner team profile with ranking, efficiency, tournament resume, and recent form.")

    if team_stats_df.empty:
        st.warning("No team stats found. Run pipeline.py first.")
    else:
        team_stats_df["Team"] = team_stats_df["Team"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
        teams = sorted(team_stats_df["Team"].dropna().unique())

        search = st.text_input("Search Team")
        if search:
            teams = [t for t in teams if search.lower() in t.lower()]

        if not teams:
            st.warning("No teams found.")
        else:
            team = st.selectbox("Select a Team", teams)
            row = team_stats_df[team_stats_df["Team"] == team].iloc[0]
            logo = get_team_logo(team)

            rank_match = team_rankings_df[team_rankings_df["Team"] == team]
            if not rank_match.empty and "Rank" in rank_match.columns:
                national_rank = int(rank_match["Rank"].iloc[0])
                percentile = round((1 - (national_rank / len(team_rankings_df))) * 100, 1)
            else:
                national_rank, percentile = "N/A", "N/A"

            with st.container(border=True):
                h1, h2 = st.columns([1, 4])
                with h1:
                    if logo is not None:
                        st.image(str(logo), width=115)
                with h2:
                    st.markdown(f"## {team}")
                    st.caption("Team profile, predictive strength, tournament positioning, and performance splits.")

                    status = row.get("tournament_status", "N/A")
                    seed = row.get("projected_seed", "N/A")
                    bubble = row.get("bubble_status", "N/A")

                    badge_cols = st.columns(3)
                    badge_cols[0].metric("National Rank", national_rank)
                    badge_cols[1].metric("Projected Seed", seed)
                    badge_cols[2].metric("Status", status)

            t1, t2, t3, t4 = st.tabs(["Overview", "Tournament Resume", "Efficiency", "Recent Form"])

            with t1:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Overall Rating", safe_metric_value(row, "Overall_Rating"))
                c2.metric("Power Rating", safe_metric_value(row, "Power_Rating"))
                c3.metric("Offense", safe_metric_value(row, "off_eff"))
                c4.metric("Defense", safe_metric_value(row, "def_eff"))
                c5.metric("Tempo", safe_metric_value(row, "possessions", decimals=1))

                st.markdown("### Quick Read")
                insights = []

                if "off_eff" in row.index and pd.notna(row["off_eff"]):
                    if row["off_eff"] >= team_stats_df["off_eff"].quantile(0.80):
                        insights.append(f"<b>{team}</b> profiles as a strong offensive team.")
                if "def_eff" in row.index and pd.notna(row["def_eff"]):
                    if row["def_eff"] <= team_stats_df["def_eff"].quantile(0.20):
                        insights.append(f"<b>{team}</b> profiles as a strong defensive team.")
                if "SOS_Rating" in row.index and pd.notna(row["SOS_Rating"]):
                    if row["SOS_Rating"] >= team_stats_df["SOS_Rating"].quantile(0.80):
                        insights.append(f"<b>{team}</b> has been tested by a strong schedule.")

                if not insights:
                    insights.append(f"<b>{team}</b> has a balanced statistical profile based on current model data.")

                render_insight("<br>".join(insights))

            with t2:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Tournament Rank", int(row.get("Tournament_Rank", 0)))
                r2.metric("Projected Seed", row.get("projected_seed", "N/A"))
                r3.metric("Resume Score", safe_metric_value(row, "Resume_Score"))
                r4.metric("SOS Rating", safe_metric_value(row, "SOS_Rating"))

                r5, r6, r7 = st.columns(3)
                r5.metric("Quality Wins", int(row.get("quality_win", 0)))
                r6.metric("Bad Losses", int(row.get("bad_loss", 0)))
                r7.metric("Bubble Status", bubble)

            with t3:
                left, right = st.columns([1, 1])

                with left:
                    st.markdown("### Efficiency Profile")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    labels = ["Off Eff", "Def Eff", "D1 Avg"]
                    d1_avg = round(team_stats_df["off_eff"].mean(), 2)
                    values = [
                        row.get("off_eff", 0),
                        row.get("def_eff", 0),
                        d1_avg,
                    ]
                    ax.bar(labels, values)
                    ax.set_ylabel("Efficiency")
                    ax.set_title(f"{team} Efficiency Profile")
                    st.pyplot(fig)

                with right:
                    st.markdown("### Percentile Snapshot")

                    percentile_rows = []
                    for label, col, lower_better in [
                        ("Overall Rating", "Overall_Rating", False),
                        ("Offensive Efficiency", "off_eff", False),
                        ("Defensive Efficiency", "def_eff", True),
                        ("Tempo", "possessions", False),
                        ("SOS Rating", "SOS_Rating", False),
                    ]:
                        if col in team_stats_df.columns and pd.notna(row.get(col)):
                            if lower_better:
                                pct = (team_stats_df[col] >= row[col]).mean() * 100
                            else:
                                pct = (team_stats_df[col] <= row[col]).mean() * 100
                            percentile_rows.append({
                                "Metric": label,
                                "Value": round(float(row[col]), 2),
                                "Percentile": f"{pct:.1f}%",
                            })

                    st.dataframe(pd.DataFrame(percentile_rows), use_container_width=True, hide_index=True)

            with t4:
                season_off = row.get("season_off_eff", None)
                last5_off = row.get("last5_off_eff", None)

                if pd.notna(season_off) and pd.notna(last5_off):
                    trend = last5_off - season_off
                    if trend > 0:
                        st.success(f"Trending Up: +{round(trend, 2)} offensive efficiency over last 5.")
                    elif trend < 0:
                        st.error(f"Trending Down: {round(trend, 2)} offensive efficiency over last 5.")
                    else:
                        st.info("Recent offensive form is flat compared to season average.")
                else:
                    st.info("Recent form data unavailable.")

                split_df = pd.DataFrame({
                    "Split": ["Season", "Last 5", "Home", "Away", "Neutral"],
                    "Off Eff": [
                        row.get("season_off_eff"),
                        row.get("last5_off_eff"),
                        row.get("home_off_eff"),
                        row.get("away_off_eff"),
                        row.get("neutral_off_eff"),
                    ],
                    "Def Eff": [
                        row.get("season_def_eff"),
                        row.get("last5_def_eff"),
                        row.get("home_def_eff"),
                        row.get("away_def_eff"),
                        row.get("neutral_def_eff"),
                    ],
                    "Pace": [
                        row.get("season_possessions"),
                        row.get("last5_possessions"),
                        row.get("home_possessions"),
                        row.get("away_possessions"),
                        row.get("neutral_possessions"),
                    ],
                })

                st.markdown("### Performance Splits")
                st.dataframe(clean_numeric(split_df), use_container_width=True, hide_index=True)

elif page == "Bubble Board":
    st.title("Bubble Board")
    st.caption("Tournament positioning based on resume score, SOS, quality wins, bad losses, and tournament rank.")
    if team_stats_df.empty:
        st.warning("No team stats found. Run pipeline.py first.")
    else:
        bubble_df = team_stats_df.copy()
        needed_cols = ["Team", "Tournament_Rank", "Tournament_Score", "Resume_Score", "SOS_Rating", "quality_win", "bad_loss", "projected_seed", "tournament_status", "Difficulty_Adjusted_Effectiveness", "bubble_status"]
        available_cols = [col for col in needed_cols if col in bubble_df.columns]
        bubble_df = bubble_df[available_cols].copy()
        if "Tournament_Rank" in bubble_df.columns:
            bubble_df = bubble_df.sort_values("Tournament_Rank", ascending=True)

        locks = bubble_df[bubble_df["tournament_status"].isin(["Lock", "Likely In"])]
        bubble = bubble_df[bubble_df["bubble_status"].astype(str).str.contains("Bubble", case=False, na=False)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Projected Field", len(bubble_df[bubble_df["Tournament_Rank"] <= 76]))
        c2.metric("Locks / Likely In", len(locks))
        c3.metric("Bubble Teams", len(bubble))
        c4.metric("First 10 Out", len(bubble_df[(bubble_df["Tournament_Rank"] > 76) & (bubble_df["Tournament_Rank"] <= 86)]))

        view = st.radio("View", ["Projected Field", "Bubble Watch", "First Four Out", "Next Teams Out", "Full Board"], horizontal=True)
        if view == "Projected Field":
            display_df = bubble_df[bubble_df["Tournament_Rank"] <= 76].copy()
        elif view == "Bubble Watch":
            display_df = bubble_df[(bubble_df["Tournament_Rank"] >= 53) & (bubble_df["Tournament_Rank"] <= 86)].copy()
        elif view == "First Four Out":
            display_df = bubble_df[(bubble_df["Tournament_Rank"] > 76) & (bubble_df["Tournament_Rank"] <= 80)].copy()
        elif view == "Next Teams Out":
            display_df = bubble_df[(bubble_df["Tournament_Rank"] > 81) & (bubble_df["Tournament_Rank"] <= 84)].copy()
        else:
            display_df = bubble_df.copy()

        rename_cols = {"Tournament_Rank": "Rank", "Tournament_Score": "Tournament Score", "Resume_Score": "Resume Score", "SOS_Rating": "SOS", "quality_win": "Quality Wins", "bad_loss": "Bad Losses", "projected_seed": "Projected Seed", "tournament_status": "Status", "Difficulty_Adjusted_Effectiveness": "Difficulty Adj.", "bubble_status": "Bubble Status"}
        display_df = display_df.rename(columns=rename_cols)
        st.dataframe(clean_numeric(display_df), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.markdown("### Last Four In")
            st.dataframe(clean_numeric(bubble_df[(bubble_df["Tournament_Rank"] >= 73) & (bubble_df["Tournament_Rank"] <= 76)].rename(columns=rename_cols)), use_container_width=True, hide_index=True)
        with right:
            st.markdown("### First Four Out")
            st.dataframe(clean_numeric(bubble_df[(bubble_df["Tournament_Rank"] >= 77) & (bubble_df["Tournament_Rank"] <= 80)].rename(columns=rename_cols)), use_container_width=True, hide_index=True)

elif page == "Team Comparison":
    st.title("Team Comparison")
    st.caption("Compare two teams across predictive strength, efficiency, schedule, and tempo.")
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
            "Overall Rating": ("Overall_Rating", True),
            "Difficulty-Adjusted Effectiveness": ("Difficulty_Adjusted_Effectiveness", True),
            "Power Rating": ("Power_Rating", True),
            "Offensive Efficiency": ("off_eff", True),
            "Defensive Efficiency": ("def_eff", False),
            "Tempo": ("possessions", True),
            "SOS Rating": ("SOS_Rating", True),
            "Resume Score": ("Resume_Score", True),
        }

        selected_stats = st.multiselect("Choose stats to compare", options=list(stat_options.keys()), default=["Overall Rating", "Difficulty-Adjusted Effectiveness", "Offensive Efficiency", "Defensive Efficiency", "Tempo"])
        row1 = team_stats_df[team_stats_df["Team"] == team1].reset_index(drop=True)
        row2 = team_stats_df[team_stats_df["Team"] == team2].reset_index(drop=True)

        if row1.empty or row2.empty:
            st.error("One or both teams could not be found.")
        elif not selected_stats:
            st.warning("Select at least one stat to compare.")
        else:
            row1 = row1.iloc[0]
            row2 = row2.iloc[0]
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
                    comparison_rows.append({"Stat": stat_label, team1: round(float(row1[col_name]), 2), team2: round(float(row2[col_name]), 2), "Edge": winner_label(row1[col_name], row2[col_name], higher_is_better)})

            compare_df = pd.DataFrame(comparison_rows)
            edge_counts = compare_df["Edge"].value_counts().to_dict() if not compare_df.empty else {}
            a1, a2, a3 = st.columns(3)
            a1.metric(team1, edge_counts.get(team1, 0))
            a2.metric(team2, edge_counts.get(team2, 0))
            a3.metric("Even", edge_counts.get("Even", 0))
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

elif page == "Model Accuracy":
    st.title("Model Accuracy Dashboard")
    st.caption("Backtested model performance across regular season and postseason games.")

    if backtest_results_df.empty:
        st.warning("No backtest results found. Run backtest_current_model.py first.")
    else:

        df = backtest_results_df.copy()
        error_col = "spread_error" if "spread_error" in df.columns else "margin_error"

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Games Tested", f"{len(df):,}")

        if "winner_correct" in df.columns:
            winner_acc = df["winner_correct"].mean()
        else:
            winner_acc = (
                (df["pred_margin"] > 0)
                == (df["actual_margin"] > 0)
            ).mean()

        m2.metric("Winner Accuracy", f"{winner_acc:.1%}")

        m3.metric(
            "Avg Spread Error",
            f"{df[error_col].mean():.2f}"
        )

        m4.metric(
            "Avg Total Error",
            f"{df['total_error'].mean():.2f}"
        )

        st.markdown("---")

        # Competitive games use actual margin <= 25 to match model_error_lab.py
        competitive = df[df["actual_margin"].abs() <= 25].copy()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Competitive Games",
            f"{len(competitive):,}"
        )

        c2.metric(
            "Competitive Spread Error",
            f"{competitive[error_col].mean():.2f}"
        )

        c3.metric(
            "Competitive Total Error",
            f"{competitive['total_error'].mean():.2f}"
        )

        st.markdown("---")

        # Postseason metrics
        if not postseason_results_df.empty:
            postseason_error_col = "spread_error" if "spread_error" in postseason_results_df.columns else "margin_error"

            st.markdown("## Postseason Performance")

            p1, p2, p3 = st.columns(3)

            postseason_acc = (
                (postseason_results_df["pred_margin"] > 0)
                == (postseason_results_df["actual_margin"] > 0)
            ).mean()

            p1.metric(
                "Postseason Accuracy",
                f"{postseason_acc:.1%}"
            )

            p2.metric(
                "Postseason Spread Error",
                f"{postseason_results_df[postseason_error_col].mean():.2f}"
            )

            p3.metric(
                "Postseason Total Error",
                f"{postseason_results_df['total_error'].mean():.2f}"
            )

        st.markdown("---")

        left, right = st.columns(2)

        with left:
            st.markdown("### Spread Error Distribution")

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.hist(
                df[error_col].dropna(),
                bins=25
            )

            ax.set_xlabel("Spread Error")
            ax.set_ylabel("Games")
            ax.set_title("Spread Error")

            st.pyplot(fig)

        with right:
            st.markdown("### Total Error Distribution")

            fig2, ax2 = plt.subplots(figsize=(6, 4))

            ax2.hist(
                df["total_error"].dropna(),
                bins=25
            )

            ax2.set_xlabel("Total Error")
            ax2.set_ylabel("Games")
            ax2.set_title("Total Error")

            st.pyplot(fig2)

        st.markdown("---")

        with st.expander("View Backtest Results"):
            st.dataframe(
                df.sort_values(error_col, ascending=False),
                use_container_width=True,
                hide_index=True,
            )
render_footer()
