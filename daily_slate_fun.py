
"""
FronEdge Metrics
Daily Slate 2.0

Includes:
- ESPN daily schedule
- Model-ready matchup filtering
- Featured matchup projections
- NCAA-proportioned basketball court
- Home-team center-court logo
- FronEdge center-court logo for neutral games
- Tale of the Tape
- Projected Box Score
- Model Insights
"""

from __future__ import annotations

import base64
import math
import random
import re

from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from daily_schedule import get_daily_schedule


CT = ZoneInfo("America/Chicago")


# ============================================================
# SCHEDULE
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def _schedule(day):
    return get_daily_schedule(day)


# ============================================================
# HELPERS
# ============================================================

def _render_html(markup):
    cleaned = re.sub(r"(?m)^[ \t]+", "", markup).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


def _num(value, digits=1):
    try:
        if pd.notna(value):
            return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        pass

    return "—"


def _pct(value):
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "—"


def _tip(value):
    try:
        return (
            pd.to_datetime(value, utc=True)
            .tz_convert(CT)
            .strftime("%-I:%M %p")
        )
    except (TypeError, ValueError):
        return "TBD"


def _status(value):
    value = str(value or "").upper()

    if "FINAL" in value:
        return "Final"

    if any(
        word in value
        for word in (
            "IN_PROGRESS",
            "INPROGRESS",
            "HALFTIME",
        )
    ):
        return "Live"

    return "Scheduled"


def _is_neutral(value):
    if pd.isna(value):
        return False

    if isinstance(value, str):
        return value.strip().lower() in (
            "true",
            "1",
            "yes",
            "y",
        )

    return bool(value)


def _logo_uri(name, getter):
    if not callable(getter):
        return None

    try:
        path = getter(name)

        if not path:
            return None

        path = Path(path)

        if not path.is_file():
            return None

        if path.stat().st_size > 2_000_000:
            return None

        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }

        mime = mime_types.get(path.suffix.lower())

        if not mime:
            return None

        encoded = base64.b64encode(
            path.read_bytes()
        ).decode()

        return f"data:{mime};base64,{encoded}"

    except (OSError, ValueError, TypeError):
        return None


def _logo(name, getter):
    uri = _logo_uri(name, getter)

    if uri:
        return (
            f'<img class="fe2-logo" '
            f'src="{escape(uri, quote=True)}" '
            f'alt="{escape(name, quote=True)} logo">'
        )

    initials = "".join(
        part[0]
        for part in str(name).split()[:2]
    ).upper()

    return (
        f'<div class="fe2-monogram">'
        f'{escape(initials)}'
        f'</div>'
    )


def _rating(team, candidates, table):
    if table.empty or "Team" not in table.columns:
        return None

    found = table.loc[
        table["Team"] == team
    ]

    if found.empty:
        return None

    for column in candidates:
        if column in found.columns:
            value = pd.to_numeric(
                found.iloc[0][column],
                errors="coerce",
            )

            if pd.notna(value):
                return float(value)

    return None


# ============================================================
# PAGE STYLES
# ============================================================

def _styles():
    st.markdown(
        """
<style>

.block-container {
    max-width: 1550px;
    padding-top: 1.2rem;
}

.fe2-top {
    background: #ffffff;
    border: 1px solid #dce4ed;
    border-top: 4px solid #c82e34;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}

.fe2-kicker {
    font: 700 11px Arial, sans-serif;
    letter-spacing: 2px;
    color: #64748b;
    margin-bottom: 7px;
}

.fe2-title {
    font: 900 clamp(26px, 3vw, 39px) Arial, sans-serif;
    letter-spacing: -1px;
    color: #071b33;
}

.fe2-tag {
    font: 800 12px Arial, sans-serif;
    color: #c82e34;
    background: #fff2f3;
    border: 1px solid #f0d8db;
    border-radius: 6px;
    padding: 9px 12px;
}

.fe2-label {
    font: 800 13px Arial, sans-serif;
    letter-spacing: 1.1px;
    color: #071b33;
    margin: 8px 0 12px;
    padding-left: 10px;
    border-left: 3px solid #c82e34;
}

.fe2-arena {
    background: #ffffff !important;
    border: 1px solid #dce4ed;
    border-top: 4px solid #c82e34;
    border-radius: 10px;
    padding: 22px;
    margin-bottom: 18px;
    color: #071b33;
}

.fe2-hero {
    display: grid;
    grid-template-columns:
        minmax(0, 1fr)
        150px
        minmax(0, 1fr);
    align-items: center;
    gap: 12px;
    min-height: 210px;
}

.fe2-team {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 8px;
    min-width: 0;
}

.fe2-logo {
    height: 92px;
    max-width: 125px;
    object-fit: contain;
    background: transparent;
    filter: none !important;
    box-shadow: none !important;
    mix-blend-mode: multiply;
}

.fe2-monogram {
    width: 92px;
    height: 92px;
    display: grid;
    place-items: center;
    background: #edf3fa;
    border: 1px solid #dce4ed;
    border-radius: 8px;
    color: #071b33;
    font: 900 29px Arial, sans-serif;
}

.fe2-teamname {
    font: 900 clamp(
        15px,
        1.5vw,
        22px
    ) Arial, sans-serif;
    color: #071b33 !important;
    text-transform: uppercase;
}

.fe2-role {
    font: 700 11px Arial, sans-serif;
    color: #64748b;
    letter-spacing: 1.4px;
}

.fe2-center {
    text-align: center;
    color: #071b33;
}

.fe2-vs {
    font: 900 25px Arial, sans-serif;
    color: #64748b;
}

.fe2-score {
    font: 900 clamp(
        23px,
        2.8vw,
        42px
    ) Arial, sans-serif;
    color: #071b33;
    white-space: nowrap;
    margin: 8px 0;
}

.fe2-scorelabel {
    font: 700 10px Arial, sans-serif;
    letter-spacing: 1px;
    color: #64748b;
}

.fe2-probs {
    display: flex;
    width: 100%;
    height: 11px;
    margin-top: 18px;
    background: #e8edf3;
    overflow: hidden;
    border-radius: 6px;
}

.fe2-prob-away {
    background: #3165a7;
}

.fe2-prob-home {
    background: #bd3039;
}

.fe2-summary {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 14px;
    color: #243b53;
    font: 700 12px Arial, sans-serif;
    margin-top: 14px;
}

.fe2-summary span {
    background: #f7f9fc;
    border: 1px solid #e5eaf0;
    border-radius: 5px;
    padding: 7px 10px;
}

.fe2-hint {
    color: #64748b;
    font: 12px Arial, sans-serif;
    margin-top: 14px;
    text-align: center;
}

.fe2-court-heading {
    color: #071b33;
    font: 800 13px Arial, sans-serif;
    letter-spacing: 1.1px;
    margin: 12px 0;
    padding-left: 10px;
    border-left: 3px solid #c82e34;
}

@media (max-width: 900px) {

    .fe2-hero {
        grid-template-columns:
            1fr 100px 1fr;
        gap: 4px;
    }

    .fe2-logo,
    .fe2-monogram {
        height: 65px;
        max-width: 75px;
    }

    .fe2-arena {
        padding: 14px;
    }

}

</style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# NCAA MATCHUP COURT
# ============================================================

def _render_matchup_court(
    away,
    home,
    getter,
    neutral=False,
):
    """
    NCAA men's basketball court.

    Court dimensions:
    - 94 feet long
    - 50 feet wide
    - 12-foot lanes
    - 6-foot center-circle radius
    - 6-foot free-throw-circle radius
    - 4-foot restricted-area radius
    - 22 ft 1.75 in three-point arc
    - 21 ft 7.875 in corner three

    Center court:
    - Home games: home-team logo
    - Neutral games: FronEdge logo
    """

    # --------------------------------------------------------
    # COURT COORDINATES
    # --------------------------------------------------------

    SCALE = 10

    X0 = 20
    Y0 = 20

    LENGTH = 940
    WIDTH = 500

    CX = 490
    CY = 270

    def px(feet):
        return X0 + feet * SCALE

    def py(feet):
        return Y0 + feet * SCALE

    # --------------------------------------------------------
    # CENTER-COURT LOGO
    # --------------------------------------------------------

    brand_path = (
        Path(__file__).resolve().parent
        / "assets"
        / "fronedge_logos"
        / "TheJersey1.svg"
    )

    center_uri = None

    if neutral:
        if brand_path.is_file():
            center_uri = (
                "data:image/svg+xml;base64,"
                + base64.b64encode(
                    brand_path.read_bytes()
                ).decode()
            )

    else:
        center_uri = _logo_uri(
            home,
            getter,
        )

    if center_uri:
        center_mark = (
            f'<image href="{escape(center_uri, quote=True)}" '
            'x="425" y="205" '
            'width="130" height="130" '
            'preserveAspectRatio="xMidYMid meet" '
            'style="mix-blend-mode:multiply"/>'
        )

    else:
        label = (
            "FRONEDGE METRICS"
            if neutral
            else home
        )

        center_mark = (
            '<text x="490" y="275" '
            'text-anchor="middle" '
            'font-family="Arial" '
            'font-weight="900" '
            'font-size="13" '
            'fill="#24354a">'
            f'{escape(label[:24])}'
            '</text>'
        )

    # --------------------------------------------------------
    # HARDWOOD FLOOR
    # --------------------------------------------------------

    rng = random.Random(2026)

    wood_colors = [
        "#E8C794",
        "#EBCB99",
        "#E7C38F",
        "#EFD2A3",
        "#E9C895",
        "#E5C08B",
    ]

    planks = []

    board_height = 12.5

    for row in range(40):
        y = Y0 + row * board_height

        x = X0 - (row % 4) * 65

        while x < X0 + LENGTH:
            board_length = rng.randint(
                110,
                215,
            )

            color = rng.choice(
                wood_colors
            )

            planks.append(
                f'<rect x="{x:.1f}" '
                f'y="{y:.1f}" '
                f'width="{board_length}" '
                f'height="{board_height}" '
                f'fill="{color}"/>'
            )

            planks.append(
                f'<path d="M{x:.1f} {y:.1f} '
                f'h{board_length}" '
                'stroke="#956C42" '
                'stroke-opacity=".16" '
                'stroke-width=".5"/>'
            )

            for offset in (3, 7, 10):
                if rng.random() < 0.7:
                    planks.append(
                        f'<path d="M{x+10:.1f} '
                        f'{y+offset:.1f} '
                        f'h{max(5, board_length-20)}" '
                        'stroke="#9D764B" '
                        'stroke-opacity=".08" '
                        'stroke-width=".5"/>'
                    )

            x += board_length

    floorboards = "".join(planks)

    # --------------------------------------------------------
    # THREE-POINT GEOMETRY
    # --------------------------------------------------------

    arc_radius = (
        22 + 1.75 / 12
    ) * SCALE

    corner_distance = (
        21 + 7.875 / 12
    ) * SCALE

    left_rim = px(5.25)

    right_rim = px(
        94 - 5.25
    )

    corner_top = (
        CY - corner_distance
    )

    corner_bottom = (
        CY + corner_distance
    )

    angle = math.asin(
        corner_distance / arc_radius
    )

    def make_three_point_path(
        rim_x,
        direction,
    ):
        points = []

        for i in range(121):
            theta = (
                -angle
                + 2 * angle * i / 120
            )

            x = (
                rim_x
                + direction
                * arc_radius
                * math.cos(theta)
            )

            y = (
                CY
                + arc_radius
                * math.sin(theta)
            )

            points.append((x, y))

        start_x = points[0][0]

        baseline_x = (
            X0
            if direction == 1
            else X0 + LENGTH
        )

        path = (
            f"M{baseline_x:.2f} "
            f"{corner_top:.2f} "
            f"L{start_x:.2f} "
            f"{corner_top:.2f} "
        )

        for x, y in points[1:]:
            path += (
                f"L{x:.2f} "
                f"{y:.2f} "
            )

        path += (
            f"L{baseline_x:.2f} "
            f"{corner_bottom:.2f}"
        )

        return path

    left_three = make_three_point_path(
        left_rim,
        1,
    )

    right_three = make_three_point_path(
        right_rim,
        -1,
    )

    # --------------------------------------------------------
    # LANE AND FREE THROW LINES
    # --------------------------------------------------------

    left_ft = px(19)

    right_ft = px(
        94 - 19
    )

    lane_top = py(19)

    lane_bottom = py(31)

    free_throw_arcs = f"""
        <path d="
            M{left_ft} {lane_top}
            A60 60 0 0 1
            {left_ft} {lane_bottom}
        "/>

        <path d="
            M{right_ft} {lane_top}
            A60 60 0 0 0
            {right_ft} {lane_bottom}
        "/>
    """

    # --------------------------------------------------------
    # RESTRICTED AREAS
    # --------------------------------------------------------

    restricted_radius = 40

    restricted_left = (
        f"M{left_rim} "
        f"{CY-restricted_radius} "
        f"A40 40 0 0 1 "
        f"{left_rim} "
        f"{CY+restricted_radius}"
    )

    restricted_right = (
        f"M{right_rim} "
        f"{CY-restricted_radius} "
        f"A40 40 0 0 0 "
        f"{right_rim} "
        f"{CY+restricted_radius}"
    )

    # --------------------------------------------------------
    # SVG
    # --------------------------------------------------------

    svg = f"""
<svg viewBox="0 0 980 540"
     xmlns="http://www.w3.org/2000/svg"
     role="img"
     aria-label="NCAA men's basketball court">

    <defs>

        <clipPath id="ncaa-floor-clip">
            <rect x="20" y="20"
                  width="940" height="500"/>
        </clipPath>

        <linearGradient
            id="ncaa-finish"
            x1="0" y1="0"
            x2="1" y2="1">

            <stop offset="0"
                  stop-color="#ffffff"
                  stop-opacity=".14"/>

            <stop offset=".5"
                  stop-color="#ffffff"
                  stop-opacity="0"/>

            <stop offset="1"
                  stop-color="#754820"
                  stop-opacity=".07"/>

        </linearGradient>

    </defs>

    <!-- Court background -->

    <rect
        width="980"
        height="540"
        fill="#f8fafc"
    />

    <!-- Hardwood -->

    <g clip-path="url(#ncaa-floor-clip)">
        {floorboards}
    </g>

    <!-- Finish -->

    <rect
        x="20"
        y="20"
        width="940"
        height="500"
        fill="url(#ncaa-finish)"
    />

    <!-- Boundary -->

    <rect
        x="20"
        y="20"
        width="940"
        height="500"
        fill="none"
        stroke="#233B53"
        stroke-width="2"
    />

    <!-- Court lines -->

    <g
        fill="none"
        stroke="#ffffff"
        stroke-width="1.67"
        stroke-linejoin="round"
        stroke-linecap="round"
    >

        <!-- Half court -->

        <path d="M490 20 V520"/>

        <!-- Center circle -->

        <circle
            cx="490"
            cy="270"
            r="60"
        />

        <!-- Left lane -->

        <rect
            x="20"
            y="{lane_top}"
            width="{left_ft-X0}"
            height="120"
        />

        <!-- Right lane -->

        <rect
            x="{right_ft}"
            y="{lane_top}"
            width="{X0+LENGTH-right_ft}"
            height="120"
        />

        <!-- Free throw arcs -->

        {free_throw_arcs}

        <!-- Three-point lines -->

        <path d="{left_three}"/>

        <path d="{right_three}"/>

        <!-- Restricted areas -->

        <path d="{restricted_left}"/>

        <path d="{restricted_right}"/>

        <!-- Backboards -->

        <path
            d="
                M60 240 V300
                M920 240 V300
            "
            stroke-width="3"
        />

        <!-- Left rim -->

        <circle
            cx="{left_rim}"
            cy="270"
            r="7.5"
            stroke="#B63A2B"
            stroke-width="2.5"
        />

        <!-- Right rim -->

        <circle
            cx="{right_rim}"
            cy="270"
            r="7.5"
            stroke="#B63A2B"
            stroke-width="2.5"
        />

        <!-- Lane marks -->

        <path d="
            M60 210 V202
            M90 210 V202
            M120 210 V202
            M150 210 V202

            M60 330 V338
            M90 330 V338
            M120 330 V338
            M150 330 V338

            M920 210 V202
            M890 210 V202
            M860 210 V202
            M830 210 V202

            M920 330 V338
            M890 330 V338
            M860 330 V338
            M830 330 V338
        "/>

    </g>

    <!-- Center court logo -->

    {center_mark}

</svg>
    """

    html = (
        "<html><head><style>"
        "html,body{"
        "margin:0;"
        "padding:0;"
        "background:transparent;"
        "overflow:hidden;"
        "}"
        "svg{"
        "display:block;"
        "width:100%;"
        "height:auto;"
        "}"
        "</style></head><body>"
        + svg
        + "</body></html>"
    )

    components.html(
        html,
        height=400,
        scrolling=False,
    )


# ============================================================
# DAILY SLATE 2.0
# ============================================================

def render_fun_daily_slate(
    team_stats_df,
    simulate_matchup,
    get_team_logo=None,
):
    """
    Main Daily Slate 2.0 function.
    Called by app.py / app2.py.
    """

    _styles()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    _render_html(
        """
<div class="fe2-top">

    <div>
        <div class="fe2-kicker">
            FRONEDGE METRICS / COLLEGE BASKETBALL
        </div>

        <div class="fe2-title">
            DAILY SLATE 2.0
        </div>
    </div>

    <div class="fe2-tag">
        MATCHUP CENTER
    </div>

</div>
        """
    )

    # --------------------------------------------------------
    # DATE NAVIGATION
    # --------------------------------------------------------

    if "fe2_date" not in st.session_state:
        st.session_state.fe2_date = (
            datetime.now(CT).date()
        )

    nav1, nav2, nav3 = st.columns(
        [1, 3, 1]
    )

    with nav1:
        if st.button(
            "◀ Previous",
            key="fe2_prev",
            use_container_width=True,
        ):
            st.session_state.fe2_date -= (
                timedelta(days=1)
            )

    with nav3:
        if st.button(
            "Next ▶",
            key="fe2_next",
            use_container_width=True,
        ):
            st.session_state.fe2_date += (
                timedelta(days=1)
            )

    with nav2:
        st.date_input(
            "Game date",
            key="fe2_date",
            label_visibility="collapsed",
        )

    # --------------------------------------------------------
    # LOAD SCHEDULE
    # --------------------------------------------------------

    try:
        slate = _schedule(
            st.session_state.fe2_date
        )

    except Exception as exc:
        st.error(
            f"Could not load ESPN schedule: {exc}"
        )
        return

    if slate.empty:
        st.info(
            "No games scheduled on this date."
        )
        return

    if (
        team_stats_df.empty
        or "Team" not in team_stats_df.columns
    ):
        st.warning(
            "Team statistics are unavailable."
        )
        return

    # --------------------------------------------------------
    # MODEL-READY GAMES
    # --------------------------------------------------------

    valid = set(
        team_stats_df["Team"]
        .dropna()
        .astype(str)
    )

    slate = slate.copy()

    slate["eligible"] = (
        slate["home_team"].isin(valid)
        & slate["away_team"].isin(valid)
    )

    eligible = slate[
        slate["eligible"]
    ].copy()

    other = slate[
        ~slate["eligible"]
    ].copy()

    st.caption(
        f"{len(slate)} scheduled games · "
        f"{len(eligible)} model-ready matchups · "
        f"{len(other)} without projections"
    )

    if eligible.empty:
        st.info(
            "No model-ready matchups on this date."
        )

        if not other.empty:
            with st.expander(
                "Other scheduled games"
            ):
                st.dataframe(
                    other[
                        [
                            "away_team",
                            "home_team",
                            "status_detail",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

        return

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    search_col, site_col, sort_col = (
        st.columns([2.5, 1.1, 1.3])
    )

    with search_col:
        query = st.text_input(
            "Find a team",
            placeholder="Search the slate…",
            key="fe2_query",
        )

    with site_col:
        site = st.selectbox(
            "Site",
            [
                "All sites",
                "Neutral",
                "Home",
            ],
            key="fe2_site",
        )

    with sort_col:
        sort = st.selectbox(
            "Order",
            [
                "Tipoff",
                "Team A–Z",
            ],
            key="fe2_sort",
        )

    visible = eligible.copy()

    if query.strip():
        q = query.strip()

        visible = visible[
            visible["away_team"].str.contains(
                q,
                case=False,
                regex=False,
                na=False,
            )
            |
            visible["home_team"].str.contains(
                q,
                case=False,
                regex=False,
                na=False,
            )
        ]

    if site != "All sites":
        visible = visible[
            visible["neutral_site"].apply(
                _is_neutral
            )
            == (site == "Neutral")
        ]

    if sort == "Team A–Z":
        visible = visible.sort_values(
            [
                "away_team",
                "home_team",
            ]
        )

    else:
        visible = visible.sort_values(
            "date"
        )

    ids = (
        visible["game_id"]
        .astype(str)
        .tolist()
    )

    if (
        "fe2_selected"
        not in st.session_state
        or st.session_state.fe2_selected
        not in ids
    ):
        st.session_state.fe2_selected = (
            ids[0] if ids else None
        )

    # --------------------------------------------------------
    # PAGE COLUMNS
    # --------------------------------------------------------

    left, right = st.columns(
        [1.05, 2.2],
        gap="medium",
    )

    # --------------------------------------------------------
    # MATCHUP LIST
    # --------------------------------------------------------

    with left:
        _render_html(
            '<div class="fe2-label">'
            'SELECT A MATCHUP'
            '</div>'
        )

        if visible.empty:
            st.info(
                "No matchups match these filters."
            )

        else:
            with st.container(
                height=575,
                border=True,
            ):
                for _, game in visible.iterrows():
                    gid = str(
                        game["game_id"]
                    )

                    marker = (
                        "● "
                        if gid
                        == st.session_state.fe2_selected
                        else ""
                    )

                    neutral_marker = (
                        " · N"
                        if _is_neutral(
                            game["neutral_site"]
                        )
                        else ""
                    )
                    separator = (
                        "vs"
                        if _is_neutral(game["neutral_site"])
                        else "@"
                    )

                    label = (
                        f"{marker}"
                        f"{game['away_team']} "
                        f"vs {game['home_team']}\n"
                        f"{_tip(game['date'])}"
                        f"{neutral_marker} · "
                        f"{_status(game.get('status'))}"
                    )

                    if st.button(
                        label,
                        key=f"fe2_game_{gid}",
                        use_container_width=True,
                        type=(
                            "primary"
                            if gid
                            == st.session_state.fe2_selected
                            else "secondary"
                        ),
                    ):
                        st.session_state.fe2_selected = gid
                        st.rerun()

        if not other.empty:
            with st.expander(
                f"Other games ({len(other)})"
            ):
                st.caption(
                    "One or both teams are absent "
                    "from the model dataset. "
                    "This does not always mean non-D1."
                )

                st.dataframe(
                    other[
                        [
                            "away_team",
                            "home_team",
                            "status_detail",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

    # --------------------------------------------------------
    # FEATURED MATCHUP
    # --------------------------------------------------------

    with right:
        if not st.session_state.fe2_selected:
            st.info(
                "Select a matchup."
            )
            return

        chosen = visible.loc[
            visible["game_id"].astype(str)
            == st.session_state.fe2_selected
        ]

        if chosen.empty:
            st.info(
                "Select a matchup."
            )
            return

        game = chosen.iloc[0]

        home = str(
            game["home_team"]
        )

        away = str(
            game["away_team"]
        )

        neutral = _is_neutral(
            game["neutral_site"]
        )

        # ----------------------------------------------------
        # EXISTING MODEL
        # ----------------------------------------------------

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

        except Exception as exc:
            separator = "vs" if neutral else "@"
            
            st.error(
                f"Projection unavailable for "
                f"{away} vs {home}: {exc}"
            )
            return

        p_away = result.get(
            "win_prob2"
        )

        p_home = result.get(
            "win_prob1"
        )

        try:
            away_width = max(
                0,
                min(
                    100,
                    float(p_away) * 100,
                ),
            )

        except (TypeError, ValueError):
            away_width = 50

        home_width = (
            100 - away_width
        )

        # Use existing FronEdge team-branding colors.
        # Load only the branding CSV; do not alter the model.
        import pandas as pd
        from pathlib import Path
        import re

        branding_file = Path(__file__).resolve().parent / "data" / "team_branding.csv"

        def team_primary_color(team_name, fallback):
            if not branding_file.exists():
                return fallback

            branding = pd.read_csv(branding_file)
            if "team" not in branding.columns or "primary_color" not in branding.columns:
                return fallback

            matches = branding[
                branding["team"].astype(str).str.strip().str.casefold()
                == str(team_name).strip().casefold()
            ]

            if matches.empty:
                return fallback

            value = matches.iloc[0]["primary_color"]

            if pd.isna(value):
                return fallback

            value = str(value).strip().lstrip("#")

            if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
                return fallback

            return "#" + value.upper()

        away_color = team_primary_color(away, "#2563EB")
        home_color = team_primary_color(home, "#DC2626")

        def team_secondary_color(team_name):
            if not branding_file.exists():
                return None

            branding = pd.read_csv(branding_file)

            if "team" not in branding.columns or "secondary_color" not in branding.columns:
                return None

            matches = branding[
                branding["team"].astype(str).str.strip().str.casefold()
                == str(team_name).strip().casefold()
            ]

            if matches.empty:
                return None

            value = matches.iloc[0]["secondary_color"]

            if pd.isna(value):
                return None

            value = str(value).strip().lstrip("#")

            if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
                return None

            return "#" + value.upper()

        def colors_too_similar(color_a, color_b):
            a = tuple(
                int(color_a[i:i+2], 16)
                for i in (1, 3, 5)
            )
            b = tuple(
                int(color_b[i:i+2], 16)
                for i in (1, 3, 5)
            )

            distance = sum(
                (x - y) ** 2
                for x, y in zip(a, b)
            ) ** 0.5

            return distance < 90

        def visible_on_white(color):
            if not color:
                return False

            rgb = [
                int(color[i:i+2], 16) / 255
                for i in (1, 3, 5)
            ]

            linear = [
                c / 12.92 if c <= 0.04045
                else ((c + 0.055) / 1.055) ** 2.4
                for c in rgb
            ]

            luminance = (
                0.2126 * linear[0]
                + 0.7152 * linear[1]
                + 0.0722 * linear[2]
            )

            # Exclude colors that are too light against white.
            return luminance < 0.70

        # Replace overly light primary colors as well.
        if not visible_on_white(away_color):
            away_color = (
                team_secondary_color(away)
                if visible_on_white(team_secondary_color(away))
                else "#2563EB"
            )

        if not visible_on_white(home_color):
            home_color = (
                team_secondary_color(home)
                if visible_on_white(team_secondary_color(home))
                else "#475569"
            )

        if colors_too_similar(away_color, home_color):
            home_secondary = team_secondary_color(home)
            away_secondary = team_secondary_color(away)

            if (
                visible_on_white(home_secondary)
                and not colors_too_similar(away_color, home_secondary)
            ):
                home_color = home_secondary

            elif (
                visible_on_white(away_secondary)
                and not colors_too_similar(away_secondary, home_color)
            ):
                away_color = away_secondary

            else:
                home_color = "#475569"

                if colors_too_similar(away_color, home_color):
                    home_color = "#D97706"

        score = (
            f"{_num(result.get('proj_score2'))}"
            " – "
            f"{_num(result.get('proj_score1'))}"
        )

        # ----------------------------------------------------
        # FEATURED MATCHUP CARD
        # ----------------------------------------------------

        _render_html(
            '<div class="fe2-label">'
            'FEATURED MATCHUP'
            '</div>'
        )

        _render_html(
            f"""
<div class="fe2-arena">

    <div class="fe2-hero">

        <div class="fe2-team">

            {_logo(away, get_team_logo)}

            <div class="fe2-teamname">
                {escape(away)}
            </div>

            <div class="fe2-role">
                AWAY
            </div>

        </div>

        <div class="fe2-center">

            <div class="fe2-vs">
                {"VS" if neutral else "@"}
            </div>

            <div class="fe2-score">
                {score}
            </div>

            <div class="fe2-scorelabel">
                PROJECTED SCORE · AWAY / HOME
            </div>

        </div>

        <div class="fe2-team">

            {_logo(home, get_team_logo)}

            <div class="fe2-teamname">
                {escape(home)}
            </div>

            <div class="fe2-role">
                {"NEUTRAL" if neutral else "HOME"}
            </div>

        </div>

    </div>

    <div class="fe2-probs">

        <div
            class="fe2-prob-away"
            style="width:{away_width:.1f}%;background-color:{away_color}">
        </div>

        <div
            class="fe2-prob-home"
            style="width:{home_width:.1f}%;background-color:{home_color}">
        </div>

    </div>

    <div class="fe2-summary">

        <span>
            {escape(away)}: {_pct(p_away)}
        </span>

        <span>
            {escape(home)}: {_pct(p_home)}
        </span>

    </div>

    <div class="fe2-summary">

        <span>
            FAVORITE:
            {escape(str(result.get("favorite", "—")))}
        </span>

        <span>
            SPREAD:
            {_num(result.get("spread"))}
        </span>

        <span>
            TOTAL:
            {_num(result.get("total"))}
        </span>

    </div>

    <div class="fe2-hint">

        {escape(_tip(game["date"]))} CT ·

        {"Neutral court" if neutral else "Home court"} ·

        {escape(str(game.get("status_detail", "")))}

    </div>

</div>
            """
        )

        # ----------------------------------------------------
        # NCAA MATCHUP COURT
        # ----------------------------------------------------

        _render_html(
            '<div class="fe2-court-heading">'
            'MATCHUP COURT'
            '</div>'
        )

        _render_matchup_court(
            away=away,
            home=home,
            getter=get_team_logo,
            neutral=neutral,
        )

        st.caption(
            "NCAA-proportioned basketball court. "
            "Home-team branding for home games; "
            "FronEdge branding for neutral-site games."
        )

        # ----------------------------------------------------
        # ANALYSIS TABS
        # ----------------------------------------------------

        factors, boxscore, insights = st.tabs(
            [
                "TALE OF THE TAPE",
                "PROJECTED BOX SCORE",
                "MODEL INSIGHTS",
            ]
        )

        # ----------------------------------------------------
        # TALE OF THE TAPE
        # ----------------------------------------------------

        with factors:
            st.caption(
                "Team statistics from the "
                "current dataset."
            )

            definitions = [
                (
                    "Offensive efficiency",
                    [
                        "weighted_off_eff",
                        "season_off_eff",
                        "off_eff",
                    ],
                ),
                (
                    "Defensive efficiency",
                    [
                        "weighted_def_eff",
                        "season_def_eff",
                        "def_eff",
                    ],
                ),
                (
                    "Tempo / possessions",
                    [
                        "weighted_possessions",
                        "season_possessions",
                        "possessions",
                    ],
                ),
                (
                    "Rebounding",
                    [
                        "season_reb",
                        "season_rebounds",
                        "rebounds",
                    ],
                ),
                (
                    "Turnovers",
                    [
                        "season_tov",
                        "season_turnovers",
                        "turnovers",
                    ],
                ),
            ]

            data = []

            for label, candidates in definitions:
                away_value = _rating(
                    away,
                    candidates,
                    team_stats_df,
                )

                home_value = _rating(
                    home,
                    candidates,
                    team_stats_df,
                )

                if (
                    away_value is not None
                    or home_value is not None
                ):
                    data.append(
                        {
                            "Category": label,
                            away: _num(away_value),
                            home: _num(home_value),
                        }
                    )

            if data:
                st.dataframe(
                    pd.DataFrame(data),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    "No comparable team factors found."
                )

        # ----------------------------------------------------
        # PROJECTED BOX SCORE
        # ----------------------------------------------------

        with boxscore:
            away_box = (
                result.get("box_score_team2")
                or {}
            )

            home_box = (
                result.get("box_score_team1")
                or {}
            )

            if (
                isinstance(away_box, dict)
                and isinstance(home_box, dict)
                and (away_box or home_box)
            ):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Team": away,
                                **away_box,
                            },
                            {
                                "Team": home,
                                **home_box,
                            },
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            else:
                st.info(
                    "Projected box score unavailable."
                )

        # ----------------------------------------------------
        # MODEL INSIGHTS
        # ----------------------------------------------------

        with insights:
            for label, key in [
                (
                    "Confidence",
                    "confidence",
                ),
                (
                    "Game style",
                    "game_style",
                ),
                (
                    "Volatility",
                    "volatility_tag",
                ),
                (
                    "Upset alert",
                    "upset_alert",
                ),
            ]:
                value = result.get(key)

                if value is not None:
                    st.markdown(
                        f"**{label}:** "
                        f"{escape(str(value))}"
                    )

            st.caption(
                "FronEdge estimates are projections, "
                "not guaranteed outcomes."
            )
