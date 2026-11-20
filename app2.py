
"""
FronEdge Metrics — Comparison App

Runs the original app.py with an additional
Daily Slate 2.0 page.

Original app.py remains unchanged.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_APP = BASE_DIR / "app.py"

if not ORIGINAL_APP.exists():
    raise FileNotFoundError(
        "app.py must be in the same folder as app2.py."
    )

# Read the original application.
source = ORIGINAL_APP.read_text(encoding="utf-8")

import os

SEASON = os.environ.get("FRONEDGE_SEASON", "2025-26")

if SEASON == "2026-27":
    replacements = {
        'team_stats_path = DATA_DIR / "team_stats_current.csv"':
            'team_stats_path = DATA_DIR / "2026_27_candidate" / "team_stats_current.csv"',

        'team_rankings_path = DATA_DIR / "team_rankings.csv"':
            'team_rankings_path = DATA_DIR / "2026_27_candidate" / "team_rankings.csv"',

        'metadata_path = DATA_DIR / "model_metadata.json"':
            'metadata_path = DATA_DIR / "2026_27_candidate" / "model_metadata.json"',
    }

    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(
                f"Could not safely locate: {old}"
            )

        source = source.replace(old, new, 1)

# Add Daily Slate 2.0 to the existing sidebar navigation.
navigation_entry = '            "Daily Slate 2.0",\n'

if '"Daily Slate 2.0"' not in source:
    old_entry = '            "Daily Slate",\n'

    if old_entry not in source:
        raise RuntimeError(
            "Could not locate Daily Slate in the navigation."
        )

    source = source.replace(
        old_entry,
        old_entry + navigation_entry,
        1,
    )

# Remove any previous standalone original-slate import
# if it was added to app.py during earlier experiments.
source = source.replace(
    """    from daily_slate_o import render_original_daily_slate

    render_original_daily_slate(
        team_stats_df=team_stats_df,
        simulate_matchup=simulate_matchup,
        get_team_logo=get_team_logo,
    )
""",
    "",
)

# Add the new page without touching the original Daily Slate.
if 'elif page == "Daily Slate 2.0":' not in source:
    new_page = '''
elif page == "Daily Slate 2.0":
    from daily_slate_fun import render_fun_daily_slate

    render_fun_daily_slate(
        team_stats_df=team_stats_df,
        simulate_matchup=simulate_matchup,
        get_team_logo=get_team_logo,
    )

'''

    teams_marker = 'elif page == "Teams":'

    if teams_marker not in source:
        raise RuntimeError(
            "Could not locate the Teams page."
        )

    source = source.replace(
        teams_marker,
        new_page + teams_marker,
        1,
    )

# Execute the complete application in this Streamlit session.
exec(
    compile(source, str(ORIGINAL_APP), "exec"),
    {
        "__name__": "__main__",
        "__file__": str(ORIGINAL_APP),
    },
)
