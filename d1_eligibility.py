"""
FronEdge Metrics — Division I eligibility.

ESPN Division I roster plus explicitly documented
corrections for known missing teams.
"""

from pathlib import Path
import json
import requests

ROOT = Path(__file__).resolve().parent

CACHE_FILE = (
    ROOT / "data" / "ml_early_season"
    / "d1_roster_2026_27.json"
)

ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/mens-college-basketball/teams"
)

KNOWN_D1_CORRECTIONS = {
    "2511": "Queens University Royals",
    "2737": "Winthrop Eagles",
}


def normalize_id(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in ("nan", "none", "<na>", ""):
        return ""

    if value.endswith(".0"):
        value = value[:-2]

    return value


def collect_teams(payload):
    teams = {}

    def walk(value):
        if isinstance(value, dict):
            team = value.get("team")

            if isinstance(team, dict):
                team_id = normalize_id(team.get("id"))
                name = (
                    team.get("displayName")
                    or team.get("name")
                )

                if team_id and name:
                    teams[team_id] = {
                        "id": team_id,
                        "name": name,
                    }

            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)

        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return teams


def load_espn_roster(refresh=False):
    if CACHE_FILE.exists() and not refresh:
        data = json.loads(CACHE_FILE.read_text())

        if (
            data.get("season") == "2026_27"
            and data.get("group") == "50"
        ):
            teams = data.get("teams", {})

            if 350 <= len(teams) <= 380:
                return teams

    response = requests.get(
        ESPN_TEAMS_URL,
        params={
            "groups": "50",
            "limit": 500,
            "season": "2027",
        },
        timeout=30,
    )

    response.raise_for_status()
    teams = collect_teams(response.json())

    if not 350 <= len(teams) <= 380:
        raise RuntimeError(
            f"Unexpected ESPN roster size: {len(teams)}"
        )

    CACHE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_FILE.write_text(
        json.dumps(
            {
                "season": "2026_27",
                "group": "50",
                "source": ESPN_TEAMS_URL,
                "teams": teams,
            },
            indent=2,
        )
    )

    return teams


def get_d1_ids(refresh=False):
    teams = load_espn_roster(refresh=refresh)

    ids = set(teams.keys())
    ids.update(KNOWN_D1_CORRECTIONS.keys())

    if not 350 <= len(ids) <= 385:
        raise RuntimeError(
            f"Unexpected corrected roster size: {len(ids)}"
        )

    return ids


def is_d1_matchup(home_team_id, away_team_id, d1_ids):
    home_id = normalize_id(home_team_id)
    away_id = normalize_id(away_team_id)

    return (
        bool(home_id)
        and bool(away_id)
        and home_id in d1_ids
        and away_id in d1_ids
    )


if __name__ == "__main__":
    raw = load_espn_roster()
    ids = get_d1_ids()

    print("ESPN roster teams:", len(raw))
    print("Corrected Division I teams:", len(ids))
    print("Queens included:", "2511" in ids)
    print("Winthrop included:", "2737" in ids)
    print(
        "Queens vs Winthrop:",
        is_d1_matchup("2511", "2737", ids),
    )
