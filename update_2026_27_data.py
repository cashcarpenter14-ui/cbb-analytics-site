
#!/usr/bin/env python3
"""Collect 2026-27 college basketball results and team box scores."""

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from d1_eligibility import get_d1_ids, is_d1_matchup, normalize_id

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "raw_data" / "2026_27"

SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/mens-college-basketball/scoreboard"
)

SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/mens-college-basketball/summary"
)

GAME_COLUMNS = [
    "game_id", "date", "status", "home_team", "away_team",
    "home_team_id", "away_team_id", "home_score", "away_score",
    "neutral_site", "start_time_utc", "home_division", "away_division"
]

BOX_COLUMNS = [
    "game_id", "team", "team_id", "points",
    "FG", "FG_PCT", "3PT", "3PT_PCT", "FT", "FT_PCT",
    "totalRebounds", "offensiveRebounds", "defensiveRebounds",
    "assists", "steals", "blocks", "turnovers",
    "teamTurnovers", "totalTurnovers", "turnoverPoints",
    "fastBreakPoints", "pointsInPaint", "fouls", "largestLead",
    "FGM", "FGA", "3PM", "3PA", "FTM", "FTA",
    "possessions", "off_eff"
]

STAT_MAP = {
    "fieldGoalPct": "FG_PCT",
    "threePointFieldGoalPct": "3PT_PCT",
    "freeThrowPct": "FT_PCT",
    "totalRebounds": "totalRebounds",
    "offensiveRebounds": "offensiveRebounds",
    "defensiveRebounds": "defensiveRebounds",
    "assists": "assists",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "teamTurnovers": "teamTurnovers",
    "totalTurnovers": "totalTurnovers",
    "turnoverPoints": "turnoverPoints",
    "fastBreakPoints": "fastBreakPoints",
    "pointsInPaint": "pointsInPaint",
    "fouls": "fouls",
    "largestLead": "largestLead",
}

PAIRS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("FGM", "FGA", "FG"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted":
        ("3PM", "3PA", "3PT"),
    "freeThrowsMade-freeThrowsAttempted": ("FTM", "FTA", "FT"),
}


def fetch(session, url, params):
    for attempt in range(3):
        try:
            response = session.get(
                url,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()

        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise

            time.sleep(2 ** attempt)


def number(value):
    try:
        return float(
            str(value).replace(",", "").replace("%", "")
        )
    except (TypeError, ValueError):
        return None


def parse_game(event):
    competitions = event.get("competitions") or []

    if not competitions:
        return None

    comp = competitions[0]

    status = (
        comp.get("status")
        or event.get("status")
        or {}
    ).get("type", {}).get("name")

    if status != "STATUS_FINAL":
        return None

    sides = {
        c.get("homeAway"): c
        for c in comp.get("competitors", [])
    }

    if "home" not in sides or "away" not in sides:
        return None

    home = sides["home"]
    away = sides["away"]

    if (
        number(home.get("score")) is None
        or number(away.get("score")) is None
    ):
        return None

    raw_date = event.get("date") or comp.get("date")

    if not raw_date:
        return None

    home_team = home.get("team", {})
    away_team = away.get("team", {})

    home_groups = home_team.get("groups")
    away_groups = away_team.get("groups")

    return {
        "game_id": str(event["id"]),
        "date": raw_date[:10],
        "status": "STATUS_FINAL",
        "home_team": home_team.get("displayName"),
        "away_team": away_team.get("displayName"),
        "home_team_id": str(home_team.get("id", "")),
        "away_team_id": str(away_team.get("id", "")),
        "home_score": int(float(home["score"])),
        "away_score": int(float(away["score"])),
        "neutral_site": bool(comp.get("neutralSite", False)),
        "start_time_utc": raw_date,
        "home_division": (
            home_groups.get("id")
            if isinstance(home_groups, dict)
            else None
        ),
        "away_division": (
            away_groups.get("id")
            if isinstance(away_groups, dict)
            else None
        ),
    }


def parse_box(summary, game):
    teams = (
        summary.get("boxscore") or {}
    ).get("teams") or []

    output = []

    for side in teams:
        team = side.get("team") or {}
        team_id = str(team.get("id", ""))

        row = {key: None for key in BOX_COLUMNS}

        row.update({
            "game_id": game["game_id"],
            "team": team.get("displayName"),
            "team_id": team_id,
        })

        if team_id == game["home_team_id"]:
            row["points"] = game["home_score"]
        elif team_id == game["away_team_id"]:
            row["points"] = game["away_score"]

        for stat in side.get("statistics") or []:
            name = stat.get("name", "")
            value = stat.get(
                "displayValue",
                stat.get("value")
            )

            if name in PAIRS:
                parts = str(value).split("-")

                if len(parts) == 2:
                    made, attempts, legacy = PAIRS[name]

                    row[made] = number(parts[0])
                    row[attempts] = number(parts[1])
                    row[legacy] = str(value)

            elif name in STAT_MAP:
                row[STAT_MAP[name]] = number(value)

        required = [
            row["FGA"],
            row["offensiveRebounds"],
            row["turnovers"],
            row["FTA"],
        ]

        if all(x is not None for x in required):
            possessions = (
                row["FGA"]
                - row["offensiveRebounds"]
                + row["turnovers"]
                + 0.475 * row["FTA"]
            )

            row["possessions"] = possessions

            if possessions > 0 and row["points"] is not None:
                row["off_eff"] = (
                    100 * row["points"] / possessions
                )

        if (
            row["points"] is not None
            and row["possessions"] is not None
            and row["possessions"] > 0
        ):
            output.append(row)

    expected_ids = {
        game["home_team_id"],
        game["away_team_id"],
    }

    actual_ids = {
        row["team_id"]
        for row in output
    }

    if len(output) != 2 or actual_ids != expected_ids:
        return []

    return output


def read_csv(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(
        path,
        dtype={
            "game_id": str,
            "team_id": str,
            "home_team_id": str,
            "away_team_id": str,
        }
    )

    for col in columns:
        if col not in df.columns:
            df[col] = None

    return df[columns]


def atomic_csv(df, path):
    temporary = path.with_suffix(".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Update FronEdge 2026-27 basketball data"
    )

    parser.add_argument(
        "--start",
        default="2026-11-01",
        help="YYYY-MM-DD"
    )

    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="YYYY-MM-DD"
    )

    args = parser.parse_args()

    d1_ids = get_d1_ids()

    print(f"Loaded {len(d1_ids)} Division I ESPN team IDs")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if start > end:
        parser.error("--start must be on or before --end")

    if start < date(2026, 7, 1) or end > date(2027, 6, 30):
        parser.error(
            "Dates must be within the 2026-27 season window"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    games_path = OUT / "full_season_games.csv"
    box_path = OUT / "team_boxscores_d1.csv"

    games = read_csv(games_path, GAME_COLUMNS)
    boxes = read_csv(box_path, BOX_COLUMNS)

    discovered = {}
    errors = []

    with requests.Session() as session:
        day = start

        while day <= end:
            try:
                data = fetch(
                    session,
                    SCOREBOARD,
                    {
                        "dates": day.strftime("%Y%m%d"),
                        "groups": 50,
                        "limit": 1000,
                    }
                )

                events = data.get("events") or []
                finals = 0

                for event in events:
                    game = parse_game(event)

                    if game and is_d1_matchup(
                        game["home_team_id"],
                        game["away_team_id"],
                        d1_ids,
                    ):
                        discovered[game["game_id"]] = game
                        finals += 1
                print(
                    f"{day}: {len(events)} events, "
                    f"{finals} final games"
                )

            except Exception as exc:
                errors.append({
                    "date": str(day),
                    "error": str(exc),
                })

                print(f"ERROR scoreboard {day}: {exc}")

            day += timedelta(days=1)

        if discovered:
            incoming = pd.DataFrame(
                discovered.values(),
                columns=GAME_COLUMNS
            )

            games = pd.concat(
                [games, incoming],
                ignore_index=True
            )

            games = games.drop_duplicates(
                "game_id",
                keep="last"
            )

            games = games.sort_values(
                ["date", "game_id"]
            )

            atomic_csv(games, games_path)
         
        existing = set()

        for game_id, group in boxes.groupby("game_id"):
            team_ids = set(
                group["team_id"].dropna().map(normalize_id)
            )

            game_rows = games[
                games["game_id"].astype(str) == str(game_id)
            ]

            if game_rows.empty:
                continue

            game = game_rows.iloc[-1]

            expected_ids = {
                normalize_id(game["home_team_id"]),
                normalize_id(game["away_team_id"]),
            }

    if (
        len(group) == 2
        and team_ids == expected_ids
        and group["possessions"].notna().all()
        and (pd.to_numeric(
            group["possessions"],
            errors="coerce"
        ) > 0).all()
    ):
        existing.add(str(game_id))

        missing = [
            game
            for game_id, game in discovered.items()
            if game_id not in existing
        ]

        print(
            f"Final games discovered: {len(discovered)}; "
            f"missing box scores: {len(missing)}"
        )

        added = []

        for i, game in enumerate(missing, 1):
            try:
                summary = fetch(
                    session,
                    SUMMARY,
                    {"event": game["game_id"]}
                )

                rows = parse_box(summary, game)

                if rows:
                    added.extend(rows)

                    print(
                        f"[{i}/{len(missing)}] "
                        f"OK {game['game_id']}"
                    )
                else:
                    errors.append({
                        "game_id": game["game_id"],
                        "error": "Incomplete box score",
                    })

                    print(
                        f"[{i}/{len(missing)}] "
                        f"INCOMPLETE {game['game_id']}"
                    )

            except Exception as exc:
                errors.append({
                    "game_id": game["game_id"],
                    "error": str(exc),
                })

                print(
                    f"[{i}/{len(missing)}] "
                    f"ERROR {game['game_id']}: {exc}"
                )

        if added:
            new_boxes = pd.DataFrame(
                added,
                columns=BOX_COLUMNS
            )

            boxes = pd.concat(
                [boxes, new_boxes],
                ignore_index=True
            )

            boxes = boxes.drop_duplicates(
                ["game_id", "team_id"],
                keep="last"
            )

            atomic_csv(boxes, box_path)

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "start": str(start),
        "end": str(end),
        "completed_games_found": len(discovered),
        "games_in_season_file": len(games),
        "new_boxscore_games": len(added) // 2,
        "boxscore_rows_in_season_file": len(boxes),
        "errors": errors,
    }

    report_path = OUT / "last_update_report.json"

    report_path.write_text(
        json.dumps(report, indent=2)
    )

    print(
        "\nSUMMARY:",
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "errors"
            },
            indent=2
        )
    )

    print(
        "Errors:",
        len(errors),
        "(see last_update_report.json)"
    )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
