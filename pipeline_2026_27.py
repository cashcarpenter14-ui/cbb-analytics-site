
"""FronEdge Metrics: isolated 2026–27 candidate run.

Run from your project folder: python pipeline_2026_27.py
Reads raw_data/2026_27/ and historical Elo priors.
Writes only to data/2026_27_candidate/; never runs production pipeline.py.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'pipeline.py'
RAW = ROOT / 'raw_data' / '2026_27'
DEST = ROOT / 'data' / '2026_27_candidate'
WORK = DEST / '_workspace'
START = pd.Timestamp('2026-11-01', tz='UTC')
END = pd.Timestamp('2027-07-01', tz='UTC')
NAME_MAP = {
    'Queens Royals': 'Queens University Royals',
    'St. Thomas Tommies': 'St. Thomas-Minnesota Tommies',
    'LSU New Orleans Privateers': 'New Orleans Privateers',
}
        


def names(series):
    return series.astype(str).str.replace('\u00a0', ' ', regex=False).str.strip().replace(NAME_MAP)


def check(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f'{label}: missing {missing}')


def main():
    for path in (SOURCE, RAW / 'full_season_games.csv', RAW / 'team_boxscores_d1.csv',
                 ROOT / 'raw_data' / 'elo_ratings_d1.csv', ROOT / 'data' / 'team_conferences.csv'):
        if not path.exists():
            raise FileNotFoundError(f'Required file missing: {path}')

    games = pd.read_csv(RAW / 'full_season_games.csv', dtype={'game_id': str})
    boxes = pd.read_csv(RAW / 'team_boxscores_d1.csv', dtype={'game_id': str})
    elo = pd.read_csv(ROOT / 'raw_data' / 'elo_ratings_d1.csv')
    games.columns = games.columns.str.strip()
    boxes.columns = boxes.columns.str.strip()
    elo.columns = elo.columns.str.strip()
    if 'date' not in games and 'game_date' in games:
        games['date'] = games['game_date']
    check(games, ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score', 'neutral_site'], 'Games')
    check(boxes, ['game_id', 'team', 'points', 'possessions'], 'Boxscores')
    team_col = next((c for c in elo if c.lower() in ('team', 'school')), None)
    rating_col = next((c for c in elo if c.lower() in ('rating', 'elo')), None)
    if team_col is None or rating_col is None:
        raise ValueError('Historical Elo requires team/school and rating/elo columns')
    elo = elo[[team_col, rating_col]].rename(columns={team_col: 'team', rating_col: 'rating'})
    elo['team'] = names(elo['team'])
    elo['rating'] = pd.to_numeric(elo['rating'], errors='raise')
    if elo['team'].duplicated().any():
        raise ValueError('Duplicate historical Elo teams after normalization')

    # ------------------------------------------------------------
    # 2026-27 Division I membership and preseason Elo
    # ------------------------------------------------------------

    conferences = pd.read_csv(
        ROOT / 'data' / 'team_conferences.csv'
    )

    conferences.columns = conferences.columns.str.strip()

    check(
        conferences,
        ['Team', 'Conference'],
        '2026-27 conference membership'
    )

    conferences['Team'] = names(conferences['Team'])

    if conferences['Team'].duplicated().any():
        raise ValueError(
            'Duplicate teams in 2026-27 conference membership'
        )   

    # Membership is determined by the current conference list,
    # not by the historical Elo file.
    d1 = set(conferences['Team'])

    # Preserve historical Elo for returning programs.
    ratings = {
        team: float(rating)
        for team, rating in zip(elo['team'], elo['rating'])
        if team in d1
}

    # Provisional starting rating for teams without historical Elo.
    NEW_TEAM_ELO = 1500.0

    missing_elo = sorted(d1 - set(ratings))

    for team in missing_elo:
        ratings[team] = NEW_TEAM_ELO

    print(
        f'2026-27 D1 membership: {len(d1)} teams',
        flush=True
    )

    print(
        f'Teams initialized with provisional Elo: {missing_elo}',
        flush=True
    )

    games['date'] = pd.to_datetime(games['date'], errors='coerce', utc=True)
    games = games.loc[games['date'].between(START, END, inclusive='left')].copy()
    if 'status' in games:
        status = games['status'].astype(str).str.upper()
        games = games.loc[status.isin(['FINAL', 'STATUS_FINAL', 'POST'])].copy()
    games['home_team'] = names(games['home_team'])
    games['away_team'] = names(games['away_team'])
    for col in ['home_score', 'away_score']:
        games[col] = pd.to_numeric(games[col], errors='coerce')
    games = games.dropna(subset=['home_score', 'away_score'])
    games['neutral_site'] = games['neutral_site'].astype(str).str.lower().isin(['true', '1', 'yes'])
    games['game_id'] = games['game_id'].astype(str).str.strip()
    games = games.drop_duplicates('game_id', keep='last').sort_values(['date', 'game_id'])
    if games.empty:
        raise ValueError('No completed 2026–27 games found')

    # Update preseason Elo priors using only completed current-season D1 games.
    elo_updates = 0
    for row in games.itertuples(index=False):
        h, a = row.home_team, row.away_team
        if h not in d1 or a not in d1 or row.home_score == row.away_score:
            continue
        advantage = 0 if row.neutral_site else 65
        expected = 1 / (1 + 10 ** ((ratings[a] - ratings[h] - advantage) / 400))
        delta = 20 * (float(row.home_score > row.away_score) - expected)
        ratings[h] += delta
        ratings[a] -= delta
        elo_updates += 1
    current_elo = pd.DataFrame({'team': list(ratings), 'rating': list(ratings.values())})

    boxes['game_id'] = boxes['game_id'].astype(str).str.strip()
    boxes['team'] = names(boxes['team'])
    boxes = boxes.loc[boxes['game_id'].isin(games['game_id']) & boxes['team'].isin(d1)].copy()
    boxes = boxes.drop_duplicates(['game_id', 'team'], keep='last')
    # Require exactly two D1 teams and match the game's listed participants.
    eligible = games.set_index('game_id')[['home_team', 'away_team']].to_dict('index')
    good_ids = []
    for gid, group in boxes.groupby('game_id'):
        matchup = eligible.get(gid)
        if matchup and len(group) == 2 and set(group['team']) == {matchup['home_team'], matchup['away_team']}:
            good_ids.append(gid)
    boxes = boxes.loc[boxes['game_id'].isin(good_ids)].copy()
    if boxes.empty:
        raise ValueError('No complete D1-vs-D1 boxscore games found')

    # Isolated workspace: original source and all production inputs/outputs stay untouched.
    (WORK / 'raw_data').mkdir(parents=True, exist_ok=True)
    (WORK / 'data').mkdir(parents=True, exist_ok=True)
    games.to_csv(WORK / 'raw_data' / 'full_season_games.csv', index=False)
    boxes.to_csv(WORK / 'raw_data' / 'team_boxscores_d1.csv', index=False)
    current_elo.to_csv(WORK / 'raw_data' / 'elo_ratings_d1.csv', index=False)
    shutil.copy2(ROOT / 'data' / 'team_conferences.csv', WORK / 'data' / 'team_conferences.csv')

    source = SOURCE.read_text(encoding='utf-8')
    start_marker = '# UPDATE FULL SEASON GAMES WITH RECENT ESPN RESULTS'
    end_marker = '# CLEAN BOXSCORE VALUES'
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError('Cannot safely disable ESPN refresh: unexpected source markers')
    before, rest = source.split(start_marker, 1)
    _, after = rest.split(end_marker, 1)
    source = before + '# ESPN refresh disabled for isolated season run\n\n' + end_marker + after
    (WORK / 'pipeline_candidate_runner.py').write_text(source, encoding='utf-8')

    print(f'Completed 2026–27 games: {len(games)}', flush=True)
    print(f'Complete D1 boxscore games: {len(good_ids)}', flush=True)
    print(f'Current-season Elo updates: {elo_updates}', flush=True)
    subprocess.run([sys.executable, 'pipeline_candidate_runner.py'], cwd=WORK, check=True)

    for filename in ('team_stats_current.csv', 'team_rankings.csv', 'model_metadata.json'):
        produced = WORK / 'data' / filename
        if not produced.exists():
            raise FileNotFoundError(f'Pipeline did not produce {filename}')
        shutil.copy2(produced, DEST / filename)
    stats = pd.read_csv(DEST / 'team_stats_current.csv')
    ranks = pd.read_csv(DEST / 'team_rankings.csv')
    if stats.empty or ranks.empty or stats['Team'].duplicated().any() or ranks['Team'].duplicated().any():
        raise ValueError('Candidate output validation failed')
    counts = boxes.groupby('team').size()
    stats['Current_Season_Games'] = stats['Team'].map(counts).fillna(0).astype(int)
    ranks['Current_Season_Games'] = ranks['Team'].map(counts).fillna(0).astype(int)
    stats.to_csv(DEST / 'team_stats_current.csv', index=False)
    ranks.to_csv(DEST / 'team_rankings.csv', index=False)
    meta_path = DEST / 'model_metadata.json'
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    metadata.update({'season': '2026-27', 'candidate_only': True,
                     'completed_games': len(games), 'complete_d1_boxscore_games': len(good_ids),
                     'elo_updates': elo_updates, 'elo_k_unvalidated': 20, 'elo_home_advantage_unvalidated': 65})
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nCandidate output:', DEST)
    print(ranks[['Rank', 'Team', 'Overall_Rating', 'Current_Season_Games']].head(10).to_string(index=False))
    print('Production pipeline.py and production data files were not modified.')


if __name__ == '__main__':
    main()
