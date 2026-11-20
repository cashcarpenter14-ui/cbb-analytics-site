"""FronEdge Daily Slate UI. Does not change model.py or ESPN ingestion."""
from __future__ import annotations

from datetime import datetime, timedelta
from html import escape
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from daily_schedule import get_daily_schedule


CSS = r"""
<style>
/* Scoped to the Daily Slate content, not the sidebar or other pages. */
.fe-slate {font-family: 'Arial Narrow', 'Helvetica Neue', sans-serif; color:#eaf0f5;}
.fe-slate .fe-eyebrow {color:#e6b968; font-size:.74rem; letter-spacing:.2em; font-weight:800;}
.fe-slate .fe-title {font-size:clamp(2rem,4vw,3.4rem);font-weight:900;letter-spacing:.035em;line-height:1.05;margin:.35rem 0;}
.fe-slate .fe-sub {color:#aab7c5;font-size:.94rem;margin-bottom:1.2rem;}
.fe-slate .fe-hero {padding:1.4rem 1.6rem;border:1px solid #354557;border-left:5px solid #e6b968;border-radius:9px;background:linear-gradient(120deg,#172535 0%,#111b28 65%,#24354a 100%);position:relative;overflow:hidden;}
.fe-slate .fe-hero:after {content:'';position:absolute;width:320px;height:320px;border:1px solid #ffffff13;border-radius:50%;right:-90px;top:-160px;pointer-events:none;box-shadow:0 0 0 48px #ffffff05,0 0 0 96px #ffffff04;}
.fe-slate .fe-section {color:#e9edf1;font-size:1.1rem;letter-spacing:.09em;font-weight:850;border-bottom:1px solid #455367;padding:.5rem 0 .6rem;margin:1.1rem 0 .8rem;}
.fe-slate .fe-section span {color:#e6b968;margin-right:.55rem;}
.fe-slate .fe-match {border:1px solid #36495d;border-top:3px solid #e6b968;background:linear-gradient(120deg,#182637,#233549);border-radius:8px;padding:1rem 1.15rem;margin:.65rem 0 .3rem;}
.fe-slate .fe-meta {color:#b5c3cf;letter-spacing:.1em;font-size:.71rem;font-weight:750;text-transform:uppercase;}
.fe-slate .fe-scoregrid {display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:.7rem;margin:.7rem 0;}
.fe-slate .fe-team {min-width:0;text-align:center;}
.fe-slate .fe-team-name {font-weight:850;font-size:clamp(.9rem,1.8vw,1.3rem);line-height:1.2;overflow-wrap:anywhere;}
.fe-slate .fe-score {font-size:clamp(2.2rem,4vw,3.2rem);font-weight:900;font-variant-numeric:tabular-nums;line-height:1.1;}
.fe-slate .fe-versus {font-weight:900;color:#e6b968;font-size:.85rem;letter-spacing:.16em;}
.fe-slate .fe-projection-label {color:#aebccc;font-size:.67rem;letter-spacing:.14em;text-align:center;margin-top:.5rem;}
.fe-slate .fe-stats {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.6rem;border-top:1px solid #4b5c6c;padding-top:.8rem;margin-top:.8rem;}
.fe-slate .fe-stat {text-align:center;min-width:0;}
.fe-slate .fe-stat-label {font-size:.64rem;letter-spacing:.08em;color:#a9b8c7;}
.fe-slate .fe-stat-value {font-size:1rem;font-weight:850;color:#f1f5f8;overflow-wrap:anywhere;}
.fe-slate .fe-empty {border:1px solid #36495d;padding:1rem;border-radius:7px;color:#b9c5d0;}
/* Native Streamlit widgets remain native and accessible. */
.st-key-fe_slate_filters [data-testid='stSelectbox'] label,
.st-key-fe_slate_date [data-testid='stDateInput'] label {font-weight:700;}
.st-key-fe_slate_game_details [data-testid='stExpander'] {border:1px solid #34465b;border-radius:8px;}
@media(max-width:640px){.fe-slate .fe-hero{padding:1rem}.fe-slate .fe-match{padding:.8rem}.fe-slate .fe-stats{gap:.3rem}.fe-slate .fe-stat-value{font-size:.85rem}}
</style>
"""


def _safe(value):
    return escape(str(value if value is not None else '—'))


def _number(value, digits=1):
    try:
        n = float(value)
        return f'{n:.{digits}f}' if pd.notna(n) else '—'
    except (TypeError, ValueError):
        return '—'


def _percent(value):
    try:
        n = float(value)
        return f'{n:.0%}' if pd.notna(n) else '—'
    except (TypeError, ValueError):
        return '—'


def _tipoff(value):
    try:
        return pd.to_datetime(value, utc=True).tz_convert('America/Chicago').strftime('%-I:%M %p CT')
    except (TypeError, ValueError):
        return 'TBD'


def _status_bucket(value):
    s = str(value or '').upper()
    if 'FINAL' in s or s == 'STATUS_FINAL':
        return 'Final'
    if any(word in s for word in ('IN_PROGRESS', 'INPROGRESS', 'HALFTIME', 'STATUS_IN_PROGRESS')):
        return 'Live'
    return 'Scheduled'


@st.cache_data(ttl=300, show_spinner=False)
def _schedule(game_date):
    return get_daily_schedule(game_date)


def _team_table(team_stats_df, home, away):
    columns = [
        ('Offensive efficiency', 'off_eff', True),
        ('Defensive efficiency', 'def_eff', False),
        ('Tempo', 'possessions', None),
        ('Overall rating', 'Overall_Rating', True),
        ('Strength of schedule', 'SOS_Rating', None),
    ]
    a = team_stats_df.loc[team_stats_df['Team'].eq(away)]
    h = team_stats_df.loc[team_stats_df['Team'].eq(home)]
    if a.empty or h.empty:
        st.info('Team profile comparison is unavailable.')
        return
    a, h = a.iloc[0], h.iloc[0]
    rows = []
    for label, col, higher_better in columns:
        if col not in team_stats_df.columns:
            continue
        av, hv = pd.to_numeric(a[col], errors='coerce'), pd.to_numeric(h[col], errors='coerce')
        if pd.isna(av) or pd.isna(hv):
            continue
        if higher_better is None:
            edge = 'Style / context'
        elif av == hv:
            edge = 'Even'
        else:
            edge = away if ((av > hv) == higher_better) else home
        rows.append({'Metric': label, away: round(float(av), 2), home: round(float(hv), 2), 'Edge': edge})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info('Comparable team statistics are unavailable.')


def _match_html(game, result):
    away, home = _safe(game['away_team']), _safe(game['home_team'])
    neutral = bool(game['neutral_site'])
    site = 'NEUTRAL COURT' if neutral else 'HOME COURT'
    away_score = _number(result.get('proj_score2'))
    home_score = _number(result.get('proj_score1'))
    return f'''<div class="fe-slate"><div class="fe-match">
      <div class="fe-meta">{_safe(_tipoff(game['date']))} &nbsp; · &nbsp; {site} &nbsp; · &nbsp; {_safe(game.get('status_detail',''))}</div>
      <div class="fe-projection-label">FRONEDGE PROJECTED SCORE</div>
      <div class="fe-scoregrid">
        <div class="fe-team"><div class="fe-team-name">{away}</div><div class="fe-score">{away_score}</div></div>
        <div class="fe-versus">VS</div>
        <div class="fe-team"><div class="fe-team-name">{home}</div><div class="fe-score">{home_score}</div></div>
      </div>
      <div class="fe-stats">
        <div class="fe-stat"><div class="fe-stat-label">AWAY WIN %</div><div class="fe-stat-value">{_percent(result.get('win_prob2'))}</div></div>
        <div class="fe-stat"><div class="fe-stat-label">PROJECTED TOTAL</div><div class="fe-stat-value">{_number(result.get('total'))}</div></div>
        <div class="fe-stat"><div class="fe-stat-label">HOME WIN %</div><div class="fe-stat-value">{_percent(result.get('win_prob1'))}</div></div>
      </div>
    </div></div>'''


def render_daily_slate(team_stats_df, simulate_matchup, get_team_logo=None):
    """Render the entire Daily Slate page, independent of sidebar/navigation."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('''<div class="fe-slate"><div class="fe-hero">
      <div class="fe-eyebrow">FRONEDGE METRICS / COLLEGE BASKETBALL</div>
      <div class="fe-title">DAILY SLATE</div>
      <div class="fe-sub">The day's matchups. The model's read. Your scouting edge.</div>
    </div></div>''', unsafe_allow_html=True)

    today = datetime.now(ZoneInfo('America/Chicago')).date()
    if 'fe_slate_selected_date' not in st.session_state:
        st.session_state['fe_slate_selected_date'] = today

    with st.container(key='fe_slate_date'):
        prev_col, date_col, next_col = st.columns([1, 2, 1], vertical_alignment='bottom')
        with prev_col:
            if st.button('← Previous', use_container_width=True, key='fe_slate_prev'):
                st.session_state['fe_slate_selected_date'] -= timedelta(days=1)
                st.rerun()
        with date_col:
            selected_date = st.date_input('Game date', key='fe_slate_selected_date')
        with next_col:
            if st.button('Next →', use_container_width=True, key='fe_slate_next'):
                st.session_state['fe_slate_selected_date'] += timedelta(days=1)
                st.rerun()

    try:
        slate = _schedule(selected_date)
    except Exception as exc:
        st.error(f'Unable to retrieve the ESPN schedule: {exc}')
        return
    if slate.empty:
        st.info('No games listed for this date.')
        return

    valid = set(team_stats_df['Team'].dropna()) if not team_stats_df.empty else set()
    slate = slate.copy()
    slate['eligible'] = slate['home_team'].isin(valid) & slate['away_team'].isin(valid)
    slate['status_bucket'] = slate['status'].map(_status_bucket)
    slate['site_bucket'] = slate['neutral_site'].map(lambda x: 'Neutral' if bool(x) else 'Home')
    slate['tipoff'] = slate['date'].map(_tipoff)

    a, b, c, d = st.columns(4)
    a.metric('All games', len(slate))
    b.metric('Model matchups', int(slate['eligible'].sum()))
    c.metric('Neutral-site games', int(slate['neutral_site'].fillna(False).sum()))
    d.metric('Other matchups', int((~slate['eligible']).sum()))

    st.markdown('<div class="fe-slate"><div class="fe-section"><span>01 /</span> GAME SELECT</div></div>', unsafe_allow_html=True)
    teams = sorted(set(slate['home_team'].dropna()) | set(slate['away_team'].dropna()))
    with st.container(key='fe_slate_filters'):
        x, y, z = st.columns([2, 1, 1])
        with x:
            team = st.selectbox('Team', ['All teams'] + teams, key='fe_slate_team')
        with y:
            status = st.selectbox('Status', ['All', 'Scheduled', 'Live', 'Final'], key='fe_slate_status')
        with z:
            site = st.selectbox('Court', ['All', 'Home', 'Neutral'], key='fe_slate_site')

    filtered = slate
    if team != 'All teams':
        filtered = filtered[(filtered['home_team'] == team) | (filtered['away_team'] == team)]
    if status != 'All':
        filtered = filtered[filtered['status_bucket'] == status]
    if site != 'All':
        filtered = filtered[filtered['site_bucket'] == site]

    eligible = filtered[filtered['eligible']]
    other = filtered[~filtered['eligible']]

    st.markdown('<div class="fe-slate"><div class="fe-section"><span>02 /</span> MATCHUP BOARD</div></div>', unsafe_allow_html=True)
    st.caption('Select a game for the full scouting breakdown. Projections are pregame estimates, not live score predictions.')
    if eligible.empty:
        st.info('No model-eligible matchups match these filters.')
    else:
        # Calculate only the selected game, not 100+ simulations on every rerun.
        labels = [f"{_tipoff(row['date'])}  |  {row['away_team']} vs {row['home_team']}" for _, row in eligible.iterrows()]
        ids = eligible['game_id'].astype(str).tolist()
        selected_id = st.selectbox('Select matchup', ids, format_func=lambda v: labels[ids.index(v)], key='fe_slate_matchup')
        game = eligible.loc[eligible['game_id'].astype(str).eq(selected_id)].iloc[0]
        home, away = game['home_team'], game['away_team']
        neutral = bool(game['neutral_site'])
        try:
            # Same model and site semantics as the original page.
            result = simulate_matchup(
                team_stats_df=team_stats_df,
                team1_name=home,
                team2_name=away,
                site_value='neutral' if neutral else 'team1_home',
                n_sims=1000,
            )
        except Exception as exc:
            st.error(f'Projection failed for {away} vs {home}: {exc}')
            result = None

        if result is not None:
            st.markdown(_match_html(game, result), unsafe_allow_html=True)
            # Logo images use the app's existing branding lookup.
            if get_team_logo is not None:
                logo_left, logo_right = st.columns(2)
                for col, name in ((logo_left, away), (logo_right, home)):
                    path = get_team_logo(name)
                    if path is not None:
                        with col:
                            st.image(str(path), width=72)

            st.markdown('<div class="fe-slate"><div class="fe-section"><span>03 /</span> SCOUTING REPORT</div></div>', unsafe_allow_html=True)
            t1, t2, t3 = st.tabs(['Matchup factors', 'Projected box score', 'Model insights'])
            with t1:
                _team_table(team_stats_df, home, away)
            with t2:
                away_box = result.get('box_score_team2') or {}
                home_box = result.get('box_score_team1') or {}
                st.dataframe(pd.DataFrame([{'Team': away, **away_box}, {'Team': home, **home_box}]), hide_index=True, use_container_width=True)
            with t3:
                insight = {
                    'Favorite': result.get('favorite', '—'),
                    'Projected spread': result.get('spread', '—'),
                    'Confidence': result.get('confidence', '—'),
                    'Game style': result.get('game_style', '—'),
                    'Volatility': result.get('volatility_tag', '—'),
                    'Upset alert': result.get('upset_alert', '—'),
                }
                for k, v in insight.items():
                    st.write(f'**{k}:** {v}')

        st.markdown('#### Full D1 matchup schedule')
        st.dataframe(
            eligible[['tipoff', 'away_team', 'home_team', 'site_bucket', 'status_detail']].rename(columns={
                'tipoff': 'Tipoff (CT)', 'away_team': 'Away', 'home_team': 'Home',
                'site_bucket': 'Court', 'status_detail': 'Status',
            }),
            hide_index=True, use_container_width=True,
        )

    if not other.empty:
        with st.expander(f'Other matchups without projections ({len(other)})'):
            st.caption('At least one team is absent from the current model dataset. This can include non-D1 opponents or unmatched D1 team names.')
            st.dataframe(other[['tipoff', 'away_team', 'home_team', 'site_bucket', 'status_detail']], hide_index=True, use_container_width=True)

    st.caption('Schedule and game status: ESPN. All projections: FronEdge Metrics. Model estimates are not guaranteed outcomes.')
