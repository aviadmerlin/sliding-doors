"""Sliding Doors — interactive counterfactual match intelligence.

Pick a match, see its win-probability story, and ask "what if" about any
goal, missed chance, or red card. Heavy computation (fitting the trait
regression, simulating win probability) is cached so only the very first
load is slow; everything after that is instant.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.counterfactual import Counterfactual, CounterfactualResult, apply_counterfactual
from src.league import Club
from src.trait_estimation import (build_regression_table, compute_state_exposure, compute_weights,
                                   fit_traits_distribution, match_real_states)
from src.win_probability import (clubs_from_traits, find_most_dramatic_match, rank_turning_points,
                                  sample_team_draws, win_probability_curve_with_uncertainty)

DATA_DIR = Path(__file__).resolve().parent / "data"

N_TRAIT_DRAWS = 18
TRIALS_PER_DRAW = 120
SHRINK_COLS = ["comeback_factor", "ten_men_resilience"]
CHECKPOINTS = list(range(0, 91))
COARSE_TRIALS = 150
COARSE_CHECKPOINTS = list(range(0, 91, 10))

COLOR_HOME = "#22C55E"
COLOR_DRAW = "#9CA3AF"
COLOR_AWAY = "#EF4444"

CHANGE_LABELS = {
    "shot_becomes_goal": "Turn into a goal",
    "goal_cancelled": "Cancel the goal",
    "red_card_undone": "Undo the red card",
}
KIND_TO_CHANGE = {"goal": "goal_cancelled", "missed_penalty": "shot_becomes_goal",
                   "big_chance": "shot_becomes_goal", "red_card": "red_card_undone"}
KIND_BADGE = {"goal": ("Goal", "green"), "missed_penalty": ("Penalty missed", "red"),
              "big_chance": ("Big chance missed", "orange"), "red_card": ("Red card", "red")}

st.set_page_config(page_title="Sliding Doors", page_icon=":material/sports_soccer:", layout="wide")


@dataclass
class Engine:
    matches: pd.DataFrame
    shots: pd.DataFrame
    red_cards: pd.DataFrame
    teams: list[str]
    latest_season: int
    season_matches: pd.DataFrame
    point_clubs: dict[str, Club]
    team_draws: list[dict[str, Club]]
    default_match_id: int


@st.cache_resource(show_spinner="Fitting the trait model from 5 seasons of data (first load only, ~30s)...")
def load_engine() -> Engine:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    shots = pd.read_csv(DATA_DIR / "shots.csv")
    red_cards = pd.read_csv(DATA_DIR / "red_cards.csv")
    estimated_traits = pd.read_csv(DATA_DIR / "estimated_traits.csv").set_index("team")

    teams = sorted(matches.home_team.unique())
    latest_season = int(matches.season.max())
    season_matches = matches[matches.season == latest_season].reset_index(drop=True)
    point_clubs = clubs_from_traits(estimated_traits)

    table = build_regression_table(matches, shots, red_cards)
    weights = compute_weights(table.season, None, latest_season)
    _, params, cov = fit_traits_distribution(table, teams, weights)
    exposure = compute_state_exposure(table, teams)

    rng = np.random.default_rng(42)
    team_draws = sample_team_draws(params, cov, teams, exposure, SHRINK_COLS, N_TRAIT_DRAWS, rng)

    screen_rng = np.random.default_rng(0)
    dramatic_match, _ = find_most_dramatic_match(season_matches, shots, red_cards, point_clubs,
                                                   screen_rng, COARSE_TRIALS, COARSE_CHECKPOINTS)

    return Engine(matches=matches, shots=shots, red_cards=red_cards, teams=teams,
                  latest_season=latest_season, season_matches=season_matches, point_clubs=point_clubs,
                  team_draws=team_draws, default_match_id=int(dramatic_match.match_id))


@st.cache_data(show_spinner="Simulating thousands of replays...", max_entries=50)
def get_actual_curve(match_id: int, _engine: Engine) -> pd.DataFrame:
    match = _engine.matches[_engine.matches.match_id == match_id].iloc[0]
    states = match_real_states(match, _engine.shots, _engine.red_cards)
    rng = np.random.default_rng(1000 + match_id)
    return win_probability_curve_with_uncertainty(
        _engine.team_draws, match.home_team, match.away_team, states,
        (int(match.home_goals), int(match.away_goals)), rng, TRIALS_PER_DRAW, CHECKPOINTS)


@st.cache_data(show_spinner=False, max_entries=50)
def get_turning_points(match_id: int, _engine: Engine) -> pd.DataFrame:
    match = _engine.matches[_engine.matches.match_id == match_id].iloc[0]
    curve = get_actual_curve(match_id, _engine)
    return rank_turning_points(curve, match, _engine.shots, _engine.red_cards, top_n=3)


@st.cache_data(show_spinner="Re-simulating the counterfactual...", max_entries=100)
def get_counterfactual(match_id: int, minute: int, change: str, team: str | None,
                        _engine: Engine) -> CounterfactualResult:
    cf = Counterfactual(match_id=match_id, minute=minute, change=change, team=team)
    actual_curve = get_actual_curve(match_id, _engine)
    rng = np.random.default_rng(2000 + match_id * 97 + minute)
    return apply_counterfactual(cf, _engine.matches, _engine.shots, _engine.red_cards, _engine.team_draws,
                                 rng, TRIALS_PER_DRAW, CHECKPOINTS, actual_curve=actual_curve)


@st.cache_data(show_spinner=False, max_entries=50)
def build_timeline(match_id: int, _engine: Engine) -> pd.DataFrame:
    """Notable events for the timeline and the What-if picker: goals,
    missed penalties, missed big chances, and red cards."""
    match = _engine.matches[_engine.matches.match_id == match_id].iloc[0]
    shots = _engine.shots[_engine.shots.match_id == match_id]
    cards = _engine.red_cards[_engine.red_cards.match_id == match_id]
    notable = shots[shots.is_goal | shots.is_penalty | shots.is_big_chance].copy()

    rows = []
    for _, r in notable.iterrows():
        kind = "goal" if r.is_goal else ("missed_penalty" if r.is_penalty else "big_chance")
        if kind == "goal":
            is_home = r.team == match.home_team
            h_after = int(r.home_score) + is_home
            a_after = int(r.away_score) + (not is_home)
            score = f"{h_after}-{a_after}"
            desc = f"{r.player} ({r.team})" + (", penalty" if r.is_penalty else "")
        else:
            score = f"{int(r.home_score)}-{int(r.away_score)}"
            desc = f"{r.player} ({r.team}), xG {r.xg:.2f}"
        rows.append({"minute": int(r.minute), "team": r.team, "kind": kind,
                     "player": r.player, "description": desc, "score_after": score})

    for _, r in cards.iterrows():
        rows.append({"minute": int(r.minute), "team": r.team, "kind": "red_card",
                     "player": r.player, "description": f"{r.player} ({r.team})", "score_after": ""})

    timeline = pd.DataFrame(rows)
    return timeline.sort_values("minute").reset_index(drop=True) if len(timeline) else timeline


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def add_band(fig: go.Figure, curve: pd.DataFrame, y_col: str, se_col: str, color: str, name: str) -> None:
    upper = curve[y_col] + 1.96 * curve[se_col]
    lower = (curve[y_col] - 1.96 * curve[se_col]).clip(lower=0)
    fig.add_trace(go.Scatter(x=curve.minute, y=upper, mode="lines", line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curve.minute, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                              fillcolor=rgba(color, 0.15), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=curve.minute, y=curve[y_col], mode="lines", name=name,
                              line=dict(color=color, width=2.5),
                              hovertemplate="%{y:.0%}<extra></extra>"))


def style_figure(fig: go.Figure) -> None:
    fig.update_layout(
        xaxis_title="minute", yaxis_title="win probability",
        yaxis_tickformat=".0%", yaxis_range=[0, 1.18], xaxis_range=[0, 90],
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=30, b=40, l=10, r=10), height=420,
    )


def add_minute_markers(fig: go.Figure, minutes: list[int], labels: list[str]) -> None:
    """Vertical dotted lines + labels at the given minutes, staggering the
    label height when two markers fall close together so they don't collide."""
    order = sorted(range(len(minutes)), key=lambda i: minutes[i])
    prev_minute, level = None, 0
    for i in order:
        minute, label = minutes[i], labels[i]
        level = level + 1 if prev_minute is not None and minute - prev_minute < 6 else 0
        prev_minute = minute
        fig.add_vline(x=minute, line_dash="dot", line_color="rgba(255,255,255,0.35)")
        fig.add_annotation(x=minute, y=1.0 + level * 0.07, yref="y", text=label, showarrow=False,
                            font=dict(size=11, color="#E5E7EB"))


def build_win_prob_figure(curve: pd.DataFrame, match: pd.Series, turning_points: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    add_band(fig, curve, "p_home", "se_home", COLOR_HOME, f"{match.home_team} win")
    add_band(fig, curve, "p_draw", "se_draw", COLOR_DRAW, "Draw")
    add_band(fig, curve, "p_away", "se_away", COLOR_AWAY, f"{match.away_team} win")

    minutes = turning_points.minute.tolist()
    add_minute_markers(fig, minutes, [f"{m}'" for m in minutes])
    style_figure(fig)
    return fig


def build_counterfactual_figure(result: CounterfactualResult, fork_minute: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.actual_curve.minute, y=result.actual_curve.p_home,
                              mode="lines", name="Actual", line=dict(color=COLOR_HOME, width=2.5)))
    fig.add_trace(go.Scatter(x=result.counterfactual_curve.minute, y=result.counterfactual_curve.p_home,
                              mode="lines", name="Counterfactual",
                              line=dict(color=COLOR_AWAY, width=2.5, dash="dash")))
    add_minute_markers(fig, [fork_minute], [f"{fork_minute}' fork point"])
    style_figure(fig)
    return fig


def main() -> None:
    st.title("Sliding Doors", icon=":material/sports_soccer:")
    st.caption("Counterfactual match intelligence for a fictional 20-club league")

    engine = load_engine()

    with st.sidebar:
        st.markdown("**Match**")
        match_ids = engine.season_matches.sort_values("matchday").match_id.tolist()

        def format_match(mid: int) -> str:
            m = engine.season_matches[engine.season_matches.match_id == mid].iloc[0]
            return f"MD{m.matchday}: {m.home_team} {m.home_goals}-{m.away_goals} {m.away_team}"

        default_index = match_ids.index(engine.default_match_id)
        match_id = st.selectbox("Match", match_ids, index=default_index, format_func=format_match,
                                 label_visibility="collapsed")
        st.caption(f"Season {engine.latest_season} · {len(match_ids)} matches. Defaults to the most "
                   "dramatic match (largest win-probability swing).")

    match = engine.matches[engine.matches.match_id == match_id].iloc[0]

    # --- Header ---
    with st.container(border=True):
        cols = st.columns([4, 2, 4], vertical_alignment="center")
        with cols[0]:
            st.subheader(match.home_team, text_alignment="right")
        with cols[1]:
            st.title(f"{match.home_goals} – {match.away_goals}", text_alignment="center")
        with cols[2]:
            st.subheader(match.away_team, text_alignment="left")
        st.caption(f"Season {match.season} · Matchday {match.matchday}", text_alignment="center")

    timeline = build_timeline(match_id, engine)

    # --- Event timeline ---
    st.subheader("Timeline", icon=":material/timeline:")
    if len(timeline) == 0:
        st.caption("No notable events in this match.")
    else:
        with st.container(border=True):
            for _, ev in timeline.iterrows():
                badge_label, badge_color = KIND_BADGE[ev.kind]
                row = st.container(horizontal=True, vertical_alignment="center")
                with row:
                    st.markdown(f"**{ev.minute}'**")
                    st.badge(badge_label, color=badge_color)
                    st.write(ev.description)
                    if ev.score_after:
                        st.caption(ev.score_after)

    # --- Win probability chart ---
    curve = get_actual_curve(match_id, engine)
    turning_points = get_turning_points(match_id, engine)

    st.subheader("Win probability", icon=":material/show_chart:")
    st.caption("Bands reflect both simulation randomness and uncertainty in the estimated club traits.")
    fig = build_win_prob_figure(curve, match, turning_points)
    st.plotly_chart(fig, width="stretch",
                     alt=f"Win probability over 90 minutes for {match.home_team} vs {match.away_team}")

    # --- Turning points ---
    st.subheader("Top 3 turning points", icon=":material/bolt:")
    tp_cols = st.columns(3)
    for col, (_, tp) in zip(tp_cols, turning_points.iterrows()):
        with col, st.container(border=True, height="stretch"):
            matching = timeline[timeline.minute == tp.minute]
            desc = matching.iloc[0].description if len(matching) else tp.event
            kind = matching.iloc[0].kind if len(matching) else None
            badge_label, badge_color = KIND_BADGE.get(kind, ("Event", "gray"))
            st.markdown(f"**{tp.minute}'**")
            st.badge(badge_label, color=badge_color)
            st.caption(desc)
            st.metric("Win-probability swing", f"{tp.swing:.0%}")

    # --- What if? ---
    st.subheader("What if...?", icon=":material/alt_route:")

    if len(timeline) == 0:
        st.caption("No events available to explore in this match.")
    else:
        with st.container(border=True):
            event_cols = st.columns(2)
            with event_cols[0]:
                event_idx = st.selectbox(
                    "Event", timeline.index,
                    format_func=lambda i: f"{timeline.loc[i, 'minute']}' — "
                                           f"{KIND_BADGE[timeline.loc[i, 'kind']][0]} — {timeline.loc[i, 'description']}")
            selected_event = timeline.loc[event_idx]
            default_change = KIND_TO_CHANGE[selected_event.kind]
            change_options = list(CHANGE_LABELS.keys())

            with event_cols[1]:
                change = st.selectbox(
                    "Change", change_options, format_func=lambda c: CHANGE_LABELS[c],
                    index=change_options.index(default_change), key=f"change_{event_idx}")

            compatible = change == default_change
            if not compatible:
                st.warning(f"This event doesn't support '{CHANGE_LABELS[change]}'. "
                           f"The compatible change is '{CHANGE_LABELS[default_change]}'.",
                           icon=":material/warning:")

            run = st.button("Run counterfactual", type="primary", icon=":material/play_arrow:",
                            disabled=not compatible)

        if run:
            result = get_counterfactual(match_id, int(selected_event.minute), change,
                                         selected_event.team, engine)

            st.markdown(f"#### {result.description}")
            cf_fig = build_counterfactual_figure(result, int(selected_event.minute))
            st.plotly_chart(cf_fig, width="stretch",
                             alt=f"Actual versus counterfactual win probability: {result.description}")

            outcome_cols = st.columns(2)
            with outcome_cols[0], st.container(border=True, height="stretch"):
                st.markdown("**Actual outcome**")
                st.metric(f"{match.home_team} win", f"{result.actual_final['p_home']:.0%}")
                st.metric("Draw", f"{result.actual_final['p_draw']:.0%}")
                st.metric(f"{match.away_team} win", f"{result.actual_final['p_away']:.0%}")
            with outcome_cols[1], st.container(border=True, height="stretch"):
                st.markdown("**Counterfactual outcome**")
                st.metric(f"{match.home_team} win", f"{result.counterfactual_final['p_home']:.0%}",
                         delta=f"{result.counterfactual_final['p_home'] - result.actual_final['p_home']:+.0%}")
                st.metric("Draw", f"{result.counterfactual_final['p_draw']:.0%}",
                         delta=f"{result.counterfactual_final['p_draw'] - result.actual_final['p_draw']:+.0%}")
                st.metric(f"{match.away_team} win", f"{result.counterfactual_final['p_away']:.0%}",
                         delta=f"{result.counterfactual_final['p_away'] - result.actual_final['p_away']:+.0%}")

    # --- AI placeholder ---
    st.subheader("Ask a what-if question", icon=":material/chat:")
    with st.container(border=True):
        st.badge("Coming soon", icon=":material/schedule:", color="orange")
        st.text_input("Ask in plain English", label_visibility="collapsed", disabled=True,
                      placeholder="e.g. What if Northgate had defended better after half time?")
        st.caption("A natural-language agent (on Microsoft Foundry) will answer free-text questions "
                  "like this using the same simulation and counterfactual engine shown above.")


if __name__ == "__main__":
    main()
