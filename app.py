"""Sliding Doors — interactive counterfactual match intelligence.

Pick a match, see its win-probability story, and ask "what if" about any
goal, missed chance, or red card.

Cold-start performance: fitting the trait regression from 5 seasons of
shot-level data takes ~15-20s. scripts/precompute_featured_matches.py
fits it once offline and saves the result to data/model_team_draws.csv,
so the app always loads the model from disk (a fast CSV read) instead of
refitting. It also precomputes full curves + turning points for a
shortlist of ~11 "featured" matches, which load here instantly with no
simulation at all. Any other match -- and any counterfactual, on any
match -- still works, simulating on demand with a progress indicator,
but no longer pays the model-fitting cost.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.counterfactual import Counterfactual, CounterfactualResult, apply_counterfactual
from src.league import Club
from src.storylines import club_storylines
from src.trait_estimation import match_real_states
from src.win_probability import clubs_from_traits, rank_turning_points, win_probability_curve_with_uncertainty

DATA_DIR = Path(__file__).resolve().parent / "data"

# Must match scripts/precompute_featured_matches.py so on-demand curves are
# directly comparable to the precomputed featured ones.
TRIALS_PER_DRAW = 200
CHECKPOINTS = list(range(0, 91))

COLOR_HOME = "#22C55E"
COLOR_DRAW = "#9CA3AF"
COLOR_AWAY = "#EF4444"
COLOR_REAL = "#3B82F6"
COLOR_HYPOTHETICAL = "#F59E0B"

CHANGE_LABELS = {
    "shot_becomes_goal": "Turn into a goal",
    "goal_cancelled": "Cancel the goal",
    "red_card_undone": "Undo the red card",
}
KIND_TO_CHANGE = {"goal": "goal_cancelled", "missed_penalty": "shot_becomes_goal",
                   "big_chance": "shot_becomes_goal", "red_card": "red_card_undone"}
KIND_BADGE = {"goal": ("Goal", "green"), "missed_penalty": ("Penalty missed", "red"),
              "big_chance": ("Big chance missed", "orange"), "red_card": ("Red card", "red")}

# (high-trait phrase, low-trait phrase), shown only when a club is in the top/bottom
# 30% of the league for that trait. Confidence tags reflect the validation
# correlations from step 2 (see README "Estimated vs. true traits").
TRAIT_DESCRIPTIONS = {
    "attack": ("Potent going forward", "Struggles to create chances", "well-established"),
    "defense": ("Hard to break down", "Leaky at the back", "well-established"),
    "protect_lead_factor": ("Strong at protecting a lead", "Can let leads slip", "well-established"),
    "fatigue_factor": ("Strong legs late on", "Fades late in matches", "rougher estimate"),
    "ten_men_resilience": ("Copes well with 10 men", "Struggles with 10 men", "rougher estimate"),
    "comeback_factor": ("Dangerous when behind", "Rarely fights back from behind", "rougher estimate"),
}

st.set_page_config(page_title="Sliding Doors", page_icon=":material/sports_soccer:", layout="wide")


# --- Fast, always-cheap data: plain CSV/JSON reads, no simulation ---

@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "matches.csv")


@st.cache_data(show_spinner=False)
def load_shots() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "shots.csv")


@st.cache_data(show_spinner=False)
def load_red_cards() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "red_cards.csv")


@st.cache_data(show_spinner=False)
def load_season_matches() -> pd.DataFrame:
    matches = load_matches()
    return matches[matches.season == matches.season.max()].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_estimated_traits() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "estimated_traits.csv").set_index("team")


@st.cache_data(show_spinner=False)
def load_trait_jumps() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "trait_jumps.csv")


@st.cache_data(show_spinner=False)
def load_power_ranking() -> pd.DataFrame:
    """The model's power ranking (scripts/estimate_traits.py), already
    validated there against the real final table via rank correlation --
    this just joins the two for side-by-side display."""
    power_ranking = pd.read_csv(DATA_DIR / "power_ranking.csv")
    season_matches = load_season_matches()

    points: dict[str, int] = {team: 0 for team in power_ranking.team}
    for _, m in season_matches.iterrows():
        if m.home_goals > m.away_goals:
            points[m.home_team] += 3
        elif m.home_goals < m.away_goals:
            points[m.away_team] += 3
        else:
            points[m.home_team] += 1
            points[m.away_team] += 1

    real_table = pd.Series(points, name="real_points").rename_axis("team").reset_index()
    merged = power_ranking.merge(real_table, on="team").sort_values(
        "expected_goal_difference", ascending=False).reset_index(drop=True)
    merged.insert(0, "model_rank", range(1, len(merged) + 1))
    merged["real_rank"] = merged.real_points.rank(ascending=False, method="min").astype(int)
    return merged


@st.cache_data(show_spinner=False)
def load_featured() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Precomputed curves/turning points for the featured shortlist (see
    scripts/precompute_featured_matches.py) -- instant, no model fit."""
    curves = pd.read_csv(DATA_DIR / "featured_curves.csv")
    turning_points = pd.read_csv(DATA_DIR / "featured_turning_points.csv")
    turning_points["secondary_events"] = turning_points.secondary_events.apply(json.loads)
    meta = json.loads((DATA_DIR / "featured_matches.json").read_text())
    return curves, turning_points, meta


# --- The fitted model: loaded from disk, never refit by the app itself ---

@st.cache_resource(show_spinner="Loading the trait model...")
def load_team_draws() -> list[dict[str, Club]]:
    """Loads the trait draws scripts/precompute_featured_matches.py already
    fit and saved -- a fast CSV read + reshape, not a refit."""
    path = DATA_DIR / "model_team_draws.csv"
    if not path.exists():
        st.error("Model not found: run `python scripts/precompute_featured_matches.py` first.",
                  icon=":material/error:")
        st.stop()
    df = pd.read_csv(path)
    return [clubs_from_traits(group.set_index("team")) for _, group in df.groupby("draw")]


@st.cache_data(show_spinner="Simulating thousands of replays for this match...", max_entries=50)
def simulate_curve(match_id: int, _team_draws: list[dict[str, Club]]) -> pd.DataFrame:
    matches, shots, red_cards = load_matches(), load_shots(), load_red_cards()
    match = matches[matches.match_id == match_id].iloc[0]
    states = match_real_states(match, shots, red_cards)
    return win_probability_curve_with_uncertainty(
        _team_draws, match.home_team, match.away_team, states,
        (int(match.home_goals), int(match.away_goals)), 1000 + match_id, TRIALS_PER_DRAW, CHECKPOINTS)


@st.cache_data(show_spinner="Re-simulating the counterfactual...", max_entries=100)
def compute_counterfactual(match_id: int, minute: int, change: str, team: str | None,
                            _team_draws: list[dict[str, Club]],
                            _actual_curve: pd.DataFrame) -> CounterfactualResult:
    matches, shots, red_cards = load_matches(), load_shots(), load_red_cards()
    cf = Counterfactual(match_id=match_id, minute=minute, change=change, team=team)
    seed = 2000 + match_id * 97 + minute
    return apply_counterfactual(cf, matches, shots, red_cards, _team_draws, seed, TRIALS_PER_DRAW,
                                 CHECKPOINTS, actual_curve=_actual_curve)


def get_curve_and_turning_points(match_id: int, featured_curves: pd.DataFrame,
                                  featured_turning_points: pd.DataFrame,
                                  featured_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Instant for a featured match (precomputed). Simulates on demand for
    anything else, using the model already loaded from disk."""
    if match_id in featured_ids:
        curve = (featured_curves[featured_curves.match_id == match_id]
                 .drop(columns="match_id").reset_index(drop=True))
        turning_points = (featured_turning_points[featured_turning_points.match_id == match_id]
                           .drop(columns="match_id").reset_index(drop=True))
        return curve, turning_points

    team_draws = load_team_draws()
    curve = simulate_curve(match_id, team_draws)
    matches, shots, red_cards = load_matches(), load_shots(), load_red_cards()
    match = matches[matches.match_id == match_id].iloc[0]
    turning_points = rank_turning_points(curve, match, shots, red_cards, top_n=3)
    return curve, turning_points


@st.cache_data(show_spinner=False, max_entries=50)
def build_timeline(match_id: int) -> pd.DataFrame:
    """Notable events for the timeline and the What-if picker: goals,
    missed penalties, missed big chances, and red cards. Cheap -- no
    simulation needed, so this is always instant, featured or not."""
    matches, shots, red_cards = load_matches(), load_shots(), load_red_cards()
    match = matches[matches.match_id == match_id].iloc[0]
    shots = shots[shots.match_id == match_id]
    cards = red_cards[red_cards.match_id == match_id]
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


def club_profile(team: str, estimated_traits: pd.DataFrame) -> list[dict]:
    """Traits where this club sits in the top or bottom 30% of the league,
    i.e. actually notable rather than merely average."""
    notable = []
    for trait, (high_label, low_label, confidence) in TRAIT_DESCRIPTIONS.items():
        percentile = (estimated_traits[trait] < estimated_traits.loc[team, trait]).mean()
        if percentile >= 0.7:
            notable.append({"label": high_label, "confidence": confidence})
        elif percentile <= 0.3:
            notable.append({"label": low_label, "confidence": confidence})
    return notable


def render_club_card(team: str, estimated_traits: pd.DataFrame, trait_jumps: pd.DataFrame) -> None:
    with st.container(border=True, height="stretch"):
        st.markdown(f"**{team}**")
        traits = club_profile(team, estimated_traits)
        if not traits:
            st.caption("Average across the board in our estimates — no standout traits.")
        else:
            for t in traits:
                st.caption(f"• {t['label']}")
            rough = [t["label"] for t in traits if t["confidence"] == "rougher estimate"]
            if rough:
                st.caption(f":material/info: Less certain estimate: {', '.join(rough).lower()}.")

        for line in club_storylines(team, trait_jumps):
            st.caption(f":material/history: {line}")


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


def compute_momentum(shots: pd.DataFrame, match: pd.Series, window: int = 12) -> pd.DataFrame:
    """Rolling-window REAL xG pressure per side, like a broadcast momentum
    bar -- a plain description of the real match's actual shots (not a
    model forecast): for each minute, each side's actual xG over the
    trailing `window` minutes, and the home-minus-away difference."""
    match_shots = shots[shots.match_id == match.match_id]
    home_xg, away_xg = np.zeros(91), np.zeros(91)
    for _, r in match_shots.iterrows():
        minute = int(r.minute)
        if r.team == match.home_team:
            home_xg[minute] += r.xg
        else:
            away_xg[minute] += r.xg

    rows = []
    for minute in range(1, 91):
        lo = max(1, minute - window + 1)
        home_roll, away_roll = float(home_xg[lo:minute + 1].sum()), float(away_xg[lo:minute + 1].sum())
        rows.append({"minute": minute, "home_xg": home_roll, "away_xg": away_roll,
                     "diff": home_roll - away_roll})
    return pd.DataFrame(rows)


def build_momentum_figure(momentum: pd.DataFrame, window: int) -> go.Figure:
    colors = [COLOR_HOME if d >= 0 else COLOR_AWAY for d in momentum["diff"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=momentum.minute, y=momentum["diff"], marker_color=colors,
                          hovertemplate="minute %{x}<br>xG pressure %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        xaxis_title="minute", yaxis_title=f"{window}-min rolling xG, home minus away",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_range=[0, 90], margin=dict(t=10, b=40, l=10, r=10), height=180, showlegend=False,
    )
    return fig


def build_counterfactual_figure(result: CounterfactualResult, outcome: str, match: pd.Series) -> go.Figure:
    """Plots the headline outcome's probability: the real match (a genuine
    forecast at every minute) vs. the counterfactual (identical up to the
    fork, then an honest FLAT line + band at the single post-change
    forecast -- there's no single counterfactual trajectory to trace
    minute by minute, so this doesn't fabricate one)."""
    y_col, se_col = f"p_{outcome}", f"se_{outcome}"
    fig = go.Figure()
    add_band(fig, result.actual_curve, y_col, se_col, COLOR_REAL, "Real match (forecast each minute)")
    add_band(fig, result.counterfactual_curve, y_col, se_col, COLOR_HYPOTHETICAL,
             "Counterfactual (forecast from the fork)")
    add_minute_markers(fig, [result.fork_minute], [f"{result.fork_minute}' fork point"])
    style_figure(fig)
    fig.update_layout(yaxis_title=f"{stat_label(outcome, match)} probability")
    return fig


def counterfactual_headline_outcome(result: CounterfactualResult) -> str:
    deltas = {o: result.fork_after[o] - result.fork_before[o] for o in ("home", "draw", "away")}
    return max(deltas, key=lambda o: abs(deltas[o]))


def counterfactual_headline_summary(result: CounterfactualResult, match: pd.Series, outcome: str) -> str:
    before, after = result.fork_before[outcome], result.fork_after[outcome]
    verb = "raises" if after > before else "lowers"
    return (f"This change {verb} {prose_label(outcome, match)} from {before:.0%} to {after:.0%}, "
            f"comparing forecasts right before vs. right after the {result.fork_minute}' mark "
            f"(not the final result, which is already known).")


def stat_label(outcome: str, match: pd.Series) -> str:
    return {"home": f"{match.home_team} win", "draw": "Draw", "away": f"{match.away_team} win"}[outcome]


def prose_label(outcome: str, match: pd.Series) -> str:
    return {"home": f"{match.home_team}'s chances", "draw": "the draw's chances",
            "away": f"{match.away_team}'s chances"}[outcome]


def turning_point_summary(tp: pd.Series, match: pd.Series) -> str:
    event_phrase = {"goal": f"{tp.event_player}'s {tp.minute}' goal",
                     "red_card": f"{tp.event_player}'s {tp.minute}' red card"}[tp.event_kind]
    verb = "boosts" if tp.headline_after > tp.headline_before else "hurts"
    summary = f"{event_phrase} {verb} {prose_label(tp.headline_outcome, match)}."
    if tp.secondary_events:
        extra = tp.secondary_events[0]
        summary += f" (Also this minute: {extra['player']} had a missed chance.)"
    return summary


def main() -> None:
    st.title("Sliding Doors", icon=":material/sports_soccer:")
    st.caption("Counterfactual match intelligence for a fictional 20-club league")
    st.markdown("**See how a match turned, and replay it with one moment changed.**")

    how_to = [
        (":material/list:", "Pick a match", "Choose any match from the sidebar — ⚡ ones load instantly."),
        (":material/show_chart:", "See the swings", "Watch win probability move through the 90 minutes."),
        (":material/alt_route:", "Try a What if", "Replay one moment differently and compare the outcome."),
    ]
    how_to_cols = st.columns(3)
    for col, (icon, title, desc) in zip(how_to_cols, how_to):
        with col, st.container(border=True):
            st.markdown(f"{icon} **{title}**")
            st.caption(desc)

    matches = load_matches()
    shots = load_shots()
    season_matches = load_season_matches()
    estimated_traits = load_estimated_traits()
    featured_curves, featured_turning_points, featured_meta = load_featured()
    featured_ids = set(featured_meta["featured_match_ids"])
    default_match_id = featured_meta["default_match_id"]

    with st.sidebar:
        st.markdown("**Match**")
        match_ids = season_matches.sort_values("matchday").match_id.tolist()

        def format_match(mid: int) -> str:
            m = season_matches[season_matches.match_id == mid].iloc[0]
            label = f"MD{m.matchday}: {m.home_team} {m.home_goals}-{m.away_goals} {m.away_team}"
            return label + (" ⚡" if mid in featured_ids else "")

        default_index = match_ids.index(default_match_id)
        match_id = st.selectbox("Match", match_ids, index=default_index, format_func=format_match,
                                 label_visibility="collapsed")
        st.caption(f"Season {int(season_matches.season.iloc[0])} · {len(match_ids)} matches. "
                   "⚡ matches load instantly (precomputed); others simulate on selection, "
                   "a few seconds the first time.")

        with st.expander("Model power ranking", icon=":material/leaderboard:"):
            st.caption("Each club's estimated attack/defense, turned into an expected goal "
                       "difference per match against a league-average opponent -- the model's own "
                       "view of true strength, independent of any one season's results.")
            power_ranking = load_power_ranking()
            st.dataframe(
                power_ranking[["model_rank", "team", "expected_goal_difference", "real_rank"]]
                .rename(columns={"model_rank": "Model rank", "team": "Club",
                                 "expected_goal_difference": "Exp. goal diff/match", "real_rank": "Real rank"}),
                hide_index=True, width="stretch",
                column_config={"Exp. goal diff/match": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Real rank = this season's actual final table. The two broadly agree but "
                       "aren't identical -- a season's results reflect day form and luck too, not "
                       "just underlying strength.")

    match = matches[matches.match_id == match_id].iloc[0]

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

    # --- Club profiles ---
    trait_jumps = load_trait_jumps()
    profile_cols = st.columns(2)
    with profile_cols[0]:
        render_club_card(match.home_team, estimated_traits, trait_jumps)
    with profile_cols[1]:
        render_club_card(match.away_team, estimated_traits, trait_jumps)

    timeline = build_timeline(match_id)

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
    curve, turning_points = get_curve_and_turning_points(match_id, featured_curves,
                                                           featured_turning_points, featured_ids)

    st.subheader("Win probability", icon=":material/show_chart:")
    st.caption("Bands reflect both simulation randomness and uncertainty in the estimated club traits. "
                "Moves moderately between goals too, as the model updates its belief about which side "
                "is currently playing better than their usual level (see Momentum below).")
    fig = build_win_prob_figure(curve, match, turning_points)
    st.plotly_chart(fig, width="stretch",
                     alt=f"Win probability over 90 minutes for {match.home_team} vs {match.away_team}")

    # --- Momentum chart ---
    st.subheader("Momentum", icon=":material/bolt:")
    st.caption("Real shot data only, not a model forecast: each side's actual xG over the last 12 "
                "minutes. The win-probability chart above reacts to the same kind of signal (overall "
                "xG pressure), but treats short bursts like this as noise around a team's underlying "
                "day form, not as their own tracked effect.")
    momentum = compute_momentum(shots, match)
    st.plotly_chart(build_momentum_figure(momentum, window=12), width="stretch",
                     alt=f"Rolling 12-minute xG momentum for {match.home_team} vs {match.away_team}")

    # --- Turning points ---
    st.subheader("Top 3 turning points", icon=":material/bolt:")
    if len(turning_points) == 0:
        st.caption("No turning points — a goalless, card-free match.")
    else:
        tp_cols = st.columns(3)
        for col, (_, tp) in zip(tp_cols, turning_points.iterrows()):
            with col, st.container(border=True, height="stretch"):
                badge_label, badge_color = KIND_BADGE[tp.event_kind]
                st.markdown(f"**{tp.minute}'**")
                st.badge(badge_label, color=badge_color)
                st.caption(f"{tp.event_player} ({tp.event_team})")
                st.markdown(f"**{stat_label(tp.headline_outcome, match)}: "
                            f"{tp.headline_before:.0%} → {tp.headline_after:.0%}**")
                st.caption(turning_point_summary(tp, match))

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
            compatible_change = KIND_TO_CHANGE[selected_event.kind]

            with event_cols[1]:
                # Only ever shows the one change compatible with the selected event,
                # so there's nothing incompatible to pick and nothing to warn about.
                st.selectbox("Change", [compatible_change], format_func=lambda c: CHANGE_LABELS[c],
                             key=f"change_{event_idx}")

            run = st.button("Run counterfactual", type="primary", icon=":material/play_arrow:")

        if run:
            team_draws = load_team_draws()
            result = compute_counterfactual(match_id, int(selected_event.minute), compatible_change,
                                             selected_event.team, team_draws, curve)
            headline_outcome = counterfactual_headline_outcome(result)

            st.markdown(f"#### {result.description}")
            st.markdown(f"**{stat_label(headline_outcome, match)}: "
                        f"{result.fork_before[headline_outcome]:.0%} → {result.fork_after[headline_outcome]:.0%}**")
            st.caption(counterfactual_headline_summary(result, match, headline_outcome))

            cf_fig = build_counterfactual_figure(result, headline_outcome, match)
            st.plotly_chart(cf_fig, width="stretch",
                             alt=f"Real versus counterfactual {stat_label(headline_outcome, match)} "
                                 f"probability: {result.description}")

            outcome_cols = st.columns(2)
            with outcome_cols[0], st.container(border=True, height="stretch"):
                st.markdown("**Forecast just before the change**")
                st.metric(f"{match.home_team} win", f"{result.fork_before['home']:.0%}")
                st.metric("Draw", f"{result.fork_before['draw']:.0%}")
                st.metric(f"{match.away_team} win", f"{result.fork_before['away']:.0%}")
            with outcome_cols[1], st.container(border=True, height="stretch"):
                st.markdown("**Forecast just after the change**")
                st.metric(f"{match.home_team} win", f"{result.fork_after['home']:.0%}",
                         delta=f"{result.fork_after['home'] - result.fork_before['home']:+.0%}")
                st.metric("Draw", f"{result.fork_after['draw']:.0%}",
                         delta=f"{result.fork_after['draw'] - result.fork_before['draw']:+.0%}")
                st.metric(f"{match.away_team} win", f"{result.fork_after['away']:.0%}",
                         delta=f"{result.fork_after['away'] - result.fork_before['away']:+.0%}")

            with st.container(border=True):
                st.caption("What actually happened (not a forecast -- the real, known result)")
                outcome_word = {"home": f"{match.home_team} won", "draw": "It was a draw",
                                 "away": f"{match.away_team} won"}[result.real_outcome]
                st.markdown(f"**{match.home_team} {result.real_final_score[0]}-{result.real_final_score[1]} "
                            f"{match.away_team}** — {outcome_word}.")

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
