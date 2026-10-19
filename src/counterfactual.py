"""Counterfactual engine: re-simulates a match from one real event changed.

Supported changes:
  - "shot_becomes_goal": a missed shot (including a missed penalty) scores instead
  - "goal_cancelled": a scored goal is wiped out, as if it had missed
  - "red_card_undone": a red card is removed; the team stays at 11 players

This is intentionally a thin, fully-documented, serializable interface: a
plain input dataclass and a plain output dataclass, no hidden state. It's
built to become a tool an AI agent calls directly in a later step.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.league import Club
from src.trait_estimation import _match_minute_states
from src.win_probability import (counterfactual_continuation_curve, real_state_at,
                                  win_probability_curve_with_uncertainty)

CHANGE_TYPES = ("shot_becomes_goal", "goal_cancelled", "red_card_undone")


@dataclass
class Counterfactual:
    """One hypothetical change to a real match, at a specific minute.

    match_id: which match (see data/matches.csv)
    minute: the minute of the real event being changed
    change: one of CHANGE_TYPES
    team: which team's event -- only needed if more than one qualifying
          event happened in that exact minute (rare); otherwise omit it
    """
    match_id: int
    minute: int
    change: str
    team: str | None = None


@dataclass
class CounterfactualResult:
    """actual_curve / counterfactual_curve: win probability per minute
    (0-90), directly comparable/plottable side by side -- identical up to
    `minute`, then diverging. actual_final / counterfactual_final: the
    outcome distribution at full time for each ({p_home, p_draw, p_away}).
    actual_final is the real, known result (so always exactly 0.0 or 1.0
    on one outcome); counterfactual_final is a genuine simulated
    distribution, since what happens after the change is no longer known.
    """
    description: str
    actual_curve: pd.DataFrame
    counterfactual_curve: pd.DataFrame
    actual_final: dict
    counterfactual_final: dict


def _find_event(cf: Counterfactual, shots: pd.DataFrame, red_cards: pd.DataFrame) -> pd.Series:
    match_shots = shots[shots.match_id == cf.match_id]
    match_cards = red_cards[red_cards.match_id == cf.match_id]

    if cf.change == "shot_becomes_goal":
        candidates = match_shots[(match_shots.minute == cf.minute) & (~match_shots.is_goal)]
    elif cf.change == "goal_cancelled":
        candidates = match_shots[(match_shots.minute == cf.minute) & match_shots.is_goal]
    elif cf.change == "red_card_undone":
        candidates = match_cards[match_cards.minute == cf.minute]
    else:
        raise ValueError(f"Unknown change type {cf.change!r}; must be one of {CHANGE_TYPES}")

    if cf.team is not None:
        candidates = candidates[candidates.team == cf.team]
    if len(candidates) == 0:
        raise ValueError(f"No matching '{cf.change}' event at minute {cf.minute} in match {cf.match_id}")
    if len(candidates) > 1:
        options = candidates[["team"]].drop_duplicates().team.tolist()
        raise ValueError(f"Multiple matching events at minute {cf.minute} -- pass `team` to disambiguate "
                          f"(options: {options})")
    return candidates.iloc[0]


def _describe(cf: Counterfactual, event: pd.Series) -> str:
    if cf.change == "shot_becomes_goal":
        kind = "penalty" if bool(event.get("is_penalty", False)) else "shot"
        return f"What if the {cf.minute}' {kind} by {event.player} ({event.team}) had scored?"
    if cf.change == "goal_cancelled":
        return f"What if the {cf.minute}' goal by {event.player} ({event.team}) hadn't happened?"
    return f"What if {event.player} ({event.team}) hadn't been sent off in the {cf.minute}'?"


def _forked_state(cf: Counterfactual, event: pd.Series, match: pd.Series,
                   real_states: pd.DataFrame, final_score: tuple[int, int]) -> tuple[int, int, bool, bool]:
    """The hypothetical (score, red-card) state at the fork minute, with
    exactly the one requested change applied on top of the real state."""
    if cf.change in ("shot_becomes_goal", "goal_cancelled"):
        # shots.csv records the score as of just BEFORE that shot -- the natural
        # base state for both "make it a goal" and "undo this goal" alike.
        h_score, a_score = int(event.home_score), int(event.away_score)
        if cf.change == "shot_becomes_goal":
            h_score += event.team == match.home_team
            a_score += event.team == match.away_team
        _, _, h_card, a_card = real_state_at(real_states, final_score, cf.minute)
        return h_score, a_score, h_card, a_card

    # red_card_undone: score is unaffected; only that team's card flag is cleared.
    h_score, a_score, h_card, a_card = real_state_at(real_states, final_score, cf.minute)
    if event.team == match.home_team:
        h_card = False
    else:
        a_card = False
    return h_score, a_score, h_card, a_card


def apply_counterfactual(cf: Counterfactual, matches: pd.DataFrame, shots: pd.DataFrame,
                          red_cards: pd.DataFrame, team_draws: list[dict[str, Club]],
                          rng: np.random.Generator, trials_per_draw: int = 150,
                          checkpoints: list[int] | None = None,
                          actual_curve: pd.DataFrame | None = None) -> CounterfactualResult:
    """Re-simulates `cf.match_id` from `cf.minute` with exactly one change
    applied, using the same estimated-trait uncertainty as the main win
    probability engine (team_draws -- see win_probability.sample_team_draws).

    actual_curve: optionally pass in an already-computed real win
    probability curve (from win_probability_curve_with_uncertainty) to
    avoid recomputing it when running several counterfactuals on the same
    match; computed fresh otherwise.
    """
    checkpoints = checkpoints if checkpoints is not None else list(range(0, 91))
    match = matches[matches.match_id == cf.match_id].iloc[0]
    event = _find_event(cf, shots, red_cards)

    match_shots = shots[shots.match_id == cf.match_id]
    match_cards = red_cards[red_cards.match_id == cf.match_id]
    real_states = _match_minute_states(match.home_team, match.away_team, match_shots, match_cards)
    final_score = (int(match.home_goals), int(match.away_goals))

    if actual_curve is None:
        actual_curve = win_probability_curve_with_uncertainty(
            team_draws, match.home_team, match.away_team, real_states, final_score,
            rng, trials_per_draw, checkpoints)

    h_score, a_score, h_card, a_card = _forked_state(cf, event, match, real_states, final_score)
    fork_checkpoints = [m for m in checkpoints if m >= cf.minute]
    continuation = counterfactual_continuation_curve(
        team_draws, match.home_team, match.away_team, rng, cf.minute,
        h_score, a_score, h_card, a_card, trials_per_draw, fork_checkpoints)
    continuation["home_score"], continuation["away_score"] = h_score, a_score

    before_fork = actual_curve[actual_curve.minute < cf.minute]
    counterfactual_curve = pd.concat([before_fork, continuation], ignore_index=True)

    outcome_cols = ["p_home", "p_draw", "p_away"]
    return CounterfactualResult(
        description=_describe(cf, event),
        actual_curve=actual_curve,
        counterfactual_curve=counterfactual_curve,
        actual_final=actual_curve.iloc[-1][outcome_cols].to_dict(),
        counterfactual_final=counterfactual_curve.iloc[-1][outcome_cols].to_dict(),
    )
