"""Re-simulates a match forward from any point, many times in parallel, to
estimate win/draw/loss probability. Vectorized with numpy: every replay's
state is a slot in an array, and one minute is advanced for all replays
at once, rather than looping replay-by-replay.

For speed, this skips individual shot/player detail (not needed for a
final-score outcome) and instead uses a single combined "goal this minute"
probability per team, built from the same shot-rate and shot-type mix as
the full match engine in match_engine.py.
"""

import numpy as np
import pandas as pd

from src.league import Club
from src.match_engine import (BIG_CHANCE_FRACTION, BIG_CHANCE_XG_RANGE, HOME_ADVANTAGE,
                               PENALTY_SHOT_FRACTION, PENALTY_XG_MEAN, RED_CARD_CHANCE_PER_MINUTE,
                               BASE_SHOT_RATE_PER_90, NORMAL_XG_RANGE)

BIG_CHANCE_XG_MEAN = sum(BIG_CHANCE_XG_RANGE) / 2
NORMAL_XG_MEAN = sum(NORMAL_XG_RANGE) / 2


def _shot_rate(team: Club, opponent: Club, own_score: np.ndarray, opp_score: np.ndarray,
               minute: int, is_home: bool, team_has_card: np.ndarray,
               opp_has_card: np.ndarray) -> np.ndarray:
    """Vectorized version of match_engine._shot_rate: same formula, but
    own_score/opp_score/*_has_card are arrays (one slot per replay)."""
    trailing = own_score < opp_score
    attack = np.where(trailing, team.attack * team.comeback_factor, team.attack)
    defense = np.where(trailing, opponent.defense * opponent.protect_lead_factor, opponent.defense)

    if minute > 75:
        defense = defense * opponent.fatigue_factor

    attack = np.where(team_has_card, attack * team.ten_men_resilience, attack)
    defense = np.where(opp_has_card, defense * opponent.ten_men_resilience, defense)

    rate = (BASE_SHOT_RATE_PER_90 / 90) * attack / defense
    if is_home:
        rate = rate * HOME_ADVANTAGE
    return rate


def _goal_given_shot(rng: np.random.Generator, n: int) -> np.ndarray:
    """Whether a shot (already known to happen) is a goal: draws a shot
    type with the same mix as the full engine, using each type's mean xG."""
    roll = rng.random(n)
    is_penalty = roll < PENALTY_SHOT_FRACTION
    is_big_chance = (~is_penalty) & (rng.random(n) < BIG_CHANCE_FRACTION)
    xg = np.where(is_penalty, PENALTY_XG_MEAN, np.where(is_big_chance, BIG_CHANCE_XG_MEAN, NORMAL_XG_MEAN))
    return rng.random(n) < xg


def simulate_forward(home: Club, away: Club, rng: np.random.Generator, start_minute: int,
                      home_goals: int, away_goals: int, home_has_card: bool, away_has_card: bool,
                      n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    """Replays the rest of the match (start_minute+1 .. 90) n_trials times
    in parallel. Returns (final_home_score, final_away_score) arrays."""
    home_score = np.full(n_trials, home_goals, dtype=np.int32)
    away_score = np.full(n_trials, away_goals, dtype=np.int32)
    home_card = np.full(n_trials, home_has_card)
    away_card = np.full(n_trials, away_has_card)

    for minute in range(start_minute + 1, 91):
        home_card |= rng.random(n_trials) < RED_CARD_CHANCE_PER_MINUTE
        away_card |= rng.random(n_trials) < RED_CARD_CHANCE_PER_MINUTE

        home_rate = _shot_rate(home, away, home_score, away_score, minute, True, home_card, away_card)
        away_rate = _shot_rate(away, home, away_score, home_score, minute, False, away_card, home_card)

        home_shot = rng.random(n_trials) < home_rate
        away_shot = rng.random(n_trials) < away_rate

        home_score += (home_shot & _goal_given_shot(rng, n_trials)).astype(np.int32)
        away_score += (away_shot & _goal_given_shot(rng, n_trials)).astype(np.int32)

    return home_score, away_score


def win_probabilities(final_home: np.ndarray, final_away: np.ndarray) -> dict:
    """Home/draw/away win shares, with a standard-error band on each,
    from a finite number of Monte Carlo replays."""
    n = len(final_home)
    p_home = (final_home > final_away).mean()
    p_draw = (final_home == final_away).mean()
    p_away = (final_home < final_away).mean()
    se = lambda p: np.sqrt(p * (1 - p) / n)
    return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "se_home": se(p_home), "se_draw": se(p_draw), "se_away": se(p_away)}


def win_probability_curve(home: Club, away: Club, real_states: pd.DataFrame, final_score: tuple[int, int],
                           rng: np.random.Generator, n_trials: int, checkpoints: list[int]) -> pd.DataFrame:
    """real_states: one row per minute 1..90 as produced by
    trait_estimation._match_minute_states (the actual game's state used for
    that minute's rate calc). final_score: the match's true final score
    (real_states only covers pre-minute-90 state, not the full-time result).
    checkpoints: minutes (0..90) at which to compute win probability, using
    the real state as of the END of that minute."""
    rows = []
    last_row = real_states.iloc[-1]  # minute-90 row: state entering minute 90

    for minute in checkpoints:
        if minute == 0:
            h_score, a_score, h_card, a_card = 0, 0, False, False
        elif minute == 90:
            h_score, a_score = final_score
            h_card, a_card = bool(last_row.home_card), bool(last_row.away_card)
        else:
            next_row = real_states[real_states.minute == minute + 1].iloc[0]
            h_score, a_score = int(next_row.home_score), int(next_row.away_score)
            h_card, a_card = bool(next_row.home_card), bool(next_row.away_card)

        final_home, final_away = simulate_forward(home, away, rng, minute, h_score, a_score,
                                                    h_card, a_card, n_trials)
        probs = win_probabilities(final_home, final_away)
        rows.append({"minute": minute, "home_score": h_score, "away_score": a_score, **probs})

    return pd.DataFrame(rows)
