"""Category 2: model logic, checked exhaustively across season 5 (not a
sample) wherever the user asked for "all events".

Performance note: the exhaustive per-event checks (goals, red cards,
notable misses) use the POINT-ESTIMATE model (src.win_probability.simulate_forward
+ win_probabilities directly) rather than the full parameter-uncertainty
production pipeline (18 trait draws x 200 trials per checkpoint) -- that
pipeline is unnecessary here (we're checking the deterministic monotonicity
of the underlying rate model, not re-deriving production-grade probability
estimates) and would make ~2,000 exhaustive comparisons impractically slow.
Each before/after pair reuses the same RNG seed (same trick as the app's
common-random-numbers smoothing) so the comparison is far less noisy than
two independent runs would be, and a generous, explicitly-justified noise
tolerance is used before flagging anything as a real violation.
"""

import numpy as np
import pandas as pd
import pytest

from src.trait_estimation import match_real_states
from src.win_probability import (
    counterfactual_fork_probabilities,
    real_state_at,
    simulate_forward,
    win_probabilities,
    win_probability_curve_with_uncertainty,
)

N_TRIALS = 1500
SEED = 20231
# Expected SE of a single proportion at N_TRIALS, worst case p=0.5: sqrt(0.25/1500) =~ 0.0129.
# Same-seed (correlated) before/after draws make the actual difference's noise much smaller
# than two independent draws would, but we use the conservative independent-draws bound and
# a ~4-sigma multiplier (chosen for ~2,000 comparisons total, i.e. roughly Bonferroni-safe)
# as the threshold for flagging a real violation rather than simulation noise.
NOISE_THRESHOLD = 0.05


def _probs_at(home, away, minute, h_score, a_score, h_card, a_card, seed=SEED, n_trials=N_TRIALS) -> dict:
    rng = np.random.default_rng(seed)
    final_home, final_away = simulate_forward(home, away, rng, minute, h_score, a_score, h_card, a_card, n_trials)
    return win_probabilities(final_home, final_away)


def _match_card_state(states: pd.DataFrame, minute: int) -> tuple[bool, bool]:
    row = states[states.minute == minute].iloc[0]
    return bool(row.home_card), bool(row.away_card)


@pytest.fixture(scope="module")
def season5_states(season5_matches: pd.DataFrame, season5_shots: pd.DataFrame,
                    season5_red_cards: pd.DataFrame) -> dict:
    """Precomputed match_real_states per match_id, reused across this
    module's tests to avoid recomputing it 3x."""
    states = {}
    for _, m in season5_matches.iterrows():
        match_shots = season5_shots[season5_shots.match_id == m.match_id]
        match_cards = season5_red_cards[season5_red_cards.match_id == m.match_id]
        states[m.match_id] = match_real_states(m, match_shots, match_cards)
    return states


def test_goal_never_lowers_scoring_teams_win_probability(
        season5_matches, season5_shots, season5_states, point_clubs) -> None:
    violations, checked = [], 0
    for _, m in season5_matches.iterrows():
        match_shots = season5_shots[season5_shots.match_id == m.match_id]
        states = season5_states[m.match_id]
        home, away = point_clubs[m.home_team], point_clubs[m.away_team]

        for _, goal in match_shots[match_shots.is_goal].iterrows():
            minute = int(goal.minute)
            h_card, a_card = _match_card_state(states, minute)
            h0, a0 = int(goal.home_score), int(goal.away_score)
            scoring_home = goal.team == m.home_team
            h1 = h0 + (1 if scoring_home else 0)
            a1 = a0 + (0 if scoring_home else 1)

            p_before = _probs_at(home, away, minute, h0, a0, h_card, a_card)
            p_after = _probs_at(home, away, minute, h1, a1, h_card, a_card)
            outcome = "home" if scoring_home else "away"
            checked += 1
            delta = p_after[f"p_{outcome}"] - p_before[f"p_{outcome}"]
            if delta < -NOISE_THRESHOLD:
                violations.append({"match_id": int(m.match_id), "minute": minute, "team": goal.team,
                                   "scoring_side": outcome, "p_before": round(p_before[f"p_{outcome}"], 3),
                                   "p_after": round(p_after[f"p_{outcome}"], 3), "delta": round(delta, 3)})

    assert not violations, (
        f"{len(violations)} of {checked} season-5 goals LOWERED the scoring team's win probability "
        f"by more than the {NOISE_THRESHOLD:.0%} noise threshold: {violations[:10]}")


def test_red_card_never_raises_punished_teams_win_probability(
        season5_matches, season5_red_cards, season5_states, point_clubs) -> None:
    violations, checked = [], 0
    for _, m in season5_matches.iterrows():
        match_cards = season5_red_cards[season5_red_cards.match_id == m.match_id]
        if len(match_cards) == 0:
            continue
        states = season5_states[m.match_id]
        home, away = point_clubs[m.home_team], point_clubs[m.away_team]

        for _, card in match_cards.iterrows():
            minute = int(card.minute)
            punished_home = card.team == m.home_team
            # Score is unaffected by a card; only the card state changes, from the
            # minute just before this card takes effect to the minute it does.
            h0, a0 = int(states[states.minute == minute].iloc[0].home_score), \
                int(states[states.minute == minute].iloc[0].away_score)
            h_card_before, a_card_before = _match_card_state(states, minute - 1) if minute > 1 else (False, False)
            h_card_after, a_card_after = _match_card_state(states, minute)

            p_before = _probs_at(home, away, minute, h0, a0, h_card_before, a_card_before)
            p_after = _probs_at(home, away, minute, h0, a0, h_card_after, a_card_after)
            outcome = "home" if punished_home else "away"
            checked += 1
            delta = p_after[f"p_{outcome}"] - p_before[f"p_{outcome}"]
            if delta > NOISE_THRESHOLD:
                violations.append({"match_id": int(m.match_id), "minute": minute, "team": card.team,
                                   "punished_side": outcome, "p_before": round(p_before[f"p_{outcome}"], 3),
                                   "p_after": round(p_after[f"p_{outcome}"], 3), "delta": round(delta, 3)})

    assert not violations, (
        f"{len(violations)} of {checked} season-5 red cards RAISED the punished team's win probability "
        f"by more than the {NOISE_THRESHOLD:.0%} noise threshold: {violations[:10]}")


def test_notable_miss_would_raise_shooting_teams_win_probability_if_scored(
        season5_matches, season5_shots, season5_states, point_clubs) -> None:
    """Hypothetical version of the goal-monotonicity check: for every
    missed penalty or missed big chance (the notable misses the app's own
    What-if picker offers), turning it into a goal should weakly raise the
    shooting team's win probability."""
    violations, checked = [], 0
    for _, m in season5_matches.iterrows():
        match_shots = season5_shots[season5_shots.match_id == m.match_id]
        states = season5_states[m.match_id]
        home, away = point_clubs[m.home_team], point_clubs[m.away_team]

        notable_misses = match_shots[(~match_shots.is_goal) & (match_shots.is_penalty | match_shots.is_big_chance)]
        for _, miss in notable_misses.iterrows():
            minute = int(miss.minute)
            h_card, a_card = _match_card_state(states, minute)
            h0, a0 = int(miss.home_score), int(miss.away_score)
            shooting_home = miss.team == m.home_team
            h1 = h0 + (1 if shooting_home else 0)
            a1 = a0 + (0 if shooting_home else 1)

            p_before = _probs_at(home, away, minute, h0, a0, h_card, a_card)
            p_after = _probs_at(home, away, minute, h1, a1, h_card, a_card)
            outcome = "home" if shooting_home else "away"
            checked += 1
            delta = p_after[f"p_{outcome}"] - p_before[f"p_{outcome}"]
            if delta < -NOISE_THRESHOLD:
                violations.append({"match_id": int(m.match_id), "minute": minute, "team": miss.team,
                                   "shooting_side": outcome, "p_before": round(p_before[f"p_{outcome}"], 3),
                                   "p_after": round(p_after[f"p_{outcome}"], 3), "delta": round(delta, 3)})

    assert not violations, (
        f"{len(violations)} of {checked} season-5 notable misses, if scored, would have LOWERED the "
        f"shooting team's win probability by more than the {NOISE_THRESHOLD:.0%} noise threshold: "
        f"{violations[:10]}")


def test_probabilities_sum_to_one(featured_curves: pd.DataFrame, season5_matches, season5_shots,
                                   season5_red_cards, point_clubs) -> None:
    totals = featured_curves.p_home + featured_curves.p_draw + featured_curves.p_away
    bad = featured_curves[(totals - 1.0).abs() > 1e-9]
    assert bad.empty, f"{len(bad)} precomputed rows where p_home+p_draw+p_away != 1: {bad.head(5)}"

    # Also check freshly computed (point-estimate) probabilities for a spread of season-5 matches/minutes.
    rng_mismatches = []
    sample = season5_matches.iloc[::40]  # every 40th match, ~10 matches
    for _, m in sample.iterrows():
        match_shots = season5_shots[season5_shots.match_id == m.match_id]
        match_cards = season5_red_cards[season5_red_cards.match_id == m.match_id]
        states = match_real_states(m, match_shots, match_cards)
        home, away = point_clubs[m.home_team], point_clubs[m.away_team]
        for minute in (0, 30, 60, 89):
            h, a, hc, ac = real_state_at(states, (int(m.home_goals), int(m.away_goals)), minute)
            probs = _probs_at(home, away, minute, h, a, hc, ac, n_trials=500)
            total = probs["p_home"] + probs["p_draw"] + probs["p_away"]
            if abs(total - 1.0) > 1e-9:
                rng_mismatches.append((int(m.match_id), minute, total))
    assert not rng_mismatches, f"freshly computed probabilities not summing to 1: {rng_mismatches[:10]}"


def test_minute_90_matches_real_final_result(season5_matches, season5_shots, season5_red_cards,
                                              point_clubs) -> None:
    mismatches = []
    sample = season5_matches.iloc[::20]  # every 20th match, ~19 matches
    for _, m in sample.iterrows():
        match_shots = season5_shots[season5_shots.match_id == m.match_id]
        match_cards = season5_red_cards[season5_red_cards.match_id == m.match_id]
        states = match_real_states(m, match_shots, match_cards)
        home, away = point_clubs[m.home_team], point_clubs[m.away_team]
        h, a, hc, ac = real_state_at(states, (int(m.home_goals), int(m.away_goals)), 90)
        probs = _probs_at(home, away, 90, h, a, hc, ac, n_trials=100)

        actual_outcome = "home" if m.home_goals > m.away_goals else ("away" if m.home_goals < m.away_goals else "draw")
        expected = {"home": 0.0, "draw": 0.0, "away": 0.0}
        expected[actual_outcome] = 1.0
        for outcome in ("home", "draw", "away"):
            if probs[f"p_{outcome}"] != expected[outcome]:
                mismatches.append((int(m.match_id), outcome, probs[f"p_{outcome}"], expected[outcome]))

    assert not mismatches, f"minute-90 probabilities don't match the deterministic real result: {mismatches[:10]}"


def test_counterfactual_with_no_change_reproduces_actual_probabilities(
        season5_matches, season5_shots, season5_red_cards, team_draws) -> None:
    """Forking "from the real state" (no hypothetical change applied)
    should reproduce the same win probability as the main engine's own
    curve at that minute -- both go through the identical
    simulate_forward/win_probabilities machinery, just via two different
    public entry points (counterfactual_fork_probabilities vs
    win_probability_curve_with_uncertainty)."""
    rng_seed = 31415
    trials_per_batch = 150
    mismatches = []

    sample = season5_matches.iloc[::60]  # a handful of matches, ~7
    for _, m in sample.iterrows():
        match_shots = season5_shots[season5_shots.match_id == m.match_id]
        match_cards = season5_red_cards[season5_red_cards.match_id == m.match_id]
        states = match_real_states(m, match_shots, match_cards)
        final_score = (int(m.home_goals), int(m.away_goals))
        minute = 45

        curve = win_probability_curve_with_uncertainty(
            team_draws, m.home_team, m.away_team, states, final_score,
            rng_seed, trials_per_batch, [minute])
        actual_at_45 = curve.iloc[0]

        h, a, hc, ac = real_state_at(states, final_score, minute)
        fork_probs = counterfactual_fork_probabilities(
            team_draws, m.home_team, m.away_team, rng_seed, minute, h, a, hc, ac, trials_per_batch)

        for outcome in ("home", "draw", "away"):
            diff = abs(actual_at_45[f"p_{outcome}"] - fork_probs[f"p_{outcome}"])
            if diff > 1e-9:
                mismatches.append((int(m.match_id), outcome, actual_at_45[f"p_{outcome}"], fork_probs[f"p_{outcome}"]))

    assert not mismatches, (
        f"no-op counterfactual (forking from the real, unmodified state) didn't exactly reproduce "
        f"the actual curve's probability at the same minute: {mismatches[:10]}")
