"""Validates the day-form Bayesian belief update (src/win_probability.py)
against two specific risks flagged in review:

1. Day form affects both the shot-rate and shot-quality channels in the
   generator, so its effect on total xG is COMPOUNDED (psi, not day_form
   itself). The update must track psi correctly -- checked here by
   injecting a KNOWN day_form into the generator and confirming the
   posterior mean matches what Bayesian theory predicts, with no
   ADDITIONAL bias beyond ordinary (and expected) posterior shrinkage.

   Note on shrinkage: a Gamma-Poisson posterior mean is NOT an unbiased
   estimate of the true per-match psi -- with a learned concentration
   k ~= 10.7 and only ~1-2 expected xG of evidence per match, a single
   match pulls the estimate noticeably toward the prior mean of 1.0 (e.g.
   true psi=0.5 lands around 0.92, not 0.5). That's correct, deliberate
   Bayesian behavior for sparse evidence, not a bug -- so the right check
   isn't "psi_hat == psi_true", it's "psi_hat matches the theoretical
   shrinkage formula (k + expected*psi_true) / (k + expected)" computed
   from each match's own actual expected-xG exposure.
2. With no observed evidence, the belief must not move at all.
"""

import numpy as np
import pandas as pd
import pytest

import src.match_engine as match_engine
from src.squad import Player
from src.trait_estimation import _match_minute_states, match_real_states
from src.win_probability import DAY_FORM_SHAPE, _expected_cumulative_xg, day_form_multiplier_curves, \
    win_probability_curve

EMPTY_RED_CARDS = pd.DataFrame(columns=["minute", "team", "player", "position"])


def _squad_from_players_csv(players: pd.DataFrame, team: str) -> list[Player]:
    rows = players[players.team == team]
    return [Player(name=r.player, position=r.position, shot_share=r.shot_share,
                   card_share=r.card_share, finishing_skill=r.finishing_skill)
            for _, r in rows.iterrows()]


@pytest.mark.parametrize("true_psi", [0.5, 0.75, 1.0, 1.3, 1.8])
def test_day_form_posterior_tracks_true_value_without_bias(true_psi, point_clubs, players, monkeypatch) -> None:
    """Injects a KNOWN day_form (psi) for the home side into the full
    generator across many independent matches, then checks the Bayesian
    posterior mean (squared back from the sqrt(psi) multiplier the engine
    actually uses -- see win_probability.py's module docstring on why psi,
    not day_form, is the quantity that must match) matches the theoretical
    Gamma-Poisson shrinkage prediction for each match's own actual
    expected-xG exposure -- not psi_true directly, since a correctly
    specified posterior mean is EXPECTED to shrink toward the prior with
    this little evidence per match (see module docstring). If the
    rate/quality compounding were wrong, this match against theory would
    fail even though the raw numbers still look "reasonable."

    Pressure spells are turned off for this test (PRESSURE_ENTRY_CHANCE_PER_MINUTE
    = 0) to isolate the day-form math itself from the separate, already-
    reported confound of pressure-spell noise (see
    scripts/estimate_traits.py's day-form spread report and the README)."""
    monkeypatch.setattr(match_engine, "PRESSURE_ENTRY_CHANCE_PER_MINUTE", 0.0)

    home_name, away_name = "Silverlake FC", "Northgate United"
    home_club, away_club = point_clubs[home_name], point_clubs[away_name]
    home_squad = _squad_from_players_csv(players, home_name)
    away_squad = _squad_from_players_csv(players, away_name)

    n_matches = 400
    rng = np.random.default_rng(13579 + int(true_psi * 1000))
    psi_hat_estimates, theoretical_predictions = [], []
    for _ in range(n_matches):
        result = match_engine.simulate_match(home_club, away_club, home_squad, away_squad, rng,
                                              home_day_form=true_psi, away_day_form=1.0)
        shots = pd.DataFrame(result["shots"])
        red_cards = pd.DataFrame(result["red_card_events"]) if result["red_card_events"] else EMPTY_RED_CARDS
        states = _match_minute_states(home_name, away_name, shots, red_cards)

        home_mult_curve, _ = day_form_multiplier_curves(home_club, away_club, states)
        psi_hat_estimates.append(home_mult_curve[90] ** 2)

        expected = _expected_cumulative_xg(home_club, away_club, states, True)[90]
        theoretical_predictions.append((DAY_FORM_SHAPE + expected * true_psi) / (DAY_FORM_SHAPE + expected))

    mean_estimate = float(np.mean(psi_hat_estimates))
    mean_theoretical = float(np.mean(theoretical_predictions))
    se = float(np.std(psi_hat_estimates, ddof=1) / np.sqrt(n_matches))

    assert abs(mean_estimate - mean_theoretical) < 6 * se, (
        f"true psi={true_psi}: mean simulated posterior = {mean_estimate:.4f} +/- {se:.4f} (SE) vs. "
        f"mean theoretical Gamma-Poisson shrinkage prediction = {mean_theoretical:.4f} -- "
        f"more than 6 SE apart, suggesting the rate/quality compounding doesn't match the generator")


def test_day_form_multiplier_is_neutral_with_no_evidence(point_clubs, matches, shots, red_cards) -> None:
    """At minute 0, before anything has been observed, the belief must
    equal the neutral prior exactly (multiplier == 1.0 for both sides) --
    checked across several real matches, not just a synthetic case."""
    sample = matches.iloc[::200]
    for _, match in sample.iterrows():
        states = match_real_states(match, shots, red_cards)
        home_club, away_club = point_clubs[match.home_team], point_clubs[match.away_team]
        home_curve, away_curve = day_form_multiplier_curves(home_club, away_club, states)
        assert home_curve[0] == 1.0, f"match {match.match_id}: home day-form multiplier at minute 0 is not neutral"
        assert away_curve[0] == 1.0, f"match {match.match_id}: away day-form multiplier at minute 0 is not neutral"


def test_day_form_never_moves_probabilities_with_no_observed_event(point_clubs, matches, shots, red_cards) -> None:
    """With the update on vs. off, the win-probability curve must be
    IDENTICAL at minute 0 -- the one checkpoint with literally zero
    observed evidence (zero elapsed minutes, zero shots). From minute 1
    onward even a scoreless stretch is itself evidence (expected xG has
    accumulated while observed hasn't), which the belief update is
    supposed to react to -- that's the point of the feature, not a bug."""
    sample = matches.iloc[::200]
    for _, match in sample.iterrows():
        states = match_real_states(match, shots, red_cards)
        home_club, away_club = point_clubs[match.home_team], point_clubs[match.away_team]
        final_score = (int(match.home_goals), int(match.away_goals))

        with_update = win_probability_curve(home_club, away_club, states, final_score,
                                             seed=42, n_trials=500, checkpoints=[0], use_day_form=True)
        without_update = win_probability_curve(home_club, away_club, states, final_score,
                                                seed=42, n_trials=500, checkpoints=[0], use_day_form=False)

        pd.testing.assert_frame_equal(with_update, without_update)
