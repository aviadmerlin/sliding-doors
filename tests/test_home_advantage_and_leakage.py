"""Category 4: home advantage and a data-leakage review flagged in code
review. Per instructions, this category only reports/quantifies -- it does
not fix anything (see test docstrings for exactly what each check means)."""

import numpy as np
import pandas as pd
import pytest

from src.match_engine import (
    BASE_SHOT_RATE_PER_90,
    BIG_CHANCE_FRACTION,
    BIG_CHANCE_XG_RANGE,
    HOME_ADVANTAGE,
    NORMAL_XG_RANGE,
    PENALTY_SHOT_FRACTION,
    PENALTY_XG_MEAN,
    RED_CARD_CHANCE_PER_MINUTE,
)
from src.win_probability import simulate_forward, win_probabilities


@pytest.mark.parametrize("team_a,team_b", [
    ("Silverlake FC", "Northgate United"),
    ("Thornfield United", "Oakhaven City"),
    ("Blackthorn Rovers", "Dunmore Town"),
    ("Castlebay FC", "Port Albion"),
])
@pytest.mark.parametrize("minute,h0,a0", [(0, 0, 0), (45, 1, 1)])
def test_home_side_has_higher_win_probability_than_if_it_were_away(
        team_a, team_b, minute, h0, a0, point_clubs) -> None:
    """Same two clubs, same game state, only the home/away assignment
    swapped: whichever club is designated home should have a higher win
    probability than it would if it were the away side instead (holding
    its own traits, the opponent, and the state fixed)."""
    club_a, club_b = point_clubs[team_a], point_clubs[team_b]
    n_trials = 4000
    seed = 909090

    rng = np.random.default_rng(seed)
    fh, fa = simulate_forward(club_a, club_b, rng, minute, h0, a0, False, False, n_trials)
    a_as_home = win_probabilities(fh, fa)["p_home"]

    rng = np.random.default_rng(seed)
    fh, fa = simulate_forward(club_b, club_a, rng, minute, a0, h0, False, False, n_trials)
    a_as_away = win_probabilities(fh, fa)["p_away"]

    assert a_as_home > a_as_away, (
        f"{team_a} vs {team_b} at minute {minute} ({h0}-{a0}): {team_a}'s win probability as "
        f"home ({a_as_home:.3f}) should be higher than as away ({a_as_away:.3f})")


def test_report_home_advantage_estimate_vs_true_value(data_dir) -> None:
    """The Poisson regression (src/trait_estimation.py) fits a "home"
    coefficient from real shot data -- a genuine, independent estimate of
    home advantage. But nothing in the product reads it:
    _traits_from_params never extracts a "home" trait, and
    src/win_probability.py instead imports the true generator constant
    HOME_ADVANTAGE directly from src/match_engine.py. This test reports how
    close the (unused) estimate is to the true value -- not a pass/fail
    correctness check, since there's no real bug in the *estimate* itself,
    just in the fact that the pipeline ignores it in favor of ground truth
    it shouldn't have access to in a genuine estimation setting."""
    params = pd.read_csv(data_dir / "model_params.csv", index_col=0)["value"]
    estimated_home_advantage = float(np.exp(params["home"]))
    true_home_advantage = HOME_ADVANTAGE
    relative_error = abs(estimated_home_advantage - true_home_advantage) / true_home_advantage

    print(f"\nHOME_ADVANTAGE: true={true_home_advantage}, estimated (unused)="
          f"{estimated_home_advantage:.5f}, relative error={relative_error:.4%}")

    # Purely informational -- always true, just gives pytest -s a place to show the numbers above.
    assert relative_error < 1.0


def test_report_constants_not_estimated_at_all_but_hardcoded_from_the_true_generator() -> None:
    """Unlike "home", these five quantities are never estimated by the
    regression at all -- src/win_probability.py imports them directly from
    src/match_engine.py (the TRUE generator) and uses them as-is for every
    simulation. There is no independent estimate to compare against the
    true value for these; the "estimate" IS the true value by construction,
    which is itself the leak -- a real system would have to infer these
    from data, and never get them exactly right. This test doesn't check
    correctness (there's nothing to get wrong when you're handed the
    answer); it pins down that the leak exists, so it starts failing (as a
    reminder to update/remove it) if the leakage is ever fixed."""
    from src.win_probability import BASE_SHOT_RATE_PER_90 as used_base_rate
    from src.win_probability import BIG_CHANCE_FRACTION as used_big_chance_fraction
    from src.win_probability import BIG_CHANCE_XG_MEAN as used_big_chance_xg_mean
    from src.win_probability import HOME_ADVANTAGE as used_home_advantage
    from src.win_probability import NORMAL_XG_MEAN as used_normal_xg_mean
    from src.win_probability import PENALTY_SHOT_FRACTION as used_penalty_fraction
    from src.win_probability import PENALTY_XG_MEAN as used_penalty_xg_mean
    from src.win_probability import RED_CARD_CHANCE_PER_MINUTE as used_red_card_chance

    true_big_chance_xg_mean = sum(BIG_CHANCE_XG_RANGE) / 2
    true_normal_xg_mean = sum(NORMAL_XG_RANGE) / 2

    leaked = {
        "BASE_SHOT_RATE_PER_90": (used_base_rate, BASE_SHOT_RATE_PER_90),
        "PENALTY_SHOT_FRACTION": (used_penalty_fraction, PENALTY_SHOT_FRACTION),
        "BIG_CHANCE_FRACTION": (used_big_chance_fraction, BIG_CHANCE_FRACTION),
        "PENALTY_XG_MEAN": (used_penalty_xg_mean, PENALTY_XG_MEAN),
        "BIG_CHANCE_XG_MEAN": (used_big_chance_xg_mean, true_big_chance_xg_mean),
        "NORMAL_XG_MEAN": (used_normal_xg_mean, true_normal_xg_mean),
        "RED_CARD_CHANCE_PER_MINUTE": (used_red_card_chance, RED_CARD_CHANCE_PER_MINUTE),
        "HOME_ADVANTAGE (also leaked, on top of home being separately estimated-but-unused)":
            (used_home_advantage, HOME_ADVANTAGE),
    }
    print("\nConstants the fast engine uses as hardcoded ground truth (no independent "
          "estimation happens for any of these):")
    for name, (used, true) in leaked.items():
        print(f"  {name}: used={used}, true generator value={true}")

    mismatches = {name: (used, true) for name, (used, true) in leaked.items() if used != true}
    assert not mismatches, f"expected all of these to be identical by construction (confirming " \
                           f"the leak is real and total): {mismatches}"
