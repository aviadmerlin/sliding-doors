"""Category 4: home advantage and the data-leakage fix that followed a code
review. src/win_probability.py now estimates every shot-rate/composition
constant from the generated data (see trait_estimation.estimate_league_constants
/ estimate_finishing_factor) instead of importing src/match_engine.py's true
generator constants. These tests enforce that boundary stays intact, and
report how close each estimate lands to the true value it's standing in for
-- tests are allowed to peek at the true generator for grading purposes;
product code (src/win_probability.py) is not."""

import ast
from pathlib import Path

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

WIN_PROBABILITY_PATH = Path(__file__).resolve().parent.parent / "src" / "win_probability.py"


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


def test_win_probability_never_imports_from_match_engine() -> None:
    """src/win_probability.py (the fast engine, used for every win-probability
    and counterfactual number in the product) must get every shot-rate and
    shot-composition constant from DATA -- never read it off the true
    generator. Parses the file's own AST rather than grepping, so a
    `from src import match_engine` or `import src.match_engine as x` would
    be caught too, not just the literal `from src.match_engine import ...`
    spelling this test previously guarded against."""
    tree = ast.parse(WIN_PROBABILITY_PATH.read_text())
    forbidden_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "match_engine" in node.module:
            forbidden_imports.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "match_engine" in alias.name:
                    forbidden_imports.append(f"import {alias.name}")

    assert not forbidden_imports, (
        f"src/win_probability.py imports from src/match_engine.py (the true generator), "
        f"which would leak ground truth into the fast engine: {forbidden_imports}")


def test_report_home_advantage_estimate_vs_true_value(data_dir) -> None:
    """Home advantage is estimated from the Poisson regression's "home"
    coefficient (trait_estimation._traits_from_params), applied per club
    via Club.home_advantage, and varies per sampled trait draw (same
    mechanism as the other club traits) -- not imported from the true
    generator. Reports how close the point estimate lands to the true
    value; a few % error is expected and healthy (perfect agreement would
    suggest leakage, not a good estimator)."""
    estimated_traits = pd.read_csv(data_dir / "estimated_traits.csv")
    estimated_home_advantage = float(estimated_traits.home_advantage.iloc[0])
    assert estimated_traits.home_advantage.nunique() == 1, \
        "home_advantage should be the same league-wide value for every club"

    true_home_advantage = HOME_ADVANTAGE
    relative_error = abs(estimated_home_advantage - true_home_advantage) / true_home_advantage
    print(f"\nHOME_ADVANTAGE: true={true_home_advantage}, estimated={estimated_home_advantage:.5f}, "
          f"relative error={relative_error:.4%}")

    assert relative_error < 0.10, (
        f"estimated home advantage ({estimated_home_advantage:.5f}) is more than 10% off the true "
        f"value ({true_home_advantage}) -- investigate the 'home' coefficient extraction")


def test_report_league_constants_estimate_vs_true_value(data_dir) -> None:
    """The remaining league-wide constants (base shot rate, shot
    composition, xG means per type, red card rate) are plain empirical
    estimates from shots.csv/red_cards.csv (base_shot_rate_per_90 also uses
    the regression's intercept) -- see
    trait_estimation.estimate_league_constants. Reports how close each
    estimate lands to the true value; a tolerance per quantity, since some
    are estimated far more precisely than others (e.g. xG means average
    over ~9,600 shots; red card rate over only ~230 events)."""
    constants = pd.read_csv(data_dir / "model_league_constants.csv", index_col=0)["value"]

    true_big_chance_xg_mean = sum(BIG_CHANCE_XG_RANGE) / 2
    true_normal_xg_mean = sum(NORMAL_XG_RANGE) / 2
    truth = {
        "base_shot_rate_per_90": (BASE_SHOT_RATE_PER_90, 0.05),
        "penalty_shot_fraction": (PENALTY_SHOT_FRACTION, 0.10),
        "big_chance_fraction": (BIG_CHANCE_FRACTION, 0.05),
        "penalty_xg_mean": (PENALTY_XG_MEAN, 0.05),
        "big_chance_xg_mean": (true_big_chance_xg_mean, 0.05),
        "normal_xg_mean": (true_normal_xg_mean, 0.05),
        "red_card_chance_per_minute": (RED_CARD_CHANCE_PER_MINUTE, 0.15),
    }

    print(f"\n{'name':28} {'estimated':>12} {'true':>12} {'rel. error':>12} {'tolerance':>10}")
    violations = []
    for name, (true_value, tolerance) in truth.items():
        estimated = float(constants[name])
        relative_error = abs(estimated - true_value) / true_value
        print(f"{name:28} {estimated:12.6f} {true_value:12.6f} {relative_error:12.2%} {tolerance:10.0%}")
        if relative_error > tolerance:
            violations.append({"name": name, "estimated": estimated, "true": true_value,
                               "relative_error": relative_error, "tolerance": tolerance})

    assert not violations, f"estimate(s) further from the true value than their tolerance allows: {violations}"
