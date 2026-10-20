"""Category 3: consistency between different parts of the pipeline --
precomputed vs. freshly computed, the fast vectorized engine vs. the full
shot-by-shot engine, the UI vs. its underlying data, and basic
reproducibility (same seed -> same result)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.counterfactual import Counterfactual, apply_counterfactual
from src.match_engine import simulate_match
from src.squad import Player
from src.trait_estimation import match_real_states
from src.win_probability import (
    rank_turning_points,
    simulate_forward,
    win_probabilities,
    win_probability_curve_with_uncertainty,
)

TRIALS_PER_DRAW = 200  # must match scripts/precompute_featured_matches.py and app.py
CHECKPOINTS = list(range(0, 91))
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_precomputed_curves_match_freshly_computed(featured_meta, featured_curves, matches, shots,
                                                     red_cards, team_draws) -> None:
    """Every featured match's saved curve should be bit-for-bit reproducible
    by rerunning exactly what scripts/precompute_featured_matches.py (and
    app.py's simulate_curve) does -- same seed convention (1000 + match_id),
    same trial count, same model loaded from disk."""
    # A tiny (1e-9) numerical tolerance, not bit-for-bit equality: with identical
    # seeds the two runs make identical random *draws*, but floating-point
    # summation order can still differ in the last bit or two between separate
    # pandas/numpy aggregation calls -- that's machine-precision noise (~1e-16
    # observed), not a reproducibility bug.
    mismatches = []
    for match_id in featured_meta["featured_match_ids"]:
        match = matches[matches.match_id == match_id].iloc[0]
        states = match_real_states(match, shots, red_cards)
        fresh = win_probability_curve_with_uncertainty(
            team_draws, match.home_team, match.away_team, states,
            (int(match.home_goals), int(match.away_goals)), 1000 + match_id, TRIALS_PER_DRAW, CHECKPOINTS
        ).reset_index(drop=True)
        precomputed = (featured_curves[featured_curves.match_id == match_id]
                       .drop(columns="match_id").reset_index(drop=True))

        max_diff = (fresh - precomputed).abs().max().max()
        if max_diff > 1e-9:
            mismatches.append({"match_id": int(match_id), "max_abs_diff": float(max_diff)})

    assert not mismatches, f"{len(mismatches)} featured matches where the precomputed curve doesn't " \
                           f"match a fresh recompute with the same seed beyond floating-point noise: {mismatches}"


def _squad_from_players_csv(players: pd.DataFrame, team: str) -> list[Player]:
    rows = players[players.team == team]
    return [Player(name=r.player, position=r.position, shot_share=r.shot_share,
                   card_share=r.card_share, finishing_skill=r.finishing_skill)
            for _, r in rows.iterrows()]


@pytest.mark.parametrize("home_team,away_team", [
    ("Silverlake FC", "Northgate United"),
    ("Thornfield United", "Oakhaven City"),
    ("Blackthorn Rovers", "Dunmore Town"),
])
def test_fast_engine_agrees_with_full_shot_by_shot_engine(
        home_team, away_team, point_clubs, players) -> None:
    """The fast vectorized engine (src.win_probability, used for every
    win-probability/counterfactual number in the product) uses a single
    combined "goal this minute" probability and skips individual
    shot/player detail. The full engine (src.match_engine, used only to
    generate the season data) draws an xG per shot and applies the
    shooting player's hidden finishing skill. These are intentionally
    different levels of detail -- this checks they still agree in
    aggregate (same final-outcome distribution from kickoff), which is
    the fast engine's actual job. A generous tolerance accounts for two
    expected approximation sources: (1) ordinary Monte Carlo noise at a
    finite trial count, and (2) the full engine's squad-average finishing
    skill being close to but not exactly 1.0 (it's random per squad), a
    source the fast engine doesn't model at all."""
    n_trials = 3000
    home, away = point_clubs[home_team], point_clubs[away_team]

    rng = np.random.default_rng(42)
    final_home, final_away = simulate_forward(home, away, rng, 0, 0, 0, False, False, n_trials)
    fast_probs = win_probabilities(final_home, final_away)

    home_squad = _squad_from_players_csv(players, home_team)
    away_squad = _squad_from_players_csv(players, away_team)
    full_rng = np.random.default_rng(42)
    full_results = [simulate_match(home, away, home_squad, away_squad, full_rng) for _ in range(n_trials)]
    full_home = np.array([r["home_goals"] for r in full_results])
    full_away = np.array([r["away_goals"] for r in full_results])
    full_probs = win_probabilities(full_home, full_away)

    # Two independent Monte Carlo estimates at n_trials each: worst-case combined SE (p=0.5 both sides).
    se_combined = np.sqrt(2 * 0.25 / n_trials)
    tolerance = 5 * se_combined  # ~5-sigma, generous given the extra (expected) finishing-skill gap

    diffs = {outcome: abs(fast_probs[f"p_{outcome}"] - full_probs[f"p_{outcome}"])
             for outcome in ("home", "draw", "away")}
    bad = {o: d for o, d in diffs.items() if d > tolerance}

    def weighted_finishing_skill(team: str) -> float:
        rows = players[players.team == team]
        return float((rows.shot_share * rows.finishing_skill).sum() / rows.shot_share.sum())

    assert not bad, (
        f"{home_team} (squad finishing skill {weighted_finishing_skill(home_team):.3f}) vs "
        f"{away_team} (squad finishing skill {weighted_finishing_skill(away_team):.3f}): "
        f"fast engine {fast_probs} vs full engine {full_probs} -- differences {diffs} exceed "
        f"tolerance {tolerance:.3f}: {bad}. The fast engine doesn't model player-level finishing "
        f"skill at all, so it's expected to drift from the full engine when a squad's shot-weighted "
        f"average skill is notably off 1.0 -- the gap's size should track how far from 1.0 it is.")


def test_ui_turning_point_cards_match_underlying_data(featured_curves, featured_turning_points,
                                                        matches, shots, red_cards, featured_meta) -> None:
    """Runs the app headlessly (Streamlit's AppTest) and checks the exact
    numbers rendered in the turning-point cards equal the numbers in the
    underlying rank_turning_points() output for the same match -- catching
    any transcription/formatting bug in app.py's rendering code."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception, f"app raised on load: {at.exception}"

    match_id = featured_meta["default_match_id"]  # whichever match the app loads by default
    match = matches[matches.match_id == match_id].iloc[0]
    curve = featured_curves[featured_curves.match_id == match_id].drop(columns="match_id").reset_index(drop=True)
    expected_tp = rank_turning_points(curve, match, shots, red_cards, top_n=3)

    rendered_stats = [m.value for m in at.markdown if "→" in m.value and ":" in m.value]
    assert len(rendered_stats) >= len(expected_tp), (
        f"expected at least {len(expected_tp)} before->after stat lines rendered, got {rendered_stats}")

    for _, tp in expected_tp.iterrows():
        expected_line = f"{tp.headline_before:.0%} → {tp.headline_after:.0%}"
        assert any(expected_line in line for line in rendered_stats), (
            f"turning point at {tp.minute}' (expected '...{expected_line}') not found verbatim "
            f"among rendered stat lines: {rendered_stats}")


def test_ui_counterfactual_outcome_panel_matches_underlying_data(matches, shots, red_cards,
                                                                   team_draws, featured_curves,
                                                                   featured_meta) -> None:
    """Runs a real counterfactual through the app UI and checks the
    rendered 'forecast before/after' metrics match apply_counterfactual's
    own fork_before/fork_after dict, called directly. The default match is
    featured, so (like the app itself) this reuses the precomputed curve
    rather than resimulating it."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    btn = [b for b in at.button if "Run counterfactual" in b.label][0]
    btn.set_value(True).run()
    assert not at.exception, f"app raised while running a counterfactual: {at.exception}"

    # Default selection: the default match's earliest notable event, which the
    # Streamlit selectbox picks (index=0) unless told otherwise. build_timeline
    # orders shots.csv's rows by minute, so the earliest goal/big-chance/penalty
    # is first; red cards never come before minute 1 so they can't preempt it.
    match_id = featured_meta["default_match_id"]
    match = matches[matches.match_id == match_id].iloc[0]
    match_shots = shots[shots.match_id == match_id]
    first_event = match_shots[match_shots.is_goal | match_shots.is_penalty |
                               match_shots.is_big_chance].sort_values("minute").iloc[0]
    change = "goal_cancelled" if first_event.is_goal else "shot_becomes_goal"

    cf = Counterfactual(match_id=match_id, minute=int(first_event.minute), change=change, team=first_event.team)
    actual_curve = featured_curves[featured_curves.match_id == match_id].drop(columns="match_id").reset_index(drop=True)
    expected = apply_counterfactual(cf, matches, shots, red_cards, team_draws,
                                     2000 + match_id * 97 + int(first_event.minute),
                                     TRIALS_PER_DRAW, CHECKPOINTS, actual_curve=actual_curve)

    rendered_metrics = {(m.label, m.value): m.delta for m in at.metric}
    for label, outcome in ((f"{match.home_team} win", "home"), ("Draw", "draw"), (f"{match.away_team} win", "away")):
        before_pct = f"{expected.fork_before[outcome]:.0%}"
        assert (label, before_pct) in rendered_metrics, (
            f"expected '{label}: {before_pct}' among rendered 'before' metrics: {list(rendered_metrics)}")
        after_pct = f"{expected.fork_after[outcome]:.0%}"
        assert (label, after_pct) in rendered_metrics, (
            f"expected '{label}: {after_pct}' among rendered 'after' metrics: {list(rendered_metrics)}")


def test_same_seed_gives_identical_results(point_clubs) -> None:
    home, away = point_clubs["Silverlake FC"], point_clubs["Northgate United"]

    rng1 = np.random.default_rng(777)
    fh1, fa1 = simulate_forward(home, away, rng1, 0, 0, 0, False, False, 1000)

    rng2 = np.random.default_rng(777)
    fh2, fa2 = simulate_forward(home, away, rng2, 0, 0, 0, False, False, 1000)

    assert np.array_equal(fh1, fh2), "same seed produced different final home scores"
    assert np.array_equal(fa1, fa2), "same seed produced different final away scores"

    probs1 = win_probabilities(fh1, fa1)
    probs2 = win_probabilities(fh2, fa2)
    assert probs1 == probs2, f"same seed produced different win probabilities: {probs1} vs {probs2}"


def test_same_seed_gives_identical_counterfactual(matches, shots, red_cards, team_draws) -> None:
    cf = Counterfactual(match_id=1781, minute=38, change="shot_becomes_goal", team="Silverlake FC")
    r1 = apply_counterfactual(cf, matches, shots, red_cards, team_draws, 55555, 150, CHECKPOINTS)
    r2 = apply_counterfactual(cf, matches, shots, red_cards, team_draws, 55555, 150, CHECKPOINTS)

    assert r1.fork_before == r2.fork_before, "same seed gave different fork_before"
    assert r1.fork_after == r2.fork_after, "same seed gave different fork_after"
    pd.testing.assert_frame_equal(r1.actual_curve, r2.actual_curve)
    pd.testing.assert_frame_equal(r1.counterfactual_curve, r2.counterfactual_curve)
