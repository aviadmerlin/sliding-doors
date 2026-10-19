"""Precomputes win-probability curves and turning points for a shortlist of
featured matches (the most dramatic one + ~10 other interesting ones), so
the deployed app can show them instantly instead of simulating on page
load -- a judge opening the link would otherwise see a blank page for
30-40 seconds. Other (non-featured) matches still compute on demand in
the app, with a progress indicator.

Run this once after (re)generating a season or re-estimating traits:

    python scripts/precompute_featured_matches.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.trait_estimation import (build_regression_table, compute_state_exposure, compute_weights,
                                   fit_traits_distribution, match_real_states)
from src.win_probability import (clubs_from_traits, rank_matches_by_drama, rank_turning_points,
                                  sample_team_draws, win_probability_curve_with_uncertainty)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_FEATURED = 11            # the most dramatic match + 10 others
N_TRAIT_DRAWS = 18         # must match app.py's N_TRAIT_DRAWS so cached + on-demand curves are comparable
TRIALS_PER_DRAW = 120      # must match app.py's TRIALS_PER_DRAW
SHRINK_COLS = ["comeback_factor", "ten_men_resilience"]
CHECKPOINTS = list(range(0, 91))
COARSE_TRIALS = 150
COARSE_CHECKPOINTS = list(range(0, 91, 10))


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    shots = pd.read_csv(DATA_DIR / "shots.csv")
    red_cards = pd.read_csv(DATA_DIR / "red_cards.csv")
    estimated_traits = pd.read_csv(DATA_DIR / "estimated_traits.csv").set_index("team")

    teams = sorted(matches.home_team.unique())
    latest_season = int(matches.season.max())
    season_matches = matches[matches.season == latest_season].reset_index(drop=True)
    point_clubs = clubs_from_traits(estimated_traits)

    print("Fitting the trait regression (same model the app would otherwise fit on first use)...")
    table = build_regression_table(matches, shots, red_cards)
    weights = compute_weights(table.season, None, latest_season)
    _, params, cov = fit_traits_distribution(table, teams, weights)
    exposure = compute_state_exposure(table, teams)

    rng = np.random.default_rng(42)
    team_draws = sample_team_draws(params, cov, teams, exposure, SHRINK_COLS, N_TRAIT_DRAWS, rng)

    print(f"Ranking all {len(season_matches)} season-{latest_season} matches by win-probability swing...")
    screen_rng = np.random.default_rng(0)
    ranked = rank_matches_by_drama(season_matches, shots, red_cards, point_clubs, screen_rng,
                                    COARSE_TRIALS, COARSE_CHECKPOINTS)
    featured_ids = [int(mid) for mid in ranked.head(N_FEATURED).match_id]
    default_match_id = featured_ids[0]

    print(f"Featured matches ({len(featured_ids)}), most dramatic first:")
    for mid in featured_ids:
        m = matches[matches.match_id == mid].iloc[0]
        swing = ranked[ranked.match_id == mid].swing.iloc[0]
        print(f"  {mid}: {m.home_team} {m.home_goals}-{m.away_goals} {m.away_team} "
              f"(matchday {m.matchday}, swing {swing:.2f})")

    all_curves, all_turning_points = [], []
    for i, match_id in enumerate(featured_ids):
        match = matches[matches.match_id == match_id].iloc[0]
        print(f"[{i + 1}/{len(featured_ids)}] Simulating {match.home_team} vs {match.away_team}...")
        states = match_real_states(match, shots, red_cards)
        curve = win_probability_curve_with_uncertainty(
            team_draws, match.home_team, match.away_team, states,
            (int(match.home_goals), int(match.away_goals)),
            np.random.default_rng(1000 + match_id), TRIALS_PER_DRAW, CHECKPOINTS)
        curve.insert(0, "match_id", match_id)
        all_curves.append(curve)

        turning_points = rank_turning_points(curve.drop(columns="match_id"), match, shots, red_cards, top_n=3)
        turning_points.insert(0, "match_id", match_id)
        all_turning_points.append(turning_points)

    pd.concat(all_curves, ignore_index=True).to_csv(DATA_DIR / "featured_curves.csv", index=False)
    pd.concat(all_turning_points, ignore_index=True).to_csv(DATA_DIR / "featured_turning_points.csv", index=False)
    with open(DATA_DIR / "featured_matches.json", "w") as f:
        json.dump({"default_match_id": default_match_id, "featured_match_ids": featured_ids}, f, indent=2)

    print(f"\nSaved {len(featured_ids)} featured matches to:")
    print(f"  {DATA_DIR / 'featured_curves.csv'}")
    print(f"  {DATA_DIR / 'featured_turning_points.csv'}")
    print(f"  {DATA_DIR / 'featured_matches.json'}")


if __name__ == "__main__":
    main()
