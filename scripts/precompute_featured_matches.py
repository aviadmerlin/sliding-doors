"""Precomputes everything the app needs to avoid a slow cold start:

1. Fits the trait regression once and saves it to disk (model params,
   covariance, and the sampled "parallel league" trait draws the
   simulation actually consumes) -- the app loads these from file instead
   of refitting (~15-20s) on first use.
2. Ranks all season-5 matches by win-probability swing and saves full
   curves + turning points for the most dramatic match plus ~10 other
   high-swing games, so a judge opening the app sees a real result in
   about a second instead of a blank page for 30-40s. Other matches still
   compute on demand in the app, using the saved model from (1).

Run this once after (re)generating a season, re-estimating traits, or
changing anything in src/win_probability.py or src/match_engine.py:

    python scripts/precompute_featured_matches.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.trait_estimation import (build_regression_table, compute_state_exposure, compute_weights,
                                   fit_traits_distribution, match_real_states, sample_trait_draws)
from src.win_probability import (clubs_from_traits, rank_matches_by_drama, rank_turning_points,
                                  shrink_traits, win_probability_curve_with_uncertainty)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_FEATURED = 11            # the most dramatic match + 10 others
N_TRAIT_DRAWS = 18         # must match app.py's N_TRAIT_DRAWS
TRIALS_PER_DRAW = 200      # must match app.py's TRIALS_PER_DRAW (bumped for smoother curves, alongside CRN)
SHRINK_COLS = ["comeback_factor", "ten_men_resilience"]
CHECKPOINTS = list(range(0, 91))
COARSE_TRIALS = 150
COARSE_CHECKPOINTS = list(range(0, 91, 10))


def save_model(params: pd.Series, cov: pd.DataFrame, team_draws: list[dict]) -> None:
    """Saves everything the app needs to simulate without refitting:
    the raw GLM coefficients + covariance (for completeness / future
    re-sampling), and the already-sampled team_draws the app actually
    loads at startup."""
    params.to_frame("value").to_csv(DATA_DIR / "model_params.csv")
    cov.to_csv(DATA_DIR / "model_cov.csv")

    rows = []
    for draw_id, draw in enumerate(team_draws):
        for team, club in draw.items():
            rows.append({"draw": draw_id, "team": team, "attack": club.attack, "defense": club.defense,
                         "comeback_factor": club.comeback_factor, "protect_lead_factor": club.protect_lead_factor,
                         "ten_men_resilience": club.ten_men_resilience, "fatigue_factor": club.fatigue_factor})
    pd.DataFrame(rows).to_csv(DATA_DIR / "model_team_draws.csv", index=False)


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    shots = pd.read_csv(DATA_DIR / "shots.csv")
    red_cards = pd.read_csv(DATA_DIR / "red_cards.csv")
    estimated_traits = pd.read_csv(DATA_DIR / "estimated_traits.csv").set_index("team")

    teams = sorted(matches.home_team.unique())
    latest_season = int(matches.season.max())
    season_matches = matches[matches.season == latest_season].reset_index(drop=True)
    point_clubs = clubs_from_traits(estimated_traits)

    print("Fitting the trait regression...")
    table = build_regression_table(matches, shots, red_cards)
    weights = compute_weights(table.season, None, latest_season)
    _, params, cov = fit_traits_distribution(table, teams, weights)
    exposure = compute_state_exposure(table, teams)

    rng = np.random.default_rng(42)
    raw_draws = sample_trait_draws(params, cov, teams, N_TRAIT_DRAWS, rng)
    team_draws = [clubs_from_traits(shrink_traits(d, exposure, SHRINK_COLS)) for d in raw_draws]

    print("Saving the fitted model to disk (data/model_*.csv) so the app never refits it...")
    save_model(params, cov, team_draws)

    print(f"Ranking all {len(season_matches)} season-{latest_season} matches by win-probability swing...")
    ranked = rank_matches_by_drama(season_matches, shots, red_cards, point_clubs, seed=0,
                                    n_trials=COARSE_TRIALS, checkpoints=COARSE_CHECKPOINTS)
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
            seed=1000 + match_id, trials_per_batch=TRIALS_PER_DRAW, checkpoints=CHECKPOINTS)
        curve.insert(0, "match_id", match_id)
        all_curves.append(curve)

        turning_points = rank_turning_points(curve.drop(columns="match_id"), match, shots, red_cards, top_n=3)
        turning_points.insert(0, "match_id", match_id)
        turning_points["secondary_events"] = turning_points.secondary_events.apply(json.dumps)
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
