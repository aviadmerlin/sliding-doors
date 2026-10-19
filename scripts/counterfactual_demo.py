"""Demos the counterfactual engine: 3 "what if" questions, 2 on the
dramatic match from win_probability.py, 1 on a different match that
actually had a red card (the dramatic match didn't have one)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.counterfactual import Counterfactual, apply_counterfactual
from src.trait_estimation import build_regression_table, compute_state_exposure, compute_weights, \
    fit_traits_distribution
from src.win_probability import sample_team_draws

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

N_TRAIT_DRAWS = 15
TRIALS_PER_DRAW = 100
SHRINK_COLS = ["comeback_factor", "ten_men_resilience"]
CHECKPOINTS = list(range(0, 91))


def plot_scenarios(results: list, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    axes = axes[0]

    for ax, (cf, result) in zip(axes, results):
        ax.plot(result.actual_curve.minute, result.actual_curve.p_home, color="#1b5e20", label="actual")
        ax.plot(result.counterfactual_curve.minute, result.counterfactual_curve.p_home,
                 color="#b71c1c", linestyle="--", label="counterfactual")
        ax.axvline(cf.minute, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.set_xlabel("minute")
        ax.set_ylabel("home win probability")
        ax.set_title(result.description, fontsize=9, wrap=True)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved chart to {out_path}")


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    shots = pd.read_csv(DATA_DIR / "shots.csv")
    red_cards = pd.read_csv(DATA_DIR / "red_cards.csv")
    teams = sorted(matches.home_team.unique())
    latest_season = matches.season.max()

    print("Fitting trait regression and sampling parameter-uncertainty draws...")
    table = build_regression_table(matches, shots, red_cards)
    weights = compute_weights(table.season, None, latest_season)
    _, params, cov = fit_traits_distribution(table, teams, weights)
    exposure = compute_state_exposure(table, teams)
    rng = np.random.default_rng(3)
    team_draws = sample_team_draws(params, cov, teams, exposure, SHRINK_COLS, N_TRAIT_DRAWS, rng)

    dramatic = matches[(matches.season == latest_season) & (matches.home_team == "Silverlake FC") &
                        (matches.away_team == "Northgate United")].iloc[0]
    red_card_match_id = 1899  # Silverlake FC 1-5 Thornfield United, Fontaine sent off at 17'

    scenarios = [
        Counterfactual(match_id=int(dramatic.match_id), minute=87, change="goal_cancelled"),
        Counterfactual(match_id=int(dramatic.match_id), minute=38, change="shot_becomes_goal",
                        team="Silverlake FC"),
        Counterfactual(match_id=red_card_match_id, minute=17, change="red_card_undone"),
    ]

    results = []
    cached_actual = {}
    for cf in scenarios:
        print(f"\nApplying: {cf}")
        actual_curve = cached_actual.get(cf.match_id)
        result = apply_counterfactual(cf, matches, shots, red_cards, team_draws, rng,
                                       TRIALS_PER_DRAW, CHECKPOINTS, actual_curve=actual_curve)
        cached_actual[cf.match_id] = result.actual_curve

        print(f"  {result.description}")
        print(f"  actual final:         home={result.actual_final['p_home']:.2f}  "
              f"draw={result.actual_final['p_draw']:.2f}  away={result.actual_final['p_away']:.2f}")
        print(f"  counterfactual final: home={result.counterfactual_final['p_home']:.2f}  "
              f"draw={result.counterfactual_final['p_draw']:.2f}  away={result.counterfactual_final['p_away']:.2f}")
        results.append((cf, result))

    plot_scenarios(results, ASSETS_DIR / "counterfactual_demo.png")


if __name__ == "__main__":
    main()
