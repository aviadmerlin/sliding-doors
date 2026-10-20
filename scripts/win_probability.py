"""Picks the most dramatic match of the latest season (largest win-probability
swing), computes its minute-by-minute win probability using the estimated
club traits, ranks the top 3 turning points, and plots the result."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.trait_estimation import build_regression_table, compute_state_exposure, compute_weights, \
    fit_traits_distribution, match_real_states
from src.win_probability import clubs_from_traits, find_most_dramatic_match, rank_turning_points, \
    sample_team_draws, win_probability_curve_with_uncertainty

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

COARSE_TRIALS = 150
COARSE_CHECKPOINTS = list(range(0, 91, 10))
FINAL_CHECKPOINTS = list(range(0, 91))
N_TRAIT_DRAWS = 25          # "parallel leagues" sampled from the GLM's uncertainty
TRIALS_PER_DRAW = 150       # simulation replays per draw, per checkpoint (25 x 150 = 3750 total)
SHRINK_COLS = ["comeback_factor", "ten_men_resilience"]


def match_states(match: pd.Series, shots: pd.DataFrame, red_cards: pd.DataFrame) -> pd.DataFrame:
    return match_real_states(match, shots, red_cards)


def plot_curve(curve: pd.DataFrame, match: pd.Series, turning_points: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(curve.minute, curve.p_home, label=f"{match.home_team} win", color="#1b5e20")
    ax.fill_between(curve.minute, curve.p_home - 1.96 * curve.se_home, curve.p_home + 1.96 * curve.se_home,
                     color="#1b5e20", alpha=0.15)

    ax.plot(curve.minute, curve.p_draw, label="Draw", color="#616161")
    ax.fill_between(curve.minute, curve.p_draw - 1.96 * curve.se_draw, curve.p_draw + 1.96 * curve.se_draw,
                     color="#616161", alpha=0.15)

    ax.plot(curve.minute, curve.p_away, label=f"{match.away_team} win", color="#b71c1c")
    ax.fill_between(curve.minute, curve.p_away - 1.96 * curve.se_away, curve.p_away + 1.96 * curve.se_away,
                     color="#b71c1c", alpha=0.15)

    sorted_tp = turning_points.sort_values("minute")
    prev_minute = None
    level = 0
    for _, tp in sorted_tp.iterrows():
        level = level + 1 if prev_minute is not None and tp.minute - prev_minute < 6 else 0
        prev_minute = tp.minute

        short_label = f"{tp.event_kind} — {tp.event_player}"
        ax.axvline(tp.minute, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.annotate(f"{tp.minute}' {short_label}", xy=(tp.minute, 1.0),
                     xytext=(tp.minute, 1.03 + level * 0.12), fontsize=8, ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="-", color="black", alpha=0.6, linewidth=0.8))

    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("minute")
    ax.set_ylabel("win probability")
    ax.set_title(f"Win probability — {match.home_team} {match.home_goals}-{match.away_goals} "
                 f"{match.away_team} (season {match.season}, matchday {match.matchday})")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    shots = pd.read_csv(DATA_DIR / "shots.csv")
    red_cards = pd.read_csv(DATA_DIR / "red_cards.csv")
    estimated_traits = pd.read_csv(DATA_DIR / "estimated_traits.csv").set_index("team")

    latest_season = matches.season.max()
    season_matches = matches[matches.season == latest_season]
    teams = sorted(matches.home_team.unique())
    clubs = clubs_from_traits(estimated_traits)

    print(f"Screening {len(season_matches)} matches from season {latest_season} "
          f"({len(COARSE_CHECKPOINTS)} checkpoints x {COARSE_TRIALS} trials each, point-estimate traits)...")
    best_match, best_swing = find_most_dramatic_match(season_matches, shots, red_cards, clubs, seed=0,
                                                        n_trials=COARSE_TRIALS, checkpoints=COARSE_CHECKPOINTS)
    print(f"Most dramatic: {best_match.home_team} {best_match.home_goals}-{best_match.away_goals} "
          f"{best_match.away_team} (matchday {best_match.matchday}), "
          f"coarse win-prob swing = {best_swing:.2f}\n")

    print("Refitting the trait regression to get its uncertainty (covariance over coefficients)...")
    table = build_regression_table(matches, shots, red_cards)
    weights = compute_weights(table.season, None, latest_season)
    _, params, cov = fit_traits_distribution(table, teams, weights)
    exposure = compute_state_exposure(table, teams)
    team_draws = sample_team_draws(params, cov, teams, exposure, SHRINK_COLS, N_TRAIT_DRAWS,
                                    np.random.default_rng(42), finishing_factor=estimated_traits.finishing_factor)
    print(f"Sampled {N_TRAIT_DRAWS} alternate leagues from the regression's uncertainty.\n")

    total_trials = N_TRAIT_DRAWS * TRIALS_PER_DRAW
    print(f"Running final curve ({len(FINAL_CHECKPOINTS)} checkpoints x {N_TRAIT_DRAWS} trait draws x "
          f"{TRIALS_PER_DRAW} trials = {total_trials}/checkpoint)...")
    states = match_states(best_match, shots, red_cards)
    curve = win_probability_curve_with_uncertainty(team_draws, best_match.home_team, best_match.away_team,
                                                     states, (best_match.home_goals, best_match.away_goals),
                                                     seed=1000, trials_per_batch=TRIALS_PER_DRAW,
                                                     checkpoints=FINAL_CHECKPOINTS)

    turning_points = rank_turning_points(curve, best_match, shots, red_cards)
    print("\nTop 3 turning points:")
    print(turning_points.to_string(index=False))

    curve.to_csv(DATA_DIR / "win_probability_curve.csv", index=False)
    plot_curve(curve, best_match, turning_points, ASSETS_DIR / "win_probability.png")


if __name__ == "__main__":
    main()
