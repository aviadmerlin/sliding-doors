"""Calibration check: across many (match, minute) checkpoints in the latest
season, do our predicted home-win probabilities match the actual share of
home wins? E.g. among all checkpoints where the model said "60-70% chance
of a home win", did the home team actually win about 60-70% of the time?

Reports two versions:
  - in-sample (leaky): traits fitted on seasons 1-5, tested on season 5 --
    the model has already "seen" the data it's being judged on.
  - out-of-sample (honest): traits fitted on seasons 1-4 only, tested on
    season 5 -- the fair test, since in a real deployment you'd only have
    past seasons when predicting the new one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.trait_estimation import (_match_minute_states, build_regression_table, compute_state_exposure,
                                   compute_weights, fit_traits, shrink_traits)
from src.win_probability import clubs_from_traits, win_probability_curve

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

N_TRIALS = 300
CHECKPOINTS = [0, 15, 30, 45, 60, 75]  # stop before full time, where predictions become trivial
N_BUCKETS = 10
SHRINK_COLS = ["comeback_factor", "ten_men_resilience"]


def fit_traits_on_seasons(matches: pd.DataFrame, shots: pd.DataFrame, red_cards: pd.DataFrame,
                           seasons: list[int]) -> pd.DataFrame:
    """Refits the trait regression using only the given seasons' data."""
    train_matches = matches[matches.season.isin(seasons)]
    train_shots = shots[shots.match_id.isin(train_matches.match_id)]
    train_cards = red_cards[red_cards.match_id.isin(train_matches.match_id)]
    teams = sorted(train_matches.home_team.unique())

    table = build_regression_table(train_matches, train_shots, train_cards)
    weights = compute_weights(table.season, None, max(seasons))
    point = fit_traits(table, teams, weights)
    exposure = compute_state_exposure(table, teams)
    return shrink_traits(point, exposure, SHRINK_COLS)


def collect_predictions(matches: pd.DataFrame, shots: pd.DataFrame, red_cards: pd.DataFrame,
                         clubs: dict, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _, match in matches.iterrows():
        match_shots = shots[shots.match_id == match.match_id]
        match_cards = red_cards[red_cards.match_id == match.match_id]
        states = _match_minute_states(match.home_team, match.away_team, match_shots, match_cards)

        actual_home_win = match.home_goals > match.away_goals
        curve = win_probability_curve(clubs[match.home_team], clubs[match.away_team], states,
                                       (match.home_goals, match.away_goals), rng, N_TRIALS, CHECKPOINTS)
        for _, row in curve.iterrows():
            rows.append({"match_id": match.match_id, "minute": row.minute,
                         "predicted_p_home": row.p_home, "actual_home_win": actual_home_win})
    return pd.DataFrame(rows)


def bucket_calibration(predictions: pd.DataFrame, n_buckets: int) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_buckets + 1)
    predictions = predictions.copy()
    predictions["bucket"] = pd.cut(predictions.predicted_p_home, bins, include_lowest=True)

    summary = predictions.groupby("bucket", observed=True).agg(
        predicted_mean=("predicted_p_home", "mean"),
        actual_rate=("actual_home_win", "mean"),
        n=("actual_home_win", "size"),
    ).reset_index()
    summary["se"] = np.sqrt(summary.actual_rate * (1 - summary.actual_rate) / summary.n)
    return summary


def expected_calibration_error(summary: pd.DataFrame) -> float:
    return float((summary.n * (summary.predicted_mean - summary.actual_rate).abs()).sum() / summary.n.sum())


def plot_calibration(summary: pd.DataFrame, overall_n: int, ece: float, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    ax.errorbar(summary.predicted_mean, summary.actual_rate, yerr=1.96 * summary.se,
                fmt="none", ecolor="#1b5e20", alpha=0.8, capsize=3)
    ax.scatter(summary.predicted_mean, summary.actual_rate, s=summary.n * 1.5,
               color="#1b5e20", alpha=0.8, zorder=3)

    ax.set_xlabel("predicted home win probability")
    ax.set_ylabel("actual home win rate")
    ax.set_title(f"{title}\n{overall_n} (match, minute) checkpoints, expected calibration error = {ece:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def run_check(label: str, clubs: dict, season_matches: pd.DataFrame, shots: pd.DataFrame,
              red_cards: pd.DataFrame, rng: np.random.Generator, out_path: Path, title: str) -> float:
    print(f"\n-- {label} --")
    predictions = collect_predictions(season_matches, shots, red_cards, clubs, rng)
    summary = bucket_calibration(predictions, N_BUCKETS)
    print(summary.round(3).to_string(index=False))
    ece = expected_calibration_error(summary)
    print(f"Expected calibration error: {ece:.4f}")
    plot_calibration(summary, len(predictions), ece, title, out_path)
    return ece


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    shots = pd.read_csv(DATA_DIR / "shots.csv")
    red_cards = pd.read_csv(DATA_DIR / "red_cards.csv")

    latest_season = matches.season.max()
    season_matches = matches[matches.season == latest_season]
    rng = np.random.default_rng(1)

    in_sample_traits = pd.read_csv(DATA_DIR / "estimated_traits.csv").set_index("team")
    ece_leaky = run_check(
        "In-sample (leaky): fit on seasons 1-5, tested on season 5",
        clubs_from_traits(in_sample_traits), season_matches, shots, red_cards, rng,
        ASSETS_DIR / "calibration_in_sample.png", "Calibration (in-sample, leaky)")

    print("\nRefitting traits on seasons 1-4 only (out-of-sample test)...")
    train_seasons = [s for s in matches.season.unique() if s != latest_season]
    out_of_sample_traits = fit_traits_on_seasons(matches, shots, red_cards, train_seasons)
    out_of_sample_traits.to_csv(DATA_DIR / "estimated_traits_seasons_1_4.csv")
    ece_honest = run_check(
        f"Out-of-sample (honest): fit on seasons {train_seasons}, tested on season {latest_season}",
        clubs_from_traits(out_of_sample_traits), season_matches, shots, red_cards, rng,
        ASSETS_DIR / "calibration.png", "Calibration (out-of-sample)")

    print(f"\nSummary: in-sample ECE = {ece_leaky:.4f}  |  out-of-sample ECE = {ece_honest:.4f}")


if __name__ == "__main__":
    main()
