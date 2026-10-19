"""Calibration check: across many (match, minute) checkpoints in the latest
season, do our predicted home-win probabilities match the actual share of
home wins? E.g. among all checkpoints where the model said "60-70% chance
of a home win", did the home team actually win about 60-70% of the time?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.trait_estimation import _match_minute_states
from src.win_probability import clubs_from_traits, win_probability_curve

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

N_TRIALS = 300
CHECKPOINTS = [0, 15, 30, 45, 60, 75]  # stop before full time, where predictions become trivial
N_BUCKETS = 10


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


def plot_calibration(summary: pd.DataFrame, overall_n: int, ece: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    ax.errorbar(summary.predicted_mean, summary.actual_rate, yerr=1.96 * summary.se,
                fmt="none", ecolor="#1b5e20", alpha=0.8, capsize=3)
    ax.scatter(summary.predicted_mean, summary.actual_rate, s=summary.n * 1.5,
               color="#1b5e20", alpha=0.8, zorder=3)

    ax.set_xlabel("predicted home win probability")
    ax.set_ylabel("actual home win rate")
    ax.set_title(f"Calibration — season 5, {overall_n} (match, minute) checkpoints\n"
                 f"expected calibration error = {ece:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
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
    clubs = clubs_from_traits(estimated_traits)
    rng = np.random.default_rng(1)

    print(f"Simulating {len(season_matches)} matches x {len(CHECKPOINTS)} checkpoints "
          f"x {N_TRIALS} trials...")
    predictions = collect_predictions(season_matches, shots, red_cards, clubs, rng)
    predictions.to_csv(DATA_DIR / "calibration_predictions.csv", index=False)
    print(f"Collected {len(predictions)} (match, minute) predictions.\n")

    summary = bucket_calibration(predictions, N_BUCKETS)
    print(summary.round(3).to_string(index=False))

    ece = float((summary.n * (summary.predicted_mean - summary.actual_rate).abs()).sum() / summary.n.sum())
    print(f"\nExpected calibration error (n-weighted avg |predicted - actual|): {ece:.4f}")

    plot_calibration(summary, len(predictions), ece, ASSETS_DIR / "calibration.png")


if __name__ == "__main__":
    main()
