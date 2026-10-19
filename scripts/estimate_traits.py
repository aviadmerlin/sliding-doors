"""Estimates each club's current hidden traits from 5 seasons of data,
tries several recency half-lives, and validates against the true traits
(see data/true_traits.csv) for the latest season."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.trait_estimation import build_regression_table, compute_weights, fit_traits

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
HALF_LIVES = [0.5, 1, 2, 3, None]  # seasons; None = no decay
TRAITS = ["attack", "defense", "comeback_factor", "protect_lead_factor",
          "ten_men_resilience", "fatigue_factor"]


def label(half_life: float | None) -> str:
    return "no decay" if half_life is None else f"half-life={half_life}"


def validate(estimated: pd.DataFrame, true: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trait in TRAITS:
        est = estimated[trait]
        truth = true.set_index("team")[trait].loc[est.index]
        mae = (est - truth).abs().mean()
        corr = np.corrcoef(est, truth)[0, 1]
        rows.append({"trait": trait, "mae": mae, "correlation": corr})
    return pd.DataFrame(rows).set_index("trait")


def plot_validation(estimated: pd.DataFrame, true: pd.DataFrame, half_life_label: str,
                     out_path: Path) -> None:
    truth = true.set_index("team")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for ax, trait in zip(axes.flat, TRAITS):
        x = truth[trait].loc[estimated.index]
        y = estimated[trait]
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = (hi - lo) * 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", linewidth=1)
        ax.scatter(x, y, alpha=0.75, color="#2a6f97")
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{trait}\n(r = {corr:.2f})")
        ax.set_xlabel("true")
        ax.set_ylabel("estimated")

    fig.suptitle(f"Estimated vs. true club traits — latest season ({half_life_label})")
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved chart to {out_path}")


def main() -> None:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    shots = pd.read_csv(DATA_DIR / "shots.csv")
    red_cards = pd.read_csv(DATA_DIR / "red_cards.csv")
    true_traits = pd.read_csv(DATA_DIR / "true_traits.csv")

    latest_season = matches.season.max()
    true_latest = true_traits[true_traits.season == latest_season]
    teams = sorted(matches.home_team.unique())

    print("Building regression table from shots + red cards...")
    table = build_regression_table(matches, shots, red_cards)
    print(f"  {len(table)} (team, match, segment) rows\n")

    results = {}
    reports = {}
    for half_life in HALF_LIVES:
        weights = compute_weights(table.season, half_life, latest_season)
        estimated = fit_traits(table, teams, weights)
        report = validate(estimated, true_latest)
        results[half_life] = estimated
        reports[half_life] = report

        print(f"-- {label(half_life)} --")
        print(report.round(3))
        print(f"  avg correlation: {report.correlation.mean():.3f}\n")

    best_half_life = max(HALF_LIVES, key=lambda hl: reports[hl].correlation.mean())
    print(f"Best half-life: {label(best_half_life)} "
          f"(avg correlation {reports[best_half_life].correlation.mean():.3f})")

    plot_validation(results[best_half_life], true_latest, label(best_half_life),
                     ASSETS_DIR / "trait_validation.png")

    results[best_half_life].to_csv(DATA_DIR / "estimated_traits.csv")
    print(f"Saved estimated traits ({label(best_half_life)}) to {DATA_DIR / 'estimated_traits.csv'}")


if __name__ == "__main__":
    main()
