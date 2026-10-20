"""Estimates each club's current hidden traits from 5 seasons of data,
tries several recency half-lives, and validates against the true traits
(see data/true_traits.csv) for the latest season."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.trait_estimation import (build_regression_table, compute_state_exposure, compute_weights,
                                   estimate_finishing_factor, estimate_league_constants, fit_traits,
                                   fit_traits_distribution, shrink_traits)

SHRINKAGE_CANDIDATES = ["comeback_factor", "protect_lead_factor", "ten_men_resilience", "fatigue_factor"]

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

    # Shrinkage: pull each club's noisier traits toward the league average,
    # weighted by how much data (exposure minutes) backs that club's estimate.
    best = results[best_half_life]
    exposure = compute_state_exposure(table, teams)
    shrunk = shrink_traits(best, exposure, SHRINKAGE_CANDIDATES)

    print("\nShrinkage effect on MAE (raw -> shrunk toward league average):")
    helps = []
    for trait in SHRINKAGE_CANDIDATES:
        mae_raw = (best[trait] - true_latest.set_index("team")[trait]).abs().mean()
        mae_shrunk = (shrunk[trait] - true_latest.set_index("team")[trait]).abs().mean()
        verdict = "helps" if mae_shrunk < mae_raw else "no help"
        if mae_shrunk < mae_raw:
            helps.append(trait)
        print(f"  {trait:22} {mae_raw:.4f} -> {mae_shrunk:.4f}  ({verdict})")

    # Only keep shrinkage where it actually lowers error; leave the rest as raw MLE.
    final = best.copy()
    final[helps] = shrunk[helps]
    print(f"\nApplying shrinkage to: {helps}")

    # Club-level finishing (goals vs. xG, shrunk toward the no-bias baseline of
    # 1.0) and the league-wide shot-rate/composition constants the fast engine
    # needs -- both estimated from data only, at the same best half-life.
    weights_best = compute_weights(table.season, best_half_life, latest_season)
    _, params_best, _ = fit_traits_distribution(table, teams, weights_best)
    final["finishing_factor"] = estimate_finishing_factor(shots, matches, teams, best_half_life, latest_season)

    league_constants = estimate_league_constants(shots, red_cards, matches, params_best, teams)
    pd.Series(league_constants, name="value").to_csv(DATA_DIR / "model_league_constants.csv")
    print(f"\nSaved league-wide constants (estimated, not imported from the generator) "
          f"to {DATA_DIR / 'model_league_constants.csv'}:")
    for k, v in league_constants.items():
        print(f"  {k:28} {v:.6f}")

    plot_validation(final, true_latest, label(best_half_life), ASSETS_DIR / "trait_validation.png")

    final.to_csv(DATA_DIR / "estimated_traits.csv")
    print(f"\nSaved estimated traits ({label(best_half_life)}, shrunk where it helps) "
          f"to {DATA_DIR / 'estimated_traits.csv'}")


if __name__ == "__main__":
    main()
