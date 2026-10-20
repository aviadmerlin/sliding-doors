"""Demos the counterfactual engine: 3 "what if" questions, 2 on the
dramatic match from win_probability.py, 1 on a different match that
actually had a red card (the dramatic match didn't have one).

Each scenario reports a fair, apples-to-apples headline: win probability
forecast right before the real event vs. right after the hypothetical
change, both at the same minute -- not the real match's known final
result against the hypothetical's forecast (see src/counterfactual.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from src.counterfactual import Counterfactual, CounterfactualResult, apply_counterfactual
from src.win_probability import clubs_from_traits

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

TRIALS_PER_DRAW = 200
CHECKPOINTS = list(range(0, 91))


def load_team_draws() -> list[dict]:
    """Loads the trait draws precomputed by scripts/precompute_featured_matches.py
    (run that first) -- instant, no refitting."""
    df = pd.read_csv(DATA_DIR / "model_team_draws.csv")
    return [clubs_from_traits(group.set_index("team")) for _, group in df.groupby("draw")]


def headline_outcome(result: CounterfactualResult) -> str:
    deltas = {o: result.fork_after[o] - result.fork_before[o] for o in ("home", "draw", "away")}
    return max(deltas, key=lambda o: abs(deltas[o]))


def outcome_label(outcome: str, match: pd.Series) -> str:
    return {"home": f"{match.home_team} win", "draw": "Draw", "away": f"{match.away_team} win"}[outcome]


def plot_scenarios(results: list, matches: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    axes = axes[0]

    for ax, (cf, result) in zip(axes, results):
        match = matches[matches.match_id == cf.match_id].iloc[0]
        outcome = headline_outcome(result)
        y_col, se_col = f"p_{outcome}", f"se_{outcome}"

        ax.plot(result.actual_curve.minute, result.actual_curve[y_col], color="#3B82F6", label="real match")
        ax.fill_between(result.actual_curve.minute,
                         result.actual_curve[y_col] - 1.96 * result.actual_curve[se_col],
                         result.actual_curve[y_col] + 1.96 * result.actual_curve[se_col],
                         color="#3B82F6", alpha=0.15)

        ax.plot(result.counterfactual_curve.minute, result.counterfactual_curve[y_col],
                 color="#F59E0B", linestyle="--", label="counterfactual (flat forecast from fork)")
        ax.fill_between(result.counterfactual_curve.minute,
                         result.counterfactual_curve[y_col] - 1.96 * result.counterfactual_curve[se_col],
                         result.counterfactual_curve[y_col] + 1.96 * result.counterfactual_curve[se_col],
                         color="#F59E0B", alpha=0.15)

        ax.axvline(cf.minute, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.set_xlabel("minute")
        ax.set_ylabel(f"{outcome_label(outcome, match)} probability")
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
    latest_season = matches.season.max()

    print("Loading the precomputed trait model (data/model_team_draws.csv)...")
    team_draws = load_team_draws()

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
    for i, cf in enumerate(scenarios):
        print(f"\nApplying: {cf}")
        match = matches[matches.match_id == cf.match_id].iloc[0]
        actual_curve = cached_actual.get(cf.match_id)
        result = apply_counterfactual(cf, matches, shots, red_cards, team_draws, 5000 + i * 97,
                                       TRIALS_PER_DRAW, CHECKPOINTS, actual_curve=actual_curve)
        cached_actual[cf.match_id] = result.actual_curve

        outcome = headline_outcome(result)
        print(f"  {result.description}")
        print(f"  {outcome_label(outcome, match)}: {result.fork_before[outcome]:.0%} (just before) -> "
              f"{result.fork_after[outcome]:.0%} (just after) -- fair, same-minute forecast comparison")
        print(f"  What actually happened: {match.home_team} {result.real_final_score[0]}-"
              f"{result.real_final_score[1]} {match.away_team} ({result.real_outcome} won)")
        results.append((cf, result))

    plot_scenarios(results, matches, ASSETS_DIR / "counterfactual_demo.png")


if __name__ == "__main__":
    main()
