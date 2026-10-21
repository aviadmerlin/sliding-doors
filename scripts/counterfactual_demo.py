"""Demos the counterfactual engine: 3 "what if" questions, 2 on the most
dramatic match (read from data/featured_matches.json, produced by
scripts/precompute_featured_matches.py), 1 on a different match that
actually had a red card (whenever the dramatic match itself didn't have
one -- true in every run so far, but checked rather than assumed).

Each scenario reports a fair, apples-to-apples headline: win probability
forecast right before the real event vs. right after the hypothetical
change, both at the same minute -- not the real match's known final
result against the hypothetical's forecast (see src/counterfactual.py)."""

import json
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

    print("Loading the precomputed trait model (data/model_team_draws.csv)...")
    team_draws = load_team_draws()

    featured = json.loads((DATA_DIR / "featured_matches.json").read_text())
    dramatic_id = featured["default_match_id"]
    dramatic = matches[matches.match_id == dramatic_id].iloc[0]
    dramatic_shots = shots[shots.match_id == dramatic_id]
    dramatic_cards = red_cards[red_cards.match_id == dramatic_id]

    last_goal = dramatic_shots[dramatic_shots.is_goal].sort_values("minute").iloc[-1]
    miss_before_last_goal = dramatic_shots[(dramatic_shots.minute < last_goal.minute) &
                                            (~dramatic_shots.is_goal)].sort_values("minute").iloc[-1]

    if len(dramatic_cards):
        red_card_match_id, red_card_minute = int(dramatic_id), int(dramatic_cards.iloc[0].minute)
    else:
        # The dramatic match has no red card -- find one where the score was
        # still close WHEN the card happened (so undoing it has something real
        # to swing -- a card in a match already decided by then makes a flat,
        # uninteresting demo), preferring earlier cards with more of the match
        # left to play. Falls back to the overall best-margin card if no
        # candidate was ever close.
        season_cards = red_cards[red_cards.match_id.isin(matches[matches.season == matches.season.max()].match_id)]
        candidates = []
        for _, card in season_cards.iterrows():
            match_shots = shots[shots.match_id == card.match_id]
            before = match_shots[match_shots.minute <= card.minute]
            h, a = (int(before.iloc[-1].home_score), int(before.iloc[-1].away_score)) if len(before) else (0, 0)
            candidates.append({"match_id": int(card.match_id), "minute": int(card.minute),
                               "margin_at_card": abs(h - a), "minutes_left": 90 - int(card.minute)})
        cand_df = pd.DataFrame(candidates)
        close_games = cand_df[cand_df.margin_at_card <= 1].sort_values("minutes_left", ascending=False)
        top = (close_games if len(close_games) else cand_df.sort_values("margin_at_card")).iloc[0]
        red_card_match_id, red_card_minute = int(top.match_id), int(top.minute)

    scenarios = [
        Counterfactual(match_id=int(dramatic.match_id), minute=int(last_goal.minute), change="goal_cancelled",
                        team=last_goal.team),
        Counterfactual(match_id=int(dramatic.match_id), minute=int(miss_before_last_goal.minute),
                        change="shot_becomes_goal", team=miss_before_last_goal.team),
        Counterfactual(match_id=red_card_match_id, minute=red_card_minute, change="red_card_undone"),
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
