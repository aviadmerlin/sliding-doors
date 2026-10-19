"""Picks the most dramatic match of the latest season (largest win-probability
swing), computes its minute-by-minute win probability using the estimated
club traits, ranks the top 3 turning points, and plots the result."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.league import Club
from src.trait_estimation import (_match_minute_states, build_regression_table, compute_state_exposure,
                                   compute_weights, fit_traits_distribution)
from src.win_probability import clubs_from_traits, sample_team_draws, win_probability_curve, \
    win_probability_curve_with_uncertainty

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

COARSE_TRIALS = 150
COARSE_CHECKPOINTS = list(range(0, 91, 10))
FINAL_CHECKPOINTS = list(range(0, 91))
N_TRAIT_DRAWS = 25          # "parallel leagues" sampled from the GLM's uncertainty
TRIALS_PER_DRAW = 150       # simulation replays per draw, per checkpoint (25 x 150 = 3750 total)
SHRINK_COLS = ["comeback_factor", "ten_men_resilience"]


def match_states(match: pd.Series, shots: pd.DataFrame, red_cards: pd.DataFrame) -> pd.DataFrame:
    match_shots = shots[shots.match_id == match.match_id]
    match_cards = red_cards[red_cards.match_id == match.match_id]
    return _match_minute_states(match.home_team, match.away_team, match_shots, match_cards)


def find_most_dramatic_match(matches: pd.DataFrame, shots: pd.DataFrame, red_cards: pd.DataFrame,
                              clubs: dict[str, Club], rng: np.random.Generator) -> tuple[pd.Series, float]:
    best_match, best_swing = None, -1.0
    for _, match in matches.iterrows():
        states = match_states(match, shots, red_cards)
        curve = win_probability_curve(clubs[match.home_team], clubs[match.away_team], states,
                                       (match.home_goals, match.away_goals), rng,
                                       COARSE_TRIALS, COARSE_CHECKPOINTS)
        swing = curve.p_home.max() - curve.p_home.min()
        if swing > best_swing:
            best_match, best_swing = match, swing
    return best_match, best_swing


def label_event(minute: int, match: pd.Series, shots: pd.DataFrame, red_cards: pd.DataFrame) -> str:
    match_shots = shots[(shots.match_id == match.match_id) & shots.is_goal]
    match_cards = red_cards[red_cards.match_id == match.match_id]

    goal = match_shots[match_shots.minute == minute]
    if len(goal):
        row = goal.iloc[0]
        side = "home" if row.team == match.home_team else "away"
        return f"{minute}' GOAL — {row.player} ({row.team}, {side}), {int(row.home_score)}-{int(row.away_score)} -> " \
               f"{int(row.home_score) + (side == 'home')}-{int(row.away_score) + (side == 'away')}"

    card = match_cards[match_cards.minute == minute]
    if len(card):
        row = card.iloc[0]
        return f"{minute}' RED CARD — {row.player} ({row.team})"

    return f"{minute}' (no scoring/card event this exact minute)"


def rank_turning_points(curve: pd.DataFrame, match: pd.Series, shots: pd.DataFrame,
                         red_cards: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Ranks minutes by total movement across all three outcome probabilities
    (not just home win) -- a goal that mostly swings the draw probability
    (e.g. an injury-time equalizer) is just as much a turning point as one
    that swings the home/away split."""
    c = curve.copy()
    c["swing"] = (c.p_home.diff().abs() + c.p_draw.diff().abs() + c.p_away.diff().abs()) / 2
    top = c.nlargest(top_n, "swing")
    top = top.assign(event=[label_event(m, match, shots, red_cards) for m in top.minute])
    return top[["minute", "swing", "p_home", "p_draw", "p_away", "event"]]


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

        short_label = tp.event.split(" — ")[1].split(",")[0] if " — " in tp.event else tp.event
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
    rng = np.random.default_rng(0)

    print(f"Screening {len(season_matches)} matches from season {latest_season} "
          f"({len(COARSE_CHECKPOINTS)} checkpoints x {COARSE_TRIALS} trials each, point-estimate traits)...")
    best_match, best_swing = find_most_dramatic_match(season_matches, shots, red_cards, clubs, rng)
    print(f"Most dramatic: {best_match.home_team} {best_match.home_goals}-{best_match.away_goals} "
          f"{best_match.away_team} (matchday {best_match.matchday}), "
          f"coarse win-prob swing = {best_swing:.2f}\n")

    print("Refitting the trait regression to get its uncertainty (covariance over coefficients)...")
    table = build_regression_table(matches, shots, red_cards)
    weights = compute_weights(table.season, None, latest_season)
    _, params, cov = fit_traits_distribution(table, teams, weights)
    exposure = compute_state_exposure(table, teams)
    team_draws = sample_team_draws(params, cov, teams, exposure, SHRINK_COLS, N_TRAIT_DRAWS, rng)
    print(f"Sampled {N_TRAIT_DRAWS} alternate leagues from the regression's uncertainty.\n")

    total_trials = N_TRAIT_DRAWS * TRIALS_PER_DRAW
    print(f"Running final curve ({len(FINAL_CHECKPOINTS)} checkpoints x {N_TRAIT_DRAWS} trait draws x "
          f"{TRIALS_PER_DRAW} trials = {total_trials}/checkpoint)...")
    states = match_states(best_match, shots, red_cards)
    curve = win_probability_curve_with_uncertainty(team_draws, best_match.home_team, best_match.away_team,
                                                     states, (best_match.home_goals, best_match.away_goals),
                                                     rng, TRIALS_PER_DRAW, FINAL_CHECKPOINTS)

    turning_points = rank_turning_points(curve, best_match, shots, red_cards)
    print("\nTop 3 turning points:")
    print(turning_points.to_string(index=False))

    curve.to_csv(DATA_DIR / "win_probability_curve.csv", index=False)
    plot_curve(curve, best_match, turning_points, ASSETS_DIR / "win_probability.png")


if __name__ == "__main__":
    main()
