"""Generates 5 synthetic seasons and prints summary stats to sanity-check them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.league import create_league
from src.season_generator import generate_seasons
from src.squad import create_squads

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NUM_SEASONS = 5


def result(row) -> str:
    if row.home_goals > row.away_goals:
        return "home_win"
    if row.home_goals < row.away_goals:
        return "away_win"
    return "draw"


def build_table(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team in sorted(set(matches.home_team) | set(matches.away_team)):
        home = matches[matches.home_team == team]
        away = matches[matches.away_team == team]
        wins = (home.result == "home_win").sum() + (away.result == "away_win").sum()
        draws = (home.result == "draw").sum() + (away.result == "draw").sum()
        losses = len(home) + len(away) - wins - draws
        goals_for = home.home_goals.sum() + away.away_goals.sum()
        goals_against = home.away_goals.sum() + away.home_goals.sum()
        rows.append({
            "team": team, "played": len(home) + len(away),
            "wins": wins, "draws": draws, "losses": losses,
            "goals_for": goals_for, "goals_against": goals_against,
            "points": wins * 3 + draws,
        })
    return pd.DataFrame(rows).sort_values("points", ascending=False).reset_index(drop=True)


def build_top_scorers(goals: pd.DataFrame, season: int, top_n: int = 5) -> pd.DataFrame:
    season_goals = goals[goals.season == season]
    counts = (season_goals.groupby(["player", "team"]).size()
              .reset_index(name="goals").sort_values("goals", ascending=False))
    return counts.head(top_n).reset_index(drop=True)


def print_summary(matches: pd.DataFrame, shots: pd.DataFrame, goals: pd.DataFrame,
                   red_cards: pd.DataFrame) -> None:
    matches = matches.copy()
    matches["total_goals"] = matches.home_goals + matches.away_goals
    matches["result"] = matches.apply(result, axis=1)
    num_matches = len(matches)

    print(f"Seasons: {NUM_SEASONS}  |  Matches: {num_matches}")
    print(f"Avg goals per match: {matches.total_goals.mean():.2f}")
    print(f"Avg home goals: {matches.home_goals.mean():.2f}  |  Avg away goals: {matches.away_goals.mean():.2f}")
    print()
    print("Result split:")
    print((matches.result.value_counts(normalize=True) * 100).round(1).astype(str) + "%")
    print()

    shots_per_match = len(shots) / num_matches / 2  # per team
    penalties = shots[shots.is_penalty]
    print(f"Shots per team per match: {shots_per_match:.2f}")
    print(f"Average xG per shot: {shots.xg.mean():.3f}")
    print(f"Average total xG per team per match: {shots.xg.sum() / num_matches / 2:.2f}")
    print(f"Penalties per match (both teams): {len(penalties) / num_matches:.2f}")
    print(f"Average penalty xG: {penalties.xg.mean():.3f}")
    print(f"Big chances per team per match: {shots.is_big_chance.sum() / num_matches / 2:.2f}")
    print()

    print(f"Red cards: {len(red_cards)} total, ~{len(red_cards) / num_matches:.2f} per match")
    print("Red cards per club:")
    print(red_cards.team.value_counts())
    print()

    print("Final standings per season (sanity check: champion ~85-95 pts, bottom ~20-35 pts):")
    for season in sorted(matches.season.unique()):
        table = build_table(matches[matches.season == season])
        print(f"\n-- Season {season} -- (champion {table.points.iloc[0]} pts, "
              f"bottom {table.points.iloc[-1]} pts)")
        print(table)
        print(f"   top scorers:")
        print(build_top_scorers(goals, season).to_string(index=False))


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    clubs = create_league(seed=42)
    squads = create_squads(clubs, seed=1)
    season = generate_seasons(clubs, squads, num_seasons=NUM_SEASONS, seed=0)

    season["matches"].to_csv(DATA_DIR / "matches.csv", index=False)
    season["shots"].to_csv(DATA_DIR / "shots.csv", index=False)
    season["goals"].to_csv(DATA_DIR / "goals.csv", index=False)
    season["red_cards"].to_csv(DATA_DIR / "red_cards.csv", index=False)
    season["true_traits"].to_csv(DATA_DIR / "true_traits.csv", index=False)

    players_rows = [
        {"team": team, "player": p.name, "position": p.position,
         "shot_share": p.shot_share, "card_share": p.card_share,
         "finishing_skill": p.finishing_skill}
        for team, squad in squads.items() for p in squad
    ]
    pd.DataFrame(players_rows).to_csv(DATA_DIR / "players.csv", index=False)

    print(f"Saved {NUM_SEASONS} seasons to {DATA_DIR}/\n")
    print_summary(season["matches"], season["shots"], season["goals"], season["red_cards"])


if __name__ == "__main__":
    main()
