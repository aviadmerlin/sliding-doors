"""Schedules and simulates multiple seasons of a double round-robin league.

Across seasons, squads change: each club's hidden traits drift a little
every season (see league.drift_traits). We save the true traits per club
per season — later, trait estimation will use this to check how much
weight recent seasons deserve relative to older ones.
"""

import numpy as np
import pandas as pd

from src.league import Club, TRAIT_BOUNDS, drift_traits
from src.match_engine import simulate_match
from src.squad import Player


def _round_robin_rounds(clubs: list[Club]) -> list[list[tuple[Club, Club]]]:
    """Standard 'circle method' scheduling: n-1 rounds, each club plays once per round."""
    teams = list(clubs)
    n = len(teams)
    rounds = []
    for _ in range(n - 1):
        pairs = [(teams[i], teams[n - 1 - i]) for i in range(n // 2)]
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds


def generate_season(clubs: list[Club], squads: dict[str, list[Player]], rng: np.random.Generator,
                     start_match_id: int = 1) -> tuple[dict[str, pd.DataFrame], int]:
    """Simulates one double round-robin season. Returns (tables, next_match_id)."""
    rounds = _round_robin_rounds(clubs)
    num_rounds = len(rounds)

    matches, shot_rows, red_card_rows = [], [], []
    match_id = start_match_id

    for leg in (1, 2):
        for round_idx, pairs in enumerate(rounds):
            matchday = round_idx + 1 if leg == 1 else round_idx + 1 + num_rounds
            for team_a, team_b in pairs:
                home, away = (team_a, team_b) if leg == 1 else (team_b, team_a)
                result = simulate_match(home, away, squads[home.name], squads[away.name], rng)

                matches.append({
                    "match_id": match_id,
                    "matchday": matchday,
                    "home_team": home.name,
                    "away_team": away.name,
                    "home_goals": result["home_goals"],
                    "away_goals": result["away_goals"],
                })
                for shot in result["shots"]:
                    shot_rows.append({"match_id": match_id, **shot})
                for ev in result["red_card_events"]:
                    red_card_rows.append({"match_id": match_id, **ev})

                match_id += 1

    tables = {
        "matches": pd.DataFrame(matches),
        "shots": pd.DataFrame(shot_rows),
        "red_cards": pd.DataFrame(red_card_rows),
    }
    return tables, match_id


def generate_seasons(initial_clubs: list[Club], squads: dict[str, list[Player]],
                      num_seasons: int = 10, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Simulates several seasons back to back, drifting traits between them.
    Squads are fixed for the whole history (no transfers yet).

    Returns matches/shots/goals/red_cards (each tagged with a 'season'
    column) plus true_traits: the hidden traits actually used in each
    season, for later validation.
    """
    rng = np.random.default_rng(seed)
    clubs = initial_clubs
    match_id = 1

    season_tables = {"matches": [], "shots": [], "red_cards": []}
    trait_rows = []

    for season in range(1, num_seasons + 1):
        for club in clubs:
            trait_rows.append({
                "season": season, "team": club.name,
                **{trait: getattr(club, trait) for trait in TRAIT_BOUNDS},
            })

        tables, match_id = generate_season(clubs, squads, rng, start_match_id=match_id)
        for name, df in tables.items():
            df = df.copy()
            df.insert(0, "season", season)
            season_tables[name].append(df)

        clubs = [drift_traits(c, rng) for c in clubs]

    combined = {name: pd.concat(dfs, ignore_index=True) for name, dfs in season_tables.items()}
    combined["true_traits"] = pd.DataFrame(trait_rows)
    combined["goals"] = combined["shots"][combined["shots"].is_goal].drop(columns=["is_goal"])

    return combined
