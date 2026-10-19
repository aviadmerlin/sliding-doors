"""Simulates a match (or part of one), minute by minute, via shots and xG.

Each minute, each team has a small chance of taking a shot. That chance
(the "shot rate") depends on:
  - the team's attack vs. the opponent's defense
  - a home-advantage bump
  - trailing/leading: a trailing team shoots more, and the leading team's
    opponent defends the lead (concedes fewer shots)
  - late-game fatigue: a tiring team concedes more shots after minute 75
  - a red card: ten_men_resilience weakens both the shot rate a team
    generates and the rate it concedes

Once a shot happens, it's classified as a penalty, a "big chance", or a
normal shot — each with its own xG range — and it's a goal if a random
draw lands below that xG.

The engine can start from any game state (minute, score, red cards), so
later we can re-simulate a match from a specific moment with one thing
changed.
"""

import numpy as np

from src.league import Club

BASE_SHOT_RATE_PER_90 = 11.0  # expected shots per team per 90 min, at attack = defense = 1.0
HOME_ADVANTAGE = 1.1
RED_CARD_CHANCE_PER_MINUTE = 0.0007  # ~6% chance of a red card per team per match

PENALTY_SHOT_FRACTION = 0.011  # share of all shots that are penalties (~0.25/match combined)
BIG_CHANCE_FRACTION = 0.12     # share of non-penalty shots that are "big chances"

PENALTY_XG_MEAN = 0.76
BIG_CHANCE_XG_RANGE = (0.15, 0.35)
NORMAL_XG_RANGE = (0.02, 0.14)


def simulate_match(
    home: Club, away: Club, rng: np.random.Generator,
    start_minute: int = 0, home_goals: int = 0, away_goals: int = 0,
    home_red_card_minute: int | None = None, away_red_card_minute: int | None = None,
) -> dict:
    """Simulates from start_minute+1 to 90, given the state at start_minute.
    Defaults simulate a full match from kickoff."""
    red_card_minute = {home.name: home_red_card_minute, away.name: away_red_card_minute}
    shots = []
    red_card_events = []

    for minute in range(start_minute + 1, 91):
        for team in (home, away):
            if red_card_minute[team.name] is None and rng.random() < RED_CARD_CHANCE_PER_MINUTE:
                red_card_minute[team.name] = minute
                red_card_events.append({"minute": minute, "team": team.name})

        home_has_card = red_card_minute[home.name] is not None
        away_has_card = red_card_minute[away.name] is not None

        shot = _maybe_shot(rng, home, away, home_goals, away_goals, minute, True,
                            home_has_card, away_has_card)
        if shot is not None:
            shots.append({"minute": minute, "team": home.name,
                          "home_score": home_goals, "away_score": away_goals, **shot})
            if shot["is_goal"]:
                home_goals += 1

        shot = _maybe_shot(rng, away, home, away_goals, home_goals, minute, False,
                            away_has_card, home_has_card)
        if shot is not None:
            shots.append({"minute": minute, "team": away.name,
                          "home_score": home_goals, "away_score": away_goals, **shot})
            if shot["is_goal"]:
                away_goals += 1

    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "shots": shots,
        "red_card_events": red_card_events,
        "home_red_card_minute": red_card_minute[home.name],
        "away_red_card_minute": red_card_minute[away.name],
    }


def _maybe_shot(rng: np.random.Generator, team: Club, opponent: Club, own_score: int,
                 opp_score: int, minute: int, is_home: bool,
                 team_has_red_card: bool, opp_has_red_card: bool) -> dict | None:
    rate = _shot_rate(team, opponent, own_score, opp_score, minute, is_home,
                       team_has_red_card, opp_has_red_card)
    if rng.random() >= rate:
        return None
    xg, is_penalty, is_big_chance = _draw_shot(rng)
    return {"xg": xg, "is_penalty": is_penalty, "is_big_chance": is_big_chance,
            "is_goal": rng.random() < xg}


def _shot_rate(team: Club, opponent: Club, own_score: int, opp_score: int, minute: int,
                is_home: bool, team_has_red_card: bool, opp_has_red_card: bool) -> float:
    attack = team.attack
    defense = opponent.defense

    if own_score < opp_score:  # team is trailing
        attack *= team.comeback_factor          # team pushes harder
        defense *= opponent.protect_lead_factor  # opponent defends the lead

    if minute > 75:
        defense *= opponent.fatigue_factor  # opponent's legs tire late

    if team_has_red_card:
        attack *= team.ten_men_resilience
    if opp_has_red_card:
        defense *= opponent.ten_men_resilience

    rate = (BASE_SHOT_RATE_PER_90 / 90) * attack / defense
    if is_home:
        rate *= HOME_ADVANTAGE
    return rate


def _draw_shot(rng: np.random.Generator) -> tuple[float, bool, bool]:
    """Picks a shot type and its xG: penalty, big chance, or normal shot."""
    if rng.random() < PENALTY_SHOT_FRACTION:
        xg = float(np.clip(rng.normal(PENALTY_XG_MEAN, 0.03), 0.55, 0.9))
        return xg, True, False
    if rng.random() < BIG_CHANCE_FRACTION:
        xg = float(rng.uniform(*BIG_CHANCE_XG_RANGE))
        return xg, False, True
    xg = float(rng.uniform(*NORMAL_XG_RANGE))
    return xg, False, False
